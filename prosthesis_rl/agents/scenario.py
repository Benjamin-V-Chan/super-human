"""Scenario agent: ADL problem identification -> a trainable task *scene*.

This is the piece that ends the "put the arm at a random floating point" busywork.
Given the perception/Gemini problem (the specific action + the ADL task), it
decides:

  * **posture**  — where the body/shoulder sits (seated, leaning to a table, or
    crouched/"bent down" so a floor object is within reach);
  * **objects**  — what is in the scene (a shoe on the floor, a bottle on a
    table, a drawer), each with a Gizmo prompt so the real geometry can be baked;
  * **waypoints** — the points the hand must actually reach to *do the task*
    (down by the laces, at the cap, on the handle) — these become the RL targets.

Two paths, mirroring `agents.requirements` / `agents.perception`:

  * **Deterministic ADL library** (primary, always available): a hand-authored set
    of reachable scenarios keyed by the action words. The "tie a shoe -> crouch and
    reach the laces near the floor" example works here with no key and no network.
  * **LLM** (optional, when a valid key is present): OpenAI (`OPENAI_API_KEY`) or
    Gemini (`GOOGLE_API_KEY`/`GEMINI_API_KEY`) proposes a novel scene for an
    action the library doesn't cover. Every LLM waypoint is clamped onto the
    arm's reachable sphere, so a hallucinated out-of-reach goal can't break
    training.

    agent = ScenarioAgent()
    scenario = agent.derive(problem_spec)        # ScenarioSpec
    # -> rl.scenario_env.ScenarioReachEnv(design, scenario=scenario)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

from prosthesis_rl.contracts import (
    POSTURE_MOUNTS,
    ProblemSpec,
    SceneObject,
    ScenarioSpec,
    TaskWaypoint,
)
from prosthesis_rl.contracts.scenario import DEFAULT_REACH_M, clamp_to_reach

OPENAI_MODEL = os.environ.get("SCENARIO_MODEL", "gpt-4.1-mini")
GEMINI_MODEL = os.environ.get("SCENARIO_GEMINI_MODEL", "gemini-2.5-flash")


# --------------------------------------------------------------------------- #
# Deterministic ADL scenario library.                                         #
#                                                                             #
# Each entry is keyed by action words and returns a posture + objects +       #
# waypoints. Positions are world metres (+y forward, +z up). They are authored #
# to sit inside the arm's reach for the chosen posture, and clamped again at   #
# build time so they stay reachable even if the default reach changes.        #
# --------------------------------------------------------------------------- #

def _wp(name, pos, *, w=1.0, tol=0.05, dwell=0.0) -> dict[str, Any]:
    return {"name": name, "pos": pos, "weight": w, "tolerance_m": tol, "dwell_s": dwell}


def _obj(name, prompt, pos, *, rgba, half) -> dict[str, Any]:
    return {"name": name, "prompt": prompt, "pos": pos, "rgba": rgba, "fallback_half": half}


# Ordered: the first entry whose keyword appears in the action/task text wins.
_LIBRARY: list[dict[str, Any]] = [
    {
        "key": "tie_shoe",
        "keywords": ["shoe", "lace", "shoelace", "tie", "sneaker", "boot", "footwear"],
        "posture": "floor_reach",
        "description": "Crouch and bring the hand down to the shoelaces on the floor in front of the foot.",
        "success": "hand reaches the laces near the floor (within 4 cm)",
        "objects": [
            # Off to the prosthesis side (-x) so the hand reaches down its open
            # side, not through the wearer's lap/thigh (the body sits at +x).
            _obj("shoe", "a single sneaker / running shoe with laces, sitting on the floor",
                 (-0.18, 0.28, 0.07), rgba=(0.55, 0.35, 0.22, 1.0), half=(0.05, 0.12, 0.045)),
        ],
        "waypoints": [
            _wp("approach", (-0.14, 0.20, 0.34), w=1.0),
            _wp("laces", (-0.18, 0.26, 0.16), w=3.0, tol=0.04, dwell=0.4),
        ],
    },
    {
        "key": "drink_bottle",
        "keywords": ["drink", "bottle", "water", "sip", "hydrate", "cup", "mug"],
        "posture": "table",
        "description": "Reach forward to a bottle standing on the table and grasp it.",
        "success": "hand reaches the bottle on the table (within 4 cm)",
        "objects": [
            _obj("bottle", "a water bottle standing upright on a table",
                 (0.0, 0.18, 0.80), rgba=(0.20, 0.45, 0.80, 1.0), half=(0.035, 0.035, 0.11)),
        ],
        "waypoints": [
            _wp("approach", (0.0, 0.12, 0.88), w=1.0),
            _wp("grasp", (0.0, 0.18, 0.83), w=3.0, tol=0.04, dwell=0.3),
        ],
    },
    {
        # A point-A -> point-B transport task: grasp the charger plug, carry it
        # across the desk, and insert it into the phone's port. The two waypoints
        # are deliberately apart (right side -> left side) so the demo shows the
        # hand *moving an object from one place to another*, not a single reach.
        "key": "plug_charger",
        "keywords": ["charger", "charging", "charge", "plug", "unplug", "cable",
                     "cord", "phone", "usb", "adapter"],
        "posture": "table",
        "description": "Pick up the charger plug on the desk, then carry it across to "
                       "the phone and insert it into the charging port.",
        "success": "hand carries the charger plug from its rest spot to the phone's port (within 3.5 cm)",
        "objects": [
            _obj("phone", "a smartphone lying flat on a desk, charging port facing the user",
                 (-0.10, 0.20, 0.80), rgba=(0.10, 0.10, 0.13, 1.0), half=(0.04, 0.08, 0.008)),
            _obj("charger", "a USB charger cable with a plug connector resting on a desk",
                 (0.10, 0.20, 0.80), rgba=(0.92, 0.92, 0.95, 1.0), half=(0.02, 0.05, 0.012)),
        ],
        # Both points sit in the arm's best desktop reach band (y~0.20, z~0.82) so
        # the hand actually arrives; the phone (insert) is the primary target.
        "waypoints": [
            _wp("grasp_plug", (0.10, 0.20, 0.83), w=2.0, tol=0.05, dwell=0.4),
            _wp("insert_port", (-0.10, 0.20, 0.82), w=3.0, tol=0.05, dwell=0.5),
        ],
    },
    {
        "key": "open_drawer",
        "keywords": ["drawer", "cabinet", "pull", "slide open", "open drawer"],
        "posture": "seated",
        "description": "Reach the drawer handle at waist height and pull it open.",
        "success": "hand reaches the drawer handle (within 4 cm)",
        "objects": [
            _obj("drawer", "a kitchen drawer that slides open on a prismatic joint",
                 (0.0, 0.20, 0.80), rgba=(0.45, 0.32, 0.24, 1.0), half=(0.18, 0.12, 0.10)),
        ],
        "waypoints": [
            _wp("handle", (0.0, 0.10, 0.88), w=3.0, tol=0.04, dwell=0.3),
            _wp("pull", (0.0, 0.00, 0.88), w=1.0, dwell=0.2),
        ],
    },
    {
        "key": "open_jar",
        "keywords": ["jar", "cap", "lid", "twist", "screw", "unscrew", "open container"],
        "posture": "table",
        "description": "Reach the cap on top of a jar on the table to twist it open.",
        "success": "hand reaches the jar cap (within 4 cm)",
        "objects": [
            _obj("jar", "a glass jar with a screw-on lid on a prismatic/revolute cap joint",
                 (0.0, 0.16, 0.80), rgba=(0.60, 0.72, 0.55, 1.0), half=(0.05, 0.05, 0.07)),
        ],
        "waypoints": [
            _wp("cap", (0.0, 0.16, 0.86), w=3.0, tol=0.04, dwell=0.4),
        ],
    },
    {
        "key": "press_switch",
        "keywords": ["switch", "button", "light", "press", "doorbell", "intercom", "elevator"],
        "posture": "seated",
        "description": "Raise the hand to a wall switch / button and press it.",
        "success": "hand reaches the switch on the wall (within 4 cm)",
        "objects": [
            _obj("switch", "a wall light switch plate",
                 (0.0, 0.12, 1.12), rgba=(0.92, 0.92, 0.92, 1.0), half=(0.04, 0.01, 0.06)),
        ],
        "waypoints": [
            _wp("press", (0.0, 0.10, 1.10), w=3.0, tol=0.04, dwell=0.3),
        ],
    },
    {
        "key": "open_door",
        "keywords": ["door", "knob", "handle", "doorknob", "lever"],
        "posture": "seated",
        "description": "Reach the door knob in front and turn it.",
        "success": "hand reaches the door knob (within 4 cm)",
        "objects": [
            _obj("doorknob", "a round door knob mounted on a door, on a revolute joint",
                 (0.0, 0.12, 0.93), rgba=(0.80, 0.70, 0.30, 1.0), half=(0.04, 0.04, 0.04)),
        ],
        "waypoints": [
            _wp("knob", (0.0, 0.12, 0.93), w=3.0, tol=0.04, dwell=0.3),
        ],
    },
    {
        "key": "pick_from_floor",
        "keywords": ["floor", "pick up", "drop", "bend down", "ground", "reach down"],
        "posture": "floor_reach",
        "description": "Crouch and pick an object up off the floor.",
        "success": "hand reaches the object on the floor (within 4 cm)",
        "objects": [
            _obj("object", "a small box / parcel resting on the floor",
                 (-0.18, 0.28, 0.07), rgba=(0.75, 0.55, 0.30, 1.0), half=(0.06, 0.06, 0.06)),
        ],
        "waypoints": [
            _wp("approach", (-0.14, 0.20, 0.34), w=1.0),
            _wp("grasp", (-0.18, 0.26, 0.16), w=3.0, tol=0.04, dwell=0.4),
        ],
    },
    {
        "key": "feeding",
        "keywords": ["eat", "feed", "spoon", "fork", "food", "plate", "bowl"],
        "posture": "table",
        "description": "Reach a bowl on the table, then bring the hand toward the mouth.",
        "success": "hand reaches the bowl, then the mouth zone (within 4 cm)",
        "objects": [
            _obj("bowl", "a cereal bowl on a table",
                 (0.0, 0.18, 0.79), rgba=(0.85, 0.85, 0.88, 1.0), half=(0.07, 0.07, 0.04)),
        ],
        "waypoints": [
            _wp("scoop", (0.0, 0.18, 0.84), w=2.0, tol=0.04, dwell=0.3),
            _wp("mouth", (0.0, -0.06, 1.02), w=2.0, tol=0.05, dwell=0.3),
        ],
    },
]

_DEFAULT_ENTRY: dict[str, Any] = {
    "key": "reach_target",
    "keywords": [],
    "posture": "seated",
    "description": "Reach forward to a target object in front of the body.",
    "success": "hand reaches the target object in front (within 5 cm)",
    "objects": [
        _obj("target", "a small graspable object in front of the person",
             (0.0, 0.16, 0.92), rgba=(0.20, 0.80, 0.30, 1.0), half=(0.04, 0.04, 0.04)),
    ],
    "waypoints": [
        _wp("reach", (0.0, 0.16, 0.92), w=2.0, tol=0.05, dwell=0.2),
    ],
}


def _match_entry(text: str) -> dict[str, Any]:
    """First library entry whose keyword appears in `text`.

    Single-word keywords match whole words only (so "plate" doesn't fire on
    "contemplate"); multi-word keywords ("slide open") match as a phrase.
    """
    t = (text or "").lower()
    words = set(re.findall(r"[a-z]+", t))
    for entry in _LIBRARY:
        for kw in entry["keywords"]:
            if (" " in kw and kw in t) or (" " not in kw and kw in words):
                return entry
    return _DEFAULT_ENTRY


def _entry_to_scenario(entry: dict[str, Any], *, task_id: str, action: str,
                       reach: float, source: str) -> ScenarioSpec:
    posture = entry.get("posture", "seated")
    mount = POSTURE_MOUNTS.get(posture, POSTURE_MOUNTS["seated"])
    objects = [SceneObject.from_dict(o) for o in entry["objects"]]
    waypoints: list[TaskWaypoint] = []
    for w in entry["waypoints"]:
        wp = TaskWaypoint.from_dict(w)
        wp.pos = clamp_to_reach(mount, wp.pos, reach)   # guarantee reachability
        waypoints.append(wp)
    return ScenarioSpec(
        task_id=task_id or entry["key"],
        primary_action=action,
        description=entry["description"],
        posture=posture,
        mount_pos=mount,
        objects=objects,
        waypoints=waypoints,
        success_condition=entry.get("success", ""),
        source=source,
    )


# --------------------------------------------------------------------------- #
# Agent                                                                        #
# --------------------------------------------------------------------------- #
class ScenarioAgent:
    """Problem identification -> ScenarioSpec (library, or LLM for novel actions)."""

    def __init__(self, *, reach: float = DEFAULT_REACH_M, use_llm: bool = True) -> None:
        self.reach = float(reach)
        self.use_llm = use_llm

    # ---- public API ------------------------------------------------------- #
    def derive(self, problem: ProblemSpec, *, prefer_llm: bool = False) -> ScenarioSpec:
        """Return a trainable ScenarioSpec for the problem's primary action/task."""
        action = problem.primary_action or ""
        task_id = ""
        if problem.tasks:
            task_id = str(problem.tasks[0].get("id") or problem.tasks[0].get("name") or "")
        text = " ".join([action, task_id, *(str(t.get("name", "")) for t in problem.tasks)])

        # The library covers the common ADLs and is always reachable; only reach
        # for the LLM when asked, or when nothing in the library matches the action.
        matched = _match_entry(text)
        is_default = matched is _DEFAULT_ENTRY
        if self.use_llm and (prefer_llm or is_default):
            llm = self._derive_llm(action, task_id, text)
            if llm is not None:
                return llm
        return _entry_to_scenario(matched, task_id=task_id, action=action,
                                  reach=self.reach, source=f"library:{matched['key']}")

    def for_action(self, action: str, *, task_id: str = "", prefer_llm: bool = False) -> ScenarioSpec:
        """Convenience: build a scenario directly from a free-text action."""
        return self.derive(ProblemSpec(primary_action=action,
                                       tasks=[{"id": task_id}] if task_id else []),
                            prefer_llm=prefer_llm)

    # ---- LLM path --------------------------------------------------------- #
    @property
    def llm_available(self) -> bool:
        return self._openai_key() is not None or self._gemini_key() is not None

    @staticmethod
    def _openai_key() -> str | None:
        return os.environ.get("OPENAI_API_KEY")

    @staticmethod
    def _gemini_key() -> str | None:
        return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    def _derive_llm(self, action: str, task_id: str, text: str) -> ScenarioSpec | None:
        """Try OpenAI then Gemini; return None on any failure (caller falls back)."""
        for call in (self._call_openai, self._call_gemini):
            try:
                raw = call(action, task_id)
            except Exception:  # noqa: BLE001 - never break the loop on a provider error
                raw = None
            if not raw:
                continue
            scenario = self._scenario_from_llm(raw, action=action, task_id=task_id)
            if scenario is not None:
                return scenario
        return None

    def _prompt(self, action: str, task_id: str) -> str:
        postures = ", ".join(sorted(POSTURE_MOUNTS))
        return (
            "You are a rehabilitation scenario designer placing a shoulder-mounted "
            "assistive arm into a physics simulation so it can practise one Activity "
            "of Daily Living. Given the action, return ONLY JSON describing the scene "
            "and the points the hand must reach to do it.\n\n"
            f"ACTION: {action!r}\nTASK_ID: {task_id!r}\n\n"
            f"Choose a posture from: {postures}. Use 'floor_reach' or 'bent_forward' "
            "for anything low/near the ground (e.g. tying a shoe), 'table'/'lap' for "
            "surface tasks, 'seated' for upright reaches.\n"
            "Coordinates are world metres: +y is forward in front of the body, +z is "
            "up. The shoulder sits roughly at the posture's mount; keep every waypoint "
            f"within ~{self.reach * 0.9:.2f} m of it so the arm can reach.\n"
            "Return JSON exactly:\n"
            "{\n"
            '  "posture": "floor_reach",\n'
            '  "description": "one sentence on what the hand does",\n'
            '  "success_condition": "measurable success",\n'
            '  "objects": [{"name":"shoe","prompt":"a sneaker with laces on the floor",'
            '"pos":[0.0,0.40,0.06],"rgba":[0.55,0.35,0.22,1.0],"fallback_half":[0.05,0.12,0.045]}],\n'
            '  "waypoints": [\n'
            '    {"name":"approach","pos":[0.0,0.34,0.24],"weight":1.0,"tolerance_m":0.05},\n'
            '    {"name":"laces","pos":[0.0,0.40,0.14],"weight":3.0,"tolerance_m":0.04,"dwell_s":0.4}\n'
            "  ]\n"
            "}\n"
            "Give the terminal/grasp waypoint the highest weight. 1-3 objects, 1-4 waypoints."
        )

    def _call_openai(self, action: str, task_id: str) -> dict[str, Any] | None:
        key = self._openai_key()
        if not key:
            return None
        from openai import OpenAI

        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": self._prompt(action, task_id)}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return _extract_json(resp.choices[0].message.content or "")

    def _call_gemini(self, action: str, task_id: str) -> dict[str, Any] | None:
        key = self._gemini_key()
        if not key:
            return None
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[self._prompt(action, task_id)],
            config=types.GenerateContentConfig(temperature=0.2),
        )
        return _extract_json(resp.text or "")

    def _scenario_from_llm(self, raw: dict[str, Any], *, action: str,
                           task_id: str) -> ScenarioSpec | None:
        posture = str(raw.get("posture", "seated"))
        if posture not in POSTURE_MOUNTS:
            posture = "seated"
        mount = POSTURE_MOUNTS[posture]
        try:
            objects = [SceneObject.from_dict(o) for o in raw.get("objects", [])]
            waypoints = []
            for w in raw.get("waypoints", []):
                wp = TaskWaypoint.from_dict(w)
                wp.pos = clamp_to_reach(mount, wp.pos, self.reach)
                waypoints.append(wp)
        except (KeyError, TypeError, ValueError):
            return None
        if not waypoints:
            return None
        model_name = OPENAI_MODEL if self._openai_key() else GEMINI_MODEL
        return ScenarioSpec(
            task_id=task_id or "llm_task",
            primary_action=action,
            description=str(raw.get("description", "")),
            posture=posture,
            mount_pos=mount,
            objects=objects,
            waypoints=waypoints,
            success_condition=str(raw.get("success_condition", "")),
            source=f"llm:{model_name}",
        )


def _extract_json(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


# Posture -> a short phrase describing the setting/surface Gizmo should build the
# whole scene around, so a terse action ("drink") becomes a full environment.
_POSTURE_SETTING = {
    "seated": "in front of a seated person",
    "table": "on a small table at sitting height in front of a seated person",
    "lap": "on a low surface in the lap of a seated person",
    "bent_forward": "on the floor just in front of a seated person leaning forward",
    "floor_reach": "on the floor directly in front of a crouched person",
}


def scene_prompt(scenario: ScenarioSpec) -> str:
    """Compose a full Gizmo *scene* prompt from a ScenarioSpec.

    A raw action ("drinking from a bottle") is too terse to generate a good
    environment, so this expands it into the setting (from the posture) plus the
    concrete objects (each SceneObject's own Gizmo prompt) and asks for a compact,
    uncluttered layout — fewer assets generate faster and the MJCF export loads
    cleanly. This drives only the *geometry* Gizmo builds; the scenario's authored
    waypoints remain the RL reach targets (so reachability stays guaranteed).

    Used by the live backend: ``bake_scene(scene_prompt(scenario))``. Caching is
    keyed on this composed string, so any two actions that resolve to the same
    scenario share one baked scene.
    """
    setting = _POSTURE_SETTING.get(scenario.posture, _POSTURE_SETTING["seated"])
    objs = [o.prompt or o.name for o in scenario.objects] or ["a small graspable object"]
    action = scenario.primary_action or scenario.task_id or "an everyday task"
    return (
        f"A simple, uncluttered indoor activity-of-daily-living scene for {action}. "
        f"Place {setting}: {'; '.join(objs)}. "
        f"Include only these objects and the surface or furniture they rest on — no "
        f"extra clutter. Keep the layout compact and toward the front so the items "
        f"sit within a seated person's arm reach."
    )


# Exposed for the orchestrator / tests: the catalogue of library task keys.
def library_keys() -> list[str]:
    return [e["key"] for e in _LIBRARY]


def library_scenarios(*, reach: float = DEFAULT_REACH_M) -> list[ScenarioSpec]:
    """Every built-in ADL as a ready-to-train ScenarioSpec — the stress-test battery.

    Used by the multi-scenario trainer and the stress-test harness to run the arm
    through the full set of daily tasks (tie shoe, drink, drawer, …) rather than
    one action at a time.
    """
    return [
        _entry_to_scenario(entry, task_id=entry["key"], action=entry["key"],
                           reach=reach, source=f"library:{entry['key']}")
        for entry in _LIBRARY
    ]
