"""Multi-clip problem synthesis — parallel perception + specialized sub-agents.

Takes N video clips, runs PerceptionAgent on each in parallel, then runs three
specialized sub-agents to extract functional limitations, and finally synthesizes
a unified set of design requirements via LLM (or deterministic fallback).

    agent = ProblemSynthesisAgent()
    observations, requirements = agent.run(["clip1.mp4", "clip2.mp4"])
"""

from __future__ import annotations

import json
import os
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import field
from pathlib import Path
from typing import Any

from prosthesis_rl.contracts import (
    ClipObservation,
    IdentifiedProblem,
    ProblemSpec,
    UnifiedRequirements,
)

__all__ = [
    "ProblemSynthesisAgent",
    "ReachEnvelopeAgent",
    "GraspPatternAgent",
    "CompensationAgent",
]


# ── Reach Envelope Agent ───────────────────────────────────────────────────────

class ReachEnvelopeAgent:
    """Flags ROM and reach-envelope limitations from a ProblemSpec.

    Checks whether the observed range-of-motion requirements are unusual or
    demanding, and proposes design solutions (wider joint ranges, longer links).
    """

    THRESHOLDS = {
        "shoulder_flexion": 100.0,   # deg — above this needs extended shoulder range
        "elbow_flexion": 125.0,      # deg — above this needs >130° elbow
        "wrist_rotation": 55.0,      # deg — above this needs powered wrist
        "reach_m": 0.52,             # m — above this needs long-reach configuration
    }

    def analyze(self, problem: ProblemSpec, clip_path: str = "") -> list[IdentifiedProblem]:
        problems: list[IdentifiedProblem] = []
        rom = problem.constraints.rom or {}
        anthro = problem.residual_anthropometrics or {}
        tasks = [t.get("id", "unknown") for t in problem.tasks]

        # Shoulder flexion
        sh = float(rom.get("shoulder_flexion", 0.0))
        if sh > self.THRESHOLDS["shoulder_flexion"]:
            problems.append(IdentifiedProblem(
                problem_id="high_shoulder_flexion_demand",
                description=f"Task requires {sh:.0f}° shoulder flexion — above standard 90° prosthetic range.",
                severity=min(1.0, (sh - 90) / 60),
                affected_tasks=tasks,
                proposed_solutions=[
                    "Extend shoulder joint range to ≥125°",
                    "Add powered shoulder assist actuator",
                    "Use extended-reach socket orientation",
                ],
            ))

        # Elbow flexion
        el = float(rom.get("elbow_flexion", 0.0))
        if el > self.THRESHOLDS["elbow_flexion"]:
            problems.append(IdentifiedProblem(
                problem_id="high_elbow_flexion_demand",
                description=f"Task requires {el:.0f}° elbow flexion — standard elbow units cap at 135°.",
                severity=min(1.0, (el - 100) / 60),
                affected_tasks=tasks,
                proposed_solutions=[
                    f"Set elbow joint limit to {min(160, el + 10):.0f}°",
                    "Use polycentric (4-bar) elbow mechanism for deeper flexion",
                    "Add powered elbow actuator with extended ROM",
                ],
            ))

        # Wrist rotation
        wr = float(rom.get("wrist_rotation", 0.0))
        if wr > self.THRESHOLDS["wrist_rotation"]:
            problems.append(IdentifiedProblem(
                problem_id="high_wrist_rotation_demand",
                description=f"Task requires ±{wr:.0f}° wrist rotation — passive wrist insufficient.",
                severity=min(1.0, wr / 80),
                affected_tasks=tasks,
                proposed_solutions=[
                    "Add powered wrist rotation unit",
                    "Use friction-lock wrist with ±80° range",
                    "Add wrist flexion/extension unit (6-DOF wrist)",
                ],
            ))

        # Reach envelope
        arm_len = float(anthro.get("upper_arm_len", 0.30)) + float(anthro.get("forearm_len", 0.26))
        if arm_len < 0.52 and float(rom.get("reach_m", 0.0)) > 0.48:
            problems.append(IdentifiedProblem(
                problem_id="insufficient_reach_envelope",
                description=f"Residual arm ({arm_len*100:.0f} cm) may not reach task targets without extension.",
                severity=0.7,
                affected_tasks=tasks,
                proposed_solutions=[
                    "Add forearm extension pylon (+4 cm)",
                    "Use angled elbow unit to maximize functional reach",
                    "Increase forearm length to 0.30 m",
                ],
            ))

        return problems


# ── Grasp Pattern Agent ────────────────────────────────────────────────────────

class GraspPatternAgent:
    """Maps observed actions + grip capacity → terminal device recommendation."""

    # action keywords → terminal device type
    _TD_MAP: list[tuple[list[str], str]] = [
        (["hook", "hang", "carry bag", "pull", "drag"], "vo_hook"),
        (["pinch", "pick", "pen", "key", "small", "thin", "writing", "typing"], "pinch_prehensor"),
        (["grasp", "bottle", "cup", "cylinder", "glass", "can", "jar", "drink"], "pinch_prehensor"),
        (["hand", "shake", "push door", "natural", "gesture", "social"], "passive_hand"),
        (["myo", "electric", "active hand", "powered", "multi-grip"], "myoelectric_hand"),
    ]

    _TD_INFO = {
        "vo_hook": {
            "grip_patterns": ["hook", "cylindrical"],
            "max_grip_force_n": 45.0,
            "weight_g": 140.0,
            "active_dof": 1,
        },
        "pinch_prehensor": {
            "grip_patterns": ["pinch", "lateral_pinch", "cylindrical"],
            "max_grip_force_n": 30.0,
            "weight_g": 180.0,
            "active_dof": 1,
        },
        "passive_hand": {
            "grip_patterns": ["passive", "static"],
            "max_grip_force_n": 0.0,
            "weight_g": 400.0,
            "active_dof": 0,
        },
        "myoelectric_hand": {
            "grip_patterns": ["power", "pinch", "lateral", "tripod", "hook"],
            "max_grip_force_n": 80.0,
            "weight_g": 520.0,
            "active_dof": 1,
        },
    }

    def analyze(self, problem: ProblemSpec, clip_path: str = "") -> list[IdentifiedProblem]:
        problems: list[IdentifiedProblem] = []
        action = (problem.primary_action or "").lower()
        grip = float(problem.constraints.grip_capacity or 0.4)
        tasks = [t.get("id", "unknown") for t in problem.tasks]

        td_type = "pinch_prehensor"
        for keywords, td in self._TD_MAP:
            if any(k in action for k in keywords):
                td_type = td
                break

        info = self._TD_INFO[td_type]

        # Flag if grip capacity is very low
        if grip < 0.3:
            problems.append(IdentifiedProblem(
                problem_id="low_grip_capacity",
                description=f"Residual grip capacity {grip:.0%} — standard terminal devices may exceed user control ability.",
                severity=0.8,
                affected_tasks=tasks,
                proposed_solutions=[
                    "Use body-powered terminal device (lower activation force)",
                    "Set grip force target to ≤ 15 N",
                    "Use voluntary-closing hook (proportional control)",
                ],
            ))
        elif grip > 0.75:
            problems.append(IdentifiedProblem(
                problem_id="high_grip_capacity",
                description=f"Strong residual grip {grip:.0%} — myoelectric control is viable.",
                severity=0.1,
                affected_tasks=tasks,
                proposed_solutions=[
                    "Myoelectric hand recommended — sufficient EMG signal strength",
                    "Consider 2-site EMG control for multi-grip switching",
                ],
            ))

        # Always emit terminal device recommendation as a finding
        problems.append(IdentifiedProblem(
            problem_id=f"terminal_device_recommendation_{td_type}",
            description=f"Recommended terminal device: {td_type} for '{action}'.",
            severity=0.0,
            affected_tasks=tasks,
            proposed_solutions=[
                f"Install {td_type.replace('_', ' ')} terminal device",
                f"Max grip force: {info['max_grip_force_n']:.0f} N",
                f"Weight budget: {info['weight_g']:.0f} g for terminal device",
            ],
        ))

        return problems

    def recommend_td(self, problem: ProblemSpec) -> str:
        action = (problem.primary_action or "").lower()
        for keywords, td in self._TD_MAP:
            if any(k in action for k in keywords):
                return td
        return "pinch_prehensor"


# ── Compensation Agent ─────────────────────────────────────────────────────────

class CompensationAgent:
    """Detects compensatory movement patterns and translates them to design needs."""

    _COMPENSATION_SIGNALS = [
        ("trunk lean", "excessive_trunk_compensation",
         "User leans trunk to reach — insufficient shoulder ROM or arm length.",
         ["Extend forearm pylon by 3–5 cm", "Add 10° shoulder abduction ROM"]),
        ("head tilt", "excessive_trunk_compensation",
         "Head tilt observed — compensating for reach limitation.",
         ["Increase shoulder flexion range", "Add lateral reach via abduction joint"]),
        ("one-handed", "bilateral_task_compensation",
         "Performing bilateral task one-handed — prosthesis needs strong assist function.",
         ["Prioritize prehension over cosmesis", "Use high-force terminal device ≥40 N"]),
        ("compensat", "general_compensation",
         "General compensatory movements detected — prosthesis undersupporting task.",
         ["Increase DOF count for more natural movement", "Add wrist flexion unit"]),
        ("awkward", "posture_compensation",
         "Awkward posture during task — poor reach geometry.",
         ["Optimize joint angle presets for this task", "Add pre-positioned wrist"]),
        ("slow", "speed_compensation",
         "Task performed slowly — control or ROM limitation.",
         ["Increase joint velocity limits in simulation", "Reduce joint stiffness"]),
    ]

    def analyze(self, problem: ProblemSpec, clip_path: str = "") -> list[IdentifiedProblem]:
        problems: list[IdentifiedProblem] = []
        pain_points = " ".join(
            (problem.tasks[i].get("pain_points") or []) if i < len(problem.tasks) else []
            for i in range(len(problem.tasks))
        ).lower()

        # Also check task pain_points lists
        all_pain: list[str] = []
        for t in problem.tasks:
            all_pain.extend(t.get("pain_points") or [])
        pain_text = " ".join(all_pain).lower()

        tasks = [t.get("id", "unknown") for t in problem.tasks]

        seen_ids: set[str] = set()
        for signal, prob_id, desc, solutions in self._COMPENSATION_SIGNALS:
            if signal in pain_text and prob_id not in seen_ids:
                seen_ids.add(prob_id)
                problems.append(IdentifiedProblem(
                    problem_id=prob_id,
                    description=desc,
                    severity=0.65,
                    affected_tasks=tasks,
                    proposed_solutions=solutions,
                ))

        return problems


# ── Problem Synthesis Agent ────────────────────────────────────────────────────

class ProblemSynthesisAgent:
    """Runs parallel perception on N clips → unified design requirements.

    Runs PerceptionAgent on each clip concurrently, then runs ReachEnvelopeAgent,
    GraspPatternAgent, and CompensationAgent on each result. Finally synthesizes
    a UnifiedRequirements via LLM or deterministic merge.
    """

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max_workers
        self._reach_agent = ReachEnvelopeAgent()
        self._grasp_agent = GraspPatternAgent()
        self._comp_agent = CompensationAgent()

    def run(
        self,
        clip_paths: list[str | Path],
        emit_cb=None,
    ) -> tuple[list[ClipObservation], UnifiedRequirements]:
        """Run full multi-clip problem synthesis.

        Returns (observations, unified_requirements).
        """
        from prosthesis_rl.agents.perception import PerceptionAgent

        clip_paths = [str(p) for p in clip_paths]
        if not clip_paths:
            raise ValueError("At least one clip path is required")

        # Parallel perception
        observations: list[ClipObservation] = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(clip_paths))) as pool:
            future_to_clip = {
                pool.submit(self._analyze_clip, clip, PerceptionAgent()): clip
                for clip in clip_paths
            }
            for future in as_completed(future_to_clip):
                clip = future_to_clip[future]
                try:
                    obs = future.result()
                    observations.append(obs)
                    if emit_cb:
                        emit_cb("clip_done", {"clip": clip, "action": obs.problem.primary_action})
                except Exception as exc:
                    if emit_cb:
                        emit_cb("clip_error", {"clip": clip, "error": str(exc)})

        # Sort by clip path for deterministic order
        observations.sort(key=lambda o: o.clip_path)

        # Synthesize unified requirements
        requirements = self._synthesize(observations)
        return observations, requirements

    def _analyze_clip(self, clip_path: str, perception) -> ClipObservation:
        problem = perception.infer_problem(clip_path)
        identified: list[IdentifiedProblem] = []
        identified.extend(self._reach_agent.analyze(problem, clip_path))
        identified.extend(self._grasp_agent.analyze(problem, clip_path))
        identified.extend(self._comp_agent.analyze(problem, clip_path))
        return ClipObservation(
            clip_path=clip_path,
            problem=problem,
            identified_problems=identified,
        )

    def _synthesize(self, observations: list[ClipObservation]) -> UnifiedRequirements:
        """Merge observations → UnifiedRequirements via LLM or deterministic fallback."""
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            try:
                return self._synthesize_llm(observations, api_key)
            except Exception:
                pass
        return self._synthesize_deterministic(observations)

    def _synthesize_llm(
        self, observations: list[ClipObservation], api_key: str
    ) -> UnifiedRequirements:
        import anthropic

        obs_summary = []
        for obs in observations:
            obs_summary.append({
                "clip": obs.clip_path,
                "primary_action": obs.problem.primary_action,
                "affected_side": obs.problem.affected_side,
                "rom": dict(obs.problem.constraints.rom),
                "grip_capacity": obs.problem.constraints.grip_capacity,
                "anthropometrics": dict(obs.problem.residual_anthropometrics or {}),
                "identified_problems": [
                    {"id": p.problem_id, "severity": p.severity, "solutions": p.proposed_solutions}
                    for p in obs.identified_problems
                ],
            })

        prompt = (
            "You are a prosthetics engineer. Given these ADL video clip observations, "
            "synthesize unified design requirements for a prosthetic arm that addresses ALL tasks.\n\n"
            f"Observations:\n{json.dumps(obs_summary, indent=2)}\n\n"
            "Return JSON with these exact keys:\n"
            "{\n"
            '  "rom_targets_deg": {<joint>: <max_degrees>},\n'
            '  "grip_capacity": <float 0-1>,\n'
            '  "design_directives": [<imperative strings>],\n'
            '  "conflicts": [<conflict descriptions>]\n'
            "}"
        )

        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text
        start = text.find("{")
        end = text.rfind("}") + 1
        data = json.loads(text[start:end])

        return self._merge_with_llm_output(observations, data)

    def _merge_with_llm_output(
        self, observations: list[ClipObservation], llm: dict
    ) -> UnifiedRequirements:
        base = self._synthesize_deterministic(observations)
        base.rom_targets_deg.update(llm.get("rom_targets_deg", {}))
        base.design_directives = llm.get("design_directives", base.design_directives)
        base.conflicts = llm.get("conflicts", base.conflicts)
        if llm.get("grip_capacity"):
            base.grip_capacity = float(llm["grip_capacity"])
        return base

    def _synthesize_deterministic(self, observations: list[ClipObservation]) -> UnifiedRequirements:
        """Merge by taking max ROM, max grip, median anthropometrics."""
        all_tasks: list[dict] = []
        rom_union: dict[str, list[float]] = {}
        grips: list[float] = []
        anthro_union: dict[str, list[float]] = {}
        sides: list[str] = []
        actions: list[str] = []

        for obs in observations:
            p = obs.problem
            all_tasks.extend(p.tasks)
            for k, v in (p.constraints.rom or {}).items():
                rom_union.setdefault(k, []).append(float(v))
            if p.constraints.grip_capacity:
                grips.append(float(p.constraints.grip_capacity))
            for k, v in (p.residual_anthropometrics or {}).items():
                anthro_union.setdefault(k, []).append(float(v))
            if p.affected_side:
                sides.append(p.affected_side)
            if p.primary_action:
                actions.append(p.primary_action)

        # Unique tasks by id
        seen_task_ids: set[str] = set()
        unique_tasks: list[dict] = []
        for t in all_tasks:
            if t.get("id") not in seen_task_ids:
                seen_task_ids.add(t.get("id", ""))
                unique_tasks.append(t)

        rom_targets = {k: max(vs) for k, vs in rom_union.items()}
        grip = max(grips) if grips else 0.4
        anthropometrics = {
            k: statistics.median(vs) for k, vs in anthro_union.items()
        }
        affected_side = max(set(sides), key=sides.count) if sides else "right"

        # Generate directives from all identified problems
        all_problems: list[IdentifiedProblem] = []
        for obs in observations:
            all_problems.extend(obs.identified_problems)

        directives: list[str] = []
        conflicts: list[str] = []
        seen_directive_ids: set[str] = set()
        for prob in sorted(all_problems, key=lambda p: p.severity, reverse=True):
            if prob.severity > 0.3 and prob.problem_id not in seen_directive_ids:
                seen_directive_ids.add(prob.problem_id)
                if prob.proposed_solutions:
                    directives.append(prob.proposed_solutions[0])

        # Detect conflicts (e.g., different sides across clips)
        if len(set(sides)) > 1:
            conflicts.append(f"Clips disagree on affected side: {set(sides)}")

        return UnifiedRequirements(
            tasks=unique_tasks,
            rom_targets_deg=rom_targets,
            grip_capacity=grip,
            anthropometrics=anthropometrics,
            design_directives=directives,
            conflicts=conflicts,
            affected_side=affected_side,
            primary_actions=list(dict.fromkeys(actions)),
        )
