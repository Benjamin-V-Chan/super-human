"""Advanced prosthetic design agent — LLM-driven variable-topology design.

Replaces the hardcoded 3-link topology with a configurable 3–7 link chain,
terminal device library, per-component material selection, and seeded stochastic
diversity so successive candidates are genuinely different.

    agent = AdvancedDesignAgent()
    candidates = agent.propose_diverse_candidates(requirements, n=3, seed=0)
    # each candidate: (DesignParams, list[ComponentSpec], TerminalDeviceSpec, str rationale, dict work)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from typing import Any

import numpy as np

from prosthesis_rl.contracts import (
    ComponentSpec,
    DesignParams,
    IdentifiedProblem,
    JointDef,
    LinkDef,
    MechanicalReport,
    TerminalDeviceSpec,
    UnifiedRequirements,
)

__all__ = ["AdvancedDesignAgent", "DesignCandidate"]


# ── Material library ───────────────────────────────────────────────────────────

_MATERIALS: dict[str, dict[str, float]] = {
    "Ti-6Al-4V": {"density": 4430.0, "yield_mpa": 880.0, "cost": 4.0},
    "CFRP-PA12": {"density": 1450.0, "yield_mpa": 210.0, "cost": 3.0},
    "PA12":      {"density": 1010.0, "yield_mpa": 50.0,  "cost": 1.0},
    "PETG":      {"density": 1270.0, "yield_mpa": 45.0,  "cost": 0.8},
}

# Terminal device specs
_TD_LIBRARY: dict[str, dict[str, Any]] = {
    "vo_hook": {
        "grip_patterns": ["hook", "cylindrical"],
        "max_grip_force_n": 45.0,
        "weight_g": 140.0,
        "active_dof": 1,
        "active_joints": [
            JointDef("gripper_open", (0, 1, 0), (0.0, 60.0)),
        ],
    },
    "pinch_prehensor": {
        "grip_patterns": ["pinch", "lateral_pinch", "cylindrical"],
        "max_grip_force_n": 30.0,
        "weight_g": 180.0,
        "active_dof": 1,
        "active_joints": [
            JointDef("pinch", (0, 1, 0), (0.0, 45.0)),
        ],
    },
    "passive_hand": {
        "grip_patterns": ["passive", "static"],
        "max_grip_force_n": 0.0,
        "weight_g": 400.0,
        "active_dof": 0,
        "active_joints": [],
    },
    "myoelectric_hand": {
        "grip_patterns": ["power", "pinch", "lateral", "tripod", "hook"],
        "max_grip_force_n": 80.0,
        "weight_g": 520.0,
        "active_dof": 1,
        "active_joints": [
            JointDef("myo_grip", (0, 1, 0), (0.0, 70.0)),
        ],
    },
}


# ── Topology presets ───────────────────────────────────────────────────────────

def _topology_minimal(upper_m, grip_w, elbow, wrist, mount) -> tuple[LinkDef, ...]:
    """3-link minimal: socket-pylon → elbow-unit → terminal-device."""
    return (
        LinkDef("socket_pylon", upper_m, 0.028,
                joints=(
                    JointDef("shoulder_flex", (0, 1, 0), (-90.0, 120.0)),
                    JointDef("shoulder_abduct", (1, 0, 0), (-60.0, 90.0)),
                )),
        LinkDef("forearm_pylon", 0.22, 0.022,
                joints=(JointDef("elbow", (0, 1, 0), elbow),)),
        LinkDef("terminal_device", 0.06, max(0.015, grip_w / 2),
                joints=(JointDef("wrist", (1, 0, 0), wrist),),
                rgba=(0.85, 0.6, 0.2, 1.0)),
    )


def _topology_standard(upper_m, forearm_m, grip_w, elbow, wrist, mount) -> tuple[LinkDef, ...]:
    """5-link standard: socket → upper-pylon → elbow-unit → forearm-pylon → terminal."""
    return (
        LinkDef("socket", 0.06, 0.032,
                joints=(
                    JointDef("shoulder_flex", (0, 1, 0), (-90.0, 120.0)),
                    JointDef("shoulder_abduct", (1, 0, 0), (-60.0, 90.0)),
                )),
        LinkDef("upper_pylon", upper_m, 0.025, joints=()),
        LinkDef("elbow_unit", 0.04, 0.030,
                joints=(JointDef("elbow", (0, 1, 0), elbow),)),
        LinkDef("forearm_pylon", forearm_m, 0.022, joints=()),
        LinkDef("terminal_device", 0.06, max(0.015, grip_w / 2),
                joints=(JointDef("wrist", (1, 0, 0), wrist),),
                rgba=(0.85, 0.6, 0.2, 1.0)),
    )


def _topology_full(upper_m, forearm_m, grip_w, elbow, wrist, mount) -> tuple[LinkDef, ...]:
    """7-link full: socket → upper-pylon → elbow-unit → forearm-pylon → wrist-flex → wrist-rot → terminal."""
    return (
        LinkDef("socket", 0.06, 0.032,
                joints=(
                    JointDef("shoulder_flex", (0, 1, 0), (-90.0, 120.0)),
                    JointDef("shoulder_abduct", (1, 0, 0), (-60.0, 90.0)),
                )),
        LinkDef("upper_pylon", upper_m, 0.025, joints=()),
        LinkDef("elbow_unit", 0.04, 0.030,
                joints=(JointDef("elbow", (0, 1, 0), elbow),)),
        LinkDef("forearm_pylon", forearm_m, 0.022, joints=()),
        LinkDef("wrist_flex_unit", 0.035, 0.026,
                joints=(JointDef("wrist_flex", (1, 0, 0), (-30.0, 30.0)),)),
        LinkDef("wrist_rot_unit", 0.025, 0.024,
                joints=(JointDef("wrist_rot", (0, 0, 1), (-80.0, 80.0)),)),
        LinkDef("terminal_device", 0.06, max(0.015, grip_w / 2),
                joints=(JointDef("td_grip", (0, 1, 0), (0.0, 60.0)),),
                rgba=(0.85, 0.6, 0.2, 1.0)),
    )


_TOPOLOGIES = {
    "minimal": _topology_minimal,
    "standard": _topology_standard,
    "full": _topology_full,
}


# ── Design Candidate ───────────────────────────────────────────────────────────

class DesignCandidate:
    """One design proposal with full BOM and show-work trace."""

    def __init__(
        self,
        params: DesignParams,
        components: list[ComponentSpec],
        terminal_device: TerminalDeviceSpec,
        rationale: str,
        work: dict[str, Any],
    ) -> None:
        self.params = params
        self.components = components
        self.terminal_device = terminal_device
        self.rationale = rationale
        self.work = work   # {problem_id → solution_chosen}

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": asdict(self.params),
            "components": [asdict(c) for c in self.components],
            "terminal_device": asdict(self.terminal_device),
            "rationale": self.rationale,
            "work": self.work,
        }


# ── Advanced Design Agent ──────────────────────────────────────────────────────

class AdvancedDesignAgent:
    """LLM-driven variable-topology prosthetic design with stochastic fallback.

    Generates genuinely diverse candidates via:
    - LLM (Claude) reasoning over identified problems → design decisions
    - Seeded stochastic parameter sampling when no API key
    - Variable topology selection (minimal / standard / full)
    - Per-component material assignment based on load profile
    """

    def propose_diverse_candidates(
        self,
        requirements: UnifiedRequirements,
        feedback: dict | None = None,
        n: int = 3,
        seed: int = 0,
    ) -> list[DesignCandidate]:
        """Generate n diverse candidates for unified requirements.

        Each candidate uses seed+i to guarantee different parameter samples.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        candidates: list[DesignCandidate] = []

        for i in range(n):
            if api_key and i == 0:
                try:
                    cand = self._propose_llm(requirements, feedback, seed + i)
                    candidates.append(cand)
                    continue
                except Exception:
                    pass
            candidates.append(self._propose_stochastic(requirements, feedback, seed=seed + i))

        return candidates

    def propose_from_feedback(
        self,
        requirements: UnifiedRequirements,
        feedback: dict,
        prev_candidate: DesignCandidate,
        seed: int = 0,
    ) -> DesignCandidate:
        """Refine design based on sim + mechanical feedback."""
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            try:
                return self._propose_llm(requirements, feedback, seed, prev_candidate=prev_candidate)
            except Exception:
                pass
        return self._propose_stochastic(requirements, feedback, seed=seed)

    # ── LLM-driven proposal ───────────────────────────────────────────────────

    def _propose_llm(
        self,
        requirements: UnifiedRequirements,
        feedback: dict | None,
        seed: int,
        prev_candidate: DesignCandidate | None = None,
    ) -> DesignCandidate:
        import anthropic

        problems_summary = self._format_problems(requirements)
        feedback_text = self._format_feedback(feedback) if feedback else "No prior feedback."
        prev_text = ""
        if prev_candidate:
            prev_text = f"\nPrevious design had: upper={prev_candidate.params.upper_arm_len:.2f}m, forearm={prev_candidate.params.forearm_len:.2f}m, IK={feedback.get('ik_success_rate', 0):.0%}"

        prompt = f"""You are a certified prosthetics engineer designing a custom upper-limb prosthesis.

IDENTIFIED PROBLEMS AND REQUIREMENTS:
{problems_summary}

SIMULATION FEEDBACK (if any):
{feedback_text}{prev_text}

Design a prosthetic arm that addresses these problems. Choose:
1. topology: "minimal" (3 links, 4 DOF) | "standard" (5 links, 5 DOF) | "full" (7 links, 7 DOF)
2. upper_arm_len: 0.20–0.38 m (mirror residual limb or extend for reach)
3. forearm_len: 0.16–0.32 m
4. elbow_range: [lo, hi] degrees — must cover ROM target
5. wrist_range: [lo, hi] degrees symmetric
6. stiffness: 0.6–2.0 N·m/rad
7. grip_width: 0.05–0.14 m
8. terminal_device: "vo_hook" | "pinch_prehensor" | "passive_hand" | "myoelectric_hand"
9. For each identified problem, state which solution you chose.

Return ONLY valid JSON:
{{
  "topology": "standard",
  "upper_arm_len": 0.30,
  "forearm_len": 0.26,
  "elbow_range": [0, 130],
  "wrist_range": [-60, 60],
  "stiffness": 1.0,
  "grip_width": 0.08,
  "terminal_device": "pinch_prehensor",
  "rationale": "Chose standard topology because...",
  "work": {{"problem_id": "solution chosen"}}
}}"""

        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text
        start = text.find("{")
        end = text.rfind("}") + 1
        data = json.loads(text[start:end])
        return self._build_from_dict(data, requirements)

    # ── Stochastic proposal ───────────────────────────────────────────────────

    def _propose_stochastic(
        self,
        requirements: UnifiedRequirements,
        feedback: dict | None,
        seed: int,
    ) -> DesignCandidate:
        rng = np.random.default_rng(seed)
        anthro = requirements.anthropometrics
        rom = requirements.rom_targets_deg

        # Sample dimensions centered on anthropometrics with task-driven perturbation
        base_upper = float(anthro.get("upper_arm_len", 0.30))
        base_fore = float(anthro.get("forearm_len", 0.26))
        base_grip = float(anthro.get("grip_span", 0.08))

        # If feedback says success is low, push toward longer links
        delta_bias = 0.0
        if feedback:
            sr = float(feedback.get("ik_success_rate", 0.25))
            if sr < 0.3:
                delta_bias = 0.03  # encourage longer reach

        upper_m = float(np.clip(
            rng.normal(base_upper + delta_bias, 0.04), 0.20, 0.38
        ))
        forearm_m = float(np.clip(
            rng.normal(base_fore + delta_bias, 0.04), 0.16, 0.32
        ))
        grip_w = float(np.clip(rng.normal(base_grip, 0.015), 0.05, 0.14))

        # Joint ranges from ROM targets with stochastic headroom
        elbow_target = float(rom.get("elbow_flexion", 130.0))
        elbow_hi = float(np.clip(
            rng.uniform(max(100.0, elbow_target - 5), min(160.0, elbow_target + 20)),
            100.0, 160.0,
        ))
        wrist_target = float(rom.get("wrist_rotation", 60.0))
        wrist_hi = float(np.clip(
            rng.uniform(max(30.0, wrist_target - 5), min(80.0, wrist_target + 15)),
            30.0, 80.0,
        ))
        stiffness = float(np.clip(
            rng.uniform(0.6, 1.8) * (1.2 - float(requirements.grip_capacity or 0.5)),
            0.6, 2.0,
        ))

        # Topology selection: more complex for higher ROM demands / more tasks
        n_tasks = len(requirements.tasks)
        need_wrist_dof = wrist_target > 55.0
        if need_wrist_dof and n_tasks >= 2:
            topology = rng.choice(["standard", "full"], p=[0.4, 0.6])
        elif n_tasks >= 2:
            topology = rng.choice(["minimal", "standard", "full"], p=[0.1, 0.6, 0.3])
        else:
            topology = rng.choice(["minimal", "standard"], p=[0.3, 0.7])

        # Terminal device
        from prosthesis_rl.agents.problem_synthesis import GraspPatternAgent
        # Use first clip's problem for TD recommendation if available
        from prosthesis_rl.contracts import ProblemSpec, Constraints
        proxy = ProblemSpec(
            primary_action=" ".join(requirements.primary_actions or ["grasp"]),
            constraints=Constraints(grip_capacity=requirements.grip_capacity),
        )
        td_type = GraspPatternAgent().recommend_td(proxy)

        work = self._build_work_map(requirements, {
            "topology": str(topology),
            "upper_arm_len": upper_m,
            "forearm_len": forearm_m,
            "terminal_device": td_type,
        })

        return self._build_candidate(
            topology=str(topology),
            upper_m=upper_m,
            forearm_m=forearm_m,
            elbow=(0.0, elbow_hi),
            wrist=(-wrist_hi, wrist_hi),
            stiffness=stiffness,
            grip_w=grip_w,
            td_type=td_type,
            requirements=requirements,
            rationale=(
                f"Stochastic candidate (seed={seed}): {topology} topology, "
                f"upper={upper_m:.2f}m, forearm={forearm_m:.2f}m, "
                f"elbow=[0,{elbow_hi:.0f}]°, wrist=[{-wrist_hi:.0f},{wrist_hi:.0f}]°, "
                f"stiffness={stiffness:.2f}, TD={td_type}"
            ),
            work=work,
        )

    # ── Builder helpers ───────────────────────────────────────────────────────

    def _build_from_dict(self, data: dict, requirements: UnifiedRequirements) -> DesignCandidate:
        topology = data.get("topology", "standard")
        upper_m = float(data.get("upper_arm_len", 0.30))
        forearm_m = float(data.get("forearm_len", 0.26))
        el = data.get("elbow_range", [0, 130])
        wr = data.get("wrist_range", [-60, 60])
        elbow = (float(el[0]), float(el[1]))
        wrist = (float(wr[0]), float(wr[1]))
        stiffness = float(data.get("stiffness", 1.0))
        grip_w = float(data.get("grip_width", 0.08))
        td_type = data.get("terminal_device", "pinch_prehensor")
        rationale = data.get("rationale", "LLM-generated design.")
        work = data.get("work", {})
        return self._build_candidate(
            topology, upper_m, forearm_m, elbow, wrist, stiffness, grip_w,
            td_type, requirements, rationale, work,
        )

    def _build_candidate(
        self,
        topology: str,
        upper_m: float,
        forearm_m: float,
        elbow: tuple[float, float],
        wrist: tuple[float, float],
        stiffness: float,
        grip_w: float,
        td_type: str,
        requirements: UnifiedRequirements,
        rationale: str,
        work: dict,
    ) -> DesignCandidate:
        mount = f"torso_{requirements.affected_side}" if requirements.affected_side in {"left", "right"} else "torso_right"

        # Build kinematic chain
        if topology == "minimal":
            links = _topology_minimal(upper_m, grip_w, elbow, wrist, mount)
        elif topology == "full":
            links = _topology_full(upper_m, forearm_m, grip_w, elbow, wrist, mount)
        else:
            links = _topology_standard(upper_m, forearm_m, grip_w, elbow, wrist, mount)

        params = DesignParams(
            upper_arm_len=upper_m,
            forearm_len=forearm_m,
            joint_stiffness=stiffness,
            grip_width=grip_w,
            joint_limits={"elbow": elbow, "wrist": wrist},
            links=links,
            mount_frame=mount,
        )

        components = self._build_bom(params, td_type, requirements)
        td_spec = TerminalDeviceSpec(
            td_type=td_type,
            grip_patterns=_TD_LIBRARY[td_type]["grip_patterns"],
            max_grip_force_n=_TD_LIBRARY[td_type]["max_grip_force_n"],
            weight_g=_TD_LIBRARY[td_type]["weight_g"],
            active_dof=_TD_LIBRARY[td_type]["active_dof"],
        )

        return DesignCandidate(
            params=params,
            components=components,
            terminal_device=td_spec,
            rationale=rationale,
            work=work,
        )

    def _build_bom(
        self,
        params: DesignParams,
        td_type: str,
        requirements: UnifiedRequirements,
    ) -> list[ComponentSpec]:
        """Assign materials and compute mass for each component."""
        # Load profile: high grip → stiffer joints → Ti at elbow
        grip = requirements.grip_capacity or 0.4
        high_load = grip > 0.6 or len(requirements.tasks) > 2

        material_map = {
            "socket": "PA12",
            "socket_pylon": "CFRP-PA12",
            "upper_pylon": "CFRP-PA12",
            "elbow_unit": "Ti-6Al-4V" if high_load else "CFRP-PA12",
            "forearm_pylon": "CFRP-PA12",
            "wrist_flex_unit": "Ti-6Al-4V" if high_load else "PA12",
            "wrist_rot_unit": "Ti-6Al-4V" if high_load else "PA12",
            "terminal_device": "Ti-6Al-4V" if td_type == "myoelectric_hand" else "PA12",
        }

        components: list[ComponentSpec] = []
        for link in params.links:
            mat = material_map.get(link.name, "PA12")
            density = _MATERIALS[mat]["density"]
            od_m = link.radius * 2
            wt_m = 0.003  # 3 mm wall thickness default
            length_m = link.length
            # Hollow cylinder mass
            ri = max(0, od_m / 2 - wt_m)
            ro = od_m / 2
            vol_m3 = math.pi * (ro**2 - ri**2) * length_m
            mass_g = vol_m3 * density * 1000

            mfg = "carbon-layup" if mat == "CFRP-PA12" else ("machined" if mat == "Ti-6Al-4V" else "FDM")

            components.append(ComponentSpec(
                name=link.name,
                component_type="joint_unit" if link.joints else "pylon",
                material=mat,
                length_mm=round(length_m * 1000, 1),
                wall_thickness_mm=round(wt_m * 1000, 1),
                outer_radius_mm=round(ro * 1000, 1),
                mass_g=round(mass_g, 1),
                manufacturing=mfg,
            ))

        # Add terminal device as a component
        td_info = _TD_LIBRARY[td_type]
        components.append(ComponentSpec(
            name=f"td_{td_type}",
            component_type="terminal_device",
            material="Ti-6Al-4V" if td_type == "myoelectric_hand" else "PA12",
            length_mm=60.0,
            wall_thickness_mm=4.0,
            outer_radius_mm=30.0,
            mass_g=float(td_info["weight_g"]),
            manufacturing="machined" if td_type in {"vo_hook", "myoelectric_hand"} else "FDM",
        ))

        return components

    def _build_work_map(
        self, requirements: UnifiedRequirements, decisions: dict
    ) -> dict[str, str]:
        """Map each design directive to the decision that addresses it."""
        work: dict[str, str] = {}
        for directive in requirements.design_directives:
            key = directive[:40]
            if "elbow" in directive.lower():
                work[key] = f"elbow range set to [0, {decisions.get('elbow_range', [0,130])[1] if isinstance(decisions.get('elbow_range'), list) else decisions.get('upper_arm_len', 0.30):.2f}]°"
            elif "wrist" in directive.lower():
                work[key] = "wrist rotation unit added" if decisions.get("topology") == "full" else "friction-lock wrist in standard config"
            elif "grip" in directive.lower():
                work[key] = f"terminal device: {decisions.get('terminal_device', 'pinch_prehensor')}"
            elif "extend" in directive.lower() or "reach" in directive.lower():
                work[key] = f"forearm pylon length: {decisions.get('forearm_len', 0.26):.2f} m"
            else:
                work[key] = f"addressed via {decisions.get('topology', 'standard')} topology"
        return work

    def _format_problems(self, requirements: UnifiedRequirements) -> str:
        lines = [
            f"Primary actions: {', '.join(requirements.primary_actions or ['unknown'])}",
            f"Affected side: {requirements.affected_side}",
            f"ROM targets: {requirements.rom_targets_deg}",
            f"Grip capacity: {requirements.grip_capacity:.0%}",
            f"Anthropometrics: {requirements.anthropometrics}",
            "Design directives:",
        ]
        for d in requirements.design_directives:
            lines.append(f"  • {d}")
        if requirements.conflicts:
            lines.append("Conflicts to resolve:")
            for c in requirements.conflicts:
                lines.append(f"  ! {c}")
        return "\n".join(lines)

    def _format_feedback(self, feedback: dict) -> str:
        if not feedback:
            return "No feedback."
        lines = []
        if "ik_success_rate" in feedback:
            lines.append(f"IK success: {feedback['ik_success_rate']:.0%}")
        if "rl_success_rate" in feedback:
            lines.append(f"RL success: {feedback['rl_success_rate']:.0%}")
        if "mean_energy" in feedback:
            lines.append(f"Mean energy: {feedback['mean_energy']:.0f} J")
        if "worst_safety_factor" in feedback:
            lines.append(f"Safety factor: {feedback['worst_safety_factor']:.2f}")
        if "total_mass_g" in feedback:
            lines.append(f"Total mass: {feedback['total_mass_g']:.0f} g")
        if "suggestions" in feedback:
            lines.append("Mechanical suggestions:")
            for s in feedback["suggestions"]:
                lines.append(f"  • {s}")
        return "\n".join(lines) if lines else "No meaningful feedback yet."
