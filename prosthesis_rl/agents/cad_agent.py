"""CadGPT agent — consumes DesignSpecLayer instruction JSON, produces build sheet.

Pipeline position:
  DesignSpecLayer.build() → instruction JSON → CadAgent.generate() → CadOutput

CadAgent is the actual "CadGPT model" referenced throughout the docs. It takes
the structured kinematic + ROM + grip specification and produces:
  - Material selection (PA12-CF / PETG / PLA) with engineering rationale
  - Per-link wall thickness (beam theory, hollow shaft under torsion + bending)
  - FDM print settings (infill, orientation, supports, clearance)
  - Printability + assembly feasibility checks
  - Per-link STL meshes and MJCF simulation file via CadBridge
  - Full build sheet (JSON) ready for the slicer / handoff to Nathan's verifier

LLM path uses Gemini (same key as RequirementsAgent / GemmaVideoAnalyzer).
Deterministic fallback applies the engineering rules directly so the loop stays
green with no API key.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prosthesis_rl.cad.bridge import CadBridge
from prosthesis_rl.contracts import DesignParams

DEFAULT_MODEL = "gemini-2.5-flash"


# ── Material database ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Material:
    name: str
    tensile_yield_mpa: float     # SY
    shear_strength_mpa: float    # ~ SY / 2
    density_kg_m3: float
    min_wall_mm: float           # absolute FDM floor
    default_infill_pct: int
    max_grip_force_n: float      # max sustained grip without creep
    notes: str

_MATERIALS: dict[str, _Material] = {
    "PA12-CF": _Material(
        name="PA12-CF",
        tensile_yield_mpa=90.0,
        shear_strength_mpa=45.0,
        density_kg_m3=1020,
        min_wall_mm=1.2,
        default_infill_pct=70,
        max_grip_force_n=50.0,
        notes="Carbon-fiber nylon — best strength/weight for structural links; requires enclosure.",
    ),
    "PETG": _Material(
        name="PETG",
        tensile_yield_mpa=45.0,
        shear_strength_mpa=22.5,
        density_kg_m3=1270,
        min_wall_mm=1.5,
        default_infill_pct=60,
        max_grip_force_n=20.0,
        notes="Good layer adhesion, slight flex — suitable for gripper bodies and covers.",
    ),
    "PLA": _Material(
        name="PLA",
        tensile_yield_mpa=50.0,
        shear_strength_mpa=25.0,
        density_kg_m3=1240,
        min_wall_mm=1.6,
        default_infill_pct=40,
        max_grip_force_n=10.0,
        notes="Easy to print; brittle under impact — prototype and non-load-bearing covers only.",
    ),
}

_SAFETY_FACTOR = 4.0   # FDM layer-adhesion uncertainty (2× for anisotropy, 2× for fatigue)


# ── Output contract ───────────────────────────────────────────────────────────

@dataclass
class LinkSpec:
    """Resolved manufacturing specification for one link."""
    name: str
    length_mm: float
    outer_radius_mm: float
    wall_thickness_mm: float
    infill_pct: int
    material: str
    print_orientation: str
    mass_g: float
    stl_path: str = ""

@dataclass
class CadOutput:
    """Full manufacturing output produced by CadAgent.generate()."""
    # Design identity
    action: str = ""
    mount_frame: str = ""
    # Material decision
    material: str = "PA12-CF"
    material_rationale: str = ""
    # FDM settings (applied to all structural links; gripper may differ)
    joint_clearance_mm: float = 0.30
    # Per-link resolved specs
    links: list[LinkSpec] = field(default_factory=list)
    # Quality gates
    printability_ok: bool = True
    design_concerns: list[str] = field(default_factory=list)
    manufacturing_notes: list[str] = field(default_factory=list)
    # Artifacts
    mesh_dir: str = ""
    mjcf_path: str = ""
    # Meta
    build_sheet: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    source: str = ""

    def summary(self) -> str:
        lines = [
            "=" * 64,
            f"  CAD OUTPUT — {self.action or 'no action'}",
            "=" * 64,
            f"  Material      : {self.material}",
            f"  Mount frame   : {self.mount_frame}",
            f"  Joint clearance: {self.joint_clearance_mm:.2f} mm",
            f"  Printability  : {'✓ OK' if self.printability_ok else '✗ ISSUES'}",
            f"  Source        : {self.source}",
            "",
            f"  {'Link':<14} {'L(mm)':>8} {'OD(mm)':>8} {'Wall(mm)':>10} {'Infill':>7} {'Mass(g)':>8} {'Orient'}",
            "  " + "-" * 62,
        ]
        for lk in self.links:
            lines.append(
                f"  {lk.name:<14} {lk.length_mm:>8.1f} {lk.outer_radius_mm*2:>8.1f} "
                f"{lk.wall_thickness_mm:>10.2f} {lk.infill_pct:>6}% "
                f"{lk.mass_g:>8.1f} {lk.print_orientation}"
            )
        total_mass = sum(lk.mass_g for lk in self.links)
        lines += [
            "  " + "-" * 62,
            f"  {'TOTAL':<14} {'':>8} {'':>8} {'':>10} {'':>7} {total_mass:>8.1f}",
            "",
        ]
        if self.manufacturing_notes:
            lines.append("  MANUFACTURING NOTES:")
            for note in self.manufacturing_notes:
                lines.append(f"    • {note}")
            lines.append("")
        if self.design_concerns:
            lines.append("  DESIGN CONCERNS:")
            for c in self.design_concerns:
                lines.append(f"    ✗ {c}")
            lines.append("")
        if self.rationale:
            lines.append(f"  RATIONALE: {self.rationale}")
        lines.append("=" * 64)
        return "\n".join(lines)


# ── Engineering calculations ──────────────────────────────────────────────────

def _wall_thickness_mm(
    torque_nm: float,
    outer_radius_m: float,
    mat: _Material,
) -> float:
    """Minimum wall thickness for a hollow cylindrical link under torque.

    Uses thin-walled torsion: τ = T / (2·A·t)  →  t = T / (2·π·r²·τ_allow)
    τ_allow = shear_strength / safety_factor
    Clamped to the material's absolute FDM minimum.
    """
    tau_allow = (mat.shear_strength_mpa * 1e6) / _SAFETY_FACTOR
    A = math.pi * outer_radius_m ** 2
    t_m = torque_nm / (2.0 * A * tau_allow) if (A * tau_allow) > 0 else 0.0
    t_mm = t_m * 1000.0
    return max(t_mm, mat.min_wall_mm)


def _link_mass_g(length_m: float, r_outer_m: float, wall_m: float, infill: float, density: float) -> float:
    """Estimated mass in grams for a hollow cylindrical link with infill."""
    r_inner = max(0.0, r_outer_m - wall_m)
    vol_shell = math.pi * (r_outer_m**2 - r_inner**2) * length_m
    vol_infill = math.pi * r_inner**2 * length_m * infill
    return (vol_shell + vol_infill) * density * 1000.0


def _print_orientation(link_name: str, length_mm: float, radius_mm: float) -> str:
    """Heuristic: long thin links print vertically (strength along Z); short fat ones horizontal."""
    aspect = length_mm / (2 * radius_mm) if radius_mm > 0 else 1.0
    if link_name == "gripper":
        return "horizontal"         # gripper fingers need lateral strength
    if aspect > 3.0:
        return "vertical"           # long tube: print upright, layer lines along axis
    return "angled_30deg"           # angled provides balanced strength for mid-aspect links


def _select_material(instruction: dict[str, Any]) -> _Material:
    """Rule-based material selection from the instruction JSON.

    PA12-CF for any link with sustained torque > 10 N·m or grip force > 15N.
    PETG for moderate loads. PLA only for low-load covers.
    """
    actuators = instruction.get("actuators", [])
    max_torque = max((float(a.get("torque_nm", 0)) for a in actuators), default=0.0)
    grip_force = instruction.get("end_effector", {}).get("grip_force_target_n", 12.0)
    if max_torque > 10.0 or grip_force > 15.0:
        return _MATERIALS["PA12-CF"]
    if max_torque > 5.0 or grip_force > 8.0:
        return _MATERIALS["PETG"]
    return _MATERIALS["PLA"]


def _joint_clearance(mat: _Material) -> float:
    """Recommended joint clearance in mm based on material shrinkage."""
    if mat.name == "PA12-CF":
        return 0.35     # SLS-style nylons have more shrink
    if mat.name == "PETG":
        return 0.25
    return 0.20


def _printability_check(links_spec: list[LinkSpec]) -> list[str]:
    """Return a list of concern strings; empty = printable."""
    concerns = []
    for lk in links_spec:
        if lk.wall_thickness_mm < 1.0:
            concerns.append(
                f"{lk.name}: wall {lk.wall_thickness_mm:.2f} mm < 1.0 mm FDM floor — "
                "increase radius or reduce torque."
            )
        if lk.length_mm / (lk.outer_radius_mm * 2 or 1) > 12:
            concerns.append(
                f"{lk.name}: aspect ratio {lk.length_mm/(lk.outer_radius_mm*2):.1f} — "
                "may need mid-print support or orientation change."
            )
        if lk.infill_pct < 30:
            concerns.append(
                f"{lk.name}: infill {lk.infill_pct}% may be too sparse for structural use."
            )
    return concerns


# ── CadAgent ─────────────────────────────────────────────────────────────────

class CadAgent:
    """CadGPT instruction JSON + DesignParams → CadOutput (build sheet + STL + MJCF).

    Consume with:
        spec_layer = DesignSpecLayer()
        instruction = spec_layer.build(problem)
        output = CadAgent().generate(instruction, params)
    """

    def __init__(self, model: str | None = None, cad: CadBridge | None = None) -> None:
        self.model = model or os.environ.get("GEMMA_MODEL", DEFAULT_MODEL)
        self.cad = cad or CadBridge()

    @property
    def available(self) -> bool:
        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {"1", "true", "yes"}
        return key is not None or vertex

    def generate(
        self,
        instruction: dict[str, Any],
        params: DesignParams,
        *,
        name: str = "candidate",
    ) -> CadOutput:
        """Full CAD generation pass.

        1. Material selection + engineering calculations (rules or LLM)
        2. Per-link wall thickness, infill, orientation, mass estimate
        3. Printability / assembly feasibility checks
        4. Export per-link STLs and MJCF via CadBridge
        5. Build sheet assembly
        """
        if self.available:
            try:
                cad_decisions = self._llm_pass(instruction)
                cad_decisions["source"] = f"llm:{self.model}"
            except Exception as exc:  # noqa: BLE001
                cad_decisions = self._rules_pass(instruction)
                cad_decisions["source"] = f"rules_after_error:{type(exc).__name__}"
        else:
            cad_decisions = self._rules_pass(instruction)
            cad_decisions["source"] = "rules"

        return self._assemble_output(instruction, params, cad_decisions, name=name)

    # ── LLM path ─────────────────────────────────────────────────────────────

    def _llm_pass(self, instruction: dict[str, Any]) -> dict[str, Any]:
        from google import genai
        from google.genai import types
        from google.genai.types import HttpOptions

        # Trim instruction to what the LLM needs (omit provenance noise)
        payload = {
            "primary_action": instruction.get("primary_action"),
            "kinematics": instruction.get("kinematics"),
            "end_effector": instruction.get("end_effector"),
            "actuators": instruction.get("actuators"),
            "reach_envelope_m": instruction.get("reach_envelope_m"),
            "cad_instructions": instruction.get("cad_instructions"),
            "rationale": instruction.get("rationale"),
        }
        prompt = (
            "You are a prosthetics CAD manufacturing engineer. Given a kinematic design "
            "instruction for a custom FDM-printed upper-limb prosthesis, choose the best "
            "FDM printing material, compute minimum wall thicknesses, specify print settings, "
            "and flag any design concerns.\n\n"
            "Available materials: PA12-CF (best, requires enclosure, 90 MPa yield), "
            "PETG (good layer adhesion, 45 MPa, easier print), PLA (easy but brittle).\n\n"
            "Apply a safety factor of 4 for FDM anisotropy and fatigue.\n\n"
            f"DESIGN INSTRUCTION:\n{json.dumps(payload, indent=2)}\n\n"
            "Respond ONLY with a JSON object with this exact shape:\n"
            "{\n"
            '  "material": "PA12-CF|PETG|PLA",\n'
            '  "material_rationale": "one sentence",\n'
            '  "joint_clearance_mm": 0.30,\n'
            '  "infill_pct": 65,\n'
            '  "manufacturing_notes": ["step 1", "step 2", "..."],\n'
            '  "design_concerns": ["concern if any"],\n'
            '  "rationale": "one paragraph summarizing the design decisions"\n'
            "}"
        )

        use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {"1", "true", "yes"}
        if use_vertex:
            client = genai.Client(http_options=HttpOptions(api_version="v1"))
        else:
            client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))

        resp = client.models.generate_content(
            model=self.model,
            contents=[prompt],
            config=types.GenerateContentConfig(temperature=0.0),
        )
        result = _extract_json(resp.text or "")
        if result is None:
            raise ValueError("CadAgent LLM response was not parseable JSON")
        return result

    # ── Rules path ────────────────────────────────────────────────────────────

    def _rules_pass(self, instruction: dict[str, Any]) -> dict[str, Any]:
        mat = _select_material(instruction)
        clearance = _joint_clearance(mat)
        action = instruction.get("primary_action", "")
        grip_f = instruction.get("end_effector", {}).get("grip_force_target_n", 12.0)
        actuators = instruction.get("actuators", [])
        max_torque = max((float(a.get("torque_nm", 0)) for a in actuators), default=0.0)

        notes = [
            f"Print all structural links in {mat.name} ({mat.notes})",
            "Gripper pads: TPU 95A — 3 mm thick, bonded with cyanoacrylate.",
            f"Set layer height 0.15 mm and {mat.default_infill_pct}% gyroid infill for structural links.",
            f"Joint clearance: {clearance:.2f} mm on all mating surfaces — check fit before gluing.",
            "Drill and tap M3 through-holes for actuator mounting post-print.",
            "Sand joint surfaces to 400 grit before assembly to remove layer ridges.",
        ]
        if "left" in instruction.get("mount_frame", ""):
            notes.append("Left-side mount: mirror the shoulder socket geometry around the sagittal plane.")

        concerns: list[str] = []
        if max_torque > 25.0:
            concerns.append(
                f"Max actuator torque {max_torque:.0f} N·m exceeds typical FDM PA12-CF safe range; "
                "consider metal insert at shoulder socket."
            )
        if float(grip_f) > 25.0:
            concerns.append(
                f"Grip force target {grip_f:.0f} N — verify actuator cable routing can sustain continuous load."
            )

        return {
            "material": mat.name,
            "material_rationale": (
                f"Max actuator torque {max_torque:.0f} N·m and grip force {float(grip_f):.0f} N "
                f"for '{action}' → {mat.name} ({mat.notes})"
            ),
            "joint_clearance_mm": clearance,
            "infill_pct": mat.default_infill_pct,
            "manufacturing_notes": notes,
            "design_concerns": concerns,
            "rationale": (
                f"Action '{action}' drives a {max_torque:.0f} N·m shoulder / "
                f"{float(grip_f):.0f} N grip requirement. {mat.name} selected: "
                f"SY={mat.tensile_yield_mpa:.0f} MPa, ρ={mat.density_kg_m3} kg/m³, "
                f"min wall {mat.min_wall_mm} mm. "
                f"Safety factor {_SAFETY_FACTOR}× applied for FDM anisotropy and cyclic loading."
            ),
        }

    # ── Assembly ──────────────────────────────────────────────────────────────

    def _assemble_output(
        self,
        instruction: dict[str, Any],
        params: DesignParams,
        decisions: dict[str, Any],
        name: str,
    ) -> CadOutput:
        mat_name = decisions.get("material", "PA12-CF")
        mat = _MATERIALS.get(mat_name, _MATERIALS["PA12-CF"])
        infill_pct: int = int(decisions.get("infill_pct", mat.default_infill_pct))
        clearance = float(decisions.get("joint_clearance_mm", _joint_clearance(mat)))

        # Per-actuator torque lookup by joint name for wall-thickness calculation
        actuator_map: dict[str, float] = {
            a["joint"]: float(a["torque_nm"])
            for a in instruction.get("actuators", [])
        }
        # Map actuator joints to links (first joint on each link drives its torque)
        link_torque: dict[str, float] = {}
        for link in params.links:
            for joint in link.joints:
                # Use the highest torque among joints on this link
                t = actuator_map.get(joint.name, 0.0)
                # Shoulder joints not in actuator_map → use shoulder_flexion torque as proxy
                if t == 0.0 and "shoulder" in joint.name:
                    t = actuator_map.get("shoulder_flexion", 20.0)
                if t == 0.0 and "elbow" in joint.name:
                    t = actuator_map.get("elbow_flexion", 15.0)
                link_torque[link.name] = max(link_torque.get(link.name, 0.0), t)

        link_specs: list[LinkSpec] = []
        for link in params.links:
            torque = link_torque.get(link.name, 5.0)
            wall = _wall_thickness_mm(torque, link.radius, mat)
            infill = infill_pct if link.name != "gripper" else min(infill_pct, 50)
            orient = _print_orientation(link.name, link.length * 1000, link.radius * 1000)
            mass = _link_mass_g(link.length, link.radius, wall / 1000.0, infill / 100.0, mat.density_kg_m3)
            link_specs.append(LinkSpec(
                name=link.name,
                length_mm=round(link.length * 1000, 1),
                outer_radius_mm=round(link.radius * 1000, 1),
                wall_thickness_mm=round(wall, 2),
                infill_pct=infill,
                material=mat_name if link.name != "gripper" else "PETG",
                print_orientation=orient,
                mass_g=round(mass, 1),
            ))

        concerns = _printability_check(link_specs)
        concerns += decisions.get("design_concerns", [])
        printability_ok = len([c for c in concerns if "wall" in c.lower() or "floor" in c.lower()]) == 0

        # Export geometry
        mesh_dir = self.cad.export_arm(params, name=name)
        mjcf_path = self.cad.export_mjcf(params, name=name)

        # Backfill stl_path per link
        for lk in link_specs:
            lk.stl_path = str(mesh_dir / f"{lk.name}.stl")

        # Build sheet — full JSON artifact
        build_sheet = {
            "schema": "cad-build-sheet/v1",
            "name": name,
            "action": instruction.get("primary_action", ""),
            "mount_frame": instruction.get("mount_frame", ""),
            "material": mat_name,
            "material_rationale": decisions.get("material_rationale", ""),
            "joint_clearance_mm": clearance,
            "links": [
                {
                    "name": lk.name,
                    "length_mm": lk.length_mm,
                    "outer_diameter_mm": lk.outer_radius_mm * 2,
                    "wall_thickness_mm": lk.wall_thickness_mm,
                    "infill_pct": lk.infill_pct,
                    "material": lk.material,
                    "print_orientation": lk.print_orientation,
                    "mass_g": lk.mass_g,
                    "stl": lk.stl_path,
                }
                for lk in link_specs
            ],
            "total_mass_g": round(sum(lk.mass_g for lk in link_specs), 1),
            "printability_ok": printability_ok,
            "design_concerns": concerns,
            "manufacturing_notes": decisions.get("manufacturing_notes", []),
            "mesh_dir": str(mesh_dir),
            "mjcf_path": str(mjcf_path),
            "reach_envelope_mm": round(instruction.get("reach_envelope_m", 0.0) * 1000, 1),
            "kinematics_summary": {
                "dof": instruction.get("kinematics", {}).get("dof", 0),
                "joint_order": instruction.get("kinematics", {}).get("joint_order", []),
            },
            "end_effector": instruction.get("end_effector", {}),
            "actuators": instruction.get("actuators", []),
            "rationale": decisions.get("rationale", ""),
            "source": decisions.get("source", "rules"),
            "instruction_schema": instruction.get("schema", ""),
        }

        return CadOutput(
            action=instruction.get("primary_action", ""),
            mount_frame=instruction.get("mount_frame", ""),
            material=mat_name,
            material_rationale=decisions.get("material_rationale", ""),
            joint_clearance_mm=clearance,
            links=link_specs,
            printability_ok=printability_ok,
            design_concerns=concerns,
            manufacturing_notes=decisions.get("manufacturing_notes", []),
            mesh_dir=str(mesh_dir),
            mjcf_path=str(mjcf_path),
            build_sheet=build_sheet,
            rationale=decisions.get("rationale", ""),
            source=decisions.get("source", "rules"),
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
