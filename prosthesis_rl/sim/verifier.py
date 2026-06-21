from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from prosthesis_rl.contracts import DesignParams, ProblemSpec, RewardBreakdown, SimFeedback


class Verifier:
    """Deterministic grading entrypoint called by the design loop.

    Builds the design's actual MuJoCo arm — skinned with the per-link CAD meshes
    when `mesh_dir` is given — and runs real reach rollouts over a few FK-sampled
    ADL targets, instead of returning a hardcoded stub. The most-loaded joint's
    torque trace feeds a service-life estimate. The mesh is no longer discarded.
    """

    def evaluate(
        self,
        problem: ProblemSpec,
        design: DesignParams,
        control_hints: dict[str, float],
        *,
        mesh_dir: str | Path | None = None,
        n_targets: int = 4,
        seconds: float = 3.0,
        seed: int = 0,
        emit_cb=None,
    ) -> SimFeedback:
        del problem, control_hints  # reserved: scene perturbations / control tuning
        import mujoco

        from prosthesis_rl.fatigue import estimate_lifespan
        from prosthesis_rl.sim.control import (
            ReachController,
            run_reach,
            sample_reachable_targets,
        )
        from prosthesis_rl.sim.mjcf_builder import build_mjcf

        model = mujoco.MjModel.from_xml_string(build_mjcf(design, mesh_dir=mesh_dir), {})
        targets = sample_reachable_targets(model, design, n=n_targets, seed=seed)

        successes, energies, roms, colls, finals = [], [], [], [], []
        elbow_torque: np.ndarray | None = None
        for target in targets:
            data = mujoco.MjData(model)
            ctrl = ReachController(model, design, target)
            metrics, log = run_reach(model, data, ctrl, seconds=seconds, fps=20,
                                     frame_cb=emit_cb)
            successes.append(metrics.reach_success)
            energies.append(metrics.energy)
            roms.append(metrics.rom_violation)
            colls.append(metrics.self_collision)
            finals.append(metrics.final_distance)
            if elbow_torque is None and "elbow" in log.joints:
                elbow_torque = log.as_array()[:, log.joints.index("elbow")]

        success = float(np.mean(successes))
        energy = float(np.mean(energies))
        breakdown = RewardBreakdown(
            success=success,
            energy_penalty=min(0.30, energy / 300.0),
            rom_penalty=float(np.mean(roms)),
            collision_penalty=0.20 * float(np.mean(colls)),
        )

        metrics_out = {
            "reach_success": success,
            "final_distance_cm": float(np.mean(finals)) * 100.0,
            "energy": energy,
            "rom_violation": float(np.mean(roms)),
            "self_collision": float(np.mean(colls)),
            "n_targets": float(len(targets)),
            "dof": float(design.dof),
        }
        notes = [
            f"real MuJoCo reach over {len(targets)} FK-sampled targets "
            f"({'mesh' if mesh_dir else 'primitive'} arm, scripted IK)"
        ]
        if elbow_torque is not None:
            life = estimate_lifespan(elbow_torque, design)
            metrics_out["predicted_life_years"] = (
                life.predicted_years if math.isfinite(life.predicted_years) else 100.0
            )
            metrics_out["peak_stress_mpa"] = life.peak_stress_mpa
            notes.append(f"est. service life: {life.display_years}")

        return SimFeedback(
            reward=breakdown.scalar,
            breakdown=breakdown,
            metrics=metrics_out,
            notes=notes,
        )
