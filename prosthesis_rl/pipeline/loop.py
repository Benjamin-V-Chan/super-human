"""Iterative CAD ↔ physics-sim RL feedback loop.

Drives the full prosthesis pipeline from raw video clip to a trained,
validated RL policy, iterating CAD design parameters until the arm passes
stress-test thresholds.

    loop = DesignOptimizationLoop()
    result = loop.run("test_vids/clip.mov", emit=emitter)

Events emitted via `emit` (PipelineEvent) are consumed by pipeline_server.py
and forwarded to the browser as SSE.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from prosthesis_rl.pipeline.events import Emitter, PipelineEvent

# ── Thresholds / loop constants ────────────────────────────────────────────────
TARGET_IK_SUCCESS  = 0.40   # scripted IK — exit early without RL if met
TARGET_RL_SUCCESS  = 0.65   # with RL policy — final pass/fail gate
MAX_DESIGN_ITER    = 4      # max design refinement iterations
RL_TIMESTEPS_QUICK = 30_000  # per loop iteration (fast feedback)
RL_TIMESTEPS_FINAL = 100_000 # final winner training


class DesignOptimizationLoop:
    """Iterative optimization loop: perception → CAD → sim → RL → repeat until pass.

    quick_mode=True: reduces seeds/targets/RL timesteps for smoke tests.
    """

    def __init__(
        self,
        quick_mode: bool = False,
        n_seeds: int = 5,
        n_targets: int = 3,
        n_candidates: int = 3,
    ) -> None:
        self.quick_mode  = quick_mode
        self.n_seeds     = 2 if quick_mode else n_seeds
        self.n_targets   = 2 if quick_mode else n_targets
        self.n_candidates = n_candidates
        self._rl_quick   = 5_000  if quick_mode else RL_TIMESTEPS_QUICK
        self._rl_final   = 10_000 if quick_mode else RL_TIMESTEPS_FINAL
        # The ADL task scene derived from the clip (posture + objects + reach
        # waypoints). Set in _run_inner; RL trains on it and the trajectory plays
        # it back so the arm practises the actual action, not a floating dot.
        self._scenario = None

    # ── Public entry point ─────────────────────────────────────────────────────

    def run(self, clip_path: str | Path, emit: Emitter) -> dict[str, Any]:
        """Run the full pipeline; emit SSE events; return result dict."""
        clip_path = str(clip_path)
        t0 = time.perf_counter()

        try:
            return self._run_inner(clip_path, emit, t0)
        except Exception as exc:
            emit.emit(PipelineEvent("error", "pipeline", {"message": str(exc)}))
            raise
        finally:
            emit.close()

    # ── Internal pipeline ──────────────────────────────────────────────────────

    def _run_inner(
        self, clip_path: str, emit: Emitter, t0: float
    ) -> dict[str, Any]:
        from prosthesis_rl.agents.cad_agent import CadAgent
        from prosthesis_rl.agents.cad_spec import DesignSpecLayer
        from prosthesis_rl.agents.design import DesignAgent
        from prosthesis_rl.agents.perception import PerceptionAgent
        from prosthesis_rl.agents.requirements import RequirementsAgent
        from prosthesis_rl.cad.bridge import CadBridge
        from prosthesis_rl.sim.verifier import Verifier

        perception  = PerceptionAgent()
        req_agent   = RequirementsAgent()
        design_agent = DesignAgent()
        spec_layer  = DesignSpecLayer()
        cad_agent   = CadAgent()
        cad_bridge  = CadBridge()
        verifier    = Verifier()

        # ── Stage 1: Perception ────────────────────────────────────────────────
        emit.emit(PipelineEvent("stage_start", "perception", {}))
        t1 = time.perf_counter()
        problem = perception.infer_problem(clip_path)
        emit.emit(PipelineEvent("stage_done", "perception", {
            "action": problem.primary_action,
            "side": problem.affected_side,
            "elapsed_s": round(time.perf_counter() - t1, 2),
            "spec": {
                "tasks": [t.get("id") for t in problem.tasks],
                "rom_deg": dict(problem.constraints.rom),
                "grip_capacity": problem.constraints.grip_capacity,
                "anthropometrics": dict(problem.residual_anthropometrics or {}),
            },
        }))

        # ── Stage 1b: Scenario — turn the observed action into a trainable task
        # scene (posture + objects + reach waypoints). This is what makes the RL
        # train on *doing the clip's action* (tie the shoe) instead of a dot.
        from prosthesis_rl.agents.scenario import ScenarioAgent

        self._scenario = ScenarioAgent().derive(problem)
        sc = self._scenario
        emit.emit(PipelineEvent("stage_done", "scenario", {
            "task_id": sc.task_id,
            "posture": sc.posture,
            "source": sc.source,
            "description": sc.description,
            "success_condition": sc.success_condition,
            "objects": [{"name": o.name, "pos": list(o.pos), "prompt": o.prompt}
                        for o in sc.objects],
            "waypoints": [{"name": w.name, "pos": list(w.pos), "weight": w.weight}
                          for w in sc.waypoints],
        }))

        # ── Stage 2: Requirements ──────────────────────────────────────────────
        emit.emit(PipelineEvent("stage_start", "requirements", {}))
        t2 = time.perf_counter()
        brief = req_agent.derive(problem)
        emit.emit(PipelineEvent("stage_done", "requirements", {
            "brief": brief,
            "elapsed_s": round(time.perf_counter() - t2, 2),
        }))

        # ── Stage 3: Generate initial candidates ──────────────────────────────
        emit.emit(PipelineEvent("stage_start", "design", {}))
        t3 = time.perf_counter()
        candidates = design_agent.propose_candidates(
            problem, brief=brief, n=self.n_candidates
        )
        instruction = spec_layer.build(problem)
        cand_summaries = [
            {
                "upper_m": round(sum(lk.length for lk in c.links if "upper" in lk.name), 3),
                "fore_m":  round(sum(lk.length for lk in c.links if "fore" in lk.name), 3),
                "dof": c.dof,
                "joints": c.joint_names,
            }
            for c, _ in candidates
        ]
        emit.emit(PipelineEvent("stage_done", "design", {
            "n_candidates": len(candidates),
            "candidates": cand_summaries,
            "elapsed_s": round(time.perf_counter() - t3, 2),
        }))

        # ── Iteration loop ─────────────────────────────────────────────────────
        best_params  = None
        best_hints   = {}
        best_eval    = None
        best_name    = "iter0_c0"
        best_mesh_dir = None
        iteration    = 0
        rl_result    = None

        current_candidates = candidates

        for iteration in range(MAX_DESIGN_ITER + 1):
            # ── Stage 4: CAD generation for each candidate ────────────────────
            emit.emit(PipelineEvent("stage_start", "cad", {
                "iteration": iteration, "n_candidates": len(current_candidates),
            }))
            cad_outputs = []
            for ci, (params, hints) in enumerate(current_candidates):
                name = f"iter{iteration}_c{ci}"
                cad_out = cad_agent.generate(instruction, params, name=name)
                cad_outputs.append((name, cad_out, params, hints))
                emit.emit(PipelineEvent("stage_done", "cad", {
                    "iteration": iteration,
                    "candidate": ci,
                    "name": name,
                    "build_sheet": cad_out.build_sheet,
                    "material": cad_out.material,
                    "mesh_dir": cad_out.mesh_dir,
                    "mjcf_path": cad_out.mjcf_path,
                }))

            # ── Stage 5: Physics sim evaluation ───────────────────────────────
            emit.emit(PipelineEvent("stage_start", "sim_eval", {"iteration": iteration}))
            eval_results = []
            for ci, (name, cad_out, params, hints) in enumerate(cad_outputs):
                mesh_dir = cad_out.mesh_dir
                full_eval = design_agent.evaluate_candidates(
                    [(params, hints)],
                    problem,
                    verifier,
                    cad_bridge,
                    n_seeds=self.n_seeds,
                    emit_cb=self._make_frame_cb(emit, iteration, ci),
                )
                eval_results.extend(full_eval)
                # Emit per-candidate summary
                er = full_eval[0]
                emit.emit(PipelineEvent("sim_frame", "sim_eval", {
                    "iteration": iteration,
                    "candidate": ci,
                    "success_rate": round(er.success_rate, 3),
                    "mean_reward": round(er.mean_reward, 3),
                    "mean_energy": round(er.mean_energy, 1),
                    "peak_stress_mpa": round(er.peak_stress_mpa, 2),
                    "predicted_life_years": min(er.predicted_life_years, 999.9),
                }))

            # Pick best candidate this iteration
            best_i, rationale = design_agent.compare(
                [c for c, _ in current_candidates], eval_results
            )
            best_er = eval_results[best_i]
            best_name_iter = cad_outputs[best_i][0]
            best_params_iter = cad_outputs[best_i][2]
            best_hints_iter = cad_outputs[best_i][3]
            best_mesh_iter = cad_outputs[best_i][1].mesh_dir

            # Update overall best
            if (best_eval is None or
                    best_er.mean_reward > best_eval.mean_reward):
                best_params  = best_params_iter
                best_hints   = best_hints_iter
                best_eval    = best_er
                best_name    = best_name_iter
                best_mesh_dir = best_mesh_iter

            emit.emit(PipelineEvent("stage_done", "sim_eval", {
                "iteration": iteration,
                "n_candidates": len(eval_results),
                "best_index": best_i,
                "best_success": round(best_er.success_rate, 3),
                "best_reward": round(best_er.mean_reward, 3),
                "rationale": rationale,
                "eval_results": [
                    {
                        "candidate": i,
                        "success_rate": round(er.success_rate, 3),
                        "mean_reward": round(er.mean_reward, 3),
                        "predicted_life_years": min(er.predicted_life_years, 999.9),
                    }
                    for i, er in enumerate(eval_results)
                ],
            }))

            # Check early exit from IK alone
            if best_er.success_rate >= TARGET_IK_SUCCESS:
                break

            if iteration >= MAX_DESIGN_ITER:
                break  # out of design iterations; proceed with RL on best

            # Generate one refined candidate from sim feedback
            sim_fb = _make_sim_feedback(best_er)
            new_cand = design_agent.propose_candidates(
                problem, feedback=sim_fb, brief=brief, n=1
            )
            current_candidates = new_cand

        # ── Stage 6: RL optimization loop ─────────────────────────────────────
        emit.emit(PipelineEvent("stage_start", "rl_loop", {
            "design_name": best_name,
            "ik_success_rate": round(best_eval.success_rate if best_eval else 0.0, 3),
        }))
        if not self.quick_mode or True:  # always run RL even in quick mode (just fewer steps)
            rl_result = self._run_rl(
                best_params, best_name, emit,
                timesteps=self._rl_quick,
                iteration=0,
            )
            rl_success = rl_result.get("eval", {}).get("success_rate", 0.0)

            # If RL still insufficient, try one more design iteration then retrain
            if rl_success < TARGET_RL_SUCCESS and not self.quick_mode:
                sim_fb_rl = _make_sim_feedback_from_rl(rl_result)
                refined = design_agent.propose_candidates(
                    problem, feedback=sim_fb_rl, brief=brief, n=1
                )
                ref_params, ref_hints = refined[0]
                ref_name = f"rl_refined"
                ref_instr = spec_layer.build(problem)
                ref_cad = cad_agent.generate(ref_instr, ref_params, name=ref_name)
                rl_result2 = self._run_rl(
                    ref_params, ref_name, emit,
                    timesteps=self._rl_final,
                    iteration=1,
                )
                if rl_result2.get("eval", {}).get("success_rate", 0.0) > rl_success:
                    best_params  = ref_params
                    best_hints   = ref_hints
                    best_name    = ref_name
                    best_mesh_dir = ref_cad.mesh_dir
                    rl_result    = rl_result2

            # Final high-quality RL training on winner
            final_rl = self._run_rl(
                best_params, f"{best_name}_final", emit,
                timesteps=self._rl_final,
                iteration="final",
            )
            rl_result = final_rl

        emit.emit(PipelineEvent("stage_done", "rl_loop", {
            "policy_path": rl_result.get("policy", ""),
            "rl_success_rate": round(rl_result.get("eval", {}).get("success_rate", 0.0), 3),
            "timesteps": rl_result.get("timesteps", 0),
        }))

        # ── Stage 7: Final product ─────────────────────────────────────────────
        emit.emit(PipelineEvent("stage_start", "final", {}))

        # Build trajectory for viewer playback
        trajectory = self._run_trajectory(best_params, rl_result, best_mesh_dir)

        stats = _build_stats(
            problem, best_eval, rl_result, best_params, best_name,
            iteration, t0,
        )
        if self._scenario is not None:
            stats["scenario_task"] = self._scenario.task_id
            stats["scenario_posture"] = self._scenario.posture

        rationale_text = design_agent.rationale_report(
            [best_params], [best_eval] if best_eval else [], 0, "Best design from optimization loop",
            action=problem.primary_action,
        ) if best_eval else "No evaluation completed."

        emit.emit(PipelineEvent("stage_done", "final", {
            "best_name": best_name,
            "rationale": rationale_text,
            "trajectory": trajectory,
            "stats": stats,
            "rl_policy": rl_result.get("policy", ""),
            "elapsed_total_s": round(time.perf_counter() - t0, 1),
        }))
        emit.emit(PipelineEvent("done", "pipeline", {"stats": stats}))

        return {
            "best_params": best_params,
            "best_eval": best_eval,
            "rl_result": rl_result,
            "stats": stats,
            "trajectory": trajectory,
            "best_name": best_name,
            "best_index": 0,
        }

    # ── RL helpers ─────────────────────────────────────────────────────────────

    def _run_rl(
        self,
        params,
        name: str,
        emit: Emitter,
        *,
        timesteps: int,
        iteration: int | str,
    ) -> dict[str, Any]:
        from prosthesis_rl.rl.train import train_reach_policy

        def _progress(info: dict) -> None:
            emit.emit(PipelineEvent("rl_step", "rl_loop", {
                "iteration": iteration,
                "timestep": info["timestep"],
                "mean_reward": info.get("mean_reward", 0.0),
                "progress": info["timestep"] / timesteps,
            }))

        return train_reach_policy(
            timesteps,
            name=name,
            design=params,
            n_envs=2 if self.quick_mode else 4,
            eval_episodes=5 if self.quick_mode else 20,
            verbose=0,
            progress_cb=_progress,
            scenario=self._scenario,   # train on the clip's ADL task scene
        )

    def _run_trajectory(self, params, rl_result, mesh_dir) -> list[dict]:
        """Roll out the best RL policy and collect qpos frames for viewer."""
        if not rl_result or not rl_result.get("policy"):
            return []
        if self._scenario is not None:
            return self._run_scenario_trajectory(params, rl_result, mesh_dir)
        try:
            import mujoco
            import numpy as np

            from prosthesis_rl.rl.rollout import load_policy, run_policy_reach
            from prosthesis_rl.sim.control import sample_reachable_targets
            from prosthesis_rl.sim.mjcf_builder import build_mjcf

            model = mujoco.MjModel.from_xml_string(build_mjcf(params, mesh_dir=mesh_dir), {})
            data  = mujoco.MjData(model)
            policy = load_policy(rl_result["policy"])
            targets = sample_reachable_targets(model, params, n=1, seed=42)

            frames: list[dict] = []

            def _cb(d):
                frames.append({
                    "qpos": d.qpos.tolist(),
                    "ee": d.site_xpos[model.site("end_effector").id].tolist(),
                })

            run_policy_reach(model, data, params, targets[0], policy,
                             seconds=4.0, fps=20, frame_cb=_cb)
            return frames
        except Exception:
            return []

    def _run_scenario_trajectory(self, params, rl_result, mesh_dir) -> list[dict]:
        """Roll the trained policy through the ADL scene, pinned to the task-
        completing waypoint, so the viewer plays back the arm doing the clip's
        action (crouch to the laces) — not a reach to a floating dot."""
        try:
            from prosthesis_rl.rl.rollout import load_policy
            from prosthesis_rl.rl.scenario_env import ScenarioReachEnv
            from prosthesis_rl.sim.mjcf_builder import EE_SITE

            scenario = self._scenario
            primary = scenario.waypoints.index(scenario.primary_waypoint())
            env = ScenarioReachEnv(scenario, params, mesh_dir=mesh_dir, seed=42,
                                   eval_waypoint=primary, add_markers=False)
            ee_id = env.model.site(EE_SITE).id
            policy = load_policy(rl_result["policy"])

            frames: list[dict] = []

            def _grab():
                # arm dof only, to match the bare-arm qpos the viewer plays back
                frames.append({
                    "qpos": env.data.qpos[: params.dof].tolist(),
                    "ee": env.data.site_xpos[ee_id].tolist(),
                })

            obs, _ = env.reset(seed=42)
            _grab()
            done = False
            while not done:
                action, _ = policy.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = env.step(action)
                _grab()
                done = term or trunc
            return frames
        except Exception:
            return []

    # ── Frame callback factory ─────────────────────────────────────────────────

    def _make_frame_cb(
        self, emit: Emitter, iteration: int, candidate: int
    ) -> Callable:
        _step = [0]

        def _cb(data) -> None:
            _step[0] += 1
            if _step[0] % 5 != 0:  # throttle: emit every 5th frame
                return
            try:
                import numpy as np
                emit.emit(PipelineEvent("sim_frame", "sim_eval", {
                    "iteration": iteration,
                    "candidate": candidate,
                    "step": _step[0],
                    "qpos": data.qpos.tolist(),
                }))
            except Exception:
                pass

        return _cb


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_sim_feedback(er) -> "SimFeedback":
    """Convert _FullEval → SimFeedback for design.propose()."""
    from prosthesis_rl.contracts import RewardBreakdown, SimFeedback

    rom_pen   = getattr(er, "rom_violation_mean", 0.0)
    coll_pen  = 0.2 * getattr(er, "collision_rate", 0.0)
    energy_pen = min(0.30, getattr(er, "mean_energy", 0.0) / 300.0)
    bd = RewardBreakdown(
        success=er.success_rate,
        energy_penalty=energy_pen,
        rom_penalty=rom_pen,
        collision_penalty=coll_pen,
    )
    return SimFeedback(
        reward=bd.scalar,
        breakdown=bd,
        metrics={"reach_success": er.success_rate},
        notes=[],
    )


def _make_sim_feedback_from_rl(rl_result: dict) -> "SimFeedback":
    from prosthesis_rl.contracts import RewardBreakdown, SimFeedback

    success = rl_result.get("eval", {}).get("success_rate", 0.0)
    bd = RewardBreakdown(success=success, energy_penalty=0.1, rom_penalty=0.0, collision_penalty=0.0)
    return SimFeedback(reward=bd.scalar, breakdown=bd, metrics={}, notes=[])


def _build_stats(
    problem, best_eval, rl_result: dict | None, params, best_name: str,
    iteration: int, t0: float,
) -> dict[str, Any]:
    """Collect final statistics for the dashboard stats panel."""
    from prosthesis_rl.contracts import DesignParams

    if params is None:
        params = DesignParams()

    reach_mm = int(sum(lk.length for lk in params.links) * 1000)

    # Joint positions (cumulative along chain)
    joint_pos: dict[str, list] = {}
    offset = 0.0
    for lk in params.links:
        for jd in lk.joints:
            joint_pos[jd.name] = [0.0, -round(offset * 1000, 1), 0.0]
        offset += lk.length

    return {
        "material":           "PA12-CF",
        "reach_envelope_mm":  reach_mm,
        "dof":                params.dof,
        "joint_names":        params.joint_names,
        "joint_positions_mm": joint_pos,
        "ik_success_rate":    round(getattr(best_eval, "success_rate", 0.0), 3) if best_eval else 0.0,
        "rl_success_rate":    round(rl_result.get("eval", {}).get("success_rate", 0.0), 3) if rl_result else 0.0,
        "mean_energy_j":      round(getattr(best_eval, "mean_energy", 0.0), 1) if best_eval else 0.0,
        "predicted_life_years": min(getattr(best_eval, "predicted_life_years", 99.0), 999.0) if best_eval else 99.0,
        "peak_stress_mpa":    round(getattr(best_eval, "peak_stress_mpa", 0.0), 2) if best_eval else 0.0,
        "primary_action":     problem.primary_action,
        "affected_side":      problem.affected_side,
        "best_name":          best_name,
        "iteration_count":    iteration,
        "rl_timesteps":       rl_result.get("timesteps", 0) if rl_result else 0,
        "elapsed_s":          round(time.perf_counter() - t0, 1),
    }
