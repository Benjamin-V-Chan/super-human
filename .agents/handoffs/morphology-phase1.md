# Agent Handoff

- Task: Phase 1 morphology — spatial validation, candidate generation, MJCF output
- Owning role: `morphology_design` (Benji)
- Status: `review`
- Writable paths assigned: `prosthesis_rl/agents/design.py`, `prosthesis_rl/cad/**`
- Upstream dependency/version: contracts at `prosthesis_rl/contracts/schemas.py` (unchanged)

## Result

- Changed files:
  - `prosthesis_rl/agents/design.py` — replaced stub with full DesignAgent: real `propose()`, `validate()`, `propose_candidates()`, `compare()`; internal `MorphologySpec`, `EvalResult` dataclasses pending contract stabilization
  - `prosthesis_rl/cad/bridge.py` — added `export_mjcf()` (MJCF XML for Nathan), `_coerce_to_design_params()` so `export_stl` accepts both `DesignParams` and `MorphologySpec`
  - `tests/test_design_agent.py` — 22 new tests covering all validation gates, propose logic, candidate comparison, MJCF output
  - `assets/mjcf/.gitkeep` — directory created for generated MJCF files

- Input fixture: `ProblemSpec` (from perception pipeline); `SimFeedback` (from verifier stub)
- Output contract/artifact: `MorphologySpec` dict; MJCF written to `assets/mjcf/candidate.xml`
- Seed/configuration: deterministic (no random seed; morphology is fully determined by `ProblemSpec` + `SimFeedback`)
- Live, fallback, or stub mode: morphology and MJCF generation are fully live; verifier remains stub (Nathan's task)

## Verification

- Command: `python3 -m pytest -q`
- Result: 28 passed
- Evidence path: `tests/test_design_agent.py` (22 new), existing suite unchanged
- Skipped gate: `npm --prefix viewer run build` — viewer untouched; `mujoco` import not available (sim is Nathan's stub)

## Follow-up

- Known risks: `MorphologySpec` and `EvalResult` defined locally in `design.py` — will need migration to `prosthesis_rl/contracts/schemas.py` once coordinator adds them; orchestrator will pick them up automatically
- Contract change requested: **Coordinator (Vihaan)** — please add to `prosthesis_rl/contracts/schemas.py`:
  - `LinkSpec(name, length_m, mass_kg)`
  - `JointSpec(name, type, limits_rad)`
  - `ActuatorSpec(joint, torque_limit_nm)`
  - `MorphologySpec(mount_frame, links, joints, actuators)`
  - `EvalResult(task_id, num_rollouts, success_rate, mean_reward, mean_energy, collision_rate, video_path)`
  - `TaskSpec(task_id, goal, objects, success_condition, episode_seconds, assumptions)` — needed for Phase 2 intake
- Next owning role: **Nathan (simulation)** — load `assets/mjcf/candidate.xml` into MuJoCo scene; confirm joint axis convention (`axis="1 0 0"`) and capsule sizing match environment expectations; provide real `EvalResult` so `compare()` can rank candidates in Phase 2
