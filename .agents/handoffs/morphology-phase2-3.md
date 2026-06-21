# Agent Handoff

- Task: Phase 2 + 3 — candidate generation, multi-seed evaluation, design rationale
- Owning role: `morphology_design` (Benji)
- Status: `review`
- Writable paths assigned: `prosthesis_rl/agents/design.py`, `prosthesis_rl/cad/**`, `scripts/compare_candidates.py`, `tests/test_design_agent.py`
- Upstream dependency/version: contracts at `prosthesis_rl/contracts/schemas.py` (mount_frame added); RequirementsAgent at `prosthesis_rl/agents/requirements.py`

## Result

### Changed files
- `prosthesis_rl/agents/design.py` — major additions:
  - `propose()` now accepts `brief` from `RequirementsAgent.derive()` → uses action-specific ROM targets (elbow/wrist range_deg), arm dimensions, grip width, stiffness
  - `propose_candidates()` generates N candidates with systematic variation: base (from brief) + long+wide + short+stiff
  - `evaluate_candidates()` runs each candidate through verifier with `n_seeds` fixed seeds → aggregates into `EvalResult` (success rate, reward variance, dist-to-goal, energy, collision, ROM violations, fatigue)
  - `rationale_report()` formats comparison table + failure-mode analysis
  - `EvalResult` dataclass updated with reward_variance, mean_final_dist_cm, rom_violation_mean, peak_stress_mpa, predicted_life_years fields
- `scripts/compare_candidates.py` — new end-to-end script: perception → requirements → candidates → validation → eval → compare → report + JSON evidence
- `tests/test_design_agent.py` — 16 new tests (46 total): brief integration, stub-verifier multi-seed, rationale flags
- `.gitignore` — added `runs/` exclusion

### Input fixture
- Clip: `test_vids/IMG_9847 (1) (1).mov`
- ProblemSpec: action="drinking water from a bottle one-handed", side=left
- RequirementsAgent brief: source=fallback (no API key), general-purpose grasp envelope

### Output artifact — real evaluation evidence (runs/candidates/20260621_012203/)
```
Candidate 0 (base: upper=0.30, fore=0.26, elbow_hi=120°):
  reward=-0.050  var=0.056  success=25%  dist=11.0cm  energy=438J  coll=0%

Candidate 1 (long+wide: upper=0.34, fore=0.30, elbow_hi=135°):
  reward=+0.050  var=0.044  success=35%  dist=11.9cm  energy=544J  coll=0%

Candidate 2 (short+stiff: upper=0.26, fore=0.22, elbow_hi=110°):  ◄ WINNER
  reward=+0.075  var=0.087  success=38%  dist=6.4cm   energy=414J  coll=0%
```

**Selected: Candidate 2** — compact arm wins on mean_reward (+0.075) and end-effector distance (6.4 cm), with lowest energy (414J) and no collisions.

**Failure modes flagged:**
- Low success rate (38%) — scripted IK struggles with FK-sampled targets; longer links or wider elbow range may help near-target reach
- Predicted service life ~0 yr — peak torques in the current Nylon PA12-CF model exceed material fatigue limits; actuator tuning or material spec needed

### Seed/configuration
- 10 seeds × 4 FK-sampled targets × 3.0 s per rollout = 120 MuJoCo episodes per candidate
- Fixed seeds 0–9 ensure reproducibility (determinism evidence: same seeds give same results)

### Live, fallback, or stub mode
- MuJoCo verifier: **LIVE** (real physics, real FK-sampled targets, scripted IK)
- RequirementsAgent: **fallback** (Gemini API key not set; deterministic action profiles used)
- Perception: **fallback** (stub detection from ADL clip)

## Verification

- Command: `python3 -m pytest -q`
- Result: 46 passed
- Evidence path: `runs/candidates/20260621_012203/report.json` (local, gitignored)
- Rationale text: `runs/candidates/20260621_012203/rationale.txt`
- Skipped gate: `npm --prefix viewer run build` (viewer not touched)

## Follow-up

- Known risks:
  - Short service life estimate is a known gap: Nylon PA12-CF material constants in `fatigue/estimate.py` are rough literature values with FDM knockdown; Phase 2 plan (FEA surrogate) is needed before this is actionable
  - 38% success rate reflects scripted IK baseline (Phase 1 floor); RL policy (Vasi) is expected to improve this substantially
  - RequirementsAgent LLM path not tested here (no API key); fallback is deterministic but doesn't tune to the specific "drinking" action beyond default profile

- Contract change requested: **Coordinator (Vihaan)** — please migrate `EvalResult` dataclass (currently in `design.py`) to `prosthesis_rl/contracts/schemas.py`. Fields: task_id, num_rollouts, success_rate, mean_reward, reward_variance, mean_energy, mean_final_dist_cm, collision_rate, rom_violation_mean, peak_stress_mpa, predicted_life_years, video_path.

- Next owning role:
  - **Vasi (policy)** — RL policy should substantially improve success rate above 38% IK baseline; use `rl/env.py` `ReachEnv` + `rl/train.py` `train_reach_policy()`
  - **Nathan (simulation)** — consider adding more ADL-realistic tasks (grasp, stabilize) beyond pure reach; failure signals from verifier (unreachable target, instability) can feed back to candidate generation
  - **Vihaan (coordinator)** — wire `compare_candidates.py` into demo runtime so the viewer can show the comparison table
