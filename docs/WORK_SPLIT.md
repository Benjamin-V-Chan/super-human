# ARMASAI — Work Split

Single source of truth for ownership, task split, milestones, and timeline. Product scope lives in [PRD.md](PRD.md); contracts and evaluation live in [TECHNICAL_PLAN.md](TECHNICAL_PLAN.md).

## Pivot Context

- Physical manufacturing is out of scope.
- CAD work feeds simulation morphology and geometry generation, not a manufacturing pipeline.
- The proof is empirical sim evidence: launch, run policy, view metrics, watch rollout.
- Perception produces a concrete `TaskSpec`; uncertain clip details become explicit assumptions.

---

## Benjamin Chan — Technical Lead, AI Design & Simulation

Benjamin owns the technical architecture of the system and leads the AI reasoning and morphology side of the simulator. He defines the design contracts that every other stage depends on, drives candidate generation and validation, and owns the empirical evidence that shows which morphology and controller perform best.

### Responsibilities

- **System architecture:** own the layered module structure (`contracts`, `agents`, `sim`, `rl`, `cad`), define stage interfaces, and make the final call on cross-cutting design decisions.
- **Shared contracts:** author and maintain `TaskSpec`, `DesignParams` / `MorphologySpec`, `SimSpec`, `PolicyArtifact`, and `EvalResult` — the source of truth every stage reads and writes.
- **Sim morphology generation:** produce `MorphologySpec` / `DesignParams` candidates with explicit `LinkDef` + `JointDef` chains, joint limits, actuator torque limits, masses, and collision geometry.
- **AI design reasoning:** drive the LLM-in-the-loop design agent — turn `TaskSpec` plus sim feedback into ranked morphology candidates and controller-interface changes.
- **Spatial reasoning and validation gates:** check reachability, workspace coverage, mount frames, joint limits, collision geometry, and task-space constraints; reject invalid kinematic trees, impossible reach, bad inertias, invalid actuator ranges, and unstable bodies before they enter sim.
- **Empirical evaluation analysis:** compare candidate morphologies across reward distributions, explain failure modes, and select the best design with quantitative evidence.
- **Integration leadership:** unblock other contributors, review cross-module PRs, and keep the stub-to-real replacement path clear as simulation and policy pieces land.

### Milestone

A candidate simulated limb is generated from a `TaskSpec`, loaded into the sim environment, evaluated across ≥ 10 fixed-seed rollouts, and justified with success rate, reward, and failure-mode analysis.

---

## Vihaan Shringi — Orchestration, APIs, Task Intake, Demo Runtime

Vihaan owns the runnable system path: task intake, contracts plumbing, HUD/eval gateway, orchestration, and demo flow.

### Responsibilities

- **Orchestration:** wire task intake → sim builder → policy runner → evaluator → report.
- **HUD integration:** maintain `tasks.py`, the gateway, and the command path that returns real eval metrics.
- **Task intake:** convert clips, prompts, or ADL examples into validated `TaskSpec` records.
- **API and provider access:** keep required provider keys, credits, and fallbacks usable — HUD, Anthropic/Gemini model access, Modal or other compute, RL training providers.
- **Demo runtime:** make the viewer or local commands launch the sim and replay results cleanly.
- **Integration discipline:** keep stubs runnable while real morphology, sim, and policy pieces replace them.

### Milestone

The sim-only loop runs end to end and produces metrics plus a replayable demo artifact.

---

## Nathan — Physics Environment and Scenario Fidelity

Nathan owns the simulation environment and task realism.

### Responsibilities

- **MuJoCo/HUD environment:** build the scene, physics config, reset logic, observations, and action space.
- **Task scenarios:** implement concrete ADL-style tasks with measurable success criteria.
- **Physics fidelity:** tune contacts, object properties, joint dynamics, and episode constraints for believable sim behavior.
- **Deterministic verifier:** make fixed-seed rollouts reproducible and cheap enough for repeated evaluation.
- **Failure diagnostics:** expose collision, instability, unreachable target, and joint-limit failure signals to evaluation.

### Milestone

At least one ADL-style scenario runs deterministically with metrics suitable for comparing candidates.

---

## Vasi — Policy, RL, Reward Optimization

Vasi owns policy behavior and learning from sim rewards.

### Responsibilities

- **Baseline controller:** scripted or IK policy first so the demo has a reliable behavior floor.
- **RL training:** train a policy when the scenario and reward are stable enough.
- **Reward shaping:** define terms for success, distance, energy, joint-limit violations, and collisions.
- **Calibration:** tune reward and scenario difficulty so rollouts produce useful variance.
- **Policy artifact:** export a `.pt` checkpoint or runnable controller config that the evaluator can load.

### Milestone

A policy or controller can run repeated rollouts and show measurable task behavior in the sim.

---

## Shared Timeline

| Phase | Benjamin (Tech Lead) | Vihaan | Nathan | Vasi |
|---|---|---|---|---|
| Phase 1 | Finalize contracts, morphology schema, spatial checks, first simulated limb | Contracts plumbing, task intake stub, eval command path | Minimal MuJoCo/HUD task scene | Scripted/IK baseline |
| Phase 2 | Candidate morphology generation, validation gates, design-agent ranking loop | End-to-end runner, viewer/replay wiring | Deterministic task metrics and reset logic | Reward shaping and calibration |
| Phase 3 | Cross-candidate comparison, design rationale, integration review | Final demo flow and report assembly | Final scenario tuning and failure signals | RL attempt or polished baseline policy |

## Hard Checkpoints

- Sim-only scope is explicit in every public doc.
- A task launches in sim from documented commands.
- At least one morphology and policy/controller run through evaluation.
- Final artifacts include metrics, rollout video/replay, and policy/controller config.
