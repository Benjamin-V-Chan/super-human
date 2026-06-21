# Prosthesis-RL SIM — Work Split

This is the single source of truth for ownership, task split, milestones, and timeline after the sim-only pivot. Product scope lives in [PRD.md](PRD.md), and contracts/evaluation live in [TECHNICAL_PLAN.md](TECHNICAL_PLAN.md).

## Pivot Impact

- Physical manufacturing is out of scope.
- CAD work becomes simulation morphology and geometry generation.
- The main proof is empirical sim evidence, not printable hardware.
- Perception is useful only insofar as it creates a concrete `TaskSpec`.
- The demo should emphasize reproducibility: launch sim, run policy, view metrics, watch rollout.

## Benji — Sim Morphology, AI Reasoning, Spatial Evaluation

Benji owns the reasoning and morphology side of the simulator: generating candidate simulated limbs, validating spatial feasibility, and deciding from empirical evidence which candidate is better.

### Responsibilities

- **Sim morphology generation:** produce `MorphologySpec` candidates with links, joints, limits, actuators, masses, and collision geometry.
- **AI design reasoning:** turn `TaskSpec` plus sim feedback into candidate morphology and controller-interface changes.
- **Spatial reasoning:** check reachability, workspace coverage, mount frames, joint limits, collision geometry, and task-space constraints.
- **Simulation geometry:** generate MJCF/URDF or mesh assets as sim inputs, not manufacturing outputs.
- **Empirical evaluation analysis:** compare candidate morphologies, inspect reward distributions, explain failure modes, and select the best sim design.
- **Validation gates:** reject invalid kinematic trees, impossible task reach, bad inertias, invalid actuator ranges, or unstable simulated bodies.

### Milestone

A candidate simulated limb can be generated, loaded into the sim, evaluated across rollouts, and justified with metrics.

## Vihaan — Orchestration, APIs, Task Intake, Demo Runtime

Vihaan owns the runnable system path: task intake, contracts, HUD/eval gateway, orchestration, and demo flow.

### Responsibilities

- **Orchestration:** wire task intake -> sim builder -> policy runner -> evaluator -> report.
- **HUD integration:** maintain `tasks.py`, the gateway, and the command path that returns real eval metrics.
- **Task intake:** convert clips, prompts, or ADL examples into validated `TaskSpec` records.
- **Shared contracts:** maintain `TaskSpec`, `MorphologySpec`, `SimSpec`, `PolicyArtifact`, and `EvalResult`.
- **Demo runtime:** make the viewer or local commands launch the sim and replay results cleanly.
- **Integration discipline:** keep stubs runnable while real morphology, sim, and policy pieces replace them.

### Milestone

The sim-only loop runs end to end and produces metrics plus a replayable demo artifact.

## Nathan — Physics Environment and Scenario Fidelity

Nathan owns the simulation environment and task realism.

### Responsibilities

- **MuJoCo/HUD environment:** build the scene, physics config, reset logic, observations, and action space.
- **Task scenarios:** implement concrete ADL-style tasks with measurable success criteria.
- **Physics fidelity:** tune contacts, object properties, joint dynamics, and episode constraints enough for believable sim behavior.
- **Deterministic verifier:** make fixed-seed rollouts reproducible and cheap enough for repeated evaluation.
- **Failure diagnostics:** expose collision, instability, unreachable target, and joint-limit failure signals to evaluation.

### Milestone

At least one ADL-style scenario runs deterministically with metrics suitable for comparing candidates.

## Vasi — Policy, RL, Reward Optimization

Vasi owns policy behavior and learning from sim rewards.

### Responsibilities

- **Baseline controller:** ship scripted or IK policy first so the demo has a reliable behavior floor.
- **RL training:** train a policy when the scenario and reward are stable enough.
- **Reward shaping:** define terms for success, distance, energy, joint-limit violations, and collisions.
- **Calibration:** tune reward and scenario difficulty so rollouts produce useful variance.
- **Policy artifact:** export a `.pt` checkpoint or runnable controller config that the evaluator can load.

### Milestone

A policy or controller can run repeated rollouts and show measurable task behavior in the sim.

## Shared Timeline

| Window | Benji | Vihaan | Nathan | Vasi |
| --- | --- | --- | --- | --- |
| Phase 1 | Morphology schema, spatial checks, first simulated limb | Contracts, task intake stub, eval command path | Minimal MuJoCo/HUD task scene | Scripted/IK baseline |
| Phase 2 | Candidate morphology generation and validation gates | End-to-end runner, viewer/replay wiring | Deterministic task metrics and reset logic | Reward shaping and calibration |
| Phase 3 | Compare candidates and write design rationale | Final demo flow and report assembly | Final scenario tuning and failure signals | RL attempt or polished baseline policy |

## Hard Checkpoints

- Sim-only scope is explicit in every public doc.
- A task launches in sim from documented commands.
- At least one morphology and policy/controller run through evaluation.
- Final artifacts include metrics, rollout video/replay, and policy/controller config.
