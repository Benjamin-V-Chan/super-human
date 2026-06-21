# Prosthesis-RL SIM — Product Requirements Document

> **Status:** Draft
> **Product pivot:** SIM-only. The end product is a reproducible simulator, trained policy, and evaluation artifact. It is not a physical prosthesis, medical device, printable product, or manufacturing workflow.
> **Related docs:** [TECHNICAL_PLAN.md](TECHNICAL_PLAN.md), [WORK_SPLIT.md](WORK_SPLIT.md)

## 1. Overview

**Prosthesis-RL SIM** is a simulation-first system for designing, testing, and optimizing assistive-limb behaviors in virtual Activities of Daily Living (ADL) scenarios. The system takes a short egocentric clip or structured task description, converts it into a simulated task scenario, generates a candidate limb morphology and controller interface, trains or evaluates a policy in simulation, and reports whether the assistive behavior works.

The product outcome is a **reproducible sim demo**:

- A generated or configured assistive-limb morphology in simulation.
- A runnable MuJoCo/HUD task environment.
- A trained or scripted policy checkpoint.
- Quantitative evaluation results.
- A visual demo of the policy completing the task in sim.

## 2. Product Bet

The core bet is that AI design reasoning plus physics simulation can rapidly search assistive-limb morphologies and control policies without claiming physical readiness. The system is valuable because it can create measurable, repeatable evidence inside simulation before any real-world hardware work exists.

## 3. Users

**Primary demo user:** a researcher, builder, or evaluator who wants to test whether a simulated assistive limb can complete a concrete ADL task.

**Secondary user:** a team member who wants to compare morphology/controller choices using reproducible rewards, videos, and metrics.

## 4. Goals

- Convert a task clip or structured prompt into a concrete simulation scenario.
- Generate a candidate simulated limb morphology with valid kinematics, joints, limits, masses, and collision geometry.
- Train or run a controller/policy for the simulated task.
- Evaluate candidate designs and policies with deterministic, repeatable metrics.
- Produce a final sim package: environment config, policy checkpoint, metrics, and demo video.

## 5. Non-Goals

- Physical prosthesis delivery.
- Clinical validation or medical claims.
- CAD for manufacturing, printable STL as a final product, or hardware-ready design.
- Human subject deployment.
- Regulatory approval.

CAD-style geometry can still exist as **simulation geometry**, but it is an internal asset for MuJoCo/HUD rather than a manufactured deliverable.

## 6. End-to-End Flow

```text
task clip or prompt
    |
    v
TaskSpec
    |
    v
SimSpec + MorphologySpec
    |
    v
MuJoCo/HUD environment
    |
    v
controller or RL policy
    |
    v
evaluation metrics + rollout video + checkpoint
```

## 7. Core Requirements

### Task Understanding

- Accept a short egocentric clip, existing ADL clip, or structured task prompt.
- Produce a `TaskSpec` with task goal, objects, success condition, observation needs, and environment assumptions.
- Avoid overclaiming patient-specific medical inference; uncertain clip details should become explicit assumptions.

### Simulation Assembly

- Convert `TaskSpec` into a `SimSpec` that defines scene geometry, objects, target poses, reward terms, initial states, and episode length.
- Generate or configure a simulated assistive-limb morphology from `MorphologySpec`.
- Validate the simulated model before policy runs: joint limits, mass/inertia, actuator ranges, collision settings, and reachable workspace.

### Policy and Control

- Support a scripted or IK controller as the baseline.
- Support RL training when time allows.
- Export the final controller as a policy artifact, such as a `.pt` checkpoint or equivalent runnable config.

### Evaluation

- Run repeated rollouts with fixed seeds.
- Report task success, reward distribution, energy, collisions, constraint violations, and stability.
- Save rollout video or viewer replay for the final demo.

## 8. Success Metrics

| Metric | Target |
| --- | --- |
| Runnable sim | A task environment launches locally or through HUD |
| Policy artifact | A scripted policy or `.pt` checkpoint can be loaded and run |
| Determinism | Fixed seed rollouts produce repeatable metrics |
| Task success | Final demo completes at least one clear ADL-style task in sim |
| Eval evidence | Report includes success rate, reward, failure modes, and rollout video |
| Scope clarity | Docs make clear that the product is sim-only |

## 9. Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Sim task is too vague | Policy cannot be evaluated cleanly | Define task success and failure before training |
| Morphology invalid | MuJoCo/HUD cannot run the environment | Add model validation gates before rollout |
| Reward has no useful variance | RL cannot improve | Tune reward and scenario difficulty with baseline rollouts |
| Clip interpretation is uncertain | Scenario becomes speculative | Record assumptions in `TaskSpec` |
| Demo depends on training instability | Final demo may fail | Keep scripted/IK baseline runnable |

## 10. Final Deliverable

The final deliverable is a sim bundle:

- Task/scenario config.
- Simulated assistive-limb morphology.
- Controller or policy artifact.
- Evaluation report.
- Rollout video or viewer replay.

Any physical interpretation is explicitly out of scope.
