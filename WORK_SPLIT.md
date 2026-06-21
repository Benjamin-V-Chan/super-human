# Prosthesis-RL — Work Split

This is the single source of truth for ownership, task split, milestones, and timeline. The README and PRD describe the project; this file describes who owns what.

AI agents follow [`AGENTS.md`](AGENTS.md). The machine-readable role registry and
delivery graph live in [`.agents/`](.agents/); they implement this human ownership
model without replacing it.

Shared contracts are the coordination boundary:

- `ProblemSpec`: task list plus patient constraints.
- `DesignParams`: candidate prosthesis geometry and control parameters.
- `Reward`: deterministic scalar score from the verifier.

## Benji — CAD, AI Design Reasoning, Spatial Evaluation

Benji owns the design-quality side of the loop: how the system reasons about prosthesis geometry, turns design parameters into manufacturable artifacts, and empirically decides whether one design is better than another.

### Tasks

- **CAD generation pipeline:** convert `DesignParams` into parametric CAD geometry through OpenSCAD/CadQuery and export STL artifacts.
- **Daytona CAD execution:** run CAD generation in the Daytona sandbox and keep generated outputs reproducible.
- **AI design models and reasoning:** build the design-agent logic that turns `ProblemSpec` plus sim feedback into candidate prosthesis parameters.
- **Spatial reasoning:** reason about workspace coverage, reachability, attachment geometry, joint limits, and collision-aware constraints.
- **Empirical evaluation:** define and inspect eval runs, compare candidate designs, track reward distributions, and diagnose design failure modes.
- **Design validation gates:** reject invalid geometry, impossible reach envelopes, unsafe joint limits, self-collisions, or non-manufacturable outputs.
- **Final design evidence:** produce the top-design STL, evaluation summary, and reasoning trace explaining why the selected design wins.

### Milestone

A generated design can be exported as STL, evaluated against verifier feedback, and justified with empirical evidence.

## Vihaan — Orchestration, APIs, Perception, Loop Execution

Vihaan owns the system plumbing that makes the project run end to end: the interfaces, APIs, task registry, perception intake, and executable demo flow.

### Tasks

- **Orchestration and API infrastructure:** wire together perceive -> design -> verify -> optimize -> manufacture.
- **HUD integration:** maintain `tasks.py`, the eval gateway, and the path that makes `hud eval tasks.py claude` return a real score.
- **Perception and task intake:** convert video or clip input into structured `ProblemSpec` data with schema validation.
- **Shared contracts:** maintain `ProblemSpec`, `DesignParams`, and `Reward` schemas so every component speaks the same language.
- **Loop execution:** close a stubbed end-to-end loop first, then replace stubs with real CAD, sim, and RL components.
- **Demo flow:** keep the system runnable and presentable from input clip through final output.

### Milestone

The end-to-end loop runs from input to score with stable contracts and replaceable components.

## Nathan — Physics Verifier and Task Realism

Nathan owns the verifier: the simulation world that decides whether a proposed design actually works on ADL tasks.

### Tasks

- **MuJoCo verifier:** build the parametric arm model, ADL scenes, and grading functions.
- **Physics and task realism:** cover reach, grasp, and feeding scenes with force windows, energy, ROM, and collision measurements.
- **Fast deterministic scoring:** make the same inputs produce the same reward and keep each verifier call cheap enough for repeated evaluation.
- **Fidelity stretch:** promote one task into Antim Worldsim/Newton if time allows.

### Milestone

The verifier can score candidate `DesignParams` deterministically on at least the first ADL task.

## Vasi — RL and Reward Optimization

Vasi owns the training layer that turns verifier feedback into better future designs.

### Tasks

- **GRPO training loop:** train the design agent using verifier rewards.
- **Controller:** ship scripted/IK control first so reward reflects design quality; attempt learned control only if time allows.
- **Reward shaping:** combine success, energy, ROM violation, and collision penalties into a useful scalar reward.
- **Calibration:** tune tasks toward useful reward variance, ideally 20-50% mean reward rather than all-pass or all-fail.

### Milestone

The first optimization run produces measurable changes in design quality or reward distribution.

## Shared Timeline

| Window | Benji | Vihaan | Nathan | Vasi |
| --- | --- | --- | --- | --- |
| Sat 12:30-7 PM | CAD bridge, design reasoning scaffold, validation gates | Schemas, orchestration scaffold, `tasks.py`, HUD gateway | Arm XML, reach task, grading | Scripted IK, reward v1 |
| Sat 7 PM-Sun 8 AM | Spatial checks, STL path, first empirical comparisons | Real intake to `ProblemSpec`, one personalized task live | Grasp/feed scenes, tune verifier variance | GRPO config, kickoff prep |
| Sun 8 AM-1 PM | Top design STL, eval summary, reasoning trace | Demo flow, loop video, integration cleanup | Final scoring and sim evidence | Pick training result, final eval |

## Hard Checkpoints

- Shared contracts locked before parallel work diverges.
- `hud eval tasks.py claude` returns a real number.
- At least one design flows through CAD, verifier scoring, and final evidence.
- Submission-ready demo and artifacts are ready by the final deadline.
