# Agent Operating Contract

This file governs AI-agent work in this repository. Human-facing product scope
lives in `docs/PRD.md`, implementation details live in
`docs/TECHNICAL_PLAN.md`, and human ownership lives in `docs/WORK_SPLIT.md`.

Machine-readable agent roles live in `.agents/agents.json`. The delivery graph
and merge gates live in `.agents/workflow.json`. Every task handoff should use
`.agents/handoff-template.md`.

## Coordinator

One coordinator owns each run. The coordinator does not duplicate feature work.
It turns the request into bounded tasks, assigns one writer per path, sequences
dependencies, reviews handoffs, runs integration checks, and reports blockers.

Before delegating, the coordinator records:

- goal and acceptance criteria;
- writable paths and read-only dependencies;
- input and output contracts;
- validation commands;
- upstream task or decision that blocks the work.

Use no more than three worker agents concurrently so the coordinator retains one
slot for integration. Parallel work is allowed only when writable paths do not
overlap and affected contracts are already stable.

## Source-of-Truth Order

When documents disagree, use this order:

1. Implemented Python contracts in `prosthesis_rl/contracts/`.
2. `docs/TECHNICAL_PLAN.md` for sim contracts, validation, and evaluation.
3. `docs/WORK_SPLIT.md` for human ownership and milestones.
4. `.agents/agents.json` for AI-agent path ownership and validation.
5. `.agents/workflow.json` for dependencies and merge gates.
6. `docs/PRD.md` for product intent.
7. `README.md` for public-facing context only.

Contract drift is an integration blocker. Do not silently reconcile conflicting
units, field shapes, task IDs, or reward semantics inside a feature module.

## Agent Roles

The canonical role definitions are in `.agents/agents.json`:

- `coordinator`: shared contracts, orchestration, task registry, integration.
- `task_intake`: clips/prompts to validated `TaskSpec`.
- `morphology_design`: simulated limb morphology, spatial checks, sim geometry.
- `simulation`: deterministic MuJoCo/HUD scenarios and verifier signals.
- `policy_optimization`: controllers, rewards, RL attempts, policy artifacts.
- `viewer_api`: local viewer and API boundary for replay/results.
- `qa_evidence`: tests, determinism checks, and demo evidence.

The QA agent is read-only over feature paths unless the coordinator explicitly
assigns a test or documentation fix. It reports defects to the owning agent.

## Workflow

Follow `.agents/workflow.json`. The normal execution order is:

1. Coordinator freezes the affected task, contract, and fixture.
2. Task intake, morphology, and simulation work in parallel when independent.
3. Coordinator integrates one deterministic sim rollout.
4. Policy optimization iterates through bounded rollout/evaluation loops.
5. Viewer/API packages replay, metrics, and policy/controller artifacts.
6. QA runs repository gates and produces evidence.

Do not turn sim geometry into manufacturing claims. Meshes, MJCF, URDF, and STL
files are simulation assets unless a task explicitly says otherwise.

## Handoffs

Every completed task must provide a handoff using
`.agents/handoff-template.md`. At minimum it includes:

- changed files and any shared files intentionally left unchanged;
- input fixture and output artifact or serialized contract;
- seed, configuration, and fallback/stub status;
- exact validation commands and results;
- unresolved risks and the next owning role.

Machine-readable outputs are preferred. Artifact paths must be relative to the
repository, and generated artifacts must not be committed unless requested.

## Change Control

- One agent writes a path at a time.
- Only the coordinator changes shared contracts, `tasks.py`, or the top-level
  orchestrator unless it delegates that exact path explicitly.
- An agent needing a contract change stops at the boundary and proposes field,
  type, units, consumers, migration, and test impact to the coordinator.
- Do not overwrite unrelated or user-authored changes.
- Do not add provider-specific schemas when an existing shared contract can be
  used.
- Mark deterministic fallbacks and stubs explicitly; a passing fallback is not
  evidence that a live integration works.

## Repository Gates

Run the smallest relevant check during feature work. Before integration handoff,
the coordinator and QA agent run the applicable gates:

```bash
python3 -m pytest -q
PYTHONPATH=. python3 scripts/smoke.py
PYTHONPATH=. python3 scripts/run_loop.py
npm --prefix viewer run build
```

Additional required evidence:

- fixed seeds produce repeatable sim metrics;
- `tasks.py` returns a numeric score;
- policy/controller artifacts can be loaded by the runner;
- rollout video or replay path is recorded when the task affects demo behavior;
- live-provider tests identify the provider and never report fallback output as
  a live success.

If an optional dependency is unavailable, report the skipped gate and the exact
dependency rather than weakening the check.
