# Agent Operating Contract

This file governs AI-agent work in this repository. `WORK_SPLIT.md` remains the
source of truth for human ownership. `.agents/agents.json` defines agent roles
and path ownership, while `.agents/workflow.json` defines the delivery graph.

## Coordinator

One coordinator owns each run. The coordinator does not duplicate feature work.
It turns the request into bounded tasks, assigns one writer per path, sequences
dependencies, reviews handoffs, runs integration checks, and reports blockers.

Before delegating, the coordinator must record for every task:

- the goal and acceptance criteria;
- writable paths and read-only dependencies;
- input and output contracts;
- validation commands;
- the upstream task or decision that blocks it.

Use no more than three worker agents concurrently so the coordinator retains one
slot for integration. Parallel work is allowed only when writable paths do not
overlap and required contracts are already stable.

## Source-of-truth order

When documents disagree, use this order:

1. `prosthesis_rl/contracts/schemas.py` for implemented Python contracts.
2. `WORK_SPLIT.md` for human ownership and milestones.
3. `.agents/agents.json` for AI-agent path ownership and validation.
4. `.agents/workflow.json` for dependencies and merge gates.
5. `docs/PRD.md` for product intent.
6. `README.md`, `plan.md`, and personal notes for context only.

Contract drift is an integration blocker. Do not silently reconcile conflicting
units, field shapes, or task IDs inside a feature module.

## Agent roles

The canonical role definitions are in `.agents/agents.json`:

- `coordinator`: shared contracts, orchestration, task registry, integration.
- `perception`: clip and frame input to validated `ProblemSpec`.
- `design_cad`: `ProblemSpec` and feedback to validated design and CAD output.
- `simulation`: deterministic ADL verification and `SimFeedback`.
- `optimization`: controller, rewards, candidate improvement, and calibration.
- `viewer_api`: UI and API adapter; it must not invent a separate schema.
- `qa_evidence`: tests, reproducibility checks, and demo evidence.

The QA agent is read-only over feature paths unless the coordinator explicitly
assigns a test or documentation fix. It reports defects to the owning agent.

## Workflow

Follow `.agents/workflow.json`. The normal execution order is:

1. Coordinator freezes the affected contract and task fixture.
2. Perception, design/CAD, and simulation work in parallel when independent.
3. Coordinator integrates one deterministic end-to-end loop.
4. Optimization iterates through design and verification with a bounded budget.
5. The winning candidate is exported and exposed through the viewer/API.
6. QA runs the repository gates and produces evidence.

Do not manufacture every candidate by default. Verify and rank candidates first,
then export the selected design unless a task explicitly tests CAD generation.

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

## Change control

- One agent writes a path at a time.
- Only the coordinator changes shared contracts, `tasks.py`, or the top-level
  orchestrator, unless it delegates that exact path explicitly.
- An agent needing a contract change stops at the boundary and proposes the
  field, type, units, consumers, migration, and test impact to the coordinator.
- Do not overwrite unrelated or user-authored changes.
- Do not add a new provider-specific schema or prompt when an existing shared
  contract can be used.
- Mark deterministic fallbacks and stubs explicitly; a passing fallback is not
  evidence that a live integration works.

## Repository gates

Run the smallest relevant check during feature work. Before integration handoff,
the coordinator and QA agent run the available applicable gates:

```bash
python3 -m pytest -q
PYTHONPATH=. python3 scripts/smoke.py
PYTHONPATH=. python3 scripts/run_loop.py
npm --prefix viewer run build
```

Additional required evidence:

- the same verifier input and seed produce the same reward;
- `tasks.py` returns a numeric score;
- selected CAD output exists and passes its validation gate;
- live-provider tests identify the provider and never report fallback output as
  a live success.

If an optional dependency is unavailable, report the skipped gate and the exact
dependency rather than weakening the check.
