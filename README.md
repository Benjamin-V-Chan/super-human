# Prosthesis-RL SIM

**Simulation-first design and evaluation of assistive-limb behavior in virtual daily-living tasks.**

Prosthesis-RL SIM turns a short egocentric clip (or a structured task prompt) into a simulated assistive-limb morphology, runs a controller or RL policy against an Activities-of-Daily-Living (ADL) task in physics simulation, and reports whether the behavior works — with metrics, failure modes, and a replayable rollout.

It is a research and demo system. It is **not** a physical prosthesis, medical device, or manufacturing workflow. CAD-style geometry exists only as simulation input, never as a manufactured deliverable.

```text
task clip / prompt
      │
      ▼
   perception ──►  TaskSpec / spec sheet
      │
      ▼
   design ──────►  MorphologySpec  ──►  MJCF / STL sim asset
      │
      ▼
   simulation ──►  MuJoCo / HUD environment
      │
      ▼
   policy ──────►  scripted / IK / RL controller
      │
      ▼
   evaluation ──►  metrics + failure modes + rollout replay
```

## Why

The bet: AI design reasoning plus physics simulation can rapidly search assistive-limb morphologies and control policies, producing measurable, repeatable evidence *inside simulation* before any real-world hardware exists. The deliverable is a reproducible sim bundle — environment config, morphology, policy artifact, evaluation report, and rollout video.

## Pipeline & modules

The system is a single Python package, `prosthesis_rl`, wired together by an orchestrator that keeps every stage runnable (stubs stand in until real pieces land).

| Stage | Package | What it does |
| --- | --- | --- |
| **Perception** | `prosthesis_rl/cv`, `agents/perception.py` | Reads an ADL clip with a vision model, infers the primary action, affected/residual side, and emits a core spec sheet for design. |
| **Design / morphology** | `agents/design.py`, `agents/spec_sheet.py` | Generates `MorphologySpec` candidates (links, joints, limits, actuators, masses), runs spatial/reachability/validity gates, and ranks candidates from sim feedback. |
| **CAD / geometry** | `prosthesis_rl/cad` | Exports simulation geometry (MJCF for MuJoCo, STL for inspection). |
| **Simulation** | `prosthesis_rl/sim` | MuJoCo/HUD scene assembly, deterministic verifier, control bindings, room/scenario assets. |
| **Policy / RL** | `prosthesis_rl/rl` | Scripted/IK baseline controller, reward shaping, rollout, and RL training. |
| **Contracts** | `prosthesis_rl/contracts` | Shared dataclasses (`ProblemSpec`, `DesignParams`, `MorphologySpec`, `SimFeedback`, `EvalResult`, …) — the source of truth every stage agrees on. |
| **Orchestration / HUD** | `agents/orchestrator.py`, `prosthesis_rl/hud`, `tasks.py` | Wires intake → design → sim → eval into one runnable loop and exposes the gateway/command path. |

A web viewer (`viewer/`, Vite) replays results in the browser.

## Status

Active, mid-build. The front half of the pipeline is real and tested; the back half is scaffolded.

- 🟢 **Perception → design → geometry** — live: vision pipeline, labeled ADL dataset + eval harness, full design agent with validation gates, STL/MJCF export. Covered by tests.
- 🟡 **Simulation & policy** — modules scaffolded but stubbed; `mujoco` and `torch` are optional extras and not required for the stub loop.
- 🟡 **Contracts & orchestration** — core contracts in place; the end-to-end loop is being reconciled as real sim/policy pieces replace stubs.

## Quick start

### Python pipeline

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,cv]"        # add ",sim" / ",cad" for live MuJoCo / CAD
pytest -q                         # run the test suite
```

Optional extras (`pyproject.toml`): `dev` (pytest), `cv` (vision model), `sim` (MuJoCo ≥3.1), `cad` (CadQuery). Provider keys for the vision model go in `.env`.

### Web viewer

```bash
cd viewer
cp .env.example .env
npm install
npm run dev          # http://localhost:5173
```

## Repository layout

```
prosthesis_rl/    core package — cv, agents, cad, sim, rl, contracts, hud
scripts/          demo, eval, benchmark, and smoke-loop entrypoints
tests/            pytest suite
viewer/           Vite web viewer for rollout replay
assets/           generated sim geometry (mjcf/, stl/)
datasets/         labeled ADL clips + labels
test_vids/        ADL test clips
docs/             PRD, technical plan, work split
.agents/          AI-agent operating contract, roles, handoffs
```

## Docs

- [Product requirements (PRD)](docs/PRD.md) — scope, goals, non-goals, success metrics
- [Technical plan](docs/TECHNICAL_PLAN.md) — contracts, validation, evaluation
- [Work split](docs/WORK_SPLIT.md) — ownership, milestones, timeline
- [Agent operating contract](AGENTS.md) — how AI agents collaborate in this repo

## Scope

In scope: simulated morphology, runnable MuJoCo/HUD environment, scripted/RL policy, deterministic evaluation, rollout replay.

Out of scope: physical prosthesis delivery, clinical/medical claims, CAD for manufacturing, human-subject deployment, regulatory approval.

## License

See [LICENSE](LICENSE).
