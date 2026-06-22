# Super Human

**AI-driven prosthetic limb design and simulation — built at the YC x HUD Hackathon.**

Super Human turns a short egocentric video clip into a personalized, physics-validated prosthetic limb design. A vision model reads the clip, infers the specific movements that are difficult, sizes a kinematic arm morphology from those measurements, runs fixed-seed rollouts in MuJoCo to score the candidate, and delivers evaluation metrics, a rollout replay, and exportable CAD geometry — all without touching physical hardware.

---

## Pipeline

```
egocentric clip (Ray-Ban / any camera)
        │
        ▼
  Gemini perception
    → TaskSpec: action, affected side, range-of-motion, grip pattern, limb measurements
        │
        ▼
  AI design agent
    → MorphologySpec / DesignParams: links, joints, limits, actuators, masses
    → spatial validation gates (reachability, inertia, collision geometry, kinematic tree)
        │
        ▼
  Simulation assembly
    → MJCF / URDF scene with the generated morphology
    → MuJoCo (native) or MuJoCo WASM (in-browser)
        │
        ▼
  Policy / controller
    → scripted IK baseline → optional RL training (PPO / SAC)
        │
        ▼
  Evaluation
    → success rate, mean reward, energy, collision rate, joint-limit violations
    → rollout video + web viewer replay
        │
        ▼
  CAD export
    → STL / OBJ per link, MJCF scene — ready for inspection or downstream manufacturing
```

---

## Architecture

The system is a single Python package (`prosthesis_rl`) with five functional layers, wired by an orchestrator:

| Layer | Module | Responsibility |
|---|---|---|
| **Perception** | `prosthesis_rl/cv`, `agents/perception.py` | Vision-model inference on ADL clips → `TaskSpec` with action label, affected/compensating side, grip type, joint ROM estimates |
| **Design / Morphology** | `agents/design.py`, `agents/spec_sheet.py` | LLM-driven candidate generation → `DesignParams` (explicit `LinkDef` + `JointDef` chains) → spatial/reachability/validity gates → sim-feedback ranking |
| **Geometry** | `prosthesis_rl/cad` | MJCF/URDF and STL export from `DesignParams`; per-link mesh generation via CadQuery |
| **Simulation** | `prosthesis_rl/sim` | MuJoCo scene assembly, deterministic verifier, action/observation spaces, room/scenario assets |
| **Policy / RL** | `prosthesis_rl/rl` | Scripted IK baseline; reward shaping (`success − distance − energy − joint_limit − collision`); RL training loop; `.pt` checkpoint export |
| **Contracts** | `prosthesis_rl/contracts` | Shared dataclasses (`TaskSpec`, `DesignParams`, `MorphologySpec`, `SimSpec`, `PolicyArtifact`, `EvalResult`) — every stage reads and writes these |
| **Orchestration** | `agents/orchestrator.py`, `prosthesis_rl/hud`, `tasks.py` | Intake → design → sim → eval loop; HUD gateway and command path |

A Vite web viewer (`viewer/`) replays rollouts in the browser and serves a live in-browser MuJoCo WASM demo of the generated arm.

---

## Data Contracts

Key schemas the pipeline exchanges between stages:

**`TaskSpec`** — what the task requires
```jsonc
{
  "task_id": "reach_shelf_v1",
  "goal": "move end effector to target object",
  "objects": ["target_block", "table"],
  "success_condition": "end_effector_distance < 0.05 for 20 consecutive frames",
  "episode_seconds": 8.0,
  "assumptions": ["target pose estimated from prompt"]
}
```

**`DesignParams` / `MorphologySpec`** — the generated limb
```jsonc
{
  "mount_frame": "torso_right",
  "links": [
    { "name": "upper", "length_m": 0.30, "mass_kg": 0.8 },
    { "name": "forearm", "length_m": 0.25, "mass_kg": 0.6 }
  ],
  "joints": [
    { "name": "shoulder_flexion", "type": "hinge", "limits_rad": [0.0, 2.1] },
    { "name": "elbow_flexion", "type": "hinge", "limits_rad": [0.0, 2.5] }
  ],
  "actuators": [
    { "joint": "shoulder_flexion", "torque_limit_nm": 20.0 },
    { "joint": "elbow_flexion", "torque_limit_nm": 15.0 }
  ]
}
```

**`EvalResult`** — what comes out
```jsonc
{
  "task_id": "reach_shelf_v1",
  "num_rollouts": 20,
  "success_rate": 0.75,
  "mean_reward": 0.42,
  "mean_energy": 0.31,
  "collision_rate": 0.05,
  "video_path": "runs/reach_shelf_v1/demo.mp4"
}
```

---

## Status

| Stage | Status |
|---|---|
| Perception → design → geometry | Live and tested — vision pipeline, labeled ADL dataset + eval harness, full design agent with validation gates, STL/MJCF export |
| Simulation & policy | Scaffolded; `mujoco` and `torch` are optional extras, stubs keep the loop runnable |
| Contracts & orchestration | Core contracts in place; end-to-end runner being reconciled as real sim/policy pieces land |
| Web viewer + WASM demo | Live — in-browser MuJoCo WASM, actuator sliders, force interaction |

---

## Quick Start

### Python pipeline

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,cv]"        # add ",sim" for live MuJoCo; ",cad" for CadQuery
pytest -q
```

Optional extras defined in `pyproject.toml`:

| Extra | Pulls in |
|---|---|
| `dev` | pytest, test utilities |
| `cv` | vision model client for perception |
| `sim` | MuJoCo ≥ 3.1 for native physics |
| `cad` | CadQuery for mesh generation |

Provider keys (Gemini / Anthropic) go in `.env`.

### Web viewer

```bash
cd viewer
cp .env.example .env
npm install
npm run dev          # http://localhost:5173
```

### In-browser MuJoCo WASM demo

```bash
# (once) generate the arm scene from the current design
python3 scripts/demo/export_web_arm.py    # → assets/scenes/arm.xml

# serve
cd webdemo && npm install
python3 -m http.server 8011
open http://localhost:8011
```

Use the Actuators panel to drive joints in real time, or drag any link to apply a physical force via `mj_applyFT`.

---

## Repository Layout

```
prosthesis_rl/     core package
  agents/          perception, design, spec_sheet, orchestrator
  cad/             MJCF/URDF/STL export (CadQuery)
  contracts/       shared dataclasses — source of truth for all stages
  cv/              vision model client + ADL dataset utilities
  hud/             HUD gateway and command path
  rl/              scripted IK baseline, reward shaping, RL training loop
  sim/             MuJoCo scene assembly, verifier, action/obs spaces
scripts/           demo, eval, benchmark, smoke-loop entrypoints
tests/             pytest suite
viewer/            Vite + Node web viewer for rollout replay
webdemo/           in-browser MuJoCo WASM live demo
assets/            generated sim geometry (mjcf/, stl/)
datasets/          labeled ADL clips + labels
test_vids/         ADL test clips
docs/              PRD, technical plan, work split, pitch context
.agents/           AI-agent operating contract, roles, handoffs
```

---

## Team

| Name | Role |
|---|---|
| **Benjamin Chan** | Technical Lead — AI design reasoning, sim morphology, spatial evaluation, architecture |
| **Vihaan Shringi** | Orchestration, APIs, task intake, demo runtime |
| **Nathan** | Physics environment and scenario fidelity (MuJoCo) |
| **Vasi** | Policy, RL, reward optimization |

---

## Docs

- [Product requirements (PRD)](docs/PRD.md) — scope, goals, non-goals, success metrics
- [Technical plan](docs/TECHNICAL_PLAN.md) — contracts, validation gates, evaluation protocol
- [Work split](docs/WORK_SPLIT.md) — ownership, milestones, timeline
- [Agent operating contract](AGENTS.md) — how AI agents collaborate in this repo

---

## Scope

**In scope:** simulated morphology, runnable MuJoCo/HUD environment, scripted or RL policy, deterministic evaluation, rollout replay, CAD geometry export.

**Out of scope:** physical prosthesis delivery, clinical or medical claims, regulatory approval, human-subject deployment.

---

## License

See [LICENSE](LICENSE).
