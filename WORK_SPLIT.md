# Prosthesis-RL — Work Split & Ownership

Companion to the PRD. Each person owns a vertical slice; the contracts between slices (`ProblemSpec`, `DesignParams`, reward scalar) are the only things everyone must agree on. Lock those by Saturday 2 PM.

---

## VIHAAN + BENJI — Agent architecture + API infra (own the loop)

You build the orchestration connecting all three stages, plus the HUD integration. If the loop doesn't close, nothing else matters — so your job is to get a **dumb end-to-end loop running by Saturday night with every component stubbed**, then let Nathan and Vasi swap in real pieces.

**Deliverables**
- **Perception agent:** clip → VLM (Claude vision) → `ProblemSpec` JSON. Strict schema, validated output.
- **Design agent scaffold:** takes `ProblemSpec` + last sim feedback → emits `DesignParams` + control hints. You build the action/observation interface; Vasi trains the policy inside it.
- **HUD task registry (`tasks.py`):** register the ADL tasks, wire the gateway, make `hud eval tasks.py claude` run end to end.
- **CAD bridge:** `DesignParams` → CadQuery/OpenSCAD → STL, executed inside the Daytona sandbox.

**Credits:** HUD (env + eval API), Anthropic (Claude vision + reasoning), Modal (if serving inference).
**Milestone:** `hud eval tasks.py claude` returns a real number by dinner Saturday.

---

## NATHAN — CV + physics sim (the verifier)

Two halves. The sim is the heart of the whole project — make it faithful enough that a design winning in your env would plausibly work on a real arm.

**Deliverables**
- **CV/perception backend:** frame extraction → VLM call → pain-point detection. Run on Modal if heavy. Produces the `ProblemSpec` that the agent consumes.
- **MuJoCo environment + grading:** parametric arm model (XML generated from `DesignParams`), ADL task scenes, and grading functions — reach success, grasp force window, energy, ROM violation, self-collision. Must be **deterministic and fast** (called thousands of times).
- **Stretch:** promote one task into Antim Worldsim/Newton for a fidelity story.

**Credits:** Modal ($250 GPU for CV), Antim Labs (Worldsim + physical validation), OpenAI (vision backup if Claude bottlenecks).
**Milestone:** Reach task (1.1) + grading callable by the design agent by dinner Saturday.

---

## VASI — RL + optimization (make the designer smart)

You train the design agent and shape the reward. You're the one making the designer learn from the verifier instead of guessing.

**Deliverables**
- **Inner controller:** scripted/IK controller first, so reward reflects *design quality* not control noise. Upgrade to a learned policy only if time allows.
- **Reward shaping:** success − energy − ROM-violation − collision, weighted per tier (see PRD §7). Tune so every task lands at **20–50% mean reward with real variance**.
- **GRPO loop (Fireworks/HUD):** roll out the design agent ~10×/task, train on trajectories where good designs got rewarded.

**Credits:** Fireworks AI ($30 RL training), HUD (platform training + evals), Google DeepMind (only if you need GCP for a bigger run).
**Milestone:** First training run kicked off by **8 AM Sunday** — hard deadline.

---

## Shared contracts (lock by Sat 2 PM)

- `ProblemSpec` = `{ tasks: [...], constraints: { rom, residual_strength, grip_capacity } }`
- `DesignParams` = `{ upper_arm_len, forearm_len, joint_stiffness, grip_width, joint_limits }`
- Reward = single deterministic scalar per episode, computed in Nathan's verifier.

## Timeline at a glance

| Window | V+B | Nathan | Vasi |
|--------|-----|--------|------|
| Sat 12:30–7 PM | Schema + agent scaffold + `tasks.py` + CAD bridge | Arm XML + Reach task + grading | Scripted IK + reward v1 |
| Sat 7 PM–Sun 8 AM | Real CV → real `ProblemSpec`; 1 personalized task live | Add grasp + feeding, tune 20–50% | GRPO config; **kick off run by 8 AM** |
| Sun 8 AM–1 PM | Loop video; converge | STL export; LeRobot demo | Pick top design; final eval |

**Sun 1 PM submission · 2:30 PM top-10 pres.**
