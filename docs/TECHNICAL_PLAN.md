# Prosthesis-RL SIM — Technical Plan

This document owns architecture, data contracts, validation, and evaluation. Product scope lives in [PRD.md](PRD.md). Ownership lives in [WORK_SPLIT.md](WORK_SPLIT.md).

## Architecture

```text
Input
  task clip | ADL clip | structured prompt
        |
        v
Task Builder
  produces TaskSpec
        |
        v
Sim Builder
  produces SimSpec + MorphologySpec
        |
        v
Runtime
  MuJoCo/HUD environment + controller/policy
        |
        v
Evaluator
  metrics + rollout video + report
```

## Core Contracts

### `TaskSpec`

```jsonc
{
  "task_id": "reach_shelf_v1",
  "goal": "move the end effector to the target object",
  "objects": ["target_block", "table"],
  "success_condition": "end_effector_distance < 0.05 for 20 consecutive frames",
  "episode_seconds": 8.0,
  "assumptions": ["target pose estimated from prompt"]
}
```

### `MorphologySpec`

> **Implemented as `DesignParams`.** The design agent emits an explicit
> `DesignParams.links` chain (`LinkDef` + `JointDef`), which *is* the morphology —
> there is no separate `MorphologySpec` class. The shape below is the conceptual
> contract; `DesignParams` is the source of truth (see `prosthesis_rl/contracts`).

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

### `SimSpec`

```jsonc
{
  "scene": "tabletop_reach",
  "physics_hz": 100,
  "control_hz": 20,
  "initial_state_seed": 7,
  "reward_terms": ["success", "distance", "energy", "collision", "joint_limit"],
  "observations": ["joint_pos", "joint_vel", "target_pose", "end_effector_pose"]
}
```

### `PolicyArtifact`

```jsonc
{
  "kind": "scripted_ik",
  "path": "policies/reach_shelf_v1.json",
  "inputs": ["observation"],
  "outputs": ["joint_targets"]
}
```

### `EvalResult`

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

## Validation Gates

- `TaskSpec` has a measurable success condition.
- `MorphologySpec` has valid masses, inertias, joint axes, joint limits, and actuator limits.
- `SimSpec` launches with fixed seeds.
- Controller output matches the action space.
- Baseline rollout completes without simulator crashes.

## Evaluation Protocol

- Run at least one ADL-style task such as reach, stabilize, grasp, or feeding motion.
- Compare a baseline controller against one candidate improved controller or morphology.
- Run at least 10 fixed-seed rollouts per candidate.
- Report success rate, mean reward, reward variance, distance-to-goal, energy, joint-limit violations, collision rate, and stability.
- Save rollout video or viewer replay.

## Reward Shape

```text
reward = success_bonus
       - distance_penalty
       - energy_penalty
       - joint_limit_penalty
       - collision_penalty
```

Rewards should create meaningful differences between candidate morphologies and policies. The task should be neither trivially solved nor impossible for the baseline.

## Sim Asset Policy

Geometry is an internal simulation asset. OBJ, STL, MJCF, or URDF files may be generated to make the simulator run and visualize behavior. They are not manufacturing outputs.
