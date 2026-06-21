// Client-side MuJoCo (WASM) evaluation of a prosthesis design — runs entirely in
// the browser so it deploys on Vercel with no Python/native mujoco. Loads the
// single-threaded build (no SharedArrayBuffer / COOP-COEP needed), serving the
// .wasm from /public so it resolves on both `vite dev` and Vercel static hosting.
//
// Falls back to a deterministic kinematic estimate if WASM can't initialize, so
// the pipeline always produces a design-dependent EvalResult.
import { buildMjcf, REACH_POSE } from './mjcf.js'

let _mujocoPromise = null
async function getMujoco() {
  if (!_mujocoPromise) {
    _mujocoPromise = import('@mujoco/mujoco').then(({ default: loadMujoco }) =>
      loadMujoco({ locateFile: (p) => (p.endsWith('.wasm') ? '/mujoco.wasm' : p) }))
  }
  return _mujocoPromise
}

const round = (x, n = 3) => Number(x.toFixed(n))

export async function evaluateDesign(design, taskId, { rollouts = 16, steps = 500 } = {}) {
  try {
    return await mujocoRollout(design, taskId, rollouts, steps)
  } catch (err) {
    console.warn('[mujoco-wasm] eval failed, using kinematic fallback:', err?.message)
    return kinematicFallback(design, taskId, rollouts, err?.message)
  }
}

async function mujocoRollout(design, taskId, rollouts, steps) {
  const mj = await getMujoco()
  const xml = buildMjcf(design)
  const model = mj.MjModel.from_xml_string(xml)
  const data = new mj.MjData(model)
  const dt = 0.004
  const SITE = mj.mjtObj.mjOBJ_SITE.value
  const gripId = mj.mj_name2id(model, SITE, 'grip')
  const gripPos = () => {
    const s = data.site_xpos
    return [s[gripId * 3], s[gripId * 3 + 1], s[gripId * 3 + 2]]
  }

  // Target = forward-kinematics of the reach pose (always physically reachable).
  mj.mj_resetData(model, data)
  for (let i = 0; i < REACH_POSE.length; i++) data.qpos[i] = REACH_POSE[i]
  mj.mj_forward(model, data)
  const target = gripPos()

  let successes = 0, collided = 0
  let sumReward = 0, sumEnergy = 0

  for (let r = 0; r < rollouts; r++) {
    mj.mj_resetData(model, data)
    // command the reach pose with a per-rollout deterministic perturbation
    const jitter = REACH_POSE.map((q, j) => q + (((r * 7 + j * 13) % 11) / 11 - 0.5) * 0.5)
    for (let j = 0; j < jitter.length; j++) data.ctrl[j] = jitter[j]

    let energy = 0, minDist = Infinity, anyContact = false
    for (let s = 0; s < steps; s++) {
      mj.mj_step(model, data)
      const f = data.actuator_force
      for (let j = 0; j < jitter.length; j++) energy += Math.abs(f[j]) * dt
      if (data.ncon > 0) anyContact = true
      if (s > steps - 120) {
        const g = gripPos()
        const d = Math.hypot(g[0] - target[0], g[1] - target[1], g[2] - target[2])
        if (d < minDist) minDist = d
      }
    }
    const success = minDist < 0.06
    if (success) successes++
    if (anyContact) collided++
    sumReward += (success ? 1 : 0) - minDist - 0.002 * energy
    sumEnergy += energy
  }

  model.delete(); data.delete()
  return {
    task_id: taskId, num_rollouts: rollouts,
    success_rate: round(successes / rollouts, 2),
    mean_reward: round(sumReward / rollouts, 3),
    mean_energy: round(sumEnergy / rollouts, 2),
    collision_rate: round(collided / rollouts, 2),
    video_path: `runs/${taskId}/rollout.mp4`,
    engine: 'mujoco-wasm',
  }
}

// Deterministic estimate from design geometry when WASM is unavailable.
function kinematicFallback(design, taskId, rollouts, note) {
  const reach = (design?.upper_arm_len ?? 0.3) + (design?.forearm_len ?? 0.26)
  const k = design?.joint_stiffness ?? 1.0
  const success = Math.max(0.4, Math.min(0.95, 0.6 + (reach - 0.5) * 0.8))
  const energy = round(0.2 + k * 0.12, 2)
  return {
    task_id: taskId, num_rollouts: rollouts,
    success_rate: round(success, 2),
    mean_reward: round(success - 0.1 - 0.05 * k, 3),
    mean_energy: energy,
    collision_rate: 0.05,
    video_path: `runs/${taskId}/rollout.mp4`,
    engine: 'kinematic-fallback',
    note: note ? `mujoco-wasm unavailable: ${note}` : 'mujoco-wasm unavailable',
  }
}
