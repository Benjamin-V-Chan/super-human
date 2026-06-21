import { useState } from 'react'
import JsonBlock from './JsonBlock.jsx'

// Human-readable renderer for a contract payload: snake_case keys → Title Case,
// numbers formatted with units, arrays as chips, nested objects as sub-sections.
// Toggle to raw JSON for the literal contract.

const LABELS = {
  primary_action: 'Primary action', affected_side: 'Affected side', residual_side: 'Residual side',
  success_condition: 'Success condition', episode_seconds: 'Episode', mount_frame: 'Mount frame',
  upper_arm_len: 'Upper arm', forearm_len: 'Forearm', grip_width: 'Grip width', grip_span: 'Grip span',
  joint_stiffness: 'Joint stiffness', arm_radius: 'Arm radius', joint_names: 'Joints', dof: 'DoF',
  num_rollouts: 'Rollouts', success_rate: 'Success rate', mean_reward: 'Mean reward',
  mean_energy: 'Mean energy', collision_rate: 'Collision rate', video_path: 'Video',
  grip_capacity: 'Grip capacity', hand_length: 'Hand length', shoulder_flexion: 'Shoulder flex',
  elbow_flexion: 'Elbow flex', wrist_rotation: 'Wrist rot', task_id: 'Task',
}
const METERS = new Set(['upper_arm_len', 'forearm_len', 'grip_width', 'grip_span', 'arm_radius', 'hand_length'])
const DEGREES = new Set(['shoulder_flexion', 'elbow_flexion', 'wrist_rotation'])
const RATES = new Set(['success_rate', 'collision_rate', 'grip_capacity'])

const humanize = (k) => LABELS[k] || k.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())

function fmt(key, v) {
  if (typeof v === 'number') {
    if (METERS.has(key)) return `${(v * 100).toFixed(1)} cm`
    if (DEGREES.has(key)) return `${v}°`
    if (RATES.has(key)) return `${Math.round(v * 100)}%`
    return String(v)
  }
  return String(v)
}

function Rows({ data }) {
  return (
    <div className="spec-rows">
      {Object.entries(data).map(([k, v]) => {
        if (k === 'source') return null
        if (Array.isArray(v)) {
          return (
            <div className="spec-row" key={k}>
              <span className="spec-k">{humanize(k)}</span>
              <span className="spec-chips">
                {v.length ? v.map((x, i) => <span className="chip" key={i}>{String(x)}</span>)
                  : <span className="spec-dim">—</span>}
              </span>
            </div>
          )
        }
        if (v && typeof v === 'object') {
          return (
            <div className="spec-sub" key={k}>
              <div className="spec-subhead">{humanize(k)}</div>
              <div className="spec-subrows">
                {Object.entries(v).map(([k2, v2]) => (
                  <div className="spec-row tight" key={k2}>
                    <span className="spec-k">{humanize(k2)}</span>
                    <span className="spec-v">{fmt(k2, v2)}</span>
                  </div>
                ))}
              </div>
            </div>
          )
        }
        return (
          <div className="spec-row" key={k}>
            <span className="spec-k">{humanize(k)}</span>
            <span className="spec-v">{fmt(k, v)}</span>
          </div>
        )
      })}
    </div>
  )
}

export default function SpecView({ data, contract }) {
  const [raw, setRaw] = useState(false)
  if (!data) return null
  return (
    <div className="spec">
      <div className="spec-head">
        <span className="spec-contract">{contract}</span>
        <div className="spec-toggle">
          <button className={!raw ? 'on' : ''} onClick={() => setRaw(false)}>readable</button>
          <button className={raw ? 'on' : ''} onClick={() => setRaw(true)}>json</button>
        </div>
      </div>
      {raw ? <JsonBlock data={data} /> : <Rows data={data} />}
    </div>
  )
}
