import { useCallback, useEffect, useRef, useState } from 'react'
import CadAssembly from './CadAssembly.jsx'
import SpecView from './SpecView.jsx'
import RayBanUpload from './RayBanUpload.jsx'
import { extractFrames } from './frames.js'
import { detectionToProblemSpec, detectionToDesign } from './mapping.js'
import { evaluateDesign } from '../sim/mujocoEval.js'
import './demo.css'

// Fully client-side single-clip pipeline (Vercel-deployable): the only server
// call is /api/analyze-frames (Gemini). Design, sim (MuJoCo WASM), policy and
// CAD all run in the browser — no Python, no filesystem.

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

const STAGES = [
  { key: 'capture', name: 'Ray-Ban Capture', role: 'Egocentric input', tech: 'Ray-Ban Meta · POV', icon: '🕶', emits: 'ADL clip',
    blurb: 'A first-person clip of the patient attempting a daily task — the single source of truth for the pipeline.' },
  { key: 'perception', name: 'Perception', role: 'Vision → task spec', tech: 'Gemini · frame sampler', icon: '👁', emits: 'ProblemSpec',
    blurb: 'Samples frames from the clip and asks Gemini for the specific action, affected/residual side, ROM, grip and limb sizing.' },
  { key: 'design', name: 'Design Agent', role: 'Morphology synthesis', tech: 'in-browser sizing', icon: '🦾', emits: 'DesignParams',
    blurb: 'Sizes the prosthesis from the perceived limb measurements, ROM and residual strength — links, joints, limits, stiffness.' },
  { key: 'simulation', name: 'Simulation', role: 'Physics verification', tech: 'MuJoCo WASM', icon: '🧪', emits: 'EvalResult',
    blurb: 'Builds an MJCF from the design and runs fixed-seed rollouts in-browser via MuJoCo WebAssembly.' },
  { key: 'policy', name: 'Policy / Control', role: 'Controller', tech: 'scripted IK', icon: '🎮', emits: 'PolicyArtifact',
    blurb: 'A scripted/IK controller artifact sized to the task — the behaviour floor the CAD model is built to run.' },
  { key: 'cad', name: 'CAD Builder', role: 'Geometry export', tech: 'three.js assembly', icon: '⬡', emits: 'CAD model',
    blurb: 'Materializes the validated morphology part-by-part into the exportable model — the end artifact.' },
]

const CAD_PARTS = [
  { key: 'socket', label: 'Shoulder socket' }, { key: 'upper', label: 'Upper-arm tube' },
  { key: 'elbow', label: 'Elbow joint' }, { key: 'forearm', label: 'Forearm tube' },
  { key: 'wrist', label: 'Wrist connector' }, { key: 'gripper', label: 'Two-jaw gripper' },
]

const SAMPLE = {
  primary_action: 'sample — upload a clip to analyze', affected_side: 'right', residual_side: 'left',
  tasks: ['reach', 'grasp'], rom: { shoulder_flexion: 110, elbow_flexion: 130, wrist_rotation: 60 },
  residual_strength: { shoulder: 0.6 }, grip_capacity: 0.4,
  residual_anthropometrics: { upper_arm_len: 0.3, forearm_len: 0.26, hand_length: 0.19, grip_span: 0.08 },
  pain_points: [], source: 'sample',
}
const FALLBACK = { ...SAMPLE, primary_action: 'analysis unavailable — default sizing' }

export default function DemoSingle() {
  const [status, setStatus] = useState({})
  const [revealed, setRevealed] = useState(0)
  const [running, setRunning] = useState(false)
  const [active, setActive] = useState('capture')
  const [clip, setClip] = useState(null)
  const [detection, setDetection] = useState(null)
  const [design, setDesign] = useState(null)
  const [evaluation, setEvaluation] = useState(null)
  const [policy, setPolicy] = useState(null)
  const runId = useRef(0)

  const cadParams = design || detectionToDesign(SAMPLE)
  const action = detection?.primary_action

  const reset = useCallback(() => {
    runId.current += 1
    setStatus({}); setRevealed(0); setRunning(false); setActive('capture')
    setDetection(null); setDesign(null); setEvaluation(null); setPolicy(null)
  }, [])

  const taskId = (a) => `${(a || 'adl_task').toLowerCase().replace(/[^a-z0-9]+/g, '_').slice(0, 28)}_v1`

  const outputFor = useCallback((key) => {
    switch (key) {
      case 'capture': return clip?.url
        ? { clip: clip.name, duration_s: clip.durationS ? Number(clip.durationS) : null, size_mb: Number(clip.sizeMB) }
        : { clip: '(none — upload a Ray-Ban clip)', view: 'egocentric' }
      case 'perception': return detectionToProblemSpec(detection)
      case 'design': return design
      case 'simulation': return evaluation
      case 'policy': return policy
      case 'cad': return { file: 'candidate.stl', parts: `${revealed}/${CAD_PARTS.length}`,
        mount_frame: cadParams.mount_frame, dof: cadParams.dof,
        status: revealed === CAD_PARTS.length ? 'complete' : 'assembling' }
      default: return null
    }
  }, [clip, detection, design, evaluation, policy, revealed, cadParams])

  const play = useCallback(async () => {
    runId.current += 1
    const myRun = runId.current
    const alive = () => runId.current === myRun
    setStatus({}); setRevealed(0); setRunning(true)
    let det = null, des = null, ev = null

    for (const stage of STAGES) {
      if (!alive()) return
      setActive(stage.key)
      setStatus((s) => ({ ...s, [stage.key]: 'running' }))
      try {
        if (stage.key === 'perception') {
          const t0 = Date.now()
          if (clip?.url) {
            const frames = await extractFrames(clip.url, 6)
            if (!alive()) return
            if (frames.length) {
              try {
                const r = await fetch('/api/analyze-frames', {
                  method: 'POST', headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ frames }),
                })
                const j = await r.json()
                det = j && j.primary_action ? j : { ...FALLBACK, source: j?.source || 'unavailable' }
              } catch { det = { ...FALLBACK, source: 'error' } }
            } else det = { ...FALLBACK, source: 'undecodable (try .mp4/H.264)' }
          } else det = { ...SAMPLE }
          des = detectionToDesign(det)
          if (!alive()) return
          setDetection(det); setDesign(des)
          await sleep(Math.max(0, 600 - (Date.now() - t0)))
        } else if (stage.key === 'design') {
          des = des || detectionToDesign(det)
          await sleep(700)
        } else if (stage.key === 'simulation') {
          ev = await evaluateDesign(des, taskId(det?.primary_action))
          if (!alive()) return
          setEvaluation(ev)
        } else if (stage.key === 'policy') {
          await sleep(600)
          setPolicy({
            kind: 'scripted_ik', path: `policies/${taskId(det?.primary_action)}.json`,
            inputs: ['observation'], outputs: ['joint_targets'],
            success_rate: ev?.success_rate ?? null,
          })
        } else if (stage.key === 'cad') {
          for (let p = 1; p <= CAD_PARTS.length; p++) {
            if (!alive()) return
            await sleep(850); setRevealed(p)
          }
        } else {
          await sleep(1200)
        }
      } catch (err) {
        if (!alive()) return
        setStatus((s) => ({ ...s, [stage.key]: 'error' }))
        setRunning(false); return
      }
      if (!alive()) return
      setStatus((s) => ({ ...s, [stage.key]: 'done' }))
    }
    setRunning(false)
  }, [clip])

  useEffect(() => () => { runId.current += 1 }, [])

  const completed = STAGES.filter((s) => status[s.key] === 'done').length
  const progress = Math.round((completed / STAGES.length) * 100)
  const activeStage = STAGES.find((s) => s.key === active)
  const activeOut = outputFor(active)

  return (
    <div className="demo">
      <header className="demo-header">
        <div className="demo-brand" style={{ cursor: 'pointer' }} onClick={() => { window.location.hash = '' }} title="Back to home">
          <span className="demo-logo">⬡</span>
          <div><div className="demo-title">ARMASAI</div><div className="demo-sub">Prosthesis Pipeline · single clip</div></div>
        </div>
        <div className="demo-flow-label">Ray-Ban clip <span className="arrow">→</span> Gemini <span className="arrow">→</span> design <span className="arrow">→</span> MuJoCo <span className="arrow">→</span> CAD</div>
        <div className="demo-actions">
          <div className="demo-progress"><div className="demo-progress-bar" style={{ width: `${progress}%` }} /><span>{progress}%</span></div>
          <button className="btn ghost" onClick={() => { window.location.hash = '#studio' }}>Studio →</button>
          <button className="btn" onClick={reset} disabled={!completed && !running}>Reset</button>
          <button className="btn primary" onClick={play} disabled={running}>{running ? '● Running…' : '▶ Run pipeline'}</button>
        </div>
      </header>

      <nav className="rail">
        {STAGES.map((stage, i) => {
          const st = status[stage.key] || 'idle'
          return (
            <div className="rail-cell" key={stage.key}>
              <button className={`chip ${st} ${active === stage.key ? 'focus' : ''}`} onClick={() => setActive(stage.key)}>
                <span className="chip-icon">{stage.icon}</span>
                <span className="chip-body"><span className="chip-name">{stage.name}</span><span className="chip-emit">{stage.emits}</span></span>
                <span className={`dot ${st}`} />
              </button>
              {i < STAGES.length - 1 && (
                <div className={`connector ${st === 'done' ? 'active' : ''} ${st === 'running' || (st === 'done' && status[STAGES[i + 1].key] === 'running') ? 'flowing' : ''}`}>
                  <div className="conn-line"><span className="packet" /></div>
                </div>
              )}
            </div>
          )
        })}
      </nav>

      <div className="body">
        <section className="col-left">
          <div className="panel input-panel">
            <div className="panel-head"><span>🕶 Ray-Ban input</span>{detection && <span className="src-tag">{detection.source}</span>}</div>
            <RayBanUpload clip={clip} onClip={setClip} sampling={status.perception === 'running'} />
          </div>
          <div className="panel detail-panel">
            <div className="panel-head"><span>{activeStage?.icon} {activeStage?.name}</span><span className="emit-tag">{activeStage?.emits}</span></div>
            <p className="detail-blurb">{activeStage?.blurb}</p>
            <SpecView data={activeOut} contract={activeStage?.emits} />
          </div>
        </section>

        <section className="col-right">
          <div className="cad-viewport">
            <div className="cad-tag">CAD OUTPUT · {action ? `“${action}”` : 'live assembly'}</div>
            <CadAssembly params={cadParams} revealed={revealed} />
            {revealed === 0 && <div className="cad-empty">Run the pipeline to materialize the model</div>}
          </div>
          <aside className="cad-side">
            <ol className="parts">
              {CAD_PARTS.map((p, i) => {
                const done = revealed > i, building = revealed === i && running
                return (
                  <li key={p.key} className={done ? 'done' : building ? 'building' : ''}>
                    <span className="part-dot" /><div><div className="part-label">{p.label}</div></div>
                    <span className="part-state">{done ? '✓' : building ? '⚙' : ''}</span>
                  </li>
                )
              })}
            </ol>
            <div className="cad-out">
              <div className="cad-out-row"><span>engine</span><b>{evaluation?.engine || '—'}</b></div>
              <div className="cad-out-row"><span>parts</span><b>{revealed}/{CAD_PARTS.length}</b></div>
            </div>
          </aside>
        </section>
      </div>
    </div>
  )
}
