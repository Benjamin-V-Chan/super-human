import { useCallback, useEffect, useRef, useState } from 'react'
import CadAssembly from './CadAssembly.jsx'
import SpecView from './SpecView.jsx'
import RayBanUpload from './RayBanUpload.jsx'
import { extractFrames } from './frames.js'
import { detectionToProblemSpec, detectionToDesign } from './mapping.js'
import { PIPELINE, CAD_PARTS, TIMING } from './demoData.js'
import { downloadStl, postJson } from '../lib/api.js'
import { evaluateDesign } from '../sim/mujocoEval.js'
import './demo.css'

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

// Used only when NO clip is uploaded (pure demo) or analysis can't run, so we
// never falsely report a specific action for an unrelated clip.
const SAMPLE = {
  primary_action: 'sample — upload a clip to analyze', affected_side: 'right', residual_side: 'left',
  tasks: ['reach', 'grasp'], rom: { shoulder_flexion: 110, elbow_flexion: 130, wrist_rotation: 60 },
  residual_strength: { shoulder: 0.6 }, grip_capacity: 0.4,
  residual_anthropometrics: { upper_arm_len: 0.3, forearm_len: 0.26, hand_length: 0.19, grip_span: 0.08 },
  pain_points: [], source: 'sample',
}

const DESIGN_FALLBACK = { ...SAMPLE, primary_action: 'analysis unavailable — default sizing' }

export default function DemoPage() {
  const [status, setStatus] = useState({})
  const [revealed, setRevealed] = useState(0)
  const [running, setRunning] = useState(false)
  const [active, setActive] = useState('capture')
  const [clip, setClip] = useState(null)
  const [detection, setDetection] = useState(null)
  const [design, setDesign] = useState(null)
  const [evaluation, setEvaluation] = useState(null)
  const [policyArtifact, setPolicyArtifact] = useState(null)
  const [cadArtifact, setCadArtifact] = useState(null)
  const [exporting, setExporting] = useState(false)
  const runId = useRef(0)

  const cadParams = design || detectionToDesign(SAMPLE)
  const action = detection?.primary_action

  const reset = useCallback(() => {
    runId.current += 1
    setStatus({}); setRevealed(0); setRunning(false)
    setDetection(null); setDesign(null); setEvaluation(null)
    setPolicyArtifact(null); setCadArtifact(null); setActive('capture')
  }, [])

  // Resolve each stage's output payload from live state.
  const outputFor = useCallback((key) => {
    switch (key) {
      case 'capture':
        return clip?.url
          ? { clip: clip.name, duration_s: clip.durationS ? Number(clip.durationS) : null,
              size_mb: Number(clip.sizeMB), saved_to: clip.serverPath || '(local preview)' }
          : { clip: '(none — upload a Ray-Ban clip)', view: 'egocentric' }
      case 'perception': return detectionToProblemSpec(detection)
      case 'design': return design
      case 'simulation': return evaluation || { status: 'not run' }
      case 'policy': return policyArtifact || { status: 'not run' }
      case 'cad': return cadArtifact || { status: 'not run', parts: `${revealed}/${CAD_PARTS.length}` }
      default: return null
    }
  }, [clip, detection, design, evaluation, policyArtifact, cadArtifact, revealed])

  const play = useCallback(async () => {
    runId.current += 1
    const myRun = runId.current
    const alive = () => runId.current === myRun
    setStatus({}); setRevealed(0); setRunning(true)
    setEvaluation(null); setPolicyArtifact(null); setCadArtifact(null)
    let det = detection, des = design, evalResult = null

    for (const stage of PIPELINE) {
      if (!alive()) return
      setActive(stage.key)
      setStatus((s) => ({ ...s, [stage.key]: 'running' }))

      try {
      if (stage.key === 'perception') {
        // Real Gemini analysis of the uploaded clip's frames.
        const t0 = Date.now()
        if (clip?.serverPath) {
          det = await postJson('/api/analyze-clip', { clip_path: clip.serverPath })
          det = { ...det, source: det.source || 'python-perception-agent' }
        } else if (clip?.url) {
          const frames = await extractFrames(clip.url, 6)
          if (!alive()) return
          if (frames.length) {
            try {
              const r = await fetch('/api/analyze-frames', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ frames }),
              })
              const j = await r.json()
              det = j && j.primary_action ? j : { ...DESIGN_FALLBACK, source: j?.source || 'unavailable' }
            } catch { det = { ...DESIGN_FALLBACK, source: 'error' } }
          } else {
            det = { ...DESIGN_FALLBACK, source: 'undecodable (try .mp4/H.264)' }
          }
        } else {
          det = { ...SAMPLE }
        }
        if (!alive()) return
        setDetection(det)
        await sleep(Math.max(0, 700 - (Date.now() - t0)))
      } else if (stage.key === 'design') {
        const baseVisuals = detectionToDesign(det)
        des = await postJson('/api/derive-design', { problem: detectionToProblemSpec(det) })
        des = { ...baseVisuals, ...des, source: 'python-design-agent' }
        if (!alive()) return
        setDesign(des)
      } else if (stage.key === 'simulation') {
        const taskId = `${(det?.primary_action || 'adl_task').toLowerCase().replace(/[^a-z0-9]+/g, '_').slice(0, 28)}_v1`
        // Run the eval in-browser via MuJoCo WASM (Vercel-friendly; no native mujoco).
        evalResult = await evaluateDesign(des, taskId)
        if (!alive()) return
        setEvaluation(evalResult)
      } else if (stage.key === 'policy') {
        const name = `${(det?.primary_action || 'adl').toLowerCase().replace(/[^a-z0-9]+/g, '_').slice(0, 24)}_policy`
        const artifact = await postJson('/api/build-policy', {
          problem: detectionToProblemSpec(det), design: des, evaluation: evalResult, name,
        })
        if (!alive()) return
        setPolicyArtifact(artifact)
      } else if (stage.key === 'cad') {
        const artifact = await postJson('/api/build-cad', { design: des, name: 'candidate' })
        if (!alive()) return
        setCadArtifact(artifact)
        for (let p = 1; p <= CAD_PARTS.length; p++) {
          if (!alive()) return
          await sleep(TIMING.cadPerPart)
          setRevealed(p)
        }
      } else {
        await sleep(TIMING[stage.key] ?? 1500)
      }
      } catch (error) {
        if (!alive()) return
        setStatus((s) => ({ ...s, [stage.key]: 'error' }))
        if (stage.key === 'simulation') setEvaluation({ status: 'error', error: error.message })
        if (stage.key === 'policy') setPolicyArtifact({ status: 'error', error: error.message })
        if (stage.key === 'cad') setCadArtifact({ status: 'error', error: error.message })
        setRunning(false)
        return
      }
      if (!alive()) return
      setStatus((s) => ({ ...s, [stage.key]: 'done' }))
    }
    setRunning(false)
  }, [clip, detection, design])

  const exportStl = useCallback(async () => {
    if (!design || exporting) return
    setExporting(true)
    try { await downloadStl(design, 'candidate') }
    finally { setExporting(false) }
  }, [design, exporting])

  useEffect(() => () => { runId.current += 1 }, [])

  const completed = PIPELINE.filter((s) => status[s.key] === 'done').length
  const progress = Math.round((completed / PIPELINE.length) * 100)
  const activeStage = PIPELINE.find((s) => s.key === active)
  const activeOut = outputFor(active)

  const partNote = (key) => {
    const cm = (m) => `${Math.round(m * 100)} cm`
    return {
      socket: `mount · ${cadParams.mount_frame}`,
      upper: `carbon-fiber · ${cm(cadParams.upper_arm_len)}`,
      elbow: `hinge · k=${cadParams.joint_stiffness}`,
      forearm: `carbon-fiber · ${cm(cadParams.forearm_len)}`,
      wrist: 'rotary coupling',
      gripper: `${cm(cadParams.grip_width)} aperture`,
    }[key]
  }

  return (
    <div className="demo">
      <header className="demo-header">
        <div className="demo-brand">
          <span className="demo-logo">⬡</span>
          <div>
            <div className="demo-title">ARMASAI</div>
            <div className="demo-sub">Multi-Agent Prosthesis Pipeline</div>
          </div>
        </div>
        <div className="demo-flow-label">
          Ray-Ban clip <span className="arrow">→</span> Gemini perception <span className="arrow">→</span> design <span className="arrow">→</span> CAD
        </div>
        <div className="demo-actions">
          <div className="demo-progress"><div className="demo-progress-bar" style={{ width: `${progress}%` }} /><span>{progress}%</span></div>
          <button className="btn ghost" onClick={() => { window.location.hash = '#studio' }}>Design Studio →</button>
          <button className="btn" onClick={reset} disabled={!completed && !running}>Reset</button>
          <button className="btn primary" onClick={play} disabled={running}>{running ? '● Running…' : '▶ Run pipeline'}</button>
        </div>
      </header>

      {/* ── Linear agent rail (all stages, one row) ── */}
      <nav className="rail">
        {PIPELINE.map((stage, i) => {
          const st = status[stage.key] || 'idle'
          return (
            <div className="rail-cell" key={stage.key}>
              <button
                className={`chip ${st} ${active === stage.key ? 'focus' : ''}`}
                onClick={() => setActive(stage.key)}
              >
                <span className="chip-icon">{stage.icon}</span>
                <span className="chip-body">
                  <span className="chip-name">{stage.name}</span>
                  <span className="chip-emit">{stage.emits}</span>
                </span>
                <StatusDot status={st} />
              </button>
              {i < PIPELINE.length - 1 && (
                <Connector active={st === 'done'} flowing={st === 'running' || (st === 'done' && status[PIPELINE[i + 1].key] === 'running')} />
              )}
            </div>
          )
        })}
      </nav>

      {/* ── Body: input + detail (left) · CAD (right) ── */}
      <div className="body">
        <section className="col-left">
          <div className="panel input-panel">
            <div className="panel-head"><span>🕶 Ray-Ban input</span>{detection && <span className="src-tag">{detection.source}</span>}</div>
            <RayBanUpload clip={clip} onClip={setClip} sampling={status.perception === 'running'} />
          </div>

          <div className="panel detail-panel">
            <div className="panel-head">
              <span>{activeStage?.icon} {activeStage?.name}</span>
              <span className="emit-tag">{activeStage?.emits}</span>
            </div>
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
                    <span className="part-dot" />
                    <div><div className="part-label">{p.label}</div><div className="part-note">{partNote(p.key)}</div></div>
                    <span className="part-state">{done ? '✓' : building ? '⚙' : ''}</span>
                  </li>
                )
              })}
            </ol>
            <button className="btn primary block" disabled={revealed < CAD_PARTS.length}
              onClick={exportStl}>{exporting ? 'Exporting…' : '⬇ Export STL'}</button>
          </aside>
        </section>
      </div>

    </div>
  )
}

function StatusDot({ status }) { return <span className={`dot ${status || 'idle'}`} /> }

function Connector({ active, flowing }) {
  return (
    <div className={`connector ${active ? 'active' : ''} ${flowing ? 'flowing' : ''}`}>
      <div className="conn-line"><span className="packet" /></div>
    </div>
  )
}
