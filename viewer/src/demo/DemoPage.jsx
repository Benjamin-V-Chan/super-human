import { useCallback, useEffect, useRef, useState } from 'react'
import CadAssembly from './CadAssembly.jsx'
import MultiClipUpload from './MultiClipUpload.jsx'
import AgentWorkPanel from './AgentWorkPanel.jsx'
import MechanicalReportPanel from './MechanicalReportPanel.jsx'
import OptimizationTrace from './OptimizationTrace.jsx'
import { detectionToDesign } from './mapping.js'
import { CAD_PARTS } from './demoData.js'
import { downloadStl } from '../lib/api.js'
import './demo.css'

// Pipeline stage definitions for the rail
const PIPELINE_STAGES = [
  { key: 'upload',      name: 'Upload',       icon: '🎥', emits: 'Clips' },
  { key: 'perception',  name: 'Perception',   icon: '👁',  emits: 'ProblemSpec × N' },
  { key: 'design',      name: 'Design Agent', icon: '🦾', emits: 'Candidates' },
  { key: 'sim_eval',    name: 'Sim + Mech',   icon: '🧪', emits: 'EvalResult + FoS' },
  { key: 'rl_loop',     name: 'RL Loop',      icon: '🎮', emits: 'PolicyArtifact' },
  { key: 'final',       name: 'Final Design', icon: '⬡',  emits: 'CAD + Report' },
]

export default function DemoPage() {
  const [clips, setClips] = useState([])
  const [running, setRunning] = useState(false)
  const [stageStatus, setStageStatus] = useState({})
  const [activeStage, setActiveStage] = useState('upload')

  // Pipeline outputs
  const [clipObservations, setClipObservations] = useState([])
  const [unifiedRequirements, setUnifiedRequirements] = useState(null)
  const [agentFindings, setAgentFindings] = useState([])
  const [workTraces, setWorkTraces] = useState([])
  const [designIterations, setDesignIterations] = useState([])
  const [mechReport, setMechReport] = useState(null)
  const [cadParams, setCadParams] = useState(null)
  const [revealed, setRevealed] = useState(0)
  const [exporting, setExporting] = useState(false)
  const [stats, setStats] = useState(null)
  const [logs, setLogs] = useState([])

  const abortRef = useRef(null)
  const runId = useRef(0)

  const reset = useCallback(() => {
    abortRef.current?.abort()
    runId.current += 1
    setRunning(false)
    setStageStatus({})
    setActiveStage('upload')
    setClipObservations([])
    setUnifiedRequirements(null)
    setAgentFindings([])
    setWorkTraces([])
    setDesignIterations([])
    setMechReport(null)
    setCadParams(null)
    setRevealed(0)
    setStats(null)
    setLogs([])
  }, [])

  const addLog = useCallback((msg) => setLogs((l) => [...l.slice(-49), msg]), [])

  const runPipeline = useCallback(async () => {
    if (!clips.length || running) return
    const serverPaths = clips.map((c) => c.serverPath).filter(Boolean)
    if (!serverPaths.length) {
      addLog('No uploaded clips — upload clips first.')
      return
    }

    runId.current += 1
    const myRun = runId.current
    const alive = () => runId.current === myRun

    abortRef.current = new AbortController()
    setRunning(true)
    setStageStatus({})
    setActiveStage('perception')
    setClipObservations([])
    setUnifiedRequirements(null)
    setAgentFindings([])
    setWorkTraces([])
    setDesignIterations([])
    setMechReport(null)
    setCadParams(null)
    setRevealed(0)
    setStats(null)
    setLogs([])

    try {
      const resp = await fetch('/api/run-multi-pipeline', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ clip_paths: serverPaths, quick_mode: false }),
        signal: abortRef.current.signal,
      })

      const reader = resp.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        if (!alive()) break

        buf += decoder.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop()

        for (const line of lines) {
          const trimmed = line.replace(/^data:\s*/, '').trim()
          if (!trimmed) continue
          try {
            const evt = JSON.parse(trimmed)
            handleEvent(evt)
          } catch {
            // not JSON — ignore
          }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') addLog(`Error: ${err.message}`)
    } finally {
      if (alive()) setRunning(false)
    }
  }, [clips, running, addLog])

  const handleEvent = useCallback((evt) => {
    const { type, stage } = evt
    addLog(`[${stage}] ${type}`)

    if (type === 'stage_start') {
      setActiveStage(stage)
      setStageStatus((s) => ({ ...s, [stage]: 'running' }))
    }

    if (type === 'stage_done') {
      setStageStatus((s) => ({ ...s, [stage]: 'done' }))

      if (stage === 'perception') {
        if (evt.unified_rom) {
          setUnifiedRequirements({
            rom_targets_deg: evt.unified_rom,
            design_directives: evt.design_directives || [],
            conflicts: evt.conflicts || [],
            primary_actions: evt.primary_actions || [],
          })
        }
      }

      if (stage === 'design' && evt.candidates?.length) {
        // Use first candidate params to update 3D view
        const c = evt.candidates[0]
        if (c) setCadParams((prev) => ({ ...(prev || {}), ...c }))
      }

      if (stage === 'final') {
        if (evt.stats) setStats(evt.stats)
        if (evt.design_iterations) setDesignIterations(evt.design_iterations)
        // Trigger CAD part reveal
        revealParts()
      }
    }

    if (type === 'agent_finding') {
      if (stage === 'perception') {
        if (evt.identified_problems) {
          setClipObservations((prev) => {
            const existing = prev.findIndex((o) => o.clip === evt.clip)
            const obs = {
              clip: evt.clip,
              primary_action: evt.primary_action,
              affected_side: evt.affected_side,
              identified_problems: evt.identified_problems || [],
            }
            if (existing >= 0) {
              const next = [...prev]
              next[existing] = obs
              return next
            }
            return [...prev, obs]
          })
        }
      }
      if (stage === 'design' && (evt.rationale || evt.work)) {
        setAgentFindings((prev) => [...prev, evt])
      }
    }

    if (type === 'work_trace') {
      setWorkTraces((prev) => [...prev, evt])
    }

    if (type === 'mechanical_result') {
      setMechReport({
        components: evt.components || [],
        total_mass_g: evt.total_mass_g || 0,
        worst_safety_factor: evt.worst_safety_factor || 0,
        predicted_life_years: 0,
        weight_budget_ok: evt.weight_budget_ok || false,
        suggestions: evt.suggestions || [],
      })
    }

    if (type === 'sim_frame' && evt.candidate === 0) {
      // Update 3D params from sim step (future: animate joints)
    }

    if (type === 'done') {
      setStageStatus((s) => ({ ...s, final: 'done' }))
    }

    if (type === 'error') {
      addLog(`ERROR: ${evt.message || JSON.stringify(evt)}`)
      setStageStatus((s) => {
        const k = stage || Object.keys(s).find((k) => s[k] === 'running') || 'pipeline'
        return { ...s, [k]: 'error' }
      })
    }
  }, [addLog])

  const revealParts = useCallback(async () => {
    for (let p = 1; p <= CAD_PARTS.length; p++) {
      await new Promise((r) => setTimeout(r, 800))
      setRevealed(p)
    }
  }, [])

  const exportStl = useCallback(async () => {
    const params = cadParams || {}
    if (!params || exporting) return
    setExporting(true)
    try { await downloadStl(params, 'design') }
    finally { setExporting(false) }
  }, [cadParams, exporting])

  useEffect(() => () => { abortRef.current?.abort() }, [])

  const completed = PIPELINE_STAGES.filter((s) => stageStatus[s.key] === 'done').length
  const progress = Math.round((completed / PIPELINE_STAGES.length) * 100)
  const viable = stats?.ik_success_rate >= 0.4 && (stats?.safety_factor ?? 0) >= 2.5
  const defaultCadParams = { upper_arm_len: 0.30, forearm_len: 0.26, grip_width: 0.08,
    joint_stiffness: 1.0, arm_radius: 0.03, primaryColor: '#0d1117', accentColor: '#00d4ff' }

  return (
    <div className="demo demo-multi">
      {/* ── Header ── */}
      <header className="demo-header">
        <div className="demo-brand">
          <span className="demo-logo">⬡</span>
          <div>
            <div className="demo-title">ARMASAI</div>
            <div className="demo-sub">Multi-Agent Prosthesis Pipeline</div>
          </div>
        </div>
        <div className="demo-flow-label">
          {clips.length} clip{clips.length !== 1 ? 's' : ''} → parallel perception → advanced design → sim+mech → RL
        </div>
        <div className="demo-actions">
          <div className="demo-progress">
            <div className="demo-progress-bar" style={{ width: `${progress}%` }} />
            <span>{progress}%</span>
          </div>
          {viable && <span className="badge-viable">✓ Viable</span>}
          <button className="btn ghost" onClick={() => { window.location.hash = '#studio' }}>Design Studio →</button>
          <button className="btn" onClick={reset} disabled={!completed && !running}>Reset</button>
          <button
            className="btn primary"
            onClick={running ? () => { abortRef.current?.abort(); setRunning(false) } : runPipeline}
            disabled={!clips.length}
          >
            {running ? '■ Stop' : '▶ Run pipeline'}
          </button>
        </div>
      </header>

      {/* ── Pipeline rail ── */}
      <nav className="rail">
        {PIPELINE_STAGES.map((stage, i) => {
          const st = stageStatus[stage.key] || 'idle'
          return (
            <div className="rail-cell" key={stage.key}>
              <button
                className={`chip ${st} ${activeStage === stage.key ? 'focus' : ''}`}
                onClick={() => setActiveStage(stage.key)}
              >
                <span className="chip-icon">{stage.icon}</span>
                <span className="chip-body">
                  <span className="chip-name">{stage.name}</span>
                  <span className="chip-emit">{stage.emits}</span>
                </span>
                <span className={`dot ${st}`} />
              </button>
              {i < PIPELINE_STAGES.length - 1 && (
                <div className={`connector ${st === 'done' ? 'active' : ''}`}>
                  <div className="conn-line" />
                </div>
              )}
            </div>
          )
        })}
      </nav>

      {/* ── Main body: 3 columns ── */}
      <div className="body body-multi">

        {/* Left: upload + agent work */}
        <section className="col-left">
          <div className="panel">
            <div className="panel-head">🎥 Input Clips ({clips.length})</div>
            <MultiClipUpload clips={clips} onClips={setClips} disabled={running} />
          </div>

          <AgentWorkPanel
            clipObservations={clipObservations}
            workTraces={workTraces}
            agentFindings={agentFindings}
            unified={unifiedRequirements}
          />

          {logs.length > 0 && (
            <div className="log-panel">
              <div className="log-title">Pipeline log</div>
              <div className="log-body">
                {logs.slice(-20).map((l, i) => <div key={i} className="log-line">{l}</div>)}
              </div>
            </div>
          )}
        </section>

        {/* Center: 3D view + stats */}
        <section className="col-center">
          <div className="cad-viewport">
            <div className="cad-tag">
              CAD OUTPUT
              {stats?.primary_action ? ` · "${stats.primary_action}"` : ''}
              {stats?.affected_side ? ` · ${stats.affected_side} side` : ''}
            </div>
            <CadAssembly params={cadParams || defaultCadParams} revealed={revealed} />
            {revealed === 0 && <div className="cad-empty">Upload clips and run the pipeline</div>}
          </div>

          {/* Stats */}
          {stats && (
            <div className="stats-bar">
              <div className="stat-item">
                <div className="stat-label">IK Success</div>
                <div className={`stat-value ${stats.ik_success_rate >= 0.4 ? 'ok' : 'warn'}`}>
                  {(stats.ik_success_rate * 100).toFixed(0)}%
                </div>
              </div>
              <div className="stat-item">
                <div className="stat-label">RL Success</div>
                <div className={`stat-value ${stats.rl_success_rate >= 0.65 ? 'ok' : 'warn'}`}>
                  {(stats.rl_success_rate * 100).toFixed(0)}%
                </div>
              </div>
              <div className="stat-item">
                <div className="stat-label">DOF</div>
                <div className="stat-value ok">{stats.dof}</div>
              </div>
              <div className="stat-item">
                <div className="stat-label">Mass</div>
                <div className={`stat-value ${(stats.total_mass_g || 0) <= 900 ? 'ok' : 'warn'}`}>
                  {(stats.total_mass_g || 0).toFixed(0)} g
                </div>
              </div>
              <div className="stat-item">
                <div className="stat-label">Safety Factor</div>
                <div className={`stat-value ${(stats.safety_factor || 0) >= 2.5 ? 'ok' : 'warn'}`}>
                  {(stats.safety_factor || 0).toFixed(2)}
                </div>
              </div>
            </div>
          )}

          <OptimizationTrace iterations={designIterations} />
        </section>

        {/* Right: parts list + mechanical analysis */}
        <section className="col-right">
          <aside className="cad-side">
            <div className="cad-side-title">Components</div>
            <ol className="parts">
              {CAD_PARTS.map((p, i) => {
                const done = revealed > i
                const building = revealed === i && running
                return (
                  <li key={p.key} className={done ? 'done' : building ? 'building' : ''}>
                    <span className="part-dot" />
                    <div>
                      <div className="part-label">{p.label}</div>
                      <div className="part-note">{p.note}</div>
                    </div>
                    <span className="part-state">{done ? '✓' : building ? '⚙' : ''}</span>
                  </li>
                )
              })}
            </ol>
            <button
              className="btn primary block"
              disabled={revealed < CAD_PARTS.length || exporting}
              onClick={exportStl}
            >
              {exporting ? 'Exporting…' : '⬇ Export STL'}
            </button>
          </aside>

          <MechanicalReportPanel report={mechReport} />
        </section>
      </div>
    </div>
  )
}
