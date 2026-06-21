import { useCallback, useEffect, useRef, useState } from 'react'
import CadAssembly from './CadAssembly.jsx'
import JsonBlock from './JsonBlock.jsx'
import IntegrationGate from './IntegrationGate.jsx'
import { PIPELINE, CAD_PARTS, CAD_PARAMS, TIMING } from './demoData.js'
import './demo.css'

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

// Resolve the payload a stage emits: real output, operator override, or the
// expected placeholder for unfinished integrations.
function stageOutput(stage, overrides) {
  if (stage.status === 'pending') return overrides[stage.key] || stage.expected
  return stage.output
}

export default function DemoPage() {
  const [status, setStatus] = useState({})   // key -> 'idle'|'running'|'done'
  const [revealed, setRevealed] = useState(0) // CAD parts shown (0..6)
  const [running, setRunning] = useState(false)
  const [active, setActive] = useState(null)  // key of focused stage
  const [overrides, setOverrides] = useState({}) // operator-completed integrations
  const [gate, setGate] = useState(null)      // stage key whose modal is open
  const runId = useRef(0)

  const reset = useCallback(() => {
    runId.current += 1
    setStatus({})
    setRevealed(0)
    setRunning(false)
    setActive(null)
  }, [])

  const play = useCallback(async () => {
    runId.current += 1
    const myRun = runId.current
    const alive = () => runId.current === myRun
    setStatus({})
    setRevealed(0)
    setRunning(true)

    for (const stage of PIPELINE) {
      if (!alive()) return
      setActive(stage.key)
      setStatus((s) => ({ ...s, [stage.key]: 'running' }))

      if (stage.key === 'cad') {
        for (let p = 1; p <= CAD_PARTS.length; p++) {
          if (!alive()) return
          await sleep(TIMING.cadPerPart)
          setRevealed(p)
        }
      } else {
        await sleep(TIMING[stage.key] ?? 1800)
      }
      if (!alive()) return
      setStatus((s) => ({ ...s, [stage.key]: 'done' }))
    }
    setRunning(false)
  }, [])

  // Clean up any in-flight run on unmount.
  useEffect(() => () => { runId.current += 1 }, [])

  const completed = PIPELINE.filter((s) => status[s.key] === 'done').length
  const progress = Math.round((completed / PIPELINE.length) * 100)

  const goStudio = () => { window.location.hash = '#studio' }

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
          Ray-Ban clip <span className="arrow">→</span> agents <span className="arrow">→</span> CAD model
        </div>
        <div className="demo-actions">
          <div className="demo-progress" title={`${completed}/${PIPELINE.length} stages`}>
            <div className="demo-progress-bar" style={{ width: `${progress}%` }} />
            <span>{progress}%</span>
          </div>
          <button className="btn ghost" onClick={goStudio}>Design Studio →</button>
          <button className="btn" onClick={reset} disabled={!completed && !running}>Reset</button>
          <button className="btn primary" onClick={play} disabled={running}>
            {running ? '● Running…' : '▶ Run pipeline'}
          </button>
        </div>
      </header>

      {/* ── Linear agent rail ── */}
      <section className="rail-wrap">
        <div className="rail">
          {PIPELINE.map((stage, i) => {
            const st = status[stage.key] || 'idle'
            const out = stageOutput(stage, overrides)
            const isPlaceholder = stage.status === 'pending' && !overrides[stage.key]
            const open = active === stage.key || st === 'done' || st === 'running'
            return (
              <div className="rail-cell" key={stage.key}>
                <article
                  className={`node ${st} ${stage.status === 'pending' ? 'pending-type' : ''} ${active === stage.key ? 'focus' : ''}`}
                  onClick={() => setActive(active === stage.key ? null : stage.key)}
                >
                  <div className="node-top">
                    <span className="node-icon">{stage.icon}</span>
                    <StatusDot status={st} />
                  </div>
                  <div className="node-name">{stage.name}</div>
                  <div className="node-role">{stage.role}</div>
                  <div className="node-tech">{stage.tech}</div>

                  {stage.consumes && (
                    <div className="node-io in">◂ {stage.consumes}</div>
                  )}
                  <div className="node-io out">{stage.emits} ▸</div>

                  {stage.status === 'pending' && (
                    <div className="node-badge">
                      {overrides[stage.key] ? 'OPERATOR-PROVIDED' : 'PLACEHOLDER'}
                      {stage.owner && <span className="owner"> · {stage.owner}</span>}
                    </div>
                  )}

                  {open && (
                    <div className="node-detail">
                      <p className="node-blurb">{stage.blurb}</p>
                      <div className="out-head">
                        <span>{stage.emits}</span>
                        {isPlaceholder && <span className="exp-tag">expected output</span>}
                      </div>
                      <JsonBlock data={out} animate={st === 'running'} />
                      {stage.status === 'pending' && (
                        <button
                          className="btn tiny"
                          onClick={(e) => { e.stopPropagation(); setGate(stage.key) }}
                        >
                          {overrides[stage.key] ? 'Edit integration output' : 'Complete integration →'}
                        </button>
                      )}
                    </div>
                  )}
                </article>

                {i < PIPELINE.length - 1 && (
                  <Connector
                    label={stage.emits}
                    active={st === 'done'}
                    flowing={st === 'running' || (st === 'done' && status[PIPELINE[i + 1].key] === 'running')}
                  />
                )}
              </div>
            )
          })}
        </div>
      </section>

      {/* ── CAD output stage ── */}
      <section className="cad-stage">
        <div className="cad-viewport">
          <div className="cad-tag">CAD OUTPUT · live assembly</div>
          <CadAssembly params={CAD_PARAMS} revealed={revealed} />
          {revealed === 0 && (
            <div className="cad-empty">Run the pipeline to materialize the model</div>
          )}
        </div>
        <aside className="cad-side">
          <h3>Build sequence</h3>
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
          <div className="cad-out">
            <div className="cad-out-row"><span>file</span><b>candidate.stl</b></div>
            <div className="cad-out-row"><span>parts</span><b>{revealed} / {CAD_PARTS.length}</b></div>
            <div className="cad-out-row"><span>status</span><b>{revealed === CAD_PARTS.length ? 'complete' : 'assembling'}</b></div>
          </div>
          <button
            className="btn primary block"
            disabled={revealed < CAD_PARTS.length}
            onClick={() => alert('STL export → wire to Python CadBridge /api/export-stl')}
          >
            ⬇ Export STL
          </button>
        </aside>
      </section>

      {gate && (
        <IntegrationGate
          stage={PIPELINE.find((s) => s.key === gate)}
          current={overrides[gate]}
          onClose={() => setGate(null)}
          onSave={(data) => { setOverrides((o) => ({ ...o, [gate]: data })); setGate(null) }}
        />
      )}
    </div>
  )
}

function StatusDot({ status }) {
  const map = { idle: 'idle', running: 'running', done: 'done' }
  return <span className={`dot ${map[status] || 'idle'}`} />
}

function Connector({ label, active, flowing }) {
  return (
    <div className={`connector ${active ? 'active' : ''} ${flowing ? 'flowing' : ''}`}>
      <div className="conn-line">
        <span className="packet" />
      </div>
      <span className="conn-label">{label}</span>
    </div>
  )
}
