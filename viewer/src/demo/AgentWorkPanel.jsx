import { useState } from 'react'

function SeverityBadge({ severity }) {
  const level = severity >= 0.8 ? 'critical' : severity >= 0.5 ? 'moderate' : severity > 0.1 ? 'low' : 'info'
  const labels = { critical: '🔴 Critical', moderate: '🟡 Moderate', low: '🟢 Low', info: '💡 Info' }
  return <span className={`severity-badge ${level}`}>{labels[level]}</span>
}

function ProblemCard({ problem, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className={`problem-card ${open ? 'open' : ''}`}>
      <button className="problem-header" onClick={() => setOpen((o) => !o)}>
        <SeverityBadge severity={problem.severity ?? 0} />
        <span className="problem-desc">{problem.description || problem.id}</span>
        <span className="chevron">{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div className="problem-body">
          <div className="problem-meta">ID: <code>{problem.id || problem.problem_id}</code></div>
          {problem.affected_tasks?.length > 0 && (
            <div className="problem-tasks">
              Tasks: {problem.affected_tasks.join(', ')}
            </div>
          )}
          {problem.solutions?.length > 0 || problem.proposed_solutions?.length > 0 ? (
            <div className="solutions">
              <div className="solutions-label">Proposed solutions:</div>
              <ol>
                {(problem.solutions || problem.proposed_solutions || []).map((s, i) => (
                  <li key={i}>{s}</li>
                ))}
              </ol>
            </div>
          ) : null}
        </div>
      )}
    </div>
  )
}

function ClipSection({ obs }) {
  const [open, setOpen] = useState(true)
  const clipName = obs.clip?.split('/').pop() || obs.clip || 'Clip'
  const problems = obs.identified_problems || []
  const critical = problems.filter((p) => (p.severity ?? 0) >= 0.8)
  return (
    <div className="clip-section">
      <button className="clip-section-header" onClick={() => setOpen((o) => !o)}>
        <span className="clip-icon">🎥</span>
        <span className="clip-section-name">{clipName}</span>
        <span className="clip-action">{obs.primary_action || ''}</span>
        {critical.length > 0 && <span className="badge-critical">{critical.length} critical</span>}
        <span className="chevron">{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div className="clip-section-body">
          {problems.length === 0
            ? <p className="no-problems">No problems identified.</p>
            : problems.map((p, i) => (
                <ProblemCard key={p.id || p.problem_id || i} problem={p} defaultOpen={i === 0} />
              ))}
        </div>
      )}
    </div>
  )
}

function WorkTraceCard({ trace }) {
  const [open, setOpen] = useState(false)
  const sf = trace.safety_factor ?? 0
  const sfOk = sf >= 2.5
  return (
    <div className={`work-trace-card ${open ? 'open' : ''}`}>
      <button className="work-trace-header" onClick={() => setOpen((o) => !o)}>
        <span className="iter-num">Iter {trace.iteration}</span>
        <span className="iter-success">{(trace.best_success * 100).toFixed(0)}% IK</span>
        <span className={`iter-sf ${sfOk ? 'ok' : 'warn'}`}>FoS {sf.toFixed(2)}</span>
        <span className="iter-mass">{(trace.total_mass_g ?? 0).toFixed(0)} g</span>
        <span className="chevron">{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div className="work-trace-body">
          <div className="work-rationale">{trace.rationale}</div>
          {trace.directives_addressed?.length > 0 && (
            <div className="work-directives">
              <div className="work-label">Directives addressed:</div>
              <ul>{trace.directives_addressed.map((d, i) => <li key={i}>{d}</li>)}</ul>
            </div>
          )}
          {trace.suggestions?.length > 0 && (
            <div className="work-suggestions">
              <div className="work-label">Mechanical suggestions:</div>
              <ul>{trace.suggestions.map((s, i) => <li key={i}>{s}</li>)}</ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function AgentFindingCard({ finding }) {
  if (!finding.rationale && !finding.work) return null
  const [open, setOpen] = useState(false)
  return (
    <div className={`agent-finding-card ${open ? 'open' : ''}`}>
      <button className="agent-finding-header" onClick={() => setOpen((o) => !o)}>
        <span className="finding-icon">🤖</span>
        <span className="finding-iter">Design Agent — Iter {finding.iteration ?? '?'}</span>
        <span className="chevron">{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div className="agent-finding-body">
          {finding.rationale && <p className="finding-rationale">{finding.rationale}</p>}
          {finding.terminal_device && (
            <div className="finding-td">Terminal device: <strong>{finding.terminal_device}</strong></div>
          )}
          {finding.topology?.length > 0 && (
            <div className="finding-topology">Links: {finding.topology.join(' → ')}</div>
          )}
          {finding.work && Object.keys(finding.work).length > 0 && (
            <div className="finding-work">
              <div className="work-label">Problem → Solution:</div>
              <table className="work-table">
                <tbody>
                  {Object.entries(finding.work).map(([k, v]) => (
                    <tr key={k}><td className="work-k">{k}</td><td className="work-v">{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default function AgentWorkPanel({ clipObservations, workTraces, agentFindings, unified }) {
  const [tab, setTab] = useState('problems')
  const hasObs   = clipObservations?.length > 0
  const hasTrace = workTraces?.length > 0
  const hasFnd   = agentFindings?.length > 0

  return (
    <div className="agent-work-panel">
      <div className="awp-header">
        <span className="awp-title">Agent Work</span>
        <div className="awp-tabs">
          <button className={tab === 'problems' ? 'active' : ''} onClick={() => setTab('problems')}>
            Problems {hasObs ? `(${clipObservations.reduce((s, o) => s + (o.identified_problems?.length || 0), 0)})` : ''}
          </button>
          <button className={tab === 'iterations' ? 'active' : ''} onClick={() => setTab('iterations')}>
            Iterations {hasTrace ? `(${workTraces.length})` : ''}
          </button>
          <button className={tab === 'decisions' ? 'active' : ''} onClick={() => setTab('decisions')}>
            Decisions {hasFnd ? `(${agentFindings.length})` : ''}
          </button>
        </div>
      </div>

      <div className="awp-body">
        {tab === 'problems' && (
          <>
            {unified?.design_directives?.length > 0 && (
              <div className="directives-box">
                <div className="directives-label">Unified design directives:</div>
                <ul>{unified.design_directives.map((d, i) => <li key={i}>{d}</li>)}</ul>
                {unified.conflicts?.length > 0 && (
                  <div className="conflicts-box">
                    <div className="conflicts-label">⚠ Conflicts:</div>
                    <ul>{unified.conflicts.map((c, i) => <li key={i}>{c}</li>)}</ul>
                  </div>
                )}
              </div>
            )}
            {!hasObs && <p className="awp-empty">Run the pipeline to see identified problems.</p>}
            {clipObservations?.map((obs, i) => (
              <ClipSection key={obs.clip || i} obs={obs} />
            ))}
          </>
        )}

        {tab === 'iterations' && (
          <>
            {!hasTrace && <p className="awp-empty">Optimization iterations will appear here.</p>}
            {workTraces?.map((t, i) => <WorkTraceCard key={i} trace={t} />)}
          </>
        )}

        {tab === 'decisions' && (
          <>
            {!hasFnd && <p className="awp-empty">Agent design decisions will appear here.</p>}
            {agentFindings?.map((f, i) => <AgentFindingCard key={i} finding={f} />)}
          </>
        )}
      </div>
    </div>
  )
}
