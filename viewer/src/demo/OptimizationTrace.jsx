export default function OptimizationTrace({ iterations }) {
  if (!iterations?.length) {
    return (
      <div className="opt-trace empty">
        <div className="opt-title">Optimization Trace</div>
        <p className="opt-empty">Design iterations will appear as the pipeline runs.</p>
      </div>
    )
  }

  const maxReward = Math.max(...iterations.map((d) => d.ik_success_rate ?? 0), 0.01)

  return (
    <div className="opt-trace">
      <div className="opt-title">Optimization Trace</div>

      <div className="opt-chart">
        {iterations.map((iter, i) => {
          const pct = ((iter.ik_success_rate ?? 0) / Math.max(maxReward, 1)) * 100
          const sfOk = (iter.safety_factor ?? 0) >= 2.5
          return (
            <div key={i} className="opt-bar-group" title={`Iter ${iter.iteration}: IK=${(iter.ik_success_rate * 100).toFixed(0)}% FoS=${iter.safety_factor?.toFixed(2)}`}>
              <div className="opt-bar-wrap">
                <div
                  className={`opt-bar ${sfOk ? 'ok' : 'warn'}`}
                  style={{ height: `${Math.max(4, pct)}%` }}
                />
              </div>
              <div className="opt-bar-label">{iter.iteration}</div>
            </div>
          )
        })}
      </div>

      <div className="opt-legend">
        <span className="legend-ok">■ FoS ≥ 2.5</span>
        <span className="legend-warn">■ FoS &lt; 2.5</span>
        <span className="legend-axis">Bar height = IK success rate</span>
      </div>

      <div className="opt-rows">
        {iterations.map((iter, i) => (
          <div key={i} className="opt-row">
            <span className="opt-iter">Iter {iter.iteration}</span>
            <span className="opt-ik">{(iter.ik_success_rate * 100).toFixed(0)}% IK</span>
            {iter.rl_success_rate > 0 && (
              <span className="opt-rl">{(iter.rl_success_rate * 100).toFixed(0)}% RL</span>
            )}
            <span className={`opt-sf ${(iter.safety_factor ?? 0) >= 2.5 ? 'ok' : 'warn'}`}>
              FoS {iter.safety_factor?.toFixed(2) ?? '—'}
            </span>
            <span className="opt-mass">{iter.mass_g?.toFixed(0) ?? '—'} g</span>
          </div>
        ))}
      </div>
    </div>
  )
}
