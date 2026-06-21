export default function MechanicalReportPanel({ report }) {
  if (!report) {
    return (
      <div className="mech-panel empty">
        <div className="mech-title">Mechanical Analysis</div>
        <p className="mech-empty">Run the pipeline to see per-component analysis.</p>
      </div>
    )
  }

  const sfOk   = report.worst_safety_factor >= 2.5
  const wOk    = report.weight_budget_ok
  const sfPct  = Math.min(100, (report.worst_safety_factor / 5) * 100)
  const massPct = Math.min(100, (report.total_mass_g / 900) * 100)

  return (
    <div className="mech-panel">
      <div className="mech-title">Mechanical Analysis</div>

      <div className="mech-gauges">
        <div className="mech-gauge">
          <div className="gauge-label">Safety Factor</div>
          <div className="gauge-bar">
            <div
              className={`gauge-fill ${sfOk ? 'ok' : 'warn'}`}
              style={{ width: `${sfPct}%` }}
            />
          </div>
          <div className={`gauge-value ${sfOk ? 'ok' : 'warn'}`}>
            {report.worst_safety_factor.toFixed(2)} {sfOk ? '✓' : '✗ <2.5'}
          </div>
        </div>

        <div className="mech-gauge">
          <div className="gauge-label">Total Mass</div>
          <div className="gauge-bar">
            <div
              className={`gauge-fill ${wOk ? 'ok' : 'warn'}`}
              style={{ width: `${massPct}%` }}
            />
          </div>
          <div className={`gauge-value ${wOk ? 'ok' : 'warn'}`}>
            {report.total_mass_g.toFixed(0)} g {wOk ? '✓' : '✗ >900g'}
          </div>
        </div>

        <div className="mech-gauge">
          <div className="gauge-label">Service Life</div>
          <div className="gauge-bar">
            <div
              className="gauge-fill ok"
              style={{ width: `${Math.min(100, (report.predicted_life_years / 10) * 100)}%` }}
            />
          </div>
          <div className="gauge-value ok">
            {report.predicted_life_years >= 99 ? '≥99 yr' : `${report.predicted_life_years.toFixed(1)} yr`}
          </div>
        </div>
      </div>

      {report.components?.length > 0 && (
        <div className="mech-components">
          <div className="mech-comp-title">Per-Component</div>
          <table className="comp-table">
            <thead>
              <tr>
                <th>Component</th>
                <th>Material</th>
                <th>Stress (MPa)</th>
                <th>FoS</th>
                <th>Mass (g)</th>
              </tr>
            </thead>
            <tbody>
              {report.components.map((c, i) => (
                <tr key={i} className={c.ok ? '' : 'comp-warn'}>
                  <td className="comp-name">{c.name}</td>
                  <td className="comp-mat">{c.material}</td>
                  <td>{c.stress_mpa?.toFixed(1) ?? '—'}</td>
                  <td className={c.safety_factor < 2.5 ? 'sf-warn' : 'sf-ok'}>
                    {c.safety_factor?.toFixed(2) ?? '—'}
                  </td>
                  <td>{c.mass_g?.toFixed(1) ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {report.suggestions?.length > 0 && (
        <div className="mech-suggestions">
          <div className="mech-sugg-title">Suggestions</div>
          <ul>
            {report.suggestions.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
