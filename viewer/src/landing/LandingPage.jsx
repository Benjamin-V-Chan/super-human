import './landing.css'

const STAGES = [
  {
    n: '01', key: 'perceive', icon: '👁',
    title: 'Perceive',
    body: 'Vision AI watches a short clip of a daily task and reads the real limitation — opening a jar, tearing paper, plugging in a charger — and which limb needs help.',
    emit: 'clip → ProblemSpec',
  },
  {
    n: '02', key: 'design', icon: '⚙',
    title: 'Design',
    body: 'A reasoning agent turns the problem into engineering specs: range of motion, grip force, and segment lengths mirrored from the intact arm.',
    emit: 'ProblemSpec → DesignParams',
  },
  {
    n: '03', key: 'verify', icon: '🧪',
    title: 'Verify',
    body: 'A deterministic MuJoCo physics simulator grades every design on the actual task — reach success, grasp force, energy, and self-collision.',
    emit: 'DesignParams → Reward',
  },
  {
    n: '04', key: 'optimize', icon: '📈',
    title: 'Optimize',
    body: 'Reinforcement learning closes the loop, improving each design against the verifier’s reward until it reliably performs the task.',
    emit: 'Reward → better design',
  },
  {
    n: '05', key: 'manufacture', icon: '🦾',
    title: 'Manufacture',
    body: 'The winning design exports as a printable STL — a custom prosthetic arm, tuned to one person’s daily life.',
    emit: 'DesignParams → STL',
  },
]

function go(hash) {
  window.location.hash = hash
}

export default function LandingPage() {
  return (
    <div className="landing">
      {/* Nav */}
      <nav className="lp-nav">
        <div className="lp-brand">
          <span className="lp-logo">⬡</span>
          <span className="lp-wordmark">ARMASAI</span>
        </div>
        <div className="lp-nav-links">
          <a href="#how">How it works</a>
          <a href="#pipeline">Pipeline</a>
          <a href="#studio" onClick={(e) => { e.preventDefault(); go('#studio') }}>Studio</a>
          <button className="lp-btn lp-btn-sm" onClick={() => go('#demo')}>Launch demo</button>
        </div>
      </nav>

      {/* Hero */}
      <header className="lp-hero">
        <div className="lp-hero-glow" />
        <div className="lp-eyebrow">AI PROSTHETIC DESIGN &amp; SIMULATION PIPELINE</div>
        <h1 className="lp-title">
          Prosthetics designed from<br />
          <span className="lp-grad">how you actually live.</span>
        </h1>
        <p className="lp-sub">
          Armasai watches a short clip of a daily task, understands the limitation,
          then designs, simulates, and manufactures a custom prosthetic arm tuned to
          that exact need — closing the loop from real life to a printable part.
        </p>
        <div className="lp-cta">
          <button className="lp-btn lp-btn-primary" onClick={() => go('#demo')}>
            Try the live demo →
          </button>
          <button className="lp-btn lp-btn-ghost" onClick={() => go('#studio')}>
            Open design studio
          </button>
        </div>

        <div className="lp-flow">
          {STAGES.map((s, i) => (
            <div className="lp-flow-cell" key={s.key}>
              <div className="lp-flow-node">
                <span className="lp-flow-icon">{s.icon}</span>
                <span className="lp-flow-name">{s.title}</span>
              </div>
              {i < STAGES.length - 1 && <span className="lp-flow-arrow">→</span>}
            </div>
          ))}
        </div>
      </header>

      {/* Problem */}
      <section className="lp-section lp-problem" id="why">
        <div className="lp-band">
          <div className="lp-band-k">THE PROBLEM</div>
          <h2>Prosthetic design is manual, slow, and expensive.</h2>
          <p>
            Translating a person’s real daily limitations into a validated, manufacturable
            arm takes specialists, fittings, and weeks of iteration. Most designs are generic,
            not personalized to the tasks that actually matter to someone’s day.
          </p>
          <p className="lp-band-accent">
            Armasai automates a first-pass personalized design loop — grounded in a single
            video of the person living their life.
          </p>
        </div>
      </section>

      {/* How it works — stage cards */}
      <section className="lp-section" id="how">
        <div className="lp-section-head">
          <div className="lp-band-k">HOW IT WORKS</div>
          <h2>A closed loop from a clip to a printable arm.</h2>
        </div>
        <div className="lp-cards">
          {STAGES.map((s) => (
            <article className="lp-card" key={s.key}>
              <div className="lp-card-top">
                <span className="lp-card-icon">{s.icon}</span>
                <span className="lp-card-n">{s.n}</span>
              </div>
              <h3>{s.title}</h3>
              <p>{s.body}</p>
              <code className="lp-card-emit">{s.emit}</code>
            </article>
          ))}
        </div>
      </section>

      {/* Pipeline contracts strip */}
      <section className="lp-section lp-pipeline" id="pipeline">
        <div className="lp-section-head">
          <div className="lp-band-k">THE CONTRACTS</div>
          <h2>Three shared contracts hold the loop together.</h2>
          <p className="lp-section-lead">
            Every stage agrees on the same data shapes, so perception, design, simulation,
            and manufacturing develop independently and snap together.
          </p>
        </div>
        <div className="lp-contracts">
          <div className="lp-contract">
            <div className="lp-contract-name">ProblemSpec</div>
            <div className="lp-contract-desc">The detected action, affected side, and physical constraints from the clip.</div>
          </div>
          <span className="lp-contract-arrow">→</span>
          <div className="lp-contract">
            <div className="lp-contract-name">DesignParams</div>
            <div className="lp-contract-desc">Link lengths, joint limits, grip width — mirrored from the intact arm.</div>
          </div>
          <span className="lp-contract-arrow">→</span>
          <div className="lp-contract">
            <div className="lp-contract-name">Reward</div>
            <div className="lp-contract-desc">A single deterministic score from the physics verifier, per task.</div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="lp-cta-band">
        <div className="lp-cta-glow" />
        <h2>See it design a prosthetic, live.</h2>
        <p>Upload a clip of a daily task and watch the full pipeline run end to end.</p>
        <div className="lp-cta">
          <button className="lp-btn lp-btn-primary" onClick={() => go('#demo')}>
            Launch the demo →
          </button>
        </div>
      </section>

      <footer className="lp-footer">
        <div className="lp-brand">
          <span className="lp-logo">⬡</span>
          <span className="lp-wordmark">ARMASAI</span>
        </div>
        <div className="lp-footer-note">Creation &amp; simulation pipeline for custom prosthetics.</div>
      </footer>
    </div>
  )
}
