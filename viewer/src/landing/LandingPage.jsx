import { useState } from 'react'
import './landing.css'

const STAGES = [
  {
    n: '01', icon: '👁', title: 'Perceive',
    body: 'Vision AI watches a short clip of a daily task and reads the real limitation — opening a jar, tearing paper, plugging in a charger — and which limb needs help.',
    a: 'clip', b: 'ProblemSpec',
  },
  {
    n: '02', icon: '⚙', title: 'Design',
    body: 'A reasoning agent turns the problem into engineering specs: range of motion, grip force, and segment lengths mirrored from the intact arm.',
    a: 'ProblemSpec', b: 'DesignParams',
  },
  {
    n: '03', icon: '🧪', title: 'Verify',
    body: 'A deterministic MuJoCo physics simulator grades every design on the actual task — reach success, grasp force, energy, and self-collision.',
    a: 'DesignParams', b: 'Reward',
  },
  {
    n: '04', icon: '📈', title: 'Optimize',
    body: 'Reinforcement learning closes the loop, improving each design against the verifier’s reward until it reliably performs the task.',
    a: 'Reward', b: 'design',
  },
  {
    n: '05', icon: '🦾', title: 'Manufacture',
    body: 'The winning design exports as a printable STL — a custom prosthetic arm, tuned to one person’s daily life.',
    a: 'DesignParams', b: 'STL',
  },
]

function go(hash) { window.location.hash = hash }

// Shows the image when it exists in /website-images, a clean placeholder when not.
function ImgSlot({ src, label, className = '' }) {
  const [ok, setOk] = useState(true)
  return (
    <div className={`lp-img ${className}`}>
      {ok && <img src={src} alt={label} onError={() => setOk(false)} />}
      {!ok && (
        <div className="lp-img-ph">
          <span>[ {label} ]</span>
          <small>{src}</small>
        </div>
      )}
    </div>
  )
}

export default function LandingPage() {
  return (
    <div className="landing">
      {/* Nav */}
      <nav className="lp-nav">
        <div className="lp-brand">
          <span className="lp-logo">⬡</span>
          <span className="lp-wordmark">SUPERHUMAN</span>
        </div>
        <div className="lp-nav-links">
          <a href="#how">How it works</a>
          <a href="#pipeline">Pipeline</a>
          <a href="#demo" onClick={(e) => { e.preventDefault(); go('#demo') }}>Demo</a>
          <button className="lp-btn lp-btn-sm" onClick={() => go('#demo')}>Launch demo →</button>
        </div>
      </nav>

      <div className="lp-wrap">
        {/* Hero */}
        <header className="lp-hero">
          <div className="lp-hero-top">
            <h1 className="lp-title">
              Prosthetics designed from how you <em>actually live.</em>
            </h1>
            <div className="lp-hero-side">
              <div className="lp-kicker">// AI prosthetic design &amp; simulation</div>
              <p className="lp-sub" style={{ marginTop: 16 }}>
                Superhuman watches a short clip of a daily task, understands the
                limitation, then designs, simulates, and manufactures a custom
                prosthetic arm — from real life to a printable part.
              </p>
              <div className="lp-hero-cta">
                <button className="lp-btn" onClick={() => go('#demo')}>Launch demo →</button>
                <button className="lp-btn lp-btn-ghost" onClick={() => go('#how')}>How it works</button>
              </div>
            </div>
          </div>
          <div className="lp-hero-img">
            <ImgSlot src="/website-images/hero.jpg" label="hero image" />
          </div>
        </header>
      </div>

      <div className="lp-wrap"><div className="lp-rule" /></div>

      {/* 01 — The problem */}
      <section className="lp-section" id="why">
        <div className="lp-wrap">
          <div className="lp-sec-head">
            <span className="lp-num">// 01</span>
            <span className="lp-kicker">The problem</span>
          </div>
          <div className="lp-problem-grid">
            <div>
              <p className="lp-statement">
                Prosthetic design is manual, slow, and <span className="accent">expensive.</span>
              </p>
            </div>
            <div>
              <p>
                Translating a person’s real daily limitations into a validated,
                manufacturable arm takes specialists, fittings, and weeks of
                iteration. Most designs are generic — not personalized to the
                tasks that actually matter to someone’s day.
              </p>
              <p style={{ color: 'var(--ink)' }}>
                Superhuman automates a first-pass personalized design loop,
                grounded in a single video of the person living their life.
              </p>
            </div>
          </div>
          <div className="lp-problem-img">
            <ImgSlot src="/website-images/problem.jpg" label="daily-task photo" />
          </div>
        </div>
      </section>

      {/* 02 — How it works */}
      <section className="lp-section" id="how" style={{ background: 'var(--bg-panel)' }}>
        <div className="lp-wrap">
          <div className="lp-sec-head">
            <span className="lp-num">// 02</span>
            <div>
              <h2>A closed loop from a clip to a printable arm.</h2>
              <p className="lp-sec-lead">
                Five stages, connected by shared data contracts — each emits exactly
                what the next one consumes.
              </p>
            </div>
          </div>
          <div className="lp-steps">
            {STAGES.map((s) => (
              <div className="lp-step" key={s.n}>
                <div className="lp-step-n">// {s.n}</div>
                <div className="lp-step-title"><span className="ic">{s.icon}</span>{s.title}</div>
                <div className="lp-step-body">{s.body}</div>
                <div className="lp-step-emit"><b>{s.a}</b> → <b>{s.b}</b></div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* 03 — The system (gallery) */}
      <section className="lp-section" id="system">
        <div className="lp-wrap">
          <div className="lp-sec-head">
            <span className="lp-num">// 03</span>
            <div>
              <h2>From a moment to a mechanism.</h2>
              <p className="lp-sec-lead">
                A real clip becomes a perceived problem, a simulated design, and a
                manufacturable arm — each step you can see.
              </p>
            </div>
          </div>
          <div className="lp-gallery">
            <ImgSlot src="/website-images/clip.jpg" label="ADL clip" />
            <ImgSlot src="/website-images/perception.jpg" label="perception overlay" />
            <ImgSlot src="/website-images/sim.jpg" label="MuJoCo simulation" />
            <ImgSlot src="/website-images/cad.jpg" label="CAD render" />
          </div>
        </div>
      </section>

      {/* 04 — Pipeline / contracts */}
      <section className="lp-section" id="pipeline" style={{ background: 'var(--bg-panel)' }}>
        <div className="lp-wrap">
          <div className="lp-sec-head">
            <span className="lp-num">// 04</span>
            <div>
              <h2>Three contracts hold the loop together.</h2>
              <p className="lp-sec-lead">
                Every stage agrees on the same data shapes, so perception, design,
                simulation, and manufacturing develop independently and snap together.
              </p>
            </div>
          </div>
          <div className="lp-contracts">
            <div className="lp-contract">
              <div className="lp-contract-name">ProblemSpec</div>
              <div className="lp-contract-desc">The detected action, affected side, and physical constraints read from the clip.</div>
            </div>
            <div className="lp-contract">
              <div className="lp-contract-name">DesignParams</div>
              <div className="lp-contract-desc">Link lengths, joint limits, and grip width — mirrored from the intact arm.</div>
            </div>
            <div className="lp-contract">
              <div className="lp-contract-name">Reward</div>
              <div className="lp-contract-desc">A single deterministic score from the physics verifier, per task.</div>
            </div>
          </div>
          <div style={{ marginTop: 40 }}>
            <ImgSlot src="/website-images/pipeline.jpg" label="pipeline / result image" className="tall" />
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="lp-cta-band">
        <div className="lp-wrap">
          <span className="lp-kicker">// see it work</span>
          <h2 style={{ marginTop: 16 }}>Watch it design a prosthetic, live.</h2>
          <p>Upload a clip of a daily task and run the full pipeline end to end.</p>
          <button className="lp-btn" onClick={() => go('#demo')}>Launch the demo →</button>
        </div>
      </section>

      <footer className="lp-footer">
        <div className="lp-brand">
          <span className="lp-logo">⬡</span>
          <span className="lp-wordmark">SUPERHUMAN</span>
        </div>
        <div className="lp-footer-note">Creation &amp; simulation pipeline for custom prosthetics.</div>
      </footer>
    </div>
  )
}
