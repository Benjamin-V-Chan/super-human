// Faux Meta Ray-Ban egocentric capture viewport — a CSS-only "recording" of a
// person reaching for a bottle (the ADL task), with a recording HUD and a
// frame-sampling overlay that activates while the Perception agent consumes it.
export default function RayBanPOV({ recording = true, sampling = false, taskHint }) {
  return (
    <div className={`pov ${sampling ? 'sampling' : ''}`}>
      {/* scene */}
      <div className="pov-scene">
        <div className="pov-table" />
        <div className="pov-bottle"><span className="pov-cap" /></div>
        <div className="pov-arm">
          <div className="pov-forearm" />
          <div className="pov-hand" />
        </div>
      </div>

      {/* footage overlays */}
      <div className="pov-vignette" />
      <div className="pov-scan" />

      {/* HUD */}
      <div className="pov-hud top">
        <span className={`pov-rec ${recording ? 'on' : ''}`}>● REC</span>
        <span className="pov-dev">RAY-BAN META</span>
      </div>
      <div className="pov-hud bot">
        <span>{taskHint || 'egocentric · 1080p'}</span>
        <span className="pov-tc">00:07:12</span>
      </div>

      {/* frame-sampling brackets (while perception runs) */}
      {sampling && (
        <>
          <span className="pov-bracket tl" /><span className="pov-bracket tr" />
          <span className="pov-bracket bl" /><span className="pov-bracket br" />
          <div className="pov-sample-label">⊞ sampling frames…</div>
        </>
      )}

      <style>{`
        .pov { position:relative; width:100%; aspect-ratio:16/9; border-radius:8px; overflow:hidden;
          background:linear-gradient(160deg,#1a1622,#0c0c12); border:1px solid var(--border); }
        .pov-scene { position:absolute; inset:0; }
        .pov-table { position:absolute; left:0; right:0; bottom:0; height:38%;
          background:linear-gradient(180deg,#241c2e,#15101c); border-top:1px solid #3a2f48; }
        .pov-bottle { position:absolute; left:46%; bottom:38%; width:13%; height:34%;
          background:linear-gradient(90deg,#2a6a78,#3fb6c8,#2a6a78); border-radius:6px 6px 4px 4px;
          box-shadow:0 0 14px rgba(0,212,255,.25); }
        .pov-cap { position:absolute; top:-9px; left:25%; width:50%; height:10px; background:#cfd6dd; border-radius:2px; }
        .pov-arm { position:absolute; right:-4%; bottom:-2%; width:62%; height:62%;
          transform-origin:bottom right; animation:reach 4s ease-in-out infinite; }
        .pov-forearm { position:absolute; right:0; bottom:0; width:78%; height:30%;
          background:linear-gradient(90deg,#caa07a,#e3bd96); border-radius:24px; transform:rotate(-18deg);
          transform-origin:right center; box-shadow:inset 0 -4px 8px rgba(0,0,0,.2); }
        .pov-hand { position:absolute; left:8%; top:30%; width:20%; height:24%;
          background:#e3bd96; border-radius:40% 50% 45% 40%; transform:rotate(-18deg); }
        @keyframes reach { 0%,100%{ transform:translateX(8%) } 50%{ transform:translateX(-6%) } }
        .pov-vignette { position:absolute; inset:0; pointer-events:none;
          box-shadow:inset 0 0 60px 14px rgba(0,0,0,.55); }
        .pov-scan { position:absolute; inset:0; pointer-events:none; opacity:.18;
          background:repeating-linear-gradient(0deg,transparent 0 2px,rgba(0,0,0,.5) 2px 3px); }
        .pov-hud { position:absolute; left:8px; right:8px; display:flex; justify-content:space-between;
          font-family:var(--mono); font-size:9px; letter-spacing:.08em; color:#cfd6dd; }
        .pov-hud.top { top:7px; } .pov-hud.bot { bottom:7px; color:var(--text-muted); }
        .pov-rec { color:#6b6b80; } .pov-rec.on { color:#ff4060; animation:blink 1.2s steps(1) infinite; }
        @keyframes blink { 50%{ opacity:.25 } }
        .pov-tc { color:#9aa; }
        .pov.sampling { box-shadow:0 0 0 1px var(--accent), 0 0 22px rgba(0,212,255,.25); }
        .pov-bracket { position:absolute; width:16px; height:16px; border:2px solid var(--accent); }
        .pov-bracket.tl { top:14px; left:14px; border-right:0; border-bottom:0; }
        .pov-bracket.tr { top:14px; right:14px; border-left:0; border-bottom:0; }
        .pov-bracket.bl { bottom:14px; left:14px; border-right:0; border-top:0; }
        .pov-bracket.br { bottom:14px; right:14px; border-left:0; border-top:0; }
        .pov-sample-label { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
          font-family:var(--mono); font-size:10px; color:var(--accent); background:rgba(0,212,255,.1);
          border:1px solid rgba(0,212,255,.4); padding:3px 8px; border-radius:5px; }
      `}</style>
    </div>
  )
}
