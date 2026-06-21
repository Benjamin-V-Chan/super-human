// Task / Dataset selector for the interactive in-sim demo.
//
// Reads the shared spec at webdemo/assets/tasks.json (the SAME file the Python HUD
// env will read) and drives MuJoCoDemo.setTask(task) when you toggle a task. Toggling
// reconfigures the wasm sim live: same-arm tasks just move the mocap target; arm-variant
// tasks reload the arm scene. The active task's grader weights flow to the eval overlay.

export class TaskPanel {
  constructor(parent) {
    this.parent = parent; // MuJoCoDemo
    this.spec = null;
    this.activeId = null;
    this.buttons = {};
    this._buildDOM();
    this._load();
  }

  _buildDOM() {
    const wrap = document.createElement("div");
    wrap.style.cssText = `position:absolute;bottom:12px;right:12px;width:240px;
      background:rgba(12,16,22,0.72);color:#e6edf3;border-radius:10px;padding:12px 14px;
      font:12px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;z-index:51;pointer-events:auto;
      box-shadow:0 6px 24px rgba(0,0,0,.35);backdrop-filter:blur(6px);max-height:78vh;overflow:auto;`;
    wrap.innerHTML = `<div style="font-weight:700;letter-spacing:.04em;margin-bottom:6px;">
        TASKS <span style="opacity:.5;font-weight:400">&middot; datasets</span></div>
      <div id="tp-body" style="opacity:.6">loading tasks.json…</div>`;
    document.body.appendChild(wrap);
    this.body = wrap.querySelector("#tp-body");
  }

  async _load() {
    try {
      this.spec = await (await fetch("./assets/tasks.json")).json();
    } catch (e) {
      this.body.textContent = "tasks.json not found";
      return;
    }
    this._render();
    // Auto-select the first task matching the currently-loaded arm so weights/target
    // are set on boot WITHOUT a reload.
    const loaded = (this.parent.params.scene || "").includes("articulated")
      ? "articulated"
      : "single";
    let first = null;
    for (const ds of this.spec.datasets) {
      for (const t of ds.tasks) {
        if (!first) first = t;
        if (t.arm === loaded) {
          this.select(t);
          return;
        }
      }
    }
    if (first) this.select(first);
  }

  _render() {
    this.body.innerHTML = "";
    for (const ds of this.spec.datasets) {
      const sec = document.createElement("section");
      sec.style.marginBottom = "6px";
      const h = document.createElement("div");
      h.textContent = ds.name;
      h.style.cssText =
        "font-size:10px;text-transform:uppercase;letter-spacing:.06em;opacity:.55;margin:6px 0 4px;";
      sec.appendChild(h);
      for (const t of ds.tasks) {
        const b = document.createElement("button");
        b.textContent = t.name;
        b.title = `${t.arm} arm · target [${t.target.map((v) => v.toFixed(2)).join(", ")}]`;
        b.style.cssText = `display:block;width:100%;text-align:left;margin:3px 0;padding:6px 8px;
          border:1px solid rgba(255,255,255,.12);border-radius:6px;background:rgba(255,255,255,.04);
          color:#cfe;cursor:pointer;font:inherit;`;
        b.onclick = () => this.select(t);
        this.buttons[t.id] = b;
        sec.appendChild(b);
      }
      this.body.appendChild(sec);
    }
  }

  async select(task) {
    this.activeId = task.id;
    this._highlight();
    await this.parent.setTask(task);
  }

  _highlight() {
    for (const [id, b] of Object.entries(this.buttons)) {
      const on = id === this.activeId;
      b.style.background = on
        ? "rgba(90,169,255,.22)"
        : "rgba(255,255,255,.04)";
      b.style.borderColor = on ? "#5aa9ff" : "rgba(255,255,255,.12)";
      b.style.color = on ? "#eaf4ff" : "#cfe";
    }
  }
}
