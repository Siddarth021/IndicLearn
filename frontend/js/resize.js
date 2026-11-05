// Drag handles so the student can make the left menu and the right tutor panel as wide as they like.
// Widths are remembered in this browser. Double-click a handle to go back to the default width.
const LIMITS = { side: [72, 420, 232], tutor: [280, 760, 392] };
const KEY = "indiclearn.panelwidths";
const read = () => { try { return JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { return {}; } };
const save = (w) => { try { localStorage.setItem(KEY, JSON.stringify(w)); } catch (e) { /* private window: fine */ } };
const clamp = (k, v) => Math.max(LIMITS[k][0], Math.min(LIMITS[k][1], Math.round(v)));

export function attachResizers(app) {
  const w = { side: clamp("side", read().side ?? LIMITS.side[2]), tutor: clamp("tutor", read().tutor ?? LIMITS.tutor[2]) };
  const apply = () => { app.style.setProperty("--side-w", w.side + "px"); app.style.setProperty("--tutor-w", w.tutor + "px"); };
  apply();
  const nudgeLayout = () => window.dispatchEvent(new Event("resize")); // the mind map and charts re-fit themselves
  const make = (k, cls) => {
    const bar = document.createElement("div");
    bar.className = "resizer " + cls; bar.setAttribute("role", "separator"); bar.setAttribute("aria-orientation", "vertical");
    bar.title = "Drag to resize · double-click to reset"; bar.tabIndex = 0;
    const set = (v) => { w[k] = clamp(k, v); apply(); };
    bar.addEventListener("pointerdown", (e) => {
      e.preventDefault(); bar.setPointerCapture(e.pointerId); document.body.classList.add("resizing");
      const x0 = e.clientX, w0 = w[k];
      const move = (ev) => { set(k === "side" ? w0 + (ev.clientX - x0) : w0 - (ev.clientX - x0)); nudgeLayout(); };
      const up = () => { bar.removeEventListener("pointermove", move); bar.removeEventListener("pointerup", up); bar.removeEventListener("pointercancel", up); document.body.classList.remove("resizing"); save(w); nudgeLayout(); };
      bar.addEventListener("pointermove", move); bar.addEventListener("pointerup", up); bar.addEventListener("pointercancel", up);
    });
    bar.addEventListener("dblclick", () => { set(LIMITS[k][2]); save(w); nudgeLayout(); });
    bar.addEventListener("keydown", (e) => {  // keyboard: arrow keys resize too
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      const d = (e.key === "ArrowRight" ? 1 : -1) * 24 * (k === "side" ? 1 : -1);
      set(w[k] + d); save(w); nudgeLayout(); e.preventDefault();
    });
    app.append(bar);
  };
  make("side", "rz-side"); make("tutor", "rz-tutor");
}
