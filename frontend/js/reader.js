import { api } from "./api.js";
import { t, unitName } from "./i18n.js";
import { h, toast } from "./util.js";
import { S, emit, on, clearOwner } from "./state.js";

const reEsc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

let globalsBound = false;
export function createReader(root) {
  clearOwner("reader");
  let data = null, readerEl, tocEl, pdfUrl = null, pop = null, selPop = null, showPdf = false, termRe = null, termMap = new Map();

  function buildTermRegex() {
    const forms = [];
    for (const tm of data.terms) {
      const names = new Set([tm.label, ...(tm.aliases || [])]);
      for (const nm of names) if (nm && nm.length >= 3 && !/^[\d\W]+$/.test(nm)) { forms.push(nm); termMap.set(nm.toLowerCase().replace(/[\s\-]+/g, " "), tm.id); }
    }
    forms.sort((a, b) => b.length - a.length);
    if (!forms.length) { termRe = null; return; }
    const alt = forms.map((f) => reEsc(f).replace(/\\-/g, "[\\s\\-]").replace(/\s+/g, "[\\s\\-]+")).join("|");
    termRe = new RegExp(`(?<![\\p{L}\\p{M}\\p{N}])(?:${alt})(?:s|es)?(?![\\p{L}\\p{N}])`, "giu");
  }
  function termId(text) { return termMap.get(text.toLowerCase().replace(/s$/, "").replace(/[\s\-]+/g, " ")) || termMap.get(text.toLowerCase().replace(/[\s\-]+/g, " ")); }

  function renderText(text) {
    const frag = document.createDocumentFragment();
    if (!termRe) { frag.append(text); return frag; }
    let last = 0; termRe.lastIndex = 0; let m; let count = 0;
    while ((m = termRe.exec(text)) && count < 8) {
      const id = termId(m[0]); if (!id) continue;
      frag.append(text.slice(last, m.index));
      frag.append(h("mark", { class: "term", "data-id": id, onclick: (e) => openPop(id, e) }, m[0]));
      last = m.index + m[0].length; count++;
    }
    frag.append(text.slice(last));
    return frag;
  }

  function openPop(id, e) {
    e.stopPropagation(); closePop();
    const n = S.nodes.get(id); if (!n) return;
    const r = e.target.getBoundingClientRect(); const host = root.getBoundingClientRect();
    pop = h("div", { class: "tip", style: { pointerEvents: "auto", position: "fixed", left: Math.min(r.left, window.innerWidth - 320) + "px", top: r.bottom + 6 + "px", maxWidth: "310px" } },
      h("b", {}, n.label), h("div", { style: { color: "#4A4336", margin: "4px 0 8px" } }, (n.definition || t("no_def")).slice(0, 220)),
      h("div", { style: { display: "flex", gap: "6px", flexWrap: "wrap" } },
        h("button", { class: "btn sm", onclick: () => { closePop(); emit("ask", { text: t("sg_explain", { t: n.label }), focus: id }); } }, t("ask_this")),
        h("button", { class: "btn sm ghost", onclick: () => { closePop(); S.focus = id; emit("nav", { view: "map", select: id }); } }, t("nav_map"))));
    document.body.append(pop);
    emit("mark", { id, kind: "viewed" });
  }
  const closePop = () => { pop?.remove(); pop = null; };
  const bindGlobals = () => { if (globalsBound) return; globalsBound = true; document.addEventListener("click", () => window.__ilClosePop?.()); };
  window.__ilClosePop = () => closePop(); bindGlobals();

  function onSelection() {
    selPop?.remove(); selPop = null;
    const sel = window.getSelection(); const txt = sel?.toString().trim();
    if (!txt || txt.length < 4 || txt.length > 300 || !readerEl?.contains(sel.anchorNode)) return;
    const r = sel.getRangeAt(0).getBoundingClientRect();
    selPop = h("div", { class: "sel-pop", style: { position: "fixed", left: r.left + "px", top: r.bottom + 8 + "px" } },
      h("button", { class: "btn sm", onmousedown: (e) => e.preventDefault(), onclick: () => { const q = txt.replace(/\s+/g, " "); selPop?.remove(); emit("ask", { text: `${t("sg_explain", { t: "“" + q + "”" })}` }); } }, "✦ " + t("ask_tutor")));
    document.body.append(selPop);
  }
  window.__ilSel = onSelection; window.__ilSelClear = (e) => { if (!selPop?.contains(e.target)) { selPop?.remove(); selPop = null; } };
  if (!window.__ilSelBound) { window.__ilSelBound = true; document.addEventListener("mouseup", () => setTimeout(() => window.__ilSel?.(), 10)); document.addEventListener("mousedown", (e) => window.__ilSelClear?.(e)); }

  async function load() {
    root.innerHTML = ""; root.append(h("div", { class: "empty" }, t("loading")));
    try { data = await api.reader(S.doc.id); } catch (e) { root.innerHTML = ""; toast(e.message, true); return; }
    termMap = new Map(); buildTermRegex(); render();
  }

  function render() {
    root.innerHTML = "";
    const bar = h("div", { class: "toolbar" }, h("h2", {}, S.doc.title), h("span", { class: "chip" }, t(S.doc.unit_label === "Slide" ? "slides_n" : S.doc.unit_label === "Part" ? "parts_n" : "pages_n", { n: S.doc.n_pages })), h("span", { class: "grow" }));
    if (data.has_original) bar.append(h("button", { class: "btn sm ghost", onclick: togglePdf }, showPdf ? t("text_view") : t("original")));
    bar.append(h("button", { "data-tutor-toggle": "1", class: "btn sm", onclick: () => { S.tutorOpen = !S.tutorOpen; emit("layout"); } }, S.tutorOpen ? t("hide_tutor") : "✦ " + t("ask_tutor")));
    tocEl = h("nav", { class: "toc" }, h("div", { class: "small-caps", style: { padding: "0 8px 8px" } }, t("contents")));
    for (const e of data.toc) tocEl.append(h("button", { class: e.level > 1 ? "l2" : "", onclick: () => scrollTo(e.page, e.label) }, e.label));
    readerEl = h("div", { class: "reader", "data-lang": S.doc.lang });
    const paper = h("div", { class: "paper" });
    for (const pg of data.pages) {
      const sec = h("section", { class: "pg", "data-page": pg.page });
      sec.append(h("span", { class: "pgno small-caps" }, `— ${unitName(S.doc.unit_label)} ${pg.page} —`));
      let ul = null;
      for (const b of pg.blocks) {
        if (b.kind === "bullet") { if (!ul) { ul = h("ul"); sec.append(ul); } ul.append(h("li", {}, renderText(b.text))); continue; }
        ul = null;
        if (b.kind === "title") sec.append(h("h2", { class: "rh", style: { fontSize: "34px" } }, b.text));
        else if (b.kind === "heading") sec.append(h(b.level <= 1 ? "h2" : "h3", { class: "rh", "data-h": b.text }, b.text));
        else if (b.kind === "math") sec.append(h("div", { class: "math" }, b.text));
        else if (b.kind === "ref") sec.append(h("p", { style: { color: "var(--muted)", fontSize: "13px" } }, b.text));
        else sec.append(h("p", {}, renderText(b.text)));
      }
      paper.append(sec);
    }
    readerEl.append(paper);
    const body = h("div", { class: "study" }, tocEl, readerEl);
    root.append(bar, body);
    if (showPdf) renderPdf();
  }

  async function togglePdf() { showPdf = !showPdf; render(); }
  async function renderPdf() {
    try {
      pdfUrl ||= await api.fileBlobUrl(S.doc.id);
      readerEl.innerHTML = ""; readerEl.style.padding = "0";
      readerEl.append(h("iframe", { src: pdfUrl, style: { width: "100%", height: "100%", border: "0" }, title: S.doc.title }));
    } catch (e) { toast(e.message, true); showPdf = false; }
  }

  function scrollTo(page, headText) {
    if (showPdf) { showPdf = false; render(); }
    const pg = readerEl.querySelector(`[data-page="${page}"]`); if (!pg) return;
    const hd = headText && [...pg.querySelectorAll("[data-h]")].find((x) => x.dataset.h === headText);
    (hd || pg).scrollIntoView({ behavior: "smooth", block: "start" });
  }
  function flash(page, snippet) {
    if (!readerEl) return;
    if (showPdf) { showPdf = false; render(); }
    const pg = readerEl.querySelector(`[data-page="${page}"]`); if (!pg) return;
    const key = (snippet || "").replace(/\s+/g, " ").slice(0, 40);
    const target = [...pg.querySelectorAll("p,li")].find((p) => p.textContent.replace(/\s+/g, " ").includes(key)) || pg;
    target.scrollIntoView({ behavior: "smooth", block: "center" });
    target.classList.add("hl-flash"); target.style.transition = "background .6s"; target.style.background = "#ffe9a8";
    setTimeout(() => (target.style.background = ""), 2200);
  }

  on("goto", ({ page, snippet }) => { if (S.view !== "study") emit("nav", { view: "study", then: () => flash(page, snippet) }); else flash(page, snippet); }, "reader");
  on("doc", () => { pdfUrl = null; showPdf = false; if (S.doc) load(); }, "reader");
  on("lang", () => { if (data) render(); }, "reader");
  return { load, flash };
}
