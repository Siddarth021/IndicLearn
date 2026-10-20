// DOM helpers + a tiny, safe markdown subset (bold / italic / code / bullets).
export function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid !== null && kid !== undefined && kid !== false) el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return el;
}
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
export function md(src) {
  const lines = esc(src).split("\n");
  const out = []; let list = false;
  const inline = (s) => s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/(^|[\s(])_(.+?)_(?=[\s).,;:]|$)/g, "$1<em>$2</em>").replace(/`([^`]+)`/g, "<code>$1</code>");
  for (const ln of lines) {
    if (/^\s*[•\-*]\s+/.test(ln)) { if (list !== "ul") { if (list) out.push(`</${list}>`); out.push("<ul style='margin:4px 0 8px;padding-left:18px'>"); list = "ul"; } out.push("<li>" + inline(ln.replace(/^\s*[•\-*]\s+/, "")) + "</li>"); continue; }
    if (/^\s*\d{1,2}[.)]\s+/.test(ln)) { if (list !== "ol") { if (list) out.push(`</${list}>`); out.push("<ol style='margin:4px 0 8px;padding-left:22px'>"); list = "ol"; } out.push("<li>" + inline(ln.replace(/^\s*\d{1,2}[.)]\s+/, "")) + "</li>"); continue; }
    if (list) { out.push(`</${list}>`); list = false; }
    if (ln.trim() === "") continue;
    out.push("<p>" + inline(ln) + "</p>");
  }
  if (list) out.push(`</${list}>`);
  return out.join("");
}
// solved-problem blocks: {p: text with $math$}, {math: display LaTeX}, {table: rows}
export function renderBlocks(bs) {
  const inl = (t) => md(t).replace(/^<p>|<\/p>$/g, "");
  return (bs || []).map((b) => b.table ? h("div", { class: "tblwrap" }, h("table", { class: "ptable" }, ...b.table.map((r, i) => h("tr", {}, ...r.map((c) => h(i ? "td" : "th", { html: inl(c) }))))))
    : b.math !== undefined ? h("div", { class: "mathblk" }, "$$" + b.math + "$$")
    : h("p", { html: inl(b.p || "") }));
}
let toastEl;
export function toast(msg, err = false) {
  toastEl?.remove();
  const el = (toastEl = h("div", { class: "toast" + (err ? " err" : ""), role: err ? "alert" : "status" }, msg));
  document.body.append(el);
  setTimeout(() => el.remove(), err ? 6000 : 3200); // remove only this toast (an older timer must not kill a newer error)
}
export const debounce = (fn, ms = 150) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
export const STATUS_COLORS = { new: "#B9AE92", doubt: "#E0A53A", revise: "#C4584A", learning: "#CFA444", mastered: "#5E8F5A" };

// Maths: render $...$, $$...$$ and \( \) with KaTeX (vendored, works offline). Loaded on first use only.
let katexReady = null;
function loadKatex() {
  if (katexReady) return katexReady;
  const add = (tag, attrs) => new Promise((res, rej) => { const e = Object.assign(document.createElement(tag), attrs); e.onload = res; e.onerror = rej; document.head.append(e); });
  katexReady = add("link", { rel: "stylesheet", href: "/static/vendor/katex/katex.min.css" })
    .then(() => add("script", { src: "/static/vendor/katex/katex.min.js" }))
    .then(() => add("script", { src: "/static/vendor/katex/auto-render.min.js" }))
    .catch(() => { katexReady = null; });
  return katexReady;
}
const HAS_MATH = /\$[^$\n]+\$|\\\(|\\\[|\\(frac|lambda|sum|times|leq|geq|alpha|beta|sigma|mu|infty)\b/;
export async function mathify(el) {
  if (!el || !HAS_MATH.test(el.textContent || "")) return;
  await loadKatex();
  try {
    window.renderMathInElement?.(el, { throwOnError: false, delimiters: [
      { left: "$$", right: "$$", display: true }, { left: "\\[", right: "\\]", display: true },
      { left: "$", right: "$", display: false }, { left: "\\(", right: "\\)", display: false }] });
  } catch (e) { /* leave the text as it is */ }
}
