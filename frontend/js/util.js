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
const HAS_MATH = /\$[^$\n]+\$|\\\(|\\\[|\\[a-zA-Z]{2,}/;
const SKIP = "svg, .katex, script, style, textarea, pre, code, option, [data-nomath]";
// true when some text outside already-rendered maths still holds a formula
function needsMath(el) {
  if (!el || el.nodeType !== 1 || el.closest(SKIP)) return false;
  const w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, { acceptNode: (n) => n.parentElement?.closest(SKIP) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT });
  for (let n = w.nextNode(); n; n = w.nextNode()) if (HAS_MATH.test(n.nodeValue)) return true;
  return false;
}
export async function mathify(el) {
  if (!needsMath(el)) return;
  await loadKatex();
  try {
    window.renderMathInElement?.(el, { throwOnError: false, delimiters: [
      { left: "$$", right: "$$", display: true }, { left: "\\[", right: "\\]", display: true },
      { left: "$", right: "$", display: false }, { left: "\\(", right: "\\)", display: false }] });
  } catch (e) { /* leave the text as it is */ }
}
// Safety net: every piece of text that ever appears on a page (tooltips, map panels, chat, lessons, quizzes,
// papers ...) is typeset automatically, so a formula can never show up as raw "$p \vee q$" again.
export function autoMath(root = document.body) {
  let queue = new Set(), timer = null, busy = false;
  const run = async () => {
    timer = null; const els = [...queue]; queue = new Set(); busy = true;
    for (const el of els) if (el.isConnected) await mathify(el);
    busy = false;
  };
  const obs = new MutationObserver((muts) => {
    if (busy) return;
    for (const m of muts) {
      const t = m.target.nodeType === 1 ? m.target : m.target.parentElement;
      if (m.type === "characterData") { if (t) queue.add(t); }
      else for (const n of m.addedNodes) { const e = n.nodeType === 1 ? n : n.parentElement; if (e) queue.add(e); }
    }
    if (queue.size && !timer) timer = setTimeout(run, 30);
  });
  obs.observe(root, { childList: true, subtree: true, characterData: true });
  queue.add(root); timer = setTimeout(run, 0);
  return obs;
}
// For places that cannot hold HTML (SVG labels, native tooltips, <option>, document title): LaTeX -> readable Unicode.
const SYM = { vee: "∨", lor: "∨", wedge: "∧", land: "∧", neg: "¬", lnot: "¬", to: "→", rightarrow: "→", Rightarrow: "⇒", implies: "⇒",
  leftarrow: "←", leftrightarrow: "↔", Leftrightarrow: "⇔", iff: "⇔", forall: "∀", exists: "∃", in: "∈", notin: "∉", ni: "∋",
  subseteq: "⊆", subset: "⊂", supseteq: "⊇", supset: "⊃", cup: "∪", cap: "∩", emptyset: "∅", varnothing: "∅", setminus: "∖",
  times: "×", cdot: "·", div: "÷", pm: "±", mp: "∓", leq: "≤", le: "≤", geq: "≥", ge: "≥", neq: "≠", ne: "≠", approx: "≈", equiv: "≡",
  sim: "∼", propto: "∝", infty: "∞", sum: "∑", prod: "∏", int: "∫", partial: "∂", nabla: "∇", oplus: "⊕", otimes: "⊗", mid: "∣",
  ldots: "…", cdots: "⋯", dots: "…", circ: "∘", top: "⊤", bot: "⊥", vdash: "⊢", models: "⊨", therefore: "∴", because: "∵",
  alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ε", varepsilon: "ε", theta: "θ", lambda: "λ", mu: "μ", pi: "π", rho: "ρ",
  sigma: "σ", tau: "τ", phi: "φ", varphi: "φ", omega: "ω", Delta: "Δ", Sigma: "Σ", Omega: "Ω", Theta: "Θ", Lambda: "Λ", Pi: "Π", Phi: "Φ",
  langle: "⟨", rangle: "⟩", lceil: "⌈", rceil: "⌉", lfloor: "⌊", rfloor: "⌋", log: "log", lg: "lg", ln: "ln", max: "max", min: "min", gcd: "gcd", mod: "mod", bmod: "mod" };
const BB = { N: "ℕ", Z: "ℤ", Q: "ℚ", R: "ℝ", C: "ℂ", P: "ℙ", E: "𝔼" };
const SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹", SUB = "₀₁₂₃₄₅₆₇₈₉";
export function plainMath(src) {
  let s = String(src ?? "");
  if (!/[\\$]/.test(s)) return s;
  s = s.replace(/\\(?:mathbb|mathbf|mathrm|mathit|mathcal|text|textbf|operatorname)\s*\{([^{}]*)\}/g, (_, x) => (/^[A-Z]$/.test(x) && BB[x]) || x);
  s = s.replace(/\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g, "($1)/($2)").replace(/\\sqrt\s*\{([^{}]*)\}/g, "√($1)");
  s = s.replace(/\\(?:left|right|big|Big|bigg|Bigg)\b\s*/g, "").replace(/\\[,;:! ]/g, " ").replace(/\\\\/g, " ");
  s = s.replace(/\\([A-Za-z]+)/g, (m, c) => SYM[c] ?? c);
  s = s.replace(/\^\s*\{?(\d+)\}?/g, (_, d) => [...d].map((c) => SUP[c]).join("")).replace(/_\s*\{?(\d+)\}?/g, (_, d) => [...d].map((c) => SUB[c]).join(""));
  s = s.replace(/[{}]/g, "").replace(/\$+/g, "").replace(/\s{2,}/g, " ");
  return s.trim();
}
// Shorten text for a tooltip without ever cutting a $...$ formula in half.
export function clip(s, max) {
  s = String(s ?? ""); if (s.length <= max) return s;
  let cut = max;
  const before = s.slice(0, cut), dollars = (before.match(/(?<!\\)\$/g) || []).length;
  if (dollars % 2 === 1) { const open = before.lastIndexOf("$"); cut = open > 0 ? open : max; }
  return s.slice(0, cut).trimEnd() + "…";
}
