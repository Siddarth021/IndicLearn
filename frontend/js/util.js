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
    if (/^\s*[•\-*]\s+/.test(ln)) { if (!list) { out.push("<ul style='margin:4px 0 8px;padding-left:18px'>"); list = true; } out.push("<li>" + inline(ln.replace(/^\s*[•\-*]\s+/, "")) + "</li>"); continue; }
    if (list) { out.push("</ul>"); list = false; }
    if (ln.trim() === "") continue;
    out.push("<p>" + inline(ln) + "</p>");
  }
  if (list) out.push("</ul>");
  return out.join("");
}
let toastEl;
export function toast(msg, err = false) {
  toastEl?.remove();
  toastEl = h("div", { class: "toast" + (err ? " err" : "") }, msg);
  document.body.append(toastEl);
  setTimeout(() => toastEl?.remove(), 3200);
}
export const debounce = (fn, ms = 150) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
export const STATUS_COLORS = { new: "#B9AE92", doubt: "#E0A53A", revise: "#C4584A", learning: "#CFA444", mastered: "#5E8F5A" };
