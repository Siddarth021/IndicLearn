import { api } from "./api.js";
import { t } from "./i18n.js";
import { h, toast, debounce, STATUS_COLORS } from "./util.js";
import { S, emit, on, clearOwner } from "./state.js";
import { MindMap } from "./graph.js";

export function createMapView(root) {
  clearOwner("map");
  let map, svg, wrap, tip, detail, tourBar, legend, mode = "default", layout = "radial", g = null, tourIdx = -1, tourList = [], loadedFor = null;
  const toolBtns = {};

  function build() {
    root.innerHTML = "";
    const seg = (name, items, cur, fn) => {
      const s = h("div", { class: "seg", role: "group" });
      items.forEach(([k, label]) => { const b = h("button", { class: k === cur ? "on" : "", onclick: () => { [...s.children].forEach((c) => c.classList.remove("on")); b.classList.add("on"); fn(k); } }, label); s.append(b); });
      return s;
    };
    const search = h("input", { class: "field", style: { width: "190px", padding: "6px 10px" }, placeholder: t("search_nodes"), "aria-label": t("search_nodes") });
    search.addEventListener("input", debounce(() => map?.search(search.value), 200));
    const chk = (label, key, def) => h("label", {}, h("input", { type: "checkbox", checked: def, onchange: (e) => map.setShow({ [key]: e.target.checked }) }), label);
    const bar = h("div", { class: "maptools" },
      seg("mode", [["default", t("map_default")], ["personal", t("map_personal")]], mode, (m) => { mode = m; closeDetail(); load(); }),
      seg("layout", [["radial", t("layout_radial")], ["tree", t("layout_tree")], ["flow", t("layout_flow")]], layout, (l) => { layout = l; map.setLayout(l); }),
      chk(t("show_prereq"), "prereq", true), chk(t("show_related"), "related", false), search,
      h("button", { class: "btn sm ghost", onclick: () => map.fit() }, t("fit")), h("button", { class: "btn sm ghost", onclick: () => map.expandAll() }, t("expand")), h("button", { class: "btn sm ghost", onclick: () => map.collapseAll() }, t("collapse")),
      h("span", { style: { flex: 1 } }), h("button", { class: "btn sm", onclick: startTour }, "▶ " + t("tour")),
      h("button", { class: "btn sm ghost", onclick: () => { S.tutorOpen = !S.tutorOpen; emit("layout"); } }, S.tutorOpen ? t("hide_tutor") : "✦ " + t("ask_tutor")));
    svg = document.createElementNS("http://www.w3.org/2000/svg", "svg"); svg.setAttribute("id", "mapsvg"); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", t("nav_map"));
    tip = h("div", { class: "tip", hidden: true }); detail = h("aside", { class: "detail", hidden: true }); tourBar = h("div", { class: "tour", hidden: true }); legend = h("div", { class: "legend" });
    wrap = h("div", { class: "mapwrap" }, svg, legend, tip, detail, tourBar);
    root.append(bar, wrap);
    map = new MindMap(svg, {
      onSelect: (n) => openDetail(n), onBackground: () => { if (tourIdx < 0) closeDetail(); },
      onHover: (n, e) => hoverTip(n, e), offsetX: 0,
    });
    map.layout = layout; drawLegend();
  }

  function drawLegend() {
    legend.innerHTML = "";
    const row = (color, label, fill = "#fff", dashed = false) => h("div", { class: "row" }, h("i", { style: { borderColor: color, background: fill, borderStyle: dashed ? "dashed" : "solid" } }), label);
    if (mode === "default") legend.append(row("#B8892B", t("lg_chapter"), "#F3E7C5"), row("#B8892B", t("lg_section")), row("#8A8068", t("lg_concept"), "#FBF4E0"), h("div", { class: "small-caps", style: { marginTop: "4px" } }, "⇢ " + t("show_prereq")));
    else for (const k of ["new", "doubt", "revise", "learning", "mastered"]) legend.append(row(STATUS_COLORS[k], t("st_" + k), { mastered: "#E3EFDF", revise: "#F6DDD8", doubt: "#FAEBD0", learning: "#F3E7C5", new: "#fff" }[k], k === "doubt"));
    if (mode === "personal") legend.append(row("#E0A53A", t("lg_doubt"), "#FAEBD0", true));
  }

  async function load(keepView = false) {
    if (!S.doc) return;
    try {
      g = await api.graph(S.doc.id, mode);
      S.graph = g; S.nodes = new Map(g.nodes.map((n) => [n.id, n]));
      map.setData({ nodes: g.nodes, edges: g.edges, path: g.path, mode, title: S.doc.title });
      drawLegend(); loadedFor = S.doc.id + mode;
      if (!keepView) setTimeout(() => map.fit(), 120);
      if (map.selected && S.nodes.has(map.selected)) openDetail(S.nodes.get(map.selected), true);
      emit("nodes");
    } catch (e) { toast(e.message, true); }
  }

  function hoverTip(n, e) {
    if (!n || n.type === "root") { tip.hidden = true; return; }
    const r = wrap.getBoundingClientRect(); tip.hidden = false;
    tip.innerHTML = "";
    tip.append(h("b", {}, n.full_label || n.label));
    const type = { chapter: t("lg_chapter"), section: t("lg_section"), topic: t("lg_section"), concept: t("lg_concept"), doubt: t("lg_doubt") }[n.type];
    tip.append(h("span", { class: "small-caps" }, type + (n.mastery ? " · " + t("st_" + n.mastery.status) : "")));
    const body = n.definition || n.summary; if (body) tip.append(h("div", { style: { marginTop: "5px", color: "#4A4336" } }, body.length > 170 ? body.slice(0, 168) + "…" : body));
    tip.style.left = Math.min(e.clientX - r.left + 16, r.width - 300) + "px"; tip.style.top = Math.min(e.clientY - r.top + 14, r.height - 140) + "px";
  }

  function closeDetail() { detail.hidden = true; if (map) { map.opts.offsetX = 0; map.select(null); } }
  function openDetail(n, quiet = false) {
    if (!n || n.type === "root") { closeDetail(); return; }
    tip.hidden = true; detail.hidden = false; map.opts.offsetX = 190; detail.innerHTML = ""; S.focus = n.type === "doubt" ? S.focus : n.id; emit("focus");
    const nodes = S.nodes; const lab = (id) => nodes.get(id)?.label || id;
    const link = (id) => h("button", { class: "lnk", onclick: () => { map.focus(id, 1.2); openDetail(nodes.get(id)); } }, lab(id));
    const section = (title, ...kids) => h("div", { class: "sec" }, h("span", { class: "small-caps" }, title), ...kids);
    const typeName = { chapter: t("lg_chapter"), section: t("lg_section"), topic: t("lg_section"), concept: t("lg_concept"), doubt: t("lg_doubt") }[n.type];
    detail.append(h("button", { class: "x", onclick: closeDetail, "aria-label": "close" }, "×"), h("span", { class: "small-caps" }, typeName));
    detail.append(h("h3", {}, n.full_label || n.label));
    if (n.mastery) detail.append(h("span", { class: "chip " + ({ mastered: "good", revise: "bad", doubt: "warn", learning: "gold", new: "" }[n.mastery.status]) }, t("st_" + n.mastery.status), n.mastery.aggregate ? ` · ${n.mastery.mastered}/${n.mastery.total}` : n.mastery.asked ? ` · ${t("asked_times", { n: n.mastery.asked })}` : ""));
    if (n.parent && n.parent !== "root" && nodes.get(n.parent)) detail.append(section(t("chapter_of"), link(n.parent)));
    if (n.type === "doubt") detail.append(section(t("lg_doubt"), h("p", { style: { margin: "4px 0" } }, n.summary || "")));
    else {
      if (n.definition) detail.append(section(t("definition"), h("p", { style: { margin: "4px 0" } }, n.definition)));
      else if (n.summary) detail.append(section(t("inside"), h("p", { style: { margin: "4px 0" } }, n.summary)));
      else detail.append(h("p", { style: { color: "var(--muted)", fontSize: "13px" } }, t("no_def")));
    }
    if (n.examples?.length) detail.append(section(t("examples"), ...n.examples.slice(0, 2).map((x) => h("p", { style: { margin: "4px 0", fontSize: "13.5px" } }, x))));
    if (n.formulas?.length) detail.append(section(t("formulas"), ...n.formulas.slice(0, 3).map((x) => h("div", { class: "math", style: { fontFamily: "ui-monospace,monospace", background: "var(--surface-2)", borderRadius: "8px", padding: "6px 10px", margin: "4px 0", fontSize: "13px" } }, x))));
    const edges = (S.graph?.edges || []);
    const pre = edges.filter((e) => e.type === "prereq" && e.target === n.id).map((e) => e.source);
    const post = edges.filter((e) => e.type === "prereq" && e.source === n.id).map((e) => e.target);
    const rel = edges.filter((e) => e.type === "related" && (e.source === n.id || e.target === n.id)).map((e) => (e.source === n.id ? e.target : e.source));
    const kids = edges.filter((e) => e.type === "contains" && e.source === n.id).map((e) => e.target).filter((id) => nodes.get(id)?.type !== "doubt");
    if (pre.length) detail.append(section("⇠ " + t("prereq"), ...[...new Set(pre)].filter((i) => nodes.has(i)).map(link)));
    if (post.length) detail.append(section(t("leads_to") + " ⇢", ...[...new Set(post)].filter((i) => nodes.has(i)).map(link)));
    if (rel.length) detail.append(section(t("related"), ...[...new Set(rel)].filter((i) => nodes.has(i)).slice(0, 6).map(link)));
    if (kids.length) detail.append(section(t("inside"), ...kids.slice(0, 12).map(link)));
    const acts = h("div", { class: "acts" });
    if (n.type !== "doubt") {
      acts.append(h("button", { class: "btn sm", onclick: () => mark(n.id, "self_understood") }, "✓ " + t("understood")), h("button", { class: "btn sm ghost", onclick: () => mark(n.id, "self_confused") }, "? " + t("confused")));
    }
    acts.append(h("button", { class: "btn sm ghost", onclick: () => emit("ask", { text: n.type === "doubt" ? n.summary : t("sg_explain", { t: n.label }), focus: n.type === "doubt" ? null : n.id }) }, "✦ " + t("ask_this")));
    if (n.type !== "doubt") acts.append(h("button", { class: "btn sm ghost", onclick: () => emit("ask", { text: t("sg_quiz"), focus: n.id }) }, t("quiz_this")));
    if (n.pages?.length || n.page) acts.append(h("button", { class: "btn sm ghost", onclick: () => emit("goto", { page: n.page || n.pages[0], snippet: n.definition || "" }) }, `${t("open_reader")} · ${t("cited")}${n.page || n.pages[0]}`));
    detail.append(acts);
    if (!quiet) emit("mark", { id: n.id, kind: "viewed" });
  }

  async function mark(id, kind) {
    try { await api.mark(S.doc.id, id, kind); toast(t("thanks_marked")); emit("progress", {}); } catch (e) { toast(e.message, true); }
  }

  // ---------------- guided tour along the learning path
  function startTour() {
    tourList = (g?.path || []).filter((id) => { const n = S.nodes.get(id); return n && n.type !== "doubt" && (n.type !== "section" || n.definition || (n.n_children || 0) > 0); });
    if (!tourList.length) return;
    tourIdx = 0; stepTour(0);
  }
  function stepTour(d) {
    tourIdx = Math.max(0, Math.min(tourList.length - 1, tourIdx + d));
    const id = tourList[tourIdx]; map.focus(id, 1.3); openDetail(S.nodes.get(id), true);
    tourBar.hidden = false; tourBar.innerHTML = "";
    tourBar.append(h("span", {}, t("step_of", { i: tourIdx + 1, n: tourList.length })), h("button", { class: "btn sm ghost", onclick: () => stepTour(-1), disabled: tourIdx === 0 }, "‹ " + t("prev")),
      h("button", { class: "btn sm", onclick: () => stepTour(1), disabled: tourIdx === tourList.length - 1 }, t("next") + " ›"),
      h("button", { class: "linkbtn", onclick: () => { tourIdx = -1; tourBar.hidden = true; } }, t("tour_stop")));
  }

  const refresh = debounce(() => { if (S.view === "map") load(true); else loadedFor = null; }, 250);
  on("progress", refresh, "map");
  on("doc", () => { loadedFor = null; }, "map");
  on("lang", () => { if (root.firstChild) { build(); loadedFor = null; if (S.view === "map") load(); } }, "map");
  return {
    show(selectId) {
      if (!map) build();
      if (loadedFor !== (S.doc?.id + mode)) load().then(() => selectId && sel(selectId)); else if (selectId) sel(selectId);
      setTimeout(() => map.fit(), 80);
    },
    reload: () => load(true),
  };
  function sel(id) { if (S.nodes.has(id)) { map.focus(id, 1.2); openDetail(S.nodes.get(id), true); } }
}
