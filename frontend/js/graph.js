// Interactive mind map (d3 v7).  Three layouts over the same data:
//   radial - classic mind map, root in the centre
//   tree   - left-to-right outline
//   flow   - learning order: columns follow prerequisite / "next" edges
import { STATUS_COLORS } from "./util.js";
const d3 = window.d3;
const TAU = Math.PI * 2;

const RADIUS = { root: 30, chapter: 22, section: 16, topic: 16, concept: 10, doubt: 11 };
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

export class MindMap {
  constructor(svgEl, opts = {}) {
    this.svg = d3.select(svgEl);
    this.opts = opts;
    this.layout = "radial";
    this.mode = "default";
    this.show = { prereq: true, related: false };
    this.collapsed = new Set();
    this.selected = null;
    this.highlight = new Set();
    this.byId = new Map();
    this.children = new Map();
    this.svg.selectAll("*").remove();
    const defs = this.svg.append("defs");
    for (const [id, color] of [["arrow-prereq", "#B8892B"], ["arrow-next", "#9c9178"]]) {
      defs.append("marker").attr("id", id).attr("viewBox", "0 -5 10 10").attr("refX", 22).attr("refY", 0).attr("markerWidth", 7).attr("markerHeight", 7).attr("orient", "auto")
        .append("path").attr("d", "M0,-4L9,0L0,4Z").attr("fill", color);
    }
    this.root = this.svg.append("g");
    this.gLinks = this.root.append("g").attr("class", "links");
    this.gOver = this.root.append("g").attr("class", "overlay");
    this.gNodes = this.root.append("g").attr("class", "nodes");
    this.zoom = d3.zoom().scaleExtent([0.15, 3]).on("zoom", (e) => this.root.attr("transform", e.transform));
    this.svg.call(this.zoom).on("dblclick.zoom", null);
    this.size = { w: 800, h: 600 };
    new ResizeObserver(() => { const r = svgEl.getBoundingClientRect(); this.size = { w: r.width, h: r.height }; }).observe(svgEl);
  }

  // ------------------------------------------------------------------ data
  setData({ nodes, edges, path, mode, title }) {
    this.mode = mode || "default";
    this.edgesAll = edges;
    this.path = path || [];
    const hasRoot = nodes.some((n) => n.id === "root");
    const rootNode = { id: "root", label: title || "", full_label: title || "", type: "root", parent: null, importance: 1, level: 0, n_children: 0 };
    this.nodes = [...(hasRoot ? [] : [rootNode]), ...nodes.map((n) => ({ ...n, parent: n.id === "root" ? null : (n.parent || "root") }))];
    this.byId = new Map(this.nodes.map((n) => [n.id, n]));
    this.children = new Map();
    for (const n of this.nodes) if (n.parent && this.byId.has(n.parent)) {
      if (!this.children.has(n.parent)) this.children.set(n.parent, []);
      this.children.get(n.parent).push(n.id);
    }
    for (const arr of this.children.values()) arr.sort((a, b) => (this.byId.get(a).order ?? 0) - (this.byId.get(b).order ?? 0));
    if (!this._inited || this._doc !== title) { this._doc = title; this._inited = true; this.autoCollapse(); }
    else this.collapsed = new Set([...this.collapsed].filter((id) => this.byId.has(id)));
    for (const n of this.nodes) if (n.type === "doubt") { let c = n.parent; while (c) { this.collapsed.delete(c); c = this.byId.get(c)?.parent; } }
    this.render(false);
  }

  autoCollapse() {
    // expand level by level until ~60 nodes are visible
    this.collapsed = new Set();
    const total = this.nodes.length;
    if (total <= 60) return;
    const open = new Set(["root"]);
    let frontier = ["root"]; let visible = 1;
    while (frontier.length) {
      const next = [];
      for (const id of frontier) for (const c of this.children.get(id) || []) next.push(c);
      if (visible + next.length > 62) break;
      visible += next.length; next.forEach((c) => open.add(c)); frontier = next;
    }
    for (const n of this.nodes) if ((this.children.get(n.id) || []).length && !open.has(n.id)) this.collapsed.add(n.id);
  }

  visibleIds() {
    const out = new Set(); const walk = (id) => { out.add(id); if (this.collapsed.has(id)) return; for (const c of this.children.get(id) || []) walk(c); };
    walk("root"); return out;
  }
  ancestorVisible(id, vis) { let cur = id; while (cur && !vis.has(cur)) cur = this.byId.get(cur)?.parent; return cur; }
  statusOf(n) { return n.mastery?.status || "new"; }

  // ---------------------------------------------------------------- layout
  computeLayout() {
    const vis = this.visibleIds();
    const mk = (id) => ({ id, data: this.byId.get(id), kids: this.collapsed.has(id) ? [] : (this.children.get(id) || []).map(mk) });
    const tree = d3.hierarchy(mk("root"), (d) => d.kids);
    const pos = new Map();
    const leaves = tree.leaves().length;
    if (this.layout === "radial") {
      const R = Math.max(300, Math.min(800, leaves * 13 + 150));
      d3.tree().size([TAU, R]).separation((a, b) => (a.parent === b.parent ? 1 : 1.6) / Math.max(a.depth, 1))(tree);
      tree.each((d) => pos.set(d.data.id, { x: d.y * Math.cos(d.x - Math.PI / 2), y: d.y * Math.sin(d.x - Math.PI / 2), angle: d.x, radius: d.y, depth: d.depth }));
    } else if (this.layout === "tree") {
      d3.tree().nodeSize([30, 250])(tree);
      tree.each((d) => pos.set(d.data.id, { x: d.y, y: d.x, depth: d.depth }));
    } else {
      this.flowLayout(vis, pos);
    }
    this.vis = vis;
    return pos;
  }

  flowLayout(vis, pos) {
    // Columns follow the learning flow: containment keeps a topic's children in its column (w=0),
    // "next" / "prereq" edges push the target one column to the right (w=1).  Cycle-safe.
    const ids = [...vis].filter((id) => id !== "root");
    const idx = new Map(ids.map((id, i) => [id, i]));
    const rel = []; const seen = new Set(); const edges = [];
    for (const id of ids) { const p = this.byId.get(id).parent; if (p && p !== "root" && vis.has(p)) rel.push([p, id, 0]); }
    for (const e of this.edgesAll) {
      if (e.type !== "prereq" && e.type !== "next") continue;
      const a = this.ancestorVisible(e.source, vis); const b = this.ancestorVisible(e.target, vis);
      if (!a || !b || a === b || a === "root" || b === "root") continue;
      const key = a + ">" + b; if (seen.has(key)) continue; seen.add(key);
      // a prerequisite pointing back to an ancestor/descendant is not a flow edge
      if (this.isAncestor(a, b) || this.isAncestor(b, a)) continue;
      rel.push([a, b, 1]); edges.push([a, b, e.type]);
    }
    const layer = new Map(ids.map((i) => [i, 0]));
    for (let it = 0; it < ids.length + 2; it++) {
      let ch = false;
      for (const [a, b, w] of rel) if (layer.get(b) < layer.get(a) + w && layer.get(a) + w <= 10) { layer.set(b, layer.get(a) + w); ch = true; }
      if (!ch) break;
    }
    const cols = new Map();
    for (const id of ids) { const l = layer.get(id); if (!cols.has(l)) cols.set(l, []); cols.get(l).push(id); }
    const ord = (id) => { const n = this.byId.get(id); const par = n.parent && n.parent !== "root" ? (this.byId.get(n.parent)?.order ?? 0) : n.order ?? 0; return par * 1000 + (n.order ?? 0); };
    for (const [l, arr] of cols) { arr.sort((a, b) => ord(a) - ord(b)); arr.forEach((id, i) => pos.set(id, { x: l * 250, y: (i - (arr.length - 1) / 2) * 40, depth: l })); }
    pos.set("root", { x: -230, y: 0, depth: 0 });
    this._flowEdges = edges;
  }
  isAncestor(a, b) { let cur = this.byId.get(b)?.parent; while (cur) { if (cur === a) return true; cur = this.byId.get(cur)?.parent; } return false; }

  // ---------------------------------------------------------------- render
  setLayout(l) {
    if (l === "flow" && this.layout !== "flow") {
      this._prevCollapsed = new Set(this.collapsed); // flow reads best at topic level: fold concepts into their topics
      for (const n of this.nodes) if ((n.type === "section" || n.type === "topic") && (this.children.get(n.id) || []).length) this.collapsed.add(n.id);
    } else if (l !== "flow" && this.layout === "flow" && this._prevCollapsed) { this.collapsed = this._prevCollapsed; this._prevCollapsed = null; }
    this.layout = l; this.render(true); setTimeout(() => this.fit(), 60);
  }
  setMode(m) { this.mode = m; }
  setShow(s) { this.show = { ...this.show, ...s }; this.renderOverlay(); }

  render(animate = true) {
    const pos = this.computeLayout(); this.pos = pos;
    const vis = this.vis; const self = this;
    const data = [...vis].map((id) => ({ id, n: this.byId.get(id), p: pos.get(id) })).filter((d) => d.p);
    const t = animate ? d3.transition().duration(550).ease(d3.easeCubicInOut) : null;

    // links (containment)
    this.gLinks.selectAll("*").remove();
    if (this.layout !== "flow") {
      const links = []; for (const id of vis) if (id !== "root" && this.byId.get(id).parent && vis.has(this.byId.get(id).parent)) links.push({ s: this.byId.get(id).parent, t: id });
      const gen = this.layout === "radial" ? d3.linkRadial().angle((d) => d.angle).radius((d) => d.radius) : d3.linkHorizontal().x((d) => d.x).y((d) => d.y);
      this.gLinks.selectAll("path").data(links).join("path").attr("class", "link").attr("fill", "none")
        .attr("stroke", (d) => (this.byId.get(d.t).type === "doubt" ? "#E0A53A" : "#D9CDB0")).attr("stroke-width", (d) => (this.byId.get(d.s).type === "root" ? 2.4 : 1.5))
        .attr("stroke-dasharray", (d) => (this.byId.get(d.t).type === "doubt" ? "4 4" : null))
        .attr("d", (d) => gen({ source: pos.get(d.s), target: pos.get(d.t) })).attr("opacity", 0).transition().duration(animate ? 500 : 0).attr("opacity", 1);
    }

    // nodes
    const sel = this.gNodes.selectAll("g.node").data(data, (d) => d.id);
    sel.exit().remove();
    const enter = sel.enter().append("g").attr("class", "node").style("cursor", "pointer")
      .attr("transform", (d) => `translate(${d.p.x},${d.p.y})`);
    enter.append("circle").attr("class", "ring").attr("fill", "none");
    enter.append("circle").attr("class", "main");
    enter.append("text").attr("class", "lbl");
    enter.append("text").attr("class", "badge").attr("text-anchor", "middle").attr("dy", "0.35em").style("pointer-events", "none");
    const all = enter.merge(sel);
    all.on("click", (e, d) => { e.stopPropagation(); self.onClick(d.id); })
      .on("dblclick", (e, d) => { e.stopPropagation(); self.toggle(d.id); })
      .on("mouseenter", (e, d) => { self.hover(d.id, true); self.opts.onHover?.(d.n, e); })
      .on("mousemove", (e, d) => self.opts.onHover?.(d.n, e))
      .on("mouseleave", (e, d) => { self.hover(d.id, false); self.opts.onHover?.(null); });
    (t ? all.transition(t) : all).attr("transform", (d) => `translate(${d.p.x},${d.p.y})`);
    this.styleNodes(all);
    this.svg.on("click", () => { this.opts.onBackground?.(); });
    this.renderOverlay();
  }

  styleNodes(all) {
    const self = this;
    all.each(function (d) {
      const g = d3.select(this); const n = d.n; const st = self.statusOf(n); const personal = self.mode === "personal";
      const r = RADIUS[n.type] || 9; const hasKids = (self.children.get(n.id) || []).length > 0;
      const stroke = personal && n.type !== "root" ? STATUS_COLORS[st] : n.type === "doubt" ? "#E0A53A" : n.type === "concept" ? "#8A8068" : "#B8892B";
      let fill = "#fff";
      if (n.type === "root") fill = "#B8892B";
      else if (n.type === "chapter") fill = personal ? "#fff" : "#F3E7C5";
      else if (n.type === "concept") fill = personal ? ({ mastered: "#E3EFDF", revise: "#F6DDD8", doubt: "#FAEBD0", learning: "#F3E7C5", new: "#fff" }[st]) : "#FBF4E0";
      else if (n.type === "doubt") fill = "#FAEBD0";
      if (personal && n.type !== "concept" && n.type !== "root" && n.type !== "doubt") fill = { mastered: "#E3EFDF", revise: "#F6DDD8", doubt: "#FAEBD0", learning: "#F3E7C5", new: "#fff" }[st];
      g.select("circle.main").attr("r", r).attr("fill", fill).attr("stroke", stroke).attr("stroke-width", n.type === "concept" ? 2 : 3)
        .attr("stroke-dasharray", n.type === "doubt" ? "3 3" : null);
      // progress ring for aggregate nodes (personal map)
      const ring = g.select("circle.ring");
      if (personal && n.mastery?.aggregate && n.mastery.total > 0) {
        const c = 2 * Math.PI * (r + 5);
        ring.attr("r", r + 5).attr("stroke", "#5E8F5A").attr("stroke-width", 3).attr("stroke-linecap", "round")
          .attr("stroke-dasharray", `${c * (n.mastery.progress || 0)} ${c}`).attr("transform", "rotate(-90)").attr("opacity", n.mastery.progress > 0 ? 1 : 0);
      } else ring.attr("opacity", 0);
      // label
      const label = (n.label || "").length > 30 ? n.label.slice(0, 29) + "…" : n.label;
      const txt = g.select("text.lbl").text(label).attr("font-size", n.type === "root" ? 17 : n.type === "chapter" ? 15 : n.type === "concept" ? 12.5 : 13.5)
        .attr("font-weight", n.type === "chapter" || n.type === "root" ? 600 : 500).attr("fill", n.type === "concept" ? "#4A4336" : "#1F1B14");
      if (n.type === "root") { txt.attr("text-anchor", "middle").attr("dy", r + 16).attr("x", 0); }
      else if (self.layout === "radial") {
        const right = Math.sin(d.p.angle) >= 0;
        txt.attr("text-anchor", right ? "start" : "end").attr("x", right ? r + 8 : -(r + 8)).attr("dy", "0.32em");
      } else { txt.attr("text-anchor", "start").attr("x", r + 8).attr("dy", "0.32em"); }
      const collapsed = self.collapsed.has(n.id) && hasKids;
      g.select("text.badge").text(collapsed ? "+" + countDesc(self, n.id) : "").attr("font-size", 10).attr("font-weight", 700).attr("fill", "#7a5a14");
      g.classed("sel", self.selected === n.id);
      g.select("circle.main").attr("stroke-width", self.selected === n.id ? 5 : n.type === "concept" ? 2 : 3);
      g.classed("dim", self.highlight.size > 0 && !self.highlight.has(n.id));
    });
  }

  renderOverlay() {
    const vis = this.vis; if (!vis) return; const pos = this.pos;
    const items = [];
    if (this.layout === "flow") {
      for (const [a, b, type] of this._flowEdges || []) items.push({ a, b, type });
    } else {
      const seen = new Set();
      for (const e of this.edgesAll) {
        if (e.type !== "prereq" && e.type !== "related") continue;
        if (e.type === "prereq" && !this.show.prereq) continue;
        if (e.type === "related" && !this.show.related) continue;
        const a = this.ancestorVisible(e.source, vis); const b = this.ancestorVisible(e.target, vis);
        if (!a || !b || a === b || a === "root" || b === "root") continue;
        const key = a + ">" + b + e.type; if (seen.has(key)) continue; seen.add(key);
        items.push({ a, b, type: e.type });
      }
    }
    const curve = (d) => {
      const s = pos.get(d.a); const t = pos.get(d.b); if (!s || !t) return "";
      const mx = (s.x + t.x) / 2; const my = (s.y + t.y) / 2; const dx = t.x - s.x; const dy = t.y - s.y;
      const k = this.layout === "flow" ? 0.06 : 0.16;
      return `M${s.x},${s.y} Q${mx - dy * k},${my + dx * k} ${t.x},${t.y}`;
    };
    this.gOver.selectAll("path").data(items, (d) => d.a + d.b + d.type).join("path").attr("class", "link ov").attr("fill", "none")
      .attr("stroke", (d) => (d.type === "prereq" ? "#B8892B" : d.type === "next" ? "#B9AE92" : "#8FA7BF")).attr("stroke-width", (d) => (d.type === "prereq" ? 1.8 : 1.2))
      .attr("stroke-dasharray", (d) => (d.type === "related" ? "2 4" : d.type === "next" ? "1 0" : "6 4"))
      .attr("marker-end", (d) => (d.type === "prereq" ? "url(#arrow-prereq)" : d.type === "next" ? "url(#arrow-next)" : null))
      .attr("d", curve).attr("opacity", 0.6);
  }

  // ------------------------------------------------------------ interaction
  hover(id, on) {
    if (this.highlight.size && !on) return;
    const nb = new Set([id]);
    if (on) {
      for (const e of this.edgesAll) { if (e.source === id) nb.add(e.target); if (e.target === id) nb.add(e.source); }
      const n = this.byId.get(id); if (n?.parent) nb.add(n.parent); for (const c of this.children.get(id) || []) nb.add(c);
      const vis = this.vis; const m = new Set([...nb].map((x) => this.ancestorVisible(x, vis)).filter(Boolean));
      this.gNodes.selectAll("g.node").classed("dim", (d) => !m.has(d.id));
    } else this.gNodes.selectAll("g.node").classed("dim", false);
  }

  onClick(id) {
    this.select(id);
    if (this.collapsed.has(id)) { this.collapsed.delete(id); this.render(true); }
    this.opts.onSelect?.(this.byId.get(id));
  }
  toggle(id) {
    if (!(this.children.get(id) || []).length) return;
    this.collapsed.has(id) ? this.collapsed.delete(id) : this.collapsed.add(id);
    this.render(true);
  }
  select(id) { this.selected = id; this.styleNodes(this.gNodes.selectAll("g.node")); }
  expandAll() { this.collapsed.clear(); this.render(true); setTimeout(() => this.fit(), 80); }
  collapseAll() { this.collapsed = new Set(this.nodes.filter((n) => n.id !== "root" && (this.children.get(n.id) || []).length && n.type !== "chapter" ? true : false).map((n) => n.id)); this.render(true); setTimeout(() => this.fit(), 80); }

  reveal(id) { let cur = this.byId.get(id)?.parent; let ch = false; while (cur) { if (this.collapsed.has(cur)) { this.collapsed.delete(cur); ch = true; } cur = this.byId.get(cur)?.parent; } if (ch) this.render(false); }
  focus(id, zoom = 1.25) {
    this.reveal(id);
    const p = this.pos.get(id); if (!p) return;
    this.select(id);
    if (!this.visible()) return;
    const { w, h } = this.size;
    this.svg.transition().duration(650).call(this.zoom.transform, d3.zoomIdentity.translate(w / 2 - (this.opts.offsetX || 0), h / 2).scale(zoom).translate(-p.x, -p.y));
  }
  visible() { const n = this.svg.node(); return !!(n && n.getBoundingClientRect().width > 0); }
  fit() {
    if (!this.pos || !this.visible()) return;
    const xs = []; const ys = []; for (const id of this.vis) { const p = this.pos.get(id); if (p) { xs.push(p.x); ys.push(p.y); } }
    if (!xs.length) return;
    const pad = 150; const x0 = Math.min(...xs) - pad; const x1 = Math.max(...xs) + pad; const y0 = Math.min(...ys) - 60; const y1 = Math.max(...ys) + 60;
    const { w, h } = this.size; const k = Math.min(w / (x1 - x0), h / (y1 - y0), 1.4);
    this.svg.transition().duration(600).call(this.zoom.transform, d3.zoomIdentity.translate(w / 2 - ((x0 + x1) / 2) * k, h / 2 - ((y0 + y1) / 2) * k).scale(k));
  }
  search(q) {
    q = (q || "").trim().toLowerCase();
    if (!q) { this.highlight = new Set(); this.styleNodes(this.gNodes.selectAll("g.node")); return 0; }
    const hits = this.nodes.filter((n) => n.id !== "root" && ((n.full_label || n.label || "").toLowerCase().includes(q) || (n.aliases || []).some((a) => a.toLowerCase().includes(q))));
    hits.forEach((n) => this.reveal(n.id));
    this.highlight = new Set(hits.map((n) => n.id));
    this.render(false);
    if (hits[0]) this.focus(hits[0].id, 1.1);
    return hits.length;
  }
}

function countDesc(map, id) { let c = 0; const st = [...(map.children.get(id) || [])]; while (st.length) { const x = st.pop(); c++; st.push(...(map.children.get(x) || [])); } return c; }
