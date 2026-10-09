import { api, setToken, getToken, ApiError } from "./api.js";
import { t, getLang, setLang, LANGS, unitName } from "./i18n.js";
import { h, toast, STATUS_COLORS } from "./util.js";
import { S, on, emit } from "./state.js";
import { createTutor } from "./chat.js";
import { createReader } from "./reader.js";
import { createMapView } from "./mapview.js";
import { renderQuiz } from "./quizcard.js";

const $app = document.getElementById("app-root");
let views = {}, tutor = null, reader = null, mapView = null, pollTimer = null;
const ICON = {
  library: '<path d="M4 5h4v14H4zM10 5h4v14h-4zM16.5 6.5l3.2-.8 3 11.7-3.2.8z"/>', study: '<path d="M4 5h7a3 3 0 0 1 3 3v11a3 3 0 0 0-3-3H4zM20 5h-7a3 3 0 0 0-3 3v11a3 3 0 0 1 3-3h7z"/>',
  map: '<circle cx="12" cy="12" r="2.5"/><circle cx="5" cy="6" r="2"/><circle cx="19" cy="6" r="2"/><circle cx="5" cy="18" r="2"/><circle cx="19" cy="18" r="2"/><path d="M10 11 6.5 7.5M14 11l3.5-3.5M10 13l-3.5 3.5M14 13l3.5 3.5"/>',
  quiz: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.7.4-1 .9-1 1.7M12 17v.01"/>', progress: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
};
const svgIcon = (k) => { const s = document.createElementNS("http://www.w3.org/2000/svg", "svg"); s.setAttribute("viewBox", "0 0 24 24"); s.innerHTML = ICON[k]; return s; };

// ------------------------------------------------------------------ boot
async function boot() {
  try { S.info = await api.info(); } catch (e) { S.info = { samples: [], max_upload_mb: 40, allow_guest: true, allow_registration: true }; }
  if (getToken()) { try { S.user = await api.me(); } catch (e) { setToken(null); } }
  S.user ? shell() : authScreen();
}

function langPicker() {
  const sel = h("select", { class: "field", "aria-label": "Language", onchange: (e) => { setLang(e.target.value); emit("lang"); S.user ? shell(true) : authScreen(); } },
    ...LANGS.map((l) => h("option", { value: l.code, selected: l.code === getLang() }, l.name)));
  return sel;
}

function authScreen() {
  $app.innerHTML = "";
  let registering = false;
  const card = h("div", { class: "card auth" });
  const draw = () => {
    card.innerHTML = "";
    const u = h("input", { class: "field", placeholder: t("username"), autocomplete: "username", "aria-label": t("username") });
    const p = h("input", { class: "field", type: "password", placeholder: t("password"), autocomplete: registering ? "new-password" : "current-password", "aria-label": t("password") });
    const d = h("input", { class: "field", placeholder: t("display"), hidden: !registering, "aria-label": t("display") });
    const go = async (e) => {
      e?.preventDefault();
      try { const r = registering ? await api.register(u.value, p.value, d.value || u.value, getLang()) : await api.login(u.value, p.value); setToken(r.token); S.user = r.user; shell(); }
      catch (err) { toast(err.message, true); }
    };
    card.append(h("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center" } }, h("div", { class: "brand", style: { padding: 0 } }, h("b", {}, "Indic", h("i", {}, "Learn"))), langPicker()),
      h("h1", {}, t("welcome_t") + " ", h("em", {}, t("welcome_b"))), h("p", { style: { color: "var(--ink-2)" } }, t("welcome_p")),
      h("form", { class: "row", onsubmit: go }, u, d, p, h("button", { class: "btn", type: "submit", style: { justifyContent: "center" } }, registering ? t("register") : t("sign_in"))),
      h("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center" } },
        S.info.allow_registration ? h("button", { class: "linkbtn", onclick: () => { registering = !registering; draw(); } }, registering ? t("sign_in") : t("register")) : h("span"),
        S.info.allow_guest ? h("button", { class: "btn ghost sm", onclick: async () => { try { const r = await api.guest(getLang()); setToken(r.token); S.user = r.user; shell(); } catch (err) { toast(err.message, true); } } }, t("guest")) : null),
      S.info.allow_guest ? h("p", { class: "small-caps", style: { marginTop: "16px", textTransform: "none", letterSpacing: 0, fontWeight: 400 } }, t("guest_note")) : null);
  };
  draw(); $app.append(h("div", { class: "authwrap" }, card));
}

// ------------------------------------------------------------------ shell
function shell(keepView = false) {
  $app.innerHTML = "";
  const app = h("div", { id: "app" });
  const side = h("aside", { class: "side" });
  side.append(h("div", { class: "brand" }, h("b", {}, "Indic", h("i", {}, "Learn")), h("span", { class: "small-caps" }, t("tagline"))));
  const nav = h("nav", { class: "nav", "aria-label": "Main" });
  const items = [["library", "nav_library"], ["study", "nav_study"], ["map", "nav_map"], ["quiz", "nav_quiz"], ["progress", "nav_progress"]];
  for (const [k, label] of items) {
    const b = h("button", { "data-v": k, class: S.view === k ? "on" : "", onclick: () => go(k), disabled: k !== "library" && !S.doc }, svgIcon(k), h("span", {}, t(label)));
    nav.append(b);
  }
  side.append(nav);
  if (S.doc) side.append(h("div", { class: "docpill" }, h("span", { class: "small-caps" }, S.doc.lang?.toUpperCase() + " · " + (S.doc.stats?.concepts ?? "") + " " + t("concepts")), h("span", { class: "t" }, S.doc.title)));
  side.append(h("div", { class: "spacer" }), langPicker(),
    h("div", { class: "userbox" }, h("span", {}, S.user.display_name), h("button", { class: "linkbtn", onclick: async () => { try { await api.logout(); } catch (e) { /* ignore */ } setToken(null); S.user = null; S.doc = null; authScreen(); } }, t("sign_out"))));
  const main = h("main", {});
  const tutorEl = h("aside", { class: "tutor", id: "tutor" });
  views = {};
  for (const k of ["library", "study", "map", "quiz", "progress"]) { views[k] = h("div", { class: "view" + (k === "study" || k === "map" ? " flush" : ""), hidden: true }); main.append(views[k]); }
  app.append(side, main, tutorEl); $app.append(app);
  tutor = createTutor(tutorEl); reader = createReader(views.study); mapView = createMapView(views.map);
  if (S.doc) { reader.load(); }
  if (!keepView) S.view = "library";
  go(S.view, { keep: true });
}

function go(view, opts = {}) {
  if (view !== "library" && !S.doc) view = "library";
  S.view = view;
  for (const k in views) views[k].hidden = k !== view;
  document.querySelectorAll(".nav button").forEach((b) => b.classList.toggle("on", b.dataset.v === view));
  const app = document.getElementById("app");
  const showTutor = (view === "study" || view === "map") && S.doc && S.tutorOpen;
  app.classList.toggle("with-tutor", !!showTutor);
  document.getElementById("tutor").style.display = showTutor ? "" : "none";
  if (view === "library") renderLibrary();
  else if (view === "study") { if (!opts.keep) reader.load?.call && null; }
  else if (view === "map") mapView.show(opts.select);
  else if (view === "quiz") renderQuizPage();
  else if (view === "progress") renderProgress();
  opts.then?.();
}
on("nav", ({ view, select, then }) => { go(view, { select }); if (then) setTimeout(then, 350); });
on("layout", () => go(S.view, { keep: true }));
on("mark", ({ id, kind }) => { if (S.doc) api.mark(S.doc.id, id, kind).catch(() => {}); });

async function openDoc(id, view = "study") {
  try {
    S.doc = await api.doc(id); S.focus = null;
    try { localStorage.setItem("il_doc", id); } catch (e) { /* ignore */ }
    const g = await api.graph(id, "default"); S.graph = g; S.nodes = new Map(g.nodes.map((n) => [n.id, n]));
    shellKeepNav(); emit("doc"); go(view);
  } catch (e) { toast(e.message, true); }
}
function shellKeepNav() {
  document.querySelectorAll(".nav button").forEach((b) => (b.disabled = b.dataset.v !== "library" && !S.doc));
  const pill = document.querySelector(".docpill"); pill?.remove();
  if (S.doc) document.querySelector(".nav").after(h("div", { class: "docpill" }, h("span", { class: "small-caps" }, S.doc.lang?.toUpperCase() + " · " + (S.doc.stats?.concepts ?? "") + " " + t("concepts")), h("span", { class: "t" }, S.doc.title)));
}

// ----------------------------------------------------------------- library
async function renderLibrary() {
  const v = views.library; v.innerHTML = "";
  const file = h("input", { type: "file", hidden: true, accept: S.info.extensions?.join(",") || ".pdf,.pptx,.docx,.md,.txt", onchange: () => file.files[0] && upload(file.files[0]) });
  const drop = h("div", { class: "drop" }, h("h3", {}, t("drop_t")), h("p", {}, t("drop_p", { mb: S.info.max_upload_mb })), h("button", { class: "btn", onclick: () => file.click() }, t("choose")), file);
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) upload(f); });
  if (S.info.samples?.length) {
    drop.append(h("div", { class: "small-caps", style: { marginTop: "18px" } }, t("or_sample")));
    drop.append(h("div", { class: "samples" }, ...S.info.samples.map((s) => h("button", { class: "btn sm ghost", onclick: async () => { try { await api.sample(s.key); renderLibrary(); } catch (e) { toast(e.message, true); } } }, s.title))));
  }
  v.append(h("div", { class: "hero" }, h("div", {}, h("span", { class: "small-caps" }, "IndicLearn · " + t("tagline")), h("h1", {}, t("hero_a") + " ", h("em", {}, t("hero_b"))), h("p", {}, t("hero_p"))), drop));
  v.append(h("div", { class: "small-caps", style: { marginBottom: "12px" } }, t("your_library")));
  const grid = h("div", { class: "docs" }); v.append(grid);
  const draw = async () => {
    let docs; try { docs = await api.docs(); } catch (e) { return; }
    S.docs = docs; grid.innerHTML = "";
    if (!docs.length) grid.append(h("div", { class: "empty", style: { gridColumn: "1/-1" } }, t("no_docs")));
    for (const d of docs) grid.append(docCard(d));
    clearTimeout(pollTimer);
    if (docs.some((d) => d.status === "processing") && S.view === "library") pollTimer = setTimeout(draw, 1400);
  };
  draw();
}
function docCard(d) {
  const ready = d.status === "ready";
  const card = h("article", { class: "card doc", onclick: () => ready && openDoc(d.id) });
  card.append(h("div", { class: "meta" }, h("span", { class: "chip gold" }, (d.lang || "…").toUpperCase()), h("span", { class: "chip" }, d.ext.replace(".", "").toUpperCase()),
    ready ? h("span", { class: "chip" }, `${d.stats.concepts ?? 0} ${t("concepts")}`) : null, ready ? h("span", { class: "chip" }, t(d.unit_label === "Slide" ? "slides_n" : d.unit_label === "Part" ? "parts_n" : "pages_n", { n: d.n_pages })) : null));
  card.append(h("h3", {}, d.title));
  if (d.status === "processing") card.append(h("div", { class: "sum" }, h("span", { class: "spin" }), " " + t("processing")));
  else if (d.status === "error") card.append(h("div", { class: "sum", style: { color: "var(--bad)" } }, d.error || t("failed")));
  else card.append(h("div", { class: "sum" }, d.summary || ""));
  card.append(h("div", { class: "foot" }, h("button", { class: "btn sm", disabled: !ready, onclick: (e) => { e.stopPropagation(); openDoc(d.id); } }, t("open")),
    h("button", { class: "linkbtn", onclick: async (e) => { e.stopPropagation(); if (confirm(t("confirm_delete"))) { await api.del(d.id).catch((x) => toast(x.message, true)); if (S.doc?.id === d.id) { S.doc = null; shellKeepNav(); } renderLibrary(); } } }, t("delete"))));
  return card;
}
async function upload(f) {
  try { toast(t("processing")); await api.upload(f); renderLibrary(); } catch (e) { toast(e instanceof ApiError ? e.message : t("error"), true); }
}

// -------------------------------------------------------------------- quiz
function renderQuizPage() {
  const v = views.quiz; v.innerHTML = "";
  const chapters = [...S.nodes.values()].filter((n) => (n.type === "chapter" || n.type === "section" || n.type === "topic") && (n.n_children > 0 || n.definition));
  const scope = h("select", { class: "field" }, h("option", { value: "" }, t("whole_doc")), ...chapters.map((n) => h("option", { value: n.id, selected: n.id === S.focus }, (n.type === "chapter" ? "" : "— ") + n.label)));
  const nSel = h("select", { class: "field", style: { width: "90px" } }, ...[3, 5, 8, 10].map((n) => h("option", { value: n, selected: n === 5 }, n)));
  const out = h("div", { class: "quizpage" });
  const start = async () => {
    out.innerHTML = ""; out.append(h("div", { class: "empty" }, h("span", { class: "spin" })));
    try {
      const q = await api.quiz(S.doc.id, scope.value || null, +nSel.value, getLang()); out.innerHTML = "";
      if (!q.questions.length) { out.append(h("div", { class: "empty" }, t("quiz_none"))); return; }
      out.append(renderQuiz(q, { onDone: () => out.append(h("button", { class: "btn", style: { marginTop: "14px" }, onclick: start }, t("try_again"))) }));
    } catch (e) { out.innerHTML = ""; toast(e.message, true); }
  };
  v.append(h("div", { class: "pagehead" }, h("div", {}, h("span", { class: "small-caps" }, S.doc.title), h("h1", {}, t("quiz_title") + " ", h("em", {}, t("quiz_em"))), h("p", {}, t("quiz_p")))),
    h("div", { class: "card quizpage", style: { marginBottom: "18px", display: "flex", gap: "12px", alignItems: "end", flexWrap: "wrap" } },
      h("div", { style: { flex: 1, minWidth: "220px" } }, h("div", { class: "small-caps" }, t("nav_map")), scope), h("div", {}, h("div", { class: "small-caps" }, t("questions")), nSel), h("button", { class: "btn", onclick: start }, t("start_quiz"))), out);
}

// ---------------------------------------------------------------- progress
async function renderProgress() {
  const v = views.progress; v.innerHTML = "";
  let p; try { p = await api.progress(S.doc.id); } catch (e) { toast(e.message, true); return; }
  const stat = (n, label) => h("div", { class: "card stat" }, h("b", {}, n), h("span", {}, label));
  const total = Math.max(p.total_concepts, 1);
  const bar = h("div", { class: "bar" }, ...["mastered", "learning", "revise", "doubt", "new"].map((k) => h("i", { title: t("st_" + k) + ": " + p.counts[k], style: { width: (100 * p.counts[k] / total) + "%", background: STATUS_COLORS[k] } })));
  v.append(h("div", { class: "pagehead" }, h("div", {}, h("span", { class: "small-caps" }, S.doc.title), h("h1", {}, t("prog_title") + " ", h("em", {}, t("prog_em"))), h("p", {}, t("prog_p")))),
    h("div", { class: "stats" }, stat(p.mastered_pct + "%", t("mastered")), stat(p.touched_pct + "%", t("touched")), stat(`${p.quiz_correct}/${p.quiz_answered}`, t("quizzes")), stat(p.doubts, t("doubts"))),
    h("div", { class: "card", style: { marginBottom: "24px" } }, bar, h("div", { style: { display: "flex", gap: "14px", flexWrap: "wrap", marginTop: "12px", fontSize: "12.5px" } }, ...["mastered", "learning", "revise", "doubt", "new"].map((k) => h("span", {}, h("i", { style: { display: "inline-block", width: "10px", height: "10px", borderRadius: "50%", background: STATUS_COLORS[k], marginRight: "6px" } }), `${t("st_" + k)} · ${p.counts[k]}`)))));
  v.append(h("div", { class: "small-caps", style: { marginBottom: "10px" } }, t("study_next")));
  const recs = h("div", { class: "recs" });
  for (const r of p.recommendations) {
    const why = r.reason === "prereq_for" ? t("reason_prereq_for", { x: r.for_label }) : t({ weak_quiz: "reason_weak_quiz", asked_doubt: "reason_asked_doubt", not_started: "reason_not_started", keep_practising: "reason_keep_practising" }[r.reason] || "reason_not_started");
    recs.append(h("div", { class: "rec" }, h("div", {}, h("h4", {}, r.label), h("small", {}, why)), h("div", { style: { display: "flex", gap: "8px" } },
      h("button", { class: "btn sm ghost", onclick: () => go("map", { select: r.node_id }) }, t("nav_map")), h("button", { class: "btn sm", onclick: () => { S.focus = r.node_id; go("quiz"); } }, t("quiz_this")))));
  }
  v.append(recs, h("button", { class: "linkbtn", style: { marginTop: "26px" }, onclick: async () => { if (confirm(t("confirm_reset"))) { await api.reset(S.doc.id); emit("progress", {}); renderProgress(); } } }, t("reset")));
}
on("progress", () => { if (S.view === "progress") renderProgress(); });
on("lang", () => {});

boot();
