// Exam papers: the student sets the paper's skeleton (total marks, time, units, rows like "5 x 1 mark MCQ"),
// gets a printable question paper; answers stay hidden until asked for (per question, all, or as a PDF key).
import { api, getToken } from "./api.js";
import { getLang } from "./i18n.js";
import { h, md, toast, mathify, renderBlocks } from "./util.js";

const L = {
  en: {
    title: "Exam", em: "papers", sub: "{c} · build a practice paper the way your exam looks, print it, write it, then check the answers.", course: "Course",
    type: "Paper type", total: "Total marks", time: "Time (minutes)", diff: "Difficulty", units: "Units to cover", all: "All units",
    pattern: "Question pattern", add: "+ Add a row", qs: "questions", of: "of", marks: "marks", each: "marks each",
    sum_ok: "Pattern matches the total", sum_bad: "Pattern adds up to {got} — the paper is set to {total}",
    make: "Make my paper", making: "Writing your paper… numericals are solved exactly, theory is checked by a second AI.",
    src_notes: "Questions are drawn from the units you ticked that your uploaded notes also cover ({docs}).",
    src_fresh: "{n} written fresh from your notes.", src_syl: "Questions come from the {c} syllabus (upload your class notes to match your professor's coverage).",
    missing: "Ticked but not in your notes: {t}.",
    pdf_q: "Question paper (PDF)", pdf_a: "Answer key (PDF)", show_all: "Show all answers", hide_all: "Hide answers",
    show: "Show answer", hide: "Hide answer", answer: "Answer", scheme: "Marking scheme", diagram: "Diagram expected",
    history: "Your papers", none: "No papers yet.", open: "Open", del: "Delete", new: "New paper", no_num: "No numerical topics in the units you ticked, so numerical slots got long-answer questions.",
    short: "The question bank ran short for some slots; try more units.", match: "Match my college paper", matching: "Reading your paper…",
    match_p: "Upload a past paper (PDF, Word or a phone photo) — the marks pattern, time and units are copied for you to check.",
    matched: "Copied the pattern: {n} questions, {m} marks. Check the rows below, then make your paper.", nat_box: "Answer: ________ (type the number)",
    gate_rules: "GATE marking: MCQ wrong answer −1/3 of its marks · MSQ and NAT have no negative marks and no partial marks.",
    qt: { auto: "Any type", fill: "Fill in the blank", mcq: "Multiple choice", short: "Short answer", long: "Long answer", numerical: "Numerical", msq: "MSQ (GATE)", nat: "NAT (GATE)" },
    df: { easy: "Easy", college: "College level", tough: "Tough" },
  },
  te: {
    title: "పరీక్ష", em: "పేపర్లు", sub: "{c} · మీ పరీక్షలాగే practice paper తయారుచేసి, print తీసి రాయండి, తర్వాత జవాబులు చూడండి.", course: "కోర్సు",
    type: "పేపర్ రకం", total: "మొత్తం మార్కులు", time: "సమయం (నిమిషాలు)", diff: "కష్టం", units: "Units", all: "అన్ని units",
    pattern: "ప్రశ్నల pattern", add: "+ వరుస జోడించు", qs: "ప్రశ్నలు", of: "లో", marks: "మార్కులు", each: "మార్కులు ఒక్కొక్కటి",
    sum_ok: "Pattern మొత్తం మార్కులకు సరిపోయింది", sum_bad: "Pattern మొత్తం {got} — పేపర్ {total} మార్కులు",
    make: "నా పేపర్ తయారుచేయి", making: "మీ పేపర్ తయారవుతోంది… numericals ఖచ్చితంగా లెక్కించబడతాయి, theory రెండో AI తో check అవుతుంది.",
    src_notes: "మీరు ఎంచుకున్న units లో మీ notes ({docs}) కవర్ చేసిన topics నుండి ప్రశ్నలు.",
    src_fresh: "{n} ప్రశ్నలు మీ notes నుండి కొత్తగా రాశాము.", src_syl: "ప్రశ్నలు {c} syllabus నుండి (మీ professor కవర్ చేసినట్లు రావాలంటే మీ class notes upload చేయండి).",
    missing: "ఎంచుకున్నా మీ notes లో లేనివి: {t}.",
    pdf_q: "ప్రశ్నాపత్రం (PDF)", pdf_a: "జవాబుల key (PDF)", show_all: "అన్ని జవాబులు చూపించు", hide_all: "జవాబులు దాచు",
    show: "జవాబు చూపించు", hide: "జవాబు దాచు", answer: "జవాబు", scheme: "మార్కుల విభజన", diagram: "ఉండాల్సిన diagram",
    history: "మీ పేపర్లు", none: "ఇంకా పేపర్లు లేవు.", open: "తెరువు", del: "తొలగించు", new: "కొత్త పేపర్", no_num: "ఎంచుకున్న units లో numerical topics లేవు, అందుకే వాటి బదులు long-answer ప్రశ్నలు ఇచ్చాము.",
    short: "కొన్ని slots కి ప్రశ్నలు సరిపోలేదు; ఇంకా units ఎంచుకోండి.", match: "నా కాలేజ్ పేపర్ లాగా", matching: "మీ పేపర్ చదువుతోంది…",
    match_p: "పాత paper (PDF, Word లేదా phone photo) upload చేయండి — marks pattern, సమయం, units కాపీ అవుతాయి, మీరు check చేసుకోండి.",
    matched: "Pattern కాపీ అయింది: {n} ప్రశ్నలు, {m} మార్కులు. కింద rows check చేసి పేపర్ చేయండి.", nat_box: "జవాబు: ________ (సంఖ్య రాయండి)",
    gate_rules: "GATE marking: MCQ తప్పు జవాబుకి −1/3 మార్కులు · MSQ, NAT కి negative marks, partial marks లేవు.",
    qt: { auto: "ఏదైనా", fill: "ఖాళీలు పూరించండి", mcq: "బహుళైచ్ఛిక", short: "చిన్న జవాబు", long: "పెద్ద జవాబు", numerical: "Numerical", msq: "MSQ (GATE)", nat: "NAT (GATE)" },
    df: { easy: "సులభం", college: "కాలేజ్ స్థాయి", tough: "కష్టం" },
  },
};
const T = (k, v = {}) => { const d = L[getLang()] || L.en; let s = k.split(".").reduce((o, x) => o?.[x], d) ?? k.split(".").reduce((o, x) => o?.[x], L.en) ?? k; for (const [a, b] of Object.entries(v)) s = s.replace("{" + a + "}", b); return s; };

let courses = null;
let course = null;
let state = null;
let lastPaper = null;
let pendingUnits = null;
let pendingSubject = null;
// a lesson's "Practice paper on this unit" opens the builder for that course with just that unit ticked
export function presetUnits(ids, subject = null) {
  if (subject && course && course.id !== subject) { pendingSubject = subject; state = null; lastPaper = null; }
  else if (subject && !course) pendingSubject = subject;
  pendingUnits = ids;
  if (state && ids) state.units = [...ids];
}
function pick(subject) {
  const c = courses.find((x) => x.id === subject) || courses[0];
  if (course?.id !== c.id) { course = c; state = null; lastPaper = null; }
}

function fresh(preset) {
  return { preset: preset.id, name: preset.name, style: preset.style || "college", total: preset.total, minutes: preset.minutes, difficulty: "college",
           units: course.units.map((u) => u.id), rows: preset.rows.map((r) => ({ ...r })) };
}

export async function renderPapers(v) {
  v.innerHTML = "";
  if (!courses) {
    try { courses = await api.courses(); } catch (e) { v.append(h("div", { class: "empty" }, e.message)); return; }
  }
  let lastSubject = null; try { lastSubject = localStorage.getItem("il_course"); } catch (e) { /* ignore */ }
  pick(pendingSubject || course?.id || lastSubject || courses[0].id); pendingSubject = null;
  if (!state) state = fresh(course.presets[0]);
  if (pendingUnits) { state.units = [...pendingUnits]; pendingUnits = null; }
  v.append(h("div", { class: "pagehead" }, h("div", {}, h("span", { class: "small-caps" }, course.title), h("h1", {}, T("title") + " ", h("em", {}, T("em"))), h("p", {}, T("sub", { c: course.title })))));
  if (courses.length > 1) v.append(h("div", { class: "pcourse" }, h("span", { class: "small-caps" }, T("course")),
    h("div", { class: "pseg" }, ...courses.map((c) => h("button", { class: "segbtn" + (c.id === course.id ? " on" : ""), onclick: () => { pick(c.id); renderPapers(v); } }, c.short || c.title)))));
  const out = h("div", { class: "paperout" });
  v.append(builder(out), out, historyCard(out));
  if (lastPaper) showPaper(out, lastPaper);
}

function builder(out) {
  const card = h("div", { class: "card pbuild" });
  const draw = () => {
    card.innerHTML = "";
    // copy the pattern of a real past paper (PDF / Word / phone photo)
    const pf = h("input", { type: "file", hidden: true, accept: ".pdf,.docx,.txt,.md,.jpg,.jpeg,.png,.webp,image/*", onchange: async () => {
      const f = pf.files[0]; if (!f) return;
      mbtn.disabled = true; mbtn.textContent = T("matching");
      try {
        const r = await api.paperPattern(f);
        if (r.blueprint.subject && r.blueprint.subject !== course.id) pick(r.blueprint.subject); // a DSMA paper switches to DSMA
        state = { ...r.blueprint, preset: "matched", rows: r.blueprint.rows.map((x) => ({ ...x })) };
        toast(T("matched", { n: r.questions_found, m: r.blueprint.total }));
        lastPaper = null; out.innerHTML = ""; // the paper on screen belonged to the old pattern
        const v = card.closest(".view"); if (v) { renderPapers(v); return; }
        draw();
      } catch (e) { toast(e.message, true); mbtn.disabled = false; mbtn.textContent = "📷 " + T("match"); }
    } });
    const mbtn = h("button", { class: "btn ghost sm", onclick: () => pf.click() }, "📷 " + T("match"));
    card.append(h("div", { class: "pmatch" }, h("div", {}, h("b", {}, T("match")), h("small", {}, T("match_p"))), mbtn, pf));
    // paper type presets
    card.append(h("div", { class: "small-caps" }, T("type")),
      h("div", { class: "pseg" }, ...course.presets.map((p) => h("button", { class: "segbtn" + (state.preset === p.id ? " on" : ""), onclick: () => { state = { ...fresh(p), units: state.units }; draw(); } },
        p.name, p.id !== "custom" ? h("small", {}, ` ${p.total}m · ${p.minutes}′`) : null))));
    // totals + difficulty
    const num = (key, min, max) => h("input", { class: "field", type: "number", min, max, value: state[key], inputmode: "numeric", "aria-label": T(key === "total" ? "total" : "time"),
      oninput: (e) => { state[key] = +e.target.value || 0; state.preset = "custom"; sum(); } });
    card.append(h("div", { class: "prow" },
      h("label", {}, h("span", { class: "small-caps" }, T("total")), num("total", 1, 200)),
      h("label", {}, h("span", { class: "small-caps" }, T("time")), num("minutes", 10, 240)),
      h("div", { class: "pdiff" }, h("span", { class: "small-caps" }, T("diff")), h("div", { class: "pseg" }, ...Object.keys(course.difficulty).map((d) =>
        h("button", { class: "segbtn" + (state.difficulty === d ? " on" : ""), onclick: () => { state.difficulty = d; draw(); } }, T("df." + d)))))));
    // units
    const allOn = state.units.length === course.units.length;
    card.append(h("div", { class: "small-caps", style: { marginTop: "14px" } }, T("units")),
      h("div", { class: "units" }, h("button", { class: "chipbtn" + (allOn ? " on" : ""), onclick: () => { state.units = allOn ? [] : course.units.map((u) => u.id); draw(); } }, T("all")),
        ...course.units.map((u, i) => h("button", { class: "chipbtn" + (state.units.includes(u.id) ? " on" : ""), title: u.topics.map((t) => t.label).join(" · "),
          onclick: () => { state.units = state.units.includes(u.id) ? state.units.filter((x) => x !== u.id) : [...state.units, u.id]; draw(); } }, `${i + 1}. ${u.title}`))));
    // pattern rows
    const rowsEl = h("div", { class: "prows" });
    state.rows.forEach((r, i) => {
      const n = (key, min, max) => h("input", { class: "field n", type: "number", min, max, value: r[key], inputmode: "numeric", "aria-label": key,
        oninput: (e) => { r[key] = +e.target.value || 0; state.preset = "custom"; sum(); } });
      rowsEl.append(h("div", { class: "prow2" }, n("count", 1, 30), h("span", {}, T("qs") + " ×"), n("marks", 1, 20), h("span", {}, T("each")),
        h("select", { class: "field", "aria-label": "type", onchange: (e) => { r.qtype = e.target.value; state.preset = "custom"; } },
          ...Object.keys(course.qtypes).map((k) => h("option", { value: k, selected: r.qtype === k }, T("qt." + k)))),
        h("button", { class: "linkbtn x", "aria-label": "remove row", onclick: () => { state.rows.splice(i, 1); state.preset = "custom"; draw(); } }, "✕")));
    });
    const status = h("div", { class: "psum", role: "status" });
    const go = h("button", { class: "btn", onclick: () => make(out, go) }, T("make"));
    const sum = () => {
      const got = state.rows.reduce((a, r) => a + (r.count || 0) * (r.marks || 0), 0);
      const ok = got === state.total && got > 0;
      status.className = "psum " + (ok ? "ok" : "bad");
      status.textContent = ok ? `✓ ${T("sum_ok")} · ${got} ${T("marks")}` : `${T("sum_bad", { got, total: state.total })}`;
      go.disabled = !ok || !state.units.length;
    };
    card.append(h("div", { class: "small-caps", style: { marginTop: "14px" } }, T("pattern")), rowsEl,
      h("button", { class: "linkbtn", onclick: () => { state.rows.push({ count: 1, marks: 2, qtype: "auto" }); state.preset = "custom"; draw(); } }, T("add")),
      h("div", { class: "pgo" }, status, go));
    sum();
  };
  draw();
  return card;
}

async function make(out, btn) {
  btn.disabled = true;
  out.innerHTML = ""; out.append(h("div", { class: "card empty" }, h("span", { class: "spin" }), " " + T("making")));
  out.scrollIntoView({ behavior: "smooth", block: "start" });
  try {
    const bp = { subject: course.id, name: state.name || "Practice paper", style: state.style, total: state.total, minutes: state.minutes, difficulty: state.difficulty, units: state.units, rows: state.rows };
    lastPaper = await api.paperNew(bp);
    showPaper(out, lastPaper);
    refreshHistory();
  } catch (e) { out.innerHTML = ""; toast(e.message, true); }
  btn.disabled = false;
}

const pdfUrl = (id, answers) => `/api/papers/${id}/pdf?answers=${answers}&token=${encodeURIComponent(getToken() || "")}`;

function showPaper(out, p) {
  out.innerHTML = "";
  let full = null;
  const loadAnswers = async () => { if (!full) full = await api.paper(p.id, true); return full; };
  const bp = p.blueprint;
  const src = p.sources || {};
  const info = [];
  if (src.from_notes) info.push(T("src_notes", { docs: src.docs.map((d) => d.title).join(", ") }) + (src.fresh ? " " + T("src_fresh", { n: src.fresh }) : ""));
  else info.push(T("src_syl", { c: p.course_title }));
  if (src.missing_in_notes?.length) {
    // name whole units the notes never touch (easier to act on than 30 topic names)
    const miss = new Set(src.missing_in_notes);
    const units = course.units.filter((u) => bp.units.includes(u.id) && u.topics.every((t) => miss.has(t.id))).map((u) => u.title);
    if (units.length) info.push(T("missing", { t: units.join(", ") }));
  }
  if (p.notes?.includes("no_numerical")) info.push(T("no_num"));
  if (bp.style === "gate") info.push(T("gate_rules"));
  if (p.notes?.includes("short_of_questions")) info.push(T("short"));
  const sheet = h("article", { class: "card sheet" },
    h("header", {}, h("h2", {}, `${p.course_title} — ${bp.name}`), h("div", { class: "meta" }, h("span", {}, `${T("total")}: ${p.total}`), h("span", {}, `${T("time")}: ${bp.minutes}`))));
  const qEls = [];
  for (const q of p.questions) {
    const ans = h("div", { class: "qans", hidden: true });
    const toggle = h("button", { class: "linkbtn", onclick: async () => {
      if (ans.hidden && !ans.childElementCount) { try { const f = await loadAnswers(); fillAnswer(ans, f.questions.find((x) => x.id === q.id)); } catch (e) { toast(e.message, true); return; } }
      ans.hidden = !ans.hidden; toggle.textContent = ans.hidden ? T("show") : T("hide");
    } }, T("show"));
    const el = h("section", { class: "pq" }, h("div", { class: "pqn" }, q.n + "."), h("div", { class: "pqb" }, ...blocks(q.blocks),
      q.options ? h("ol", { class: "opts", type: "a" }, ...q.options.map((o) => h("li", { html: md(o).replace(/^<p>|<\/p>$/g, "") }))) : null,
      q.kind === "nat" ? h("div", { class: "natbox" }, T("nat_box")) : null,
      h("div", { class: "pqfoot" }, bp.style === "gate" ? h("span", { class: "chip gold" }, KIND[q.kind] + (q.negative ? ` · −${q.negative}` : "")) : null, h("span", { class: "chip" }, q.topic_label), q.source === "notes" ? h("span", { class: "chip gold" }, "from your notes") : null, q.source === "pyq" ? h("span", { class: "chip gold", title: "Asked in a previous IIIT Sri City exam" }, "★ " + q.paper_title) : null, toggle), ans),
      h("div", { class: "pqm" }, `[${q.marks}]`));
    qEls.push({ ans, toggle, q });
    sheet.append(el);
  }
  let shown = false;
  const allBtn = h("button", { class: "btn ghost sm", onclick: async () => {
    try { const f = await loadAnswers(); for (const x of qEls) { if (!x.ans.childElementCount) fillAnswer(x.ans, f.questions.find((y) => y.id === x.q.id)); x.ans.hidden = shown; x.toggle.textContent = shown ? T("show") : T("hide"); } }
    catch (e) { toast(e.message, true); return; }
    shown = !shown; allBtn.textContent = shown ? T("hide_all") : T("show_all");
  } }, T("show_all"));
  out.append(h("div", { class: "pbar" },
    h("a", { class: "btn", href: pdfUrl(p.id, false), target: "_blank", rel: "noopener" }, "⎙ " + T("pdf_q")),
    h("a", { class: "btn ghost", href: pdfUrl(p.id, true), target: "_blank", rel: "noopener" }, T("pdf_a")), allBtn),
    h("div", { class: "pinfo" }, ...info.map((s) => h("p", {}, s))), sheet);
  mathify(sheet);
}

const blocks = renderBlocks;

const KIND = { mcq: "MCQ", msq: "MSQ", nat: "NAT", fill: "Fill", short: "Short", medium: "Answer", long: "Long", numerical: "Numerical" };

function fillAnswer(el, q) {
  if (!q) return;
  el.innerHTML = "";
  el.append(h("b", {}, T("answer")));
  if (q.kind === "msq" && Array.isArray(q.correct)) el.append(h("p", {}, h("b", {}, q.correct.map((i) => `(${"abcde"[i]}) ${q.options[i]}`).join("   ·   "))));
  else if (q.answer_blocks && q.options && q.correct !== null && q.correct !== undefined) el.append(h("p", {}, h("b", {}, `(${"abcde"[q.correct]}) ${q.options[q.correct]}`)));
  if (q.answer_blocks) el.append(...blocks(q.answer_blocks));
  else {
    if (q.options && q.correct !== null && q.correct !== undefined) el.append(h("p", {}, h("b", {}, `(${"abcde"[q.correct]}) ${q.options[q.correct]}`)));
    el.append(h("div", { html: md(q.answer_text || "") }));
    if (q.diagram) el.append(h("p", { class: "small" }, `${T("diagram")}: ${q.diagram}`));
    if (q.scheme?.length) el.append(h("div", { class: "small-caps" }, T("scheme")), h("ul", {}, ...q.scheme.map((s) => h("li", {}, `${s.point} — ${s.marks}`))));
  }
  mathify(el);
}

let historyEl = null;
function historyCard(out) {
  historyEl = h("div", { class: "phist" });
  historyEl._out = out;
  refreshHistory();
  return historyEl;
}
async function refreshHistory() {
  if (!historyEl) return;
  let list = [];
  try { list = await api.papers(); } catch (e) { return; }
  historyEl.innerHTML = "";
  historyEl.append(h("div", { class: "small-caps", style: { margin: "26px 0 10px" } }, T("history")));
  if (!list.length) { historyEl.append(h("div", { class: "empty" }, T("none"))); return; }
  for (const p of list) {
    const cshort = (courses.find((c) => c.id === p.subject) || {}).short || (p.subject || "os").toUpperCase();
    historyEl.append(h("div", { class: "rec" }, h("div", {}, h("h4", {}, p.name), h("small", {}, `${cshort} · ${new Date(p.created_at * 1000).toLocaleString()} · ${p.total} ${T("marks")} · ${p.questions} Q`)),
      h("div", { style: { display: "flex", gap: "8px" } },
        h("button", { class: "btn sm ghost", onclick: async () => { try { const got = await api.paper(p.id, false);
          if (got.course && got.course !== course.id) { pick(got.course); lastPaper = got; const v = historyEl.closest(".view"); if (v) { renderPapers(v); return; } }
          lastPaper = got; showPaper(historyEl._out, lastPaper); historyEl._out.scrollIntoView({ behavior: "smooth" }); } catch (e) { toast(e.message, true); } } }, T("open")),
        h("button", { class: "linkbtn", onclick: async () => { await api.paperDel(p.id).catch(() => {}); if (lastPaper?.id === p.id) { lastPaper = null; historyEl._out.innerHTML = ""; } refreshHistory(); } }, T("del")))));
  }
}
