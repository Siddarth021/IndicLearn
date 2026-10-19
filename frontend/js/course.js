// OS course: topic list with progress, and one lesson per topic (English / Telugu) with a live solved example,
// a quick check, listen-aloud, and a jump to a practice paper on the unit.
import { api } from "./api.js";
import { getLang } from "./i18n.js";
import { h, md, toast, mathify } from "./util.js";
import { presetUnits } from "./papers.js";
import { emit } from "./state.js";

const L = {
  en: {
    title: "Operating", em: "Systems", sub: "Learn each topic, see a solved example, check yourself, then make an exam paper on it.",
    learned: "learned", of: "of", topics: "topics", back: "← All topics", listen: "🔊 Listen", stop: "■ Stop", summary: "In short",
    terms: "Key terms", example: "Worked example", newex: "New example", show_sol: "Show solution", hide_sol: "Hide solution",
    draw: "Be able to draw", tips: "How to score full marks", mistakes: "Common mistakes", check: "Quick check",
    mark: "✓ Mark as learned", marked: "✓ Learned", paper: "Practice paper on this unit", prev: "← Previous", next: "Next →",
    right: "Correct!", wrong: "Not quite.", soon: "Lesson coming soon", te_missing: "Telugu isn't available right now (the AI engines are busy) — showing English. Try తెలుగు again later.", translating: "Preparing this lesson in Telugu — first time only, about half a minute…",
    continue: "Continue", start: "Start",
  },
  te: {
    title: "Operating", em: "Systems", sub: "ప్రతి topic నేర్చుకోండి, solved example చూడండి, మిమ్మల్ని మీరు check చేసుకోండి, తర్వాత దానిపై exam paper చేయండి.",
    learned: "నేర్చుకున్నవి", of: "/", topics: "topics", back: "← అన్ని topics", listen: "🔊 వినండి", stop: "■ ఆపు", summary: "క్లుప్తంగా",
    terms: "ముఖ్య పదాలు", example: "Solved example", newex: "కొత్త example", show_sol: "Solution చూపించు", hide_sol: "Solution దాచు",
    draw: "గీయగలగాల్సిన diagram", tips: "పూర్తి మార్కులు ఎలా", mistakes: "సాధారణ తప్పులు", check: "Quick check",
    mark: "✓ నేర్చుకున్నాను", marked: "✓ నేర్చుకున్నారు", paper: "ఈ unit పై practice paper", prev: "← ముందుది", next: "తర్వాతది →",
    right: "సరైనది!", wrong: "సరైనది కాదు.", soon: "Lesson త్వరలో", te_missing: "ఇప్పుడు తెలుగు అందుబాటులో లేదు (AI engines బిజీ) — English చూపిస్తున్నాం. కాసేపటి తర్వాత మళ్ళీ ప్రయత్నించండి.", translating: "ఈ lesson ని తెలుగులో తయారుచేస్తున్నాం — మొదటిసారి మాత్రమే, సుమారు అర నిమిషం…",
    continue: "కొనసాగించు", start: "మొదలుపెట్టు",
  },
};
const T = (k) => (L[getLang()] || L.en)[k] ?? L.en[k];
let lessonLang = null; // the language the student picked for lessons (defaults to the app language)
let speaking = null;

export async function renderCourse(v, topic = null) {
  stopSpeaking();
  v.innerHTML = "";
  if (topic) return renderLesson(v, topic);
  let idx;
  try { idx = await api.lessons(); } catch (e) { v.append(h("div", { class: "empty" }, e.message)); return; }
  const pct = Math.round((100 * idx.done) / Math.max(1, idx.total));
  const firstOpen = idx.units.flatMap((u) => u.topics).find((t) => t.status !== "learned" && t.has_lesson);
  v.append(h("div", { class: "pagehead" }, h("div", {}, h("span", { class: "small-caps" }, "Course"), h("h1", {}, T("title") + " ", h("em", {}, T("em"))), h("p", {}, T("sub"))),
    h("div", { class: "cprog" }, h("b", {}, `${idx.done} ${T("of")} ${idx.total}`), h("span", {}, T("topics") + " " + T("learned")),
      h("div", { class: "bar" }, h("i", { style: { width: pct + "%", background: "var(--good)" } })),
      firstOpen ? h("button", { class: "btn sm", style: { marginTop: "10px" }, onclick: () => renderCourse(v, firstOpen.id) }, (idx.done ? T("continue") : T("start")) + ": " + firstOpen.label.split("(")[0]) : null)));
  for (const [i, u] of idx.units.entries()) {
    const done = u.topics.filter((t) => t.status === "learned").length;
    v.append(h("section", { class: "cunit card" }, h("header", {}, h("h3", {}, `${i + 1}. ${u.title}`), h("span", { class: "small-caps" }, `${done}/${u.topics.length}`)),
      h("div", { class: "ctopics" }, ...u.topics.map((t) => h("button", { class: "ctopic " + t.status, disabled: !t.has_lesson, onclick: () => renderCourse(v, t.id) },
        h("span", { class: "dot", "aria-hidden": "true" }, t.status === "learned" ? "✓" : t.status === "opened" ? "•" : ""),
        h("span", { class: "lbl" }, h("b", {}, t.label), h("small", {}, t.has_lesson ? t.summary.slice(0, 120) + (t.summary.length > 120 ? "…" : "") : T("soon"))),
        t.numerical ? h("span", { class: "chip" }, "numericals") : null)))));
  }
}

async function renderLesson(v, topic, seed = null) {
  if (!lessonLang) lessonLang = getLang() === "te" ? "te" : "en";
  let d;
  const slow = lessonLang === "te" ? setTimeout(() => { v.innerHTML = ""; v.append(h("div", { class: "card empty" }, h("span", { class: "spin" }), " " + T("translating"))); }, 700) : null;
  try { d = await api.lesson(topic, lessonLang, seed); } catch (e) { clearTimeout(slow); toast(e.message, true); return renderCourse(v); }
  clearTimeout(slow);
  v.innerHTML = "";
  const les = d.lesson;
  const langSeg = h("div", { class: "pseg" }, ...[["en", "English"], ["te", "తెలుగు"]].map(([c, n]) =>
    h("button", { class: "segbtn" + (lessonLang === c ? " on" : ""), onclick: () => { lessonLang = c; renderLesson(v, topic); } }, n)));
  const fullText = [les.summary, ...les.sections.map((s) => `${s.heading}. ${s.body}`)].join("\n").replace(/\*\*/g, "");
  const listen = h("button", { class: "btn ghost sm", onclick: () => toggleSpeak(listen, fullText, d.lang) }, T("listen"));
  v.append(h("div", { class: "lnav" }, h("button", { class: "linkbtn", onclick: () => renderCourse(v) }, T("back")), langSeg),
    h("article", { class: "lesson" },
      h("span", { class: "small-caps" }, d.unit.title), h("h1", {}, les.title || d.label),
      lessonLang === "te" && d.lang !== "te" ? h("p", { class: "notice" }, T("te_missing")) : null,
      h("div", { class: "lsum" }, h("b", {}, T("summary")), h("div", { html: md(les.summary) }), listen),
      ...les.sections.map((s) => h("section", { class: s.exact ? "lrules" : "" }, h("h3", {}, s.heading), h("div", { html: md(steps(s.body)) }))),
      les.key_terms?.length ? h("section", {}, h("h3", {}, T("terms")), h("dl", { class: "terms" }, ...les.key_terms.flatMap((k) => [h("dt", {}, k.term), h("dd", {}, k.meaning)]))) : null,
      exampleBlock(d, v, topic),
      les.diagram ? h("section", { class: "ldraw" }, h("h3", {}, "✎ " + T("draw")), h("p", {}, les.diagram)) : null,
      les.exam_tips?.length ? h("section", { class: "ltips" }, h("h3", {}, T("tips")), h("ul", {}, ...les.exam_tips.map((x) => h("li", { html: md(x) })))) : null,
      les.mistakes?.length ? h("section", { class: "lmist" }, h("h3", {}, T("mistakes")), h("ul", {}, ...les.mistakes.map((x) => h("li", { html: md(x) })))) : null,
      d.check?.length ? quickCheck(d.check) : null,
      footer(d, v)));
  v.scrollTop = 0; v.closest?.("main")?.scrollTo?.(0, 0);
  mathify(v);
}

// "1. Put ... 2. Take ... 3. Repeat" written as one paragraph -> one step per line, so it renders as a list
const steps = (t) => (/(^|\s)1[.)]\s/.test(t) && /\s2[.)]\s/.test(t) ? t.replace(/\s(\d{1,2})[.)]\s(?=[A-Z\u0C00-\u0C7F])/g, "\n$1. ") : t);

function blocks(bs) {
  return bs.map((b) => b.table ? h("div", { class: "tblwrap" }, h("table", { class: "ptable" }, ...b.table.map((r, i) => h("tr", {}, ...r.map((c) => h(i ? "td" : "th", {}, c)))))) : h("p", {}, b.p));
}

function exampleBlock(d, v, topic) {
  if (d.example) { // numerical topic: a fresh, exactly solved problem
    const sol = h("div", { class: "qans", hidden: true }, ...blocks(d.example.answer));
    const btn = h("button", { class: "btn sm", onclick: () => { sol.hidden = !sol.hidden; btn.textContent = sol.hidden ? T("show_sol") : T("hide_sol"); } }, T("show_sol"));
    return h("section", { class: "lex" }, h("h3", {}, T("example")), ...blocks(d.example.blocks),
      h("div", { class: "pbar" }, btn, h("button", { class: "btn ghost sm", onclick: () => renderLesson(v, topic, Math.floor(Math.random() * 1e9)) }, "↻ " + T("newex"))), sol);
  }
  const ex = d.lesson.example;
  return ex?.body ? h("section", { class: "lex" }, h("h3", {}, T("example") + (ex.title ? ": " + ex.title : "")), h("div", { html: md(ex.body) })) : null;
}

function quickCheck(items) {
  return h("section", { class: "lcheck" }, h("h3", {}, T("check")), ...items.map((q, qi) => {
    const fb = h("div");
    let done = false;
    const opts = q.options.map((o, i) => h("button", { class: "opt", onclick: () => {
      if (done) return; done = true;
      opts[i].classList.add(i === q.correct ? "right" : "wrong"); opts[q.correct].classList.add("right");
      fb.append(h("div", { class: "explain" }, h("b", {}, (i === q.correct ? T("right") : T("wrong")) + " "), q.why));
    } }, o));
    return h("div", { class: "qcard" }, h("div", { class: "stem" }, `${qi + 1}. ${q.q}`), ...opts, fb);
  }));
}

function footer(d, v) {
  const mark = h("button", { class: "btn" + (d.status === "learned" ? " ghost" : ""), onclick: async () => {
    try { await api.lessonMark(d.topic, d.status === "learned" ? "opened" : "learned"); d.status = d.status === "learned" ? "opened" : "learned";
      mark.textContent = d.status === "learned" ? T("marked") : T("mark"); mark.classList.toggle("ghost", d.status === "learned");
      if (d.status === "learned") emit("xp", { amount: 5 }); } catch (e) { toast(e.message, true); }
  } }, d.status === "learned" ? T("marked") : T("mark"));
  return h("footer", { class: "lfoot" }, mark,
    h("button", { class: "btn ghost", onclick: () => { presetUnits([d.unit.id]); emit("nav", { view: "papers" }); } }, "⎙ " + T("paper")),
    h("div", { class: "lpn" }, d.prev ? h("button", { class: "linkbtn", onclick: () => renderLesson(v, d.prev.id) }, T("prev")) : h("span"),
      d.next ? h("button", { class: "linkbtn", onclick: () => renderLesson(v, d.next.id) }, T("next")) : h("span")));
}

// ---- listen aloud (server voice, piece by piece so it starts quickly)
function stopSpeaking() { if (speaking) { speaking.stop = true; speaking.audio?.pause(); speaking.btn.textContent = T("listen"); speaking = null; } }
async function toggleSpeak(btn, text, lang) {
  if (speaking?.btn === btn) return stopSpeaking();
  stopSpeaking();
  const me = (speaking = { btn, stop: false, audio: null });
  btn.textContent = T("stop");
  const pieces = (text.match(/[^.!?।\n]+[.!?।]*\s*/g) || [text]).reduce((acc, s) => { if (acc.length && (acc[acc.length - 1] + s).length < 350) acc[acc.length - 1] += s; else acc.push(s); return acc; }, []);
  try {
    let next = api.tts(pieces[0], lang);
    for (let i = 0; i < pieces.length && !me.stop; i++) {
      const blob = await next;
      if (me.stop) return;
      if (i + 1 < pieces.length) next = api.tts(pieces[i + 1], lang);
      const a = (me.audio = new Audio(URL.createObjectURL(blob)));
      await a.play();
      await new Promise((res) => { a.onended = res; a.onerror = res; a.onpause = res; });
    }
  } catch (e) { toast(e.message, true); }
  if (speaking === me) stopSpeaking();
}
