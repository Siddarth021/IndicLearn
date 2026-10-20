// Courses: the list of subject courses (Operating Systems, DSMA, ...), each course's home (lessons by unit, progress,
// knowledge map, quiz, exam papers, doubt chat), and one lesson per topic (English / Telugu) with a live solved example,
// a quick check, listen-aloud, the doubt chat beside it, and a jump to a practice paper on the unit.
//
// A course is also a study document on the server (the lessons are the student's "notes"), so the Mind map, Quiz,
// Progress and Study tabs and the tutor chat all work on the course exactly as they do on uploaded notes.
import { api } from "./api.js";
import { getLang } from "./i18n.js";
import { h, md, toast, mathify, renderBlocks } from "./util.js";
import { presetUnits } from "./papers.js";
import { S, emit } from "./state.js";

const L = {
  en: {
    courses: "Courses", courses_h: "Learn a subject,", courses_em: "end to end", courses_p: "Lessons in English and Telugu, solved examples you can regenerate, quick checks, a doubt chat that knows the lessons, a knowledge map that follows your progress, and exam papers.",
    lessons_n: "{n} lessons", questions_n: "{n} checked questions", open_course: "Open course", all_courses: "← All courses",
    sub: "Learn each topic, see a solved example, check yourself, ask your doubts, then make an exam paper on it.",
    learned: "learned", of: "of", topics: "topics", back: "← All topics", listen: "🔊 Listen", stop: "■ Stop", summary: "In short",
    terms: "Key terms", example: "Worked example", newex: "New example", show_sol: "Show solution", hide_sol: "Hide solution",
    draw: "Be able to draw", tips: "How to score full marks", mistakes: "Common mistakes", check: "Quick check",
    mark: "✓ Mark as learned", marked: "✓ Learned", paper: "Practice paper on this unit", prev: "← Previous", next: "Next →",
    right: "Correct!", wrong: "Not quite.", soon: "Lesson coming soon", te_missing: "Telugu isn't available right now (the AI engines are busy) — showing English. Try తెలుగు again later.", translating: "Preparing this lesson in Telugu — first time only, about half a minute…",
    continue: "Continue", start: "Start", map: "Knowledge map", quiz: "Quiz me", progress: "My progress", papers: "Exam paper", ask: "Ask a doubt",
    ask_lesson: "✦ Ask a doubt about this lesson", read: "Read", before: "Learn first", books: "Text books and references", text: "Text book", ref: "Reference",
    links: "Free lectures and courses", numericals: "numericals", opening: "Opening the course…",
  },
  te: {
    courses: "కోర్సులు", courses_h: "ఒక subject ని", courses_em: "మొదటి నుండి చివరి వరకు", courses_p: "English, తెలుగు lessons, మళ్ళీ మళ్ళీ కొత్తగా వచ్చే solved examples, quick checks, lessons తెలిసిన doubt chat, మీ progress తో మారే knowledge map, exam papers.",
    lessons_n: "{n} lessons", questions_n: "{n} check చేసిన ప్రశ్నలు", open_course: "కోర్సు తెరువు", all_courses: "← అన్ని కోర్సులు",
    sub: "ప్రతి topic నేర్చుకోండి, solved example చూడండి, మిమ్మల్ని మీరు check చేసుకోండి, doubts అడగండి, తర్వాత దానిపై exam paper చేయండి.",
    learned: "నేర్చుకున్నవి", of: "/", topics: "topics", back: "← అన్ని topics", listen: "🔊 వినండి", stop: "■ ఆపు", summary: "క్లుప్తంగా",
    terms: "ముఖ్య పదాలు", example: "Solved example", newex: "కొత్త example", show_sol: "Solution చూపించు", hide_sol: "Solution దాచు",
    draw: "గీయగలగాల్సిన diagram", tips: "పూర్తి మార్కులు ఎలా", mistakes: "సాధారణ తప్పులు", check: "Quick check",
    mark: "✓ నేర్చుకున్నాను", marked: "✓ నేర్చుకున్నారు", paper: "ఈ unit పై practice paper", prev: "← ముందుది", next: "తర్వాతది →",
    right: "సరైనది!", wrong: "సరైనది కాదు.", soon: "Lesson త్వరలో", te_missing: "ఇప్పుడు తెలుగు అందుబాటులో లేదు (AI engines బిజీ) — English చూపిస్తున్నాం. కాసేపటి తర్వాత మళ్ళీ ప్రయత్నించండి.", translating: "ఈ lesson ని తెలుగులో తయారుచేస్తున్నాం — మొదటిసారి మాత్రమే, సుమారు అర నిమిషం…",
    continue: "కొనసాగించు", start: "మొదలుపెట్టు", map: "Knowledge map", quiz: "Quiz", progress: "నా progress", papers: "Exam paper", ask: "Doubt అడగండి",
    ask_lesson: "✦ ఈ lesson పై doubt అడగండి", read: "చదవండి", before: "ముందు నేర్చుకోండి", books: "Text books, references", text: "Text book", ref: "Reference",
    links: "ఉచిత lectures, courses", numericals: "numericals", opening: "కోర్సు తెరుస్తోంది…",
  },
};
const T = (k, v = {}) => { let s = (L[getLang()] || L.en)[k] ?? L.en[k] ?? k; for (const [a, b] of Object.entries(v)) s = s.replace("{" + a + "}", b); return s; };
let lessonLang = null; // the language the student picked for lessons (defaults to the app language)
let speaking = null;
let current = { subject: null, topic: null }; // what the Courses view shows
try { current.subject = localStorage.getItem("il_course") || null; } catch (e) { /* ignore */ }

export const courseLessonOpen = () => !!(current.subject && current.topic);
export const courseOpen = () => current.subject;

// ---- make the course the open study document (chat, map, quiz, progress follow it)
async function enter(subject) {
  const doc = await api.courseOpen(subject);
  if (S.doc?.id !== doc.id) {
    S.doc = doc; S.focus = null;
    const g = await api.graph(doc.id, "default"); S.graph = g; S.nodes = new Map(g.nodes.map((n) => [n.id, n]));
    emit("doc"); emit("course-doc");
  } else S.doc = doc;
  current.subject = subject;
  try { localStorage.setItem("il_course", subject); } catch (e) { /* ignore */ }
}

export async function renderCourse(v, topic = null, subject = undefined) {
  stopSpeaking();
  if (subject !== undefined) current.subject = subject;
  current.topic = topic;
  v.innerHTML = "";
  if (!current.subject) { emit("layout-course"); return renderCourses(v); }
  if (S.doc?.course !== current.subject) {
    v.append(h("div", { class: "card empty" }, h("span", { class: "spin" }), " " + T("opening")));
    try { await enter(current.subject); } catch (e) { current.subject = null; toast(e.message, true); return renderCourses(v); }
    v.innerHTML = "";
  }
  emit("layout-course");
  if (topic) return renderLesson(v, topic);
  return renderIndex(v);
}

// ---- all courses
async function renderCourses(v) {
  let list;
  try { list = await api.courses(); } catch (e) { v.append(h("div", { class: "empty" }, e.message)); return; }
  const prog = await Promise.all(list.map((c) => api.lessons(c.id).catch(() => null)));
  v.append(h("div", { class: "pagehead" }, h("div", {}, h("span", { class: "small-caps" }, T("courses")), h("h1", {}, T("courses_h") + " ", h("em", {}, T("courses_em"))), h("p", {}, T("courses_p")))));
  v.append(h("div", { class: "ccards" }, ...list.map((c, i) => {
    const p = prog[i];
    const pct = p ? Math.round((100 * p.done) / Math.max(1, p.total)) : 0;
    const go = () => renderCourse(v, null, c.id);
    return h("article", { class: "card ccard", onclick: go },
      h("span", { class: "small-caps" }, c.short || c.id.toUpperCase()), h("h3", {}, c.title),
      h("div", { class: "meta" }, h("span", { class: "chip" }, T("lessons_n", { n: p?.total ?? "…" })), h("span", { class: "chip" }, `${c.units.length} units`),
        c.questions ? h("span", { class: "chip gold" }, T("questions_n", { n: c.questions })) : null),
      h("div", { class: "bar" }, h("i", { style: { width: pct + "%", background: "var(--good)" } })),
      h("div", { class: "foot" }, h("small", {}, p ? `${p.done} ${T("of")} ${p.total} ${T("learned")}` : ""),
        h("button", { class: "btn sm", onclick: (e) => { e.stopPropagation(); go(); } }, p?.done ? T("continue") : T("open_course"))));
  })));
}

// ---- one course: units and lessons
async function renderIndex(v) {
  let idx;
  try { idx = await api.lessons(current.subject); } catch (e) { v.append(h("div", { class: "empty" }, e.message)); return; }
  const pct = Math.round((100 * idx.done) / Math.max(1, idx.total));
  const firstOpen = idx.units.flatMap((u) => u.topics).find((t) => t.status !== "learned" && t.has_lesson);
  const words = idx.title.split(" ");
  const nav = (view) => () => emit("nav", { view });
  v.append(h("div", { class: "lnav" }, h("button", { class: "linkbtn", onclick: () => { current.subject = null; try { localStorage.removeItem("il_course"); } catch (e) { /* ignore */ } renderCourse(v); } }, T("all_courses"))));
  v.append(h("div", { class: "pagehead" }, h("div", {}, h("span", { class: "small-caps" }, `${T("courses")} · ${idx.short}`), h("h1", {}, words.slice(0, -1).join(" ") + " ", h("em", {}, words[words.length - 1])), h("p", {}, T("sub"))),
    h("div", { class: "cprog" }, h("b", {}, `${idx.done} ${T("of")} ${idx.total}`), h("span", {}, T("topics") + " " + T("learned")),
      h("div", { class: "bar" }, h("i", { style: { width: pct + "%", background: "var(--good)" } })),
      firstOpen ? h("button", { class: "btn sm", style: { marginTop: "10px" }, onclick: () => renderCourse(v, firstOpen.id) }, (idx.done ? T("continue") : T("start")) + ": " + firstOpen.label.split("(")[0]) : null)));
  v.append(h("div", { class: "cacts" },
    h("button", { class: "btn ghost sm", onclick: nav("map") }, "🗺 " + T("map")), h("button", { class: "btn ghost sm", onclick: nav("quiz") }, "✎ " + T("quiz")),
    h("button", { class: "btn ghost sm", onclick: nav("progress") }, "📈 " + T("progress")),
    h("button", { class: "btn ghost sm", onclick: () => { presetUnits(null, current.subject); emit("nav", { view: "papers" }); } }, "⎙ " + T("papers")),
    h("button", { class: "btn sm", onclick: () => { S.focus = null; emit("ask", {}); } }, "✦ " + T("ask"))));
  for (const [i, u] of idx.units.entries()) {
    const done = u.topics.filter((t) => t.status === "learned").length;
    v.append(h("section", { class: "cunit card" }, h("header", {}, h("h3", {}, `${i + 1}. ${u.title}`), h("span", { class: "small-caps" }, `${done}/${u.topics.length}`)),
      h("div", { class: "ctopics" }, ...u.topics.map((t) => h("button", { class: "ctopic " + t.status, disabled: !t.has_lesson, onclick: () => renderCourse(v, t.id) },
        h("span", { class: "dot", "aria-hidden": "true" }, t.status === "learned" ? "✓" : t.status === "opened" ? "•" : ""),
        h("span", { class: "lbl" }, h("b", {}, t.label), h("small", { html: t.has_lesson ? md(t.summary.slice(0, 130) + (t.summary.length > 130 ? "…" : "")).replace(/^<p>|<\/p>$/g, "") : T("soon") })),
        t.numerical ? h("span", { class: "chip" }, T("numericals")) : null)))));
  }
  if (idx.books?.length || idx.links?.length) {
    v.append(h("section", { class: "card cbooks" },
      idx.books?.length ? h("div", {}, h("h3", {}, T("books")), h("ul", {}, ...idx.books.map((b) => h("li", {}, h("span", { class: "chip" + (b.kind === "text" ? " gold" : "") }, b.kind === "text" ? T("text") : T("ref")), " ", b.title)))) : null,
      idx.links?.length ? h("div", {}, h("h3", {}, T("links")), h("ul", {}, ...idx.links.map((l) => h("li", {}, h("a", { href: l.url, target: "_blank", rel: "noopener" }, l.title))))) : null));
  }
  mathify(v);
}

// ---- one lesson
async function renderLesson(v, topic, seed = null) {
  current.topic = topic;
  if (!lessonLang) lessonLang = getLang() === "te" ? "te" : "en";
  let d;
  const slow = lessonLang === "te" ? setTimeout(() => { v.innerHTML = ""; v.append(h("div", { class: "card empty" }, h("span", { class: "spin" }), " " + T("translating"))); }, 700) : null;
  try { d = await api.lesson(current.subject, topic, lessonLang, seed); } catch (e) { clearTimeout(slow); toast(e.message, true); return renderCourse(v); }
  clearTimeout(slow);
  if (current.topic !== topic) return; // the student moved on while it loaded
  v.innerHTML = "";
  S.focus = topic; emit("focus"); emit("layout-course");
  const les = d.lesson;
  const langSeg = h("div", { class: "pseg" }, ...[["en", "English"], ["te", "తెలుగు"]].map(([c, n]) =>
    h("button", { class: "segbtn" + (lessonLang === c ? " on" : ""), onclick: () => { lessonLang = c; renderLesson(v, topic); } }, n)));
  const fullText = [les.summary, ...les.sections.map((s) => `${s.heading}. ${s.body}`)].join("\n").replace(/\*\*/g, "");
  const listen = h("button", { class: "btn ghost sm", onclick: () => toggleSpeak(listen, fullText, d.lang) }, T("listen"));
  v.append(h("div", { class: "lnav" }, h("button", { class: "linkbtn", onclick: () => renderCourse(v) }, T("back")), langSeg),
    h("article", { class: "lesson" },
      h("span", { class: "small-caps" }, d.unit.title), h("h1", {}, les.title || d.label),
      d.refs?.length || d.prereq?.length ? h("div", { class: "lmeta" },
        d.prereq?.length ? h("span", {}, T("before") + ": ", ...d.prereq.map((p) => h("button", { class: "chipbtn", onclick: () => renderLesson(v, p.id) }, p.label.split(" (")[0]))) : null,
        d.refs?.length ? h("span", { class: "lrefs" }, "📖 " + T("read") + ": " + d.refs.join(" · ")) : null) : null,
      lessonLang === "te" && d.lang !== "te" ? h("p", { class: "notice" }, T("te_missing")) : null,
      h("div", { class: "lsum" }, h("b", {}, T("summary")), h("div", { html: md(les.summary) }), h("div", { class: "lsumbar" }, listen,
        h("button", { class: "btn ghost sm", onclick: () => { S.focus = topic; emit("ask", { focus: topic }); } }, T("ask_lesson")))),
      ...les.sections.map((s) => h("section", { class: s.exact ? "lrules" : "" }, h("h3", {}, s.heading), h("div", { html: md(steps(s.body)) }))),
      les.key_terms?.length ? h("section", {}, h("h3", {}, T("terms")), h("dl", { class: "terms" }, ...les.key_terms.flatMap((k) => [h("dt", { html: md(k.term).replace(/^<p>|<\/p>$/g, "") }), h("dd", { html: md(k.meaning).replace(/^<p>|<\/p>$/g, "") })]))) : null,
      exampleBlock(d, v, topic),
      les.diagram ? h("section", { class: "ldraw" }, h("h3", {}, "✎ " + T("draw")), h("div", { html: md(les.diagram) })) : null,
      les.exam_tips?.length ? h("section", { class: "ltips" }, h("h3", {}, T("tips")), h("ul", {}, ...les.exam_tips.map((x) => h("li", { html: md(x).replace(/^<p>|<\/p>$/g, "") })))) : null,
      les.mistakes?.length ? h("section", { class: "lmist" }, h("h3", {}, T("mistakes")), h("ul", {}, ...les.mistakes.map((x) => h("li", { html: md(x).replace(/^<p>|<\/p>$/g, "") })))) : null,
      d.check?.length ? quickCheck(d.check, topic) : null,
      footer(d, v)));
  v.scrollTop = 0; v.closest?.("main")?.scrollTo?.(0, 0);
  mathify(v);
}

// "1. Put ... 2. Take ... 3. Repeat" written as one paragraph -> one step per line, so it renders as a list
const steps = (t) => (/(^|\s)1[.)]\s/.test(t) && /\s2[.)]\s/.test(t) ? t.replace(/\s(\d{1,2})[.)]\s(?=[A-Zఀ-౿])/g, "\n$1. ") : t);

const blocks = renderBlocks;

function exampleBlock(d, v, topic) {
  if (d.example) { // numerical topic: a fresh, exactly solved problem
    const sol = h("div", { class: "qans", hidden: true }, ...blocks(d.example.answer));
    const btn = h("button", { class: "btn sm", onclick: () => { sol.hidden = !sol.hidden; btn.textContent = sol.hidden ? T("show_sol") : T("hide_sol"); if (!sol.hidden) mathify(sol); } }, T("show_sol"));
    const wrap = h("section", { class: "lex" }, h("h3", {}, T("example")), ...blocks(d.example.blocks),
      h("div", { class: "pbar" }, btn, h("button", { class: "btn ghost sm", onclick: () => newExample(wrap, d, v, topic) }, "↻ " + T("newex"))), sol);
    return wrap;
  }
  const ex = d.lesson.example;
  return ex?.body ? h("section", { class: "lex" }, h("h3", {}, T("example") + (ex.title ? ": " + ex.title : "")), h("div", { html: md(ex.body) })) : null;
}

// a new solved example without reloading (and losing the place in) the whole lesson
async function newExample(wrap, d, v, topic) {
  try {
    const n = await api.lesson(current.subject, topic, "en", Math.floor(Math.random() * 1e9));
    const fresh = exampleBlock({ ...d, example: n.example }, v, topic);
    wrap.replaceWith(fresh); mathify(fresh);
  } catch (e) { toast(e.message, true); }
}

function quickCheck(items, topic) {
  return h("section", { class: "lcheck" }, h("h3", {}, T("check")), ...items.map((q, qi) => {
    const fb = h("div");
    let done = false;
    const opts = q.options.map((o, i) => h("button", { class: "opt", html: md(o).replace(/^<p>|<\/p>$/g, ""), onclick: () => {
      if (done) return; done = true;
      opts[i].classList.add(i === q.correct ? "right" : "wrong"); opts[q.correct].classList.add("right");
      fb.append(h("div", { class: "explain" }, h("b", {}, (i === q.correct ? T("right") : T("wrong")) + " "), h("span", { html: md(q.why).replace(/^<p>|<\/p>$/g, "") })));
      mathify(fb);
      api.lessonCheck(current.subject, topic, i === q.correct).then(() => emit("progress", { touched: [topic] })).catch(() => {});
      emit("xp", { amount: i === q.correct ? 3 : 1 });
    } }));
    return h("div", { class: "qcard" }, h("div", { class: "stem", html: `${qi + 1}. ` + md(q.q).replace(/^<p>|<\/p>$/g, "") }), ...opts, fb);
  }));
}

function footer(d, v) {
  const mark = h("button", { class: "btn" + (d.status === "learned" ? " ghost" : ""), onclick: async () => {
    try { await api.lessonMark(current.subject, d.topic, d.status === "learned" ? "opened" : "learned"); d.status = d.status === "learned" ? "opened" : "learned";
      mark.textContent = d.status === "learned" ? T("marked") : T("mark"); mark.classList.toggle("ghost", d.status === "learned");
      if (d.status === "learned") { emit("xp", { amount: 5 }); emit("progress", { touched: [d.topic] }); } } catch (e) { toast(e.message, true); }
  } }, d.status === "learned" ? T("marked") : T("mark"));
  return h("footer", { class: "lfoot" }, mark,
    h("button", { class: "btn ghost", onclick: () => { presetUnits([d.unit.id], current.subject); emit("nav", { view: "papers" }); } }, "⎙ " + T("paper")),
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
  const spoken = text.replace(/\$([^$]+)\$/g, (_m, x) => x.replace(/\\[a-z]+/gi, " ").replace(/[{}^_]/g, " "));
  const pieces = (spoken.match(/[^.!?।\n]+[.!?।]*\s*/g) || [spoken]).reduce((acc, s) => { if (acc.length && (acc[acc.length - 1] + s).length < 350) acc[acc.length - 1] += s; else acc.push(s); return acc; }, []);
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
