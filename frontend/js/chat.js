import { api } from "./api.js";
import { t, getLang, LANGS, unitName } from "./i18n.js";
import { h, md, toast, mathify } from "./util.js";
import { S, emit, on, clearOwner } from "./state.js";
import { renderQuiz } from "./quizcard.js";

// Labels for the new chat features (kept here so the big i18n table stays reviewable as-is).
const L = {
  en: { reasoning: "How I worked this out", terms: "Key terms", check: "Quick check", listen: "Listen", stop: "Stop", convert: "Convert to…", original: "Original",
    copy: "Copy", copied: "Copied", mic: "Speak your doubt", listening: "Listening… speak now", beyond: "Partly beyond your notes", notes: "From your notes",
    translating: "Translating…", right: "Right!", wrong: "Not quite —", tryit: "Tap an answer", no_mic: "Voice input isn't available in this browser.", offline: "Offline answer (quoted from notes)",
    no_voice: "Couldn't play the voice right now — please try again.", transcribing: "Turning your voice into text…", thinking1: "Reading your notes…", thinking2: "Writing a clear explanation…", thinking3: "Almost there…" },
  te: { reasoning: "నేను ఎలా ఆలోచించానంటే", terms: "ముఖ్య పదాలు", check: "చిన్న పరీక్ష", listen: "వినండి", stop: "ఆపు", convert: "భాష మార్చు…", original: "అసలు",
    copy: "కాపీ", copied: "కాపీ అయింది", mic: "మాట్లాడి అడగండి", listening: "వింటున్నాను… మాట్లాడండి", beyond: "కొంత భాగం నోట్స్ బయటిది", notes: "మీ నోట్స్ నుండి",
    translating: "అనువదిస్తున్నాను…", right: "సరైనది!", wrong: "కాదు —", tryit: "ఒక జవాబు నొక్కండి", no_mic: "ఈ బ్రౌజర్‌లో వాయిస్ ఇన్‌పుట్ లేదు.", offline: "ఆఫ్‌లైన్ జవాబు (నోట్స్ నుండి)",
    no_voice: "ఇప్పుడు వాయిస్ ప్లే కాలేదు — మళ్ళీ ప్రయత్నించండి.", transcribing: "మీ మాటను టెక్స్ట్‌గా మారుస్తున్నాను…", thinking1: "మీ నోట్స్ చదువుతున్నాను…", thinking2: "సులభంగా వివరిస్తున్నాను…", thinking3: "దాదాపు అయిపోయింది…" },
  hi: { reasoning: "मैंने कैसे सोचा", terms: "मुख्य शब्द", check: "छोटी जाँच", listen: "सुनें", stop: "रोकें", convert: "भाषा बदलें…", original: "मूल",
    copy: "कॉपी", copied: "कॉपी हुआ", mic: "बोलकर पूछें", listening: "सुन रहा हूँ… बोलिए", beyond: "कुछ हिस्सा नोट्स से बाहर", notes: "आपके नोट्स से",
    translating: "अनुवाद हो रहा है…", right: "सही!", wrong: "नहीं —", tryit: "एक उत्तर चुनें", no_mic: "इस ब्राउज़र में वॉइस इनपुट उपलब्ध नहीं है।", offline: "ऑफ़लाइन उत्तर (नोट्स से)",
    no_voice: "अभी आवाज़ नहीं चल पाई — फिर से कोशिश करें।", transcribing: "आपकी आवाज़ को टेक्स्ट में बदल रहा हूँ…", thinking1: "आपके नोट्स पढ़ रहा हूँ…", thinking2: "आसान भाषा में समझा रहा हूँ…", thinking3: "बस हो गया…" },
  sa: { reasoning: "चिन्तनक्रमः", terms: "मुख्यपदानि", check: "लघुपरीक्षा", listen: "शृणु", stop: "विरम", convert: "भाषां परिवर्तय…", original: "मूलम्",
    copy: "प्रतिलिपिः", copied: "प्रतिलिपिः कृता", mic: "वदित्वा पृच्छ", listening: "शृणोमि… वदतु", beyond: "किञ्चित् टिप्पणीभ्यः बहिः", notes: "टिप्पणीभ्यः",
    translating: "अनूद्यते…", right: "सम्यक्!", wrong: "न —", tryit: "उत्तरं चिनोतु", no_mic: "अस्मिन् ब्राउज़रे वाक्-निवेशः नास्ति।", offline: "ऑफलाइन-उत्तरम्" },
};
const lb = (k, lang) => (L[lang] || L[getLang()] || L.en)[k] ?? L.en[k];
const SPEECH_LOCALE = { en: "en-IN", te: "te-IN", hi: "hi-IN", sa: "hi-IN", ta: "ta-IN", kn: "kn-IN", ml: "ml-IN", mr: "mr-IN", bn: "bn-IN", gu: "gu-IN", pa: "pa-IN", or: "or-IN", ur: "ur-IN" };
let LANG_LIST = null;
const languages = async () => (LANG_LIST ??= await api.languages().catch(() => LANGS.map((l) => ({ code: l.code, name: l.name, native: l.name }))));

// -------------------------------------------------------------------------- speech out
let speaking = null; // {btn, audio}
function stopSpeaking() {
  try { window.speechSynthesis?.cancel(); } catch (e) { /* ignore */ }
  if (speaking?.audio) { speaking.audio.pause(); URL.revokeObjectURL(speaking.audio.src); }
  if (speaking?.btn) speaking.btn.classList.remove("on");
  speaking = null;
}
function plain(text) { return String(text || "").replace(/\$([^$]+)\$/g, (_m, x) => x.replace(/\\[a-zA-Z]+/g, " ").replace(/[{}^_]/g, " ")).replace(/\[(?:p\.|page |slide |part |lesson )\s*\d+\]/gi, "").replace(/[*_`#>]/g, "").replace(/\s+/g, " ").trim(); }
// English words inside a Telugu/Hindi reply need an English voice: a Telugu voice skips Latin letters.
function scriptRuns(text) {
  const re = /[A-Za-z][A-Za-z0-9'\-.]*(?:[ ,]+[A-Za-z0-9][A-Za-z0-9'\-.]*)*/g;
  const out = []; let pos = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > pos) out.push({ en: false, t: text.slice(pos, m.index) });
    out.push({ en: true, t: m[0] }); pos = m.index + m[0].length;
  }
  if (pos < text.length) out.push({ en: false, t: text.slice(pos) });
  return out.filter((r) => /[\p{L}\p{N}]/u.test(r.t));
}
async function speak(btn, text, lang) {
  if (speaking?.btn === btn) { stopSpeaking(); return; }
  stopSpeaking();
  speaking = { btn }; btn.classList.add("on");
  const base = (lang || "en").split("-")[0];
  const voices = window.speechSynthesis?.getVoices?.() || [];
  const voice = voices.find((v) => v.lang?.toLowerCase().startsWith(base)) || null;
  const enVoice = voices.find((v) => v.lang?.toLowerCase().startsWith("en-in")) || voices.find((v) => v.lang?.toLowerCase().startsWith("en")) || null;
  const msg = plain(text);
  const serverFirst = !!S.info?.speech?.server_tts;  // natural Sarvam voice beats robotic device voices
  if (serverFirst) {
    try { await playServer(btn, msg, base); return; } catch (e) { if (speaking?.btn !== btn) return; /* fall through to device voice */ }
  }
  if (voice && (base === "en" || enVoice)) {
    const runs = base === "en" ? [{ en: true, t: msg }] : scriptRuns(msg);
    runs.forEach((r, i) => {
      const u = new SpeechSynthesisUtterance(r.t);
      const v = r.en ? enVoice : voice;
      u.voice = v; u.lang = v.lang; u.rate = 0.95;
      if (i === runs.length - 1) u.onend = u.onerror = () => { if (speaking?.btn === btn) stopSpeaking(); };
      window.speechSynthesis.speak(u);
    });
    return;
  }
  // no device voice (Telugu on most Windows PCs): the server reads it, English words included
  if (serverFirst) { stopSpeaking(); toast(lb("no_voice"), true); return; }
  try { await playServer(btn, msg, base); } catch (e) { stopSpeaking(); toast(e.message, true); }
}
// Speak in sentence-sized pieces: the first piece starts playing in ~1-2 s while the next one is fetched.
function speechPieces(text, max = 350) {
  const sents = text.match(/[^.!?।\n]+[.!?।]*\s*/g) || [text];
  const out = []; let cur = "";
  for (const s of sents) { if ((cur + s).length > max && cur) { out.push(cur); cur = s; } else cur += s; }
  if (cur.trim()) out.push(cur);
  return out.length > 1 && out[0].length < 60 ? [out[0] + out[1], ...out.slice(2)] : out;
}
async function playServer(btn, msg, base) {
  btn.classList.add("loading");
  const pieces = speechPieces(msg.slice(0, 3000));
  let next = api.tts(pieces[0], base);
  try {
    for (let i = 0; i < pieces.length; i++) {
      const blob = await next;
      if (speaking?.btn !== btn) return;
      if (i + 1 < pieces.length) next = api.tts(pieces[i + 1], base);
      const audio = new Audio(URL.createObjectURL(blob));
      speaking.audio = audio;
      btn.classList.remove("loading");
      await audio.play();
      await new Promise((res) => { audio.onended = res; audio.onerror = res; audio.onpause = () => { if (speaking?.btn !== btn) res(); }; });
      URL.revokeObjectURL(audio.src);
      if (speaking?.btn !== btn) return;
    }
    if (speaking?.btn === btn) stopSpeaking();
  } finally { btn.classList.remove("loading"); }
}

// -------------------------------------------------------------------------- the tutor
export function createTutor(root) {
  clearOwner("tutor");
  let msgsEl, input, focusEl, sugEl, replySel, micBtn, busy = false, loadedDoc = null, rec = null;

  function mount() {
    root.innerHTML = "";
    replySel = h("select", { class: "field", style: { width: "auto", padding: "4px 8px", fontSize: "12px" }, title: t("reply_in") },
      h("option", { value: "auto" }, t("auto")), ...LANGS.map((l) => h("option", { value: l.code }, l.name)));
    languages().then((ls) => { for (const l of ls) if (!LANGS.some((x) => x.code === l.code) && !l.code.includes("-")) replySel.append(h("option", { value: l.code }, l.native)); });
    const head = h("header", {}, h("div", {}, h("div", { class: "small-caps" }, t("tutor_sub")), h("h3", {}, t("tutor").split(" ")[0] + " ", h("em", {}, t("tutor").split(" ").slice(1).join(" ")))),
      h("div", { style: { display: "grid", gap: "6px", justifyItems: "end" } }, h("div", { style: { display: "flex", gap: "6px", alignItems: "center" } }, replySel,
        h("button", { class: "tutor-close", type: "button", "aria-label": "Close", title: "Close", onclick: () => { S.tutorOpen = false; emit("layout"); } }, "✕")),
        h("button", { class: "linkbtn", onclick: clear }, t("clear_chat"))));
    msgsEl = h("div", { class: "msgs", "aria-live": "polite" });
    sugEl = h("div", { class: "suggest" });
    focusEl = h("div", { class: "focusline" });
    input = h("textarea", { rows: "1", placeholder: t("placeholder"), maxlength: "1000", "aria-label": t("placeholder") });
    input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
    input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 120) + "px"; });
    micBtn = h("button", { class: "mic", type: "button", title: lb("mic"), "aria-label": lb("mic"), onclick: toggleMic }, micIcon());
    const form = h("form", { onsubmit: (e) => { e.preventDefault(); send(); } }, micBtn, input, h("button", { class: "btn", type: "submit" }, t("send")));
    root.append(head, msgsEl, sugEl, h("div", { class: "composer" }, focusEl, form));
    loadedDoc = null; refreshFocus(); suggestions();
    if (S.doc) load();
  }

  async function load() {
    loadedDoc = S.doc.id; msgsEl.innerHTML = "";
    try {
      const hist = await api.history(S.doc.id);
      if (!hist.length) greet();
      for (const m of hist) m.role === "user" ? userMsg(m.content) : botMsg({ text: m.content, ...m.meta }, false);
    } catch (e) { greet(); }
    scroll();
  }
  function greet() {
    const g = { en: `Hi! I'm your study tutor for “${S.doc.title}”. Ask me anything — type or speak, in English, తెలుగు, हिन्दी, संस्कृतम् or even Telugu in English letters ("convolution ante enti?"). I'll explain step by step, give examples and quiz you.`,
      te: `నమస్కారం! “${S.doc.title}” కోసం నేను మీ స్టడీ ట్యూటర్‌ని. ఏదైనా అడగండి — టైప్ చేసి లేదా మాట్లాడి, తెలుగులో, English లో, లేదా "convolution ante enti?" లాగా ఇంగ్లీష్ అక్షరాల్లో తెలుగు. దశలవారీగా వివరిస్తాను.`,
      hi: `नमस्ते! मैं “${S.doc.title}” के लिए आपका स्टडी ट्यूटर हूँ। कुछ भी पूछें — लिखकर या बोलकर, हिन्दी, English, తెలుగు या Hinglish में।`,
      sa: `नमस्ते! “${S.doc.title}” विषये अहं भवतः अध्ययनसहायकः। किमपि पृच्छतु — लिखित्वा वदित्वा वा।` }[getLang()];
    botMsg({ text: g }, false);
  }

  function userMsg(text) { msgsEl.append(h("div", { class: "msg user" }, text)); scroll(); }

  // [p.3] in an answer becomes a small chip that jumps to that page in the reader
  function linkPages(el, r) {
    el.querySelectorAll("p, li").forEach((node) => {
      node.innerHTML = node.innerHTML.replace(/\[(p\.|page |slide |part )\s*(\d+)\]/gi, (_m, _u, n) => `<button class="pg" data-page="${n}">${unitName(S.doc?.unit_label)} ${n}</button>`);
    });
    el.querySelectorAll("button.pg").forEach((b) => b.addEventListener("click", () => {
      const pg = +b.dataset.page; const c = (r.citations || []).find((x) => x.page === pg);
      emit("goto", { page: pg, snippet: c?.snippet, node: c?.node_id });
    }));
  }

  function botMsg(r, animate = true) {
    const el = h("div", { class: "msg bot" });
    const lang = r.lang || getLang();
    const body = h("div", { class: "body" });
    const extras = h("div", { class: "extras" });
    const state = { r, orig: r, shown: lang };

    function paint(cur) {
      body.innerHTML = md(cur.text || ""); linkPages(body, r);
      queueMicrotask(() => mathify(el)); // after the rest of the bubble is painted
      extras.innerHTML = "";
      if (r.notice) extras.append(h("div", { class: "notice", title: r.notice }, "⚠ " + t("ai_down"))); // technical reason only in the tooltip
      if (r.beyond_notes || r.used_general_knowledge) extras.append(h("div", { class: "beyond" }, "◐ " + (cur.beyond_notes || lb("beyond", state.shown))));
      if (cur.key_terms?.length) {
        extras.append(h("div", { class: "terms" }, h("span", { class: "small-caps" }, lb("terms", state.shown)),
          ...cur.key_terms.map((k) => h("span", { class: "chip", title: k.meaning || "" }, h("b", {}, k.term), k.meaning ? " · " + k.meaning : ""))));
      }
      if (cur.reasoning?.length) {
        extras.append(h("details", { class: "reason" }, h("summary", {}, "🧠 " + lb("reasoning", state.shown)),
          h("ol", {}, ...cur.reasoning.map((s) => h("li", { html: md(s).replace(/^<p>|<\/p>$/g, "") })))));
      }
      if (r.check && cur === state.orig) extras.append(checkCard(r.check, lang));
      if (cur.follow_ups?.length) extras.append(h("div", { class: "followups" }, ...cur.follow_ups.map((f) => h("button", { onclick: () => send(f) }, f))));
    }
    paint(r);
    el.append(body, extras);

    if (r.citations?.length) {
      const c = h("div", { class: "cites" });
      const seen = new Set();
      for (const ci of [...r.citations].sort((a, b) => a.page - b.page)) { const key = String(ci.page); if (seen.has(key)) continue; seen.add(key);
        c.append(h("button", { title: ci.snippet, onclick: () => emit("goto", { page: ci.page, snippet: ci.snippet, node: ci.node_id }) }, `${unitName(S.doc.unit_label)} ${ci.page}`)); }
      el.append(c);
    }
    if (r.new_doubt) el.append(h("div", { class: "delta doubt" }, "◌ " + ({ en: "Added to your map as a doubt", te: "మీ మ్యాప్‌లో సందేహంగా చేర్చాను", hi: "आपके मैप में शंका के रूप में जोड़ा गया", sa: "मानचित्रे संशयरूपेण योजितम्" }[lang] || "Added to your map as a doubt")));
    else if (r.touched?.length) {
      const n = S.nodes.get(r.touched[0]);
      el.append(h("div", { class: "delta" }, "✓ " + t("thanks_marked").replace(/—.*$/, "").trim() + (n ? ` · ${n.label}` : "")));
    }
    if (r.quiz) el.append(renderQuiz(r.quiz));
    if (r.text && r.intent !== "greet" && r.text.length > 40) el.append(toolbar(state, paint));
    msgsEl.append(el); if (animate) scrollToMsg(el);
    return el;
  }

  function toolbar(state, paint) {
    const bar = h("div", { class: "mtools" });
    const listen = h("button", { class: "tool", title: lb("listen", state.shown), onclick: () => speak(listen, cur().text, state.shown) }, "🔊 " + lb("listen", state.shown));
    const sel = h("select", { class: "tool", title: lb("convert", state.shown), "aria-label": lb("convert", state.shown) }, h("option", { value: "" }, "🌐 " + lb("convert", state.shown)));
    languages().then((ls) => { for (const l of ls) sel.append(h("option", { value: l.code }, l.native === l.name ? l.name : `${l.native} · ${l.name}`)); });
    const orig = h("button", { class: "tool", style: { display: "none" }, onclick: () => { state.cur = null; state.shown = state.orig.lang || getLang(); paint(state.orig); orig.style.display = "none"; sel.value = ""; } }, "↺ " + lb("original"));
    const copy = h("button", { class: "tool", onclick: async () => { try { await navigator.clipboard.writeText(plain(cur().text)); copy.textContent = "✓ " + lb("copied", state.shown); setTimeout(() => (copy.textContent = "⧉ " + lb("copy", state.shown)), 1400); } catch (e) { /* ignore */ } } }, "⧉ " + lb("copy", state.shown));
    const cur = () => state.cur || state.orig;
    state.cache = {};
    sel.addEventListener("change", async () => {
      const target = sel.value; if (!target) return;
      if (target === (state.orig.lang || "")) { orig.click(); return; }
      stopSpeaking();
      const o = state.orig;
      let tr = state.cache[target];
      if (!tr) {
        sel.disabled = true; const label = sel.options[sel.selectedIndex].text; sel.options[sel.selectedIndex].text = lb("translating");
        try {
          tr = await api.translate({ target, source: o.lang || "", text: o.text, reasoning: o.reasoning || [], key_terms: o.key_terms || [], follow_ups: o.follow_ups || [], beyond_notes: o.beyond_notes || "" });
          state.cache[target] = tr;
        } catch (e) { toast(e.message, true); }
        sel.options[sel.selectedIndex].text = label; sel.disabled = false;
      }
      if (tr) { state.cur = { ...o, ...tr }; state.shown = target; paint(state.cur); orig.style.display = ""; }
    });
    bar.append(listen, sel, orig, copy);
    if (state.orig.engine) bar.append(h("span", { class: "engine", title: state.orig.understood_as ? "Searched notes for: " + state.orig.understood_as : "" }, "✦ " + state.orig.engine));
    else if (state.orig.citations?.length) bar.append(h("span", { class: "engine" }, lb("offline", state.shown)));
    return bar;
  }

  function checkCard(c, lang) {
    const box = h("div", { class: "qcard check" }, h("div", { class: "small-caps" }, "⚡ " + lb("check", lang)), h("div", { class: "stem" }, c.q));
    const fb = h("div"); let done = false;
    const btns = c.options.map((o, i) => h("button", { class: "opt", onclick: () => {
      if (done) return; done = true; btns.forEach((b) => (b.disabled = true));
      const ok = i === c.answer; btns[i].classList.add(ok ? "right" : "wrong"); btns[c.answer].classList.add("right");
      fb.append(h("div", { class: "explain" }, h("b", {}, (ok ? lb("right", lang) : lb("wrong", lang)) + " "), c.why || ""));
      if (S.focus && S.doc) api.mark(S.doc.id, S.focus, ok ? "self_understood" : "self_confused").then(() => emit("progress", { touched: [S.focus] })).catch(() => {});
      if (S.focus) emit("xp", { amount: ok ? 4 : 2 }); // matches the server: self_understood / self_confused
    } }, o));
    box.append(...btns, fb);
    return box;
  }

  const scroll = () => { msgsEl.scrollTop = msgsEl.scrollHeight; };
  // a long answer must be read from its first line: show the student's question + the start of the reply
  const scrollToMsg = (el) => {
    const prev = el.previousElementSibling;
    const top = (prev && prev.classList.contains("user") ? prev : el).offsetTop - msgsEl.offsetTop - 8;
    msgsEl.scrollTo({ top: Math.max(0, top), behavior: "smooth" });
  };

  async function send(text) {
    text = (text ?? input.value).trim();
    if (!text || busy || !S.doc) return;
    busy = true; input.value = ""; input.style.height = "auto"; userMsg(text); stopSpeaking();
    const stage = h("span", { class: "stage" }, lb("thinking1"));
    const typing = h("div", { class: "msg bot" }, h("div", { class: "typing" }, h("i"), h("i"), h("i")), stage); msgsEl.append(typing); scroll();
    const timers = [setTimeout(() => (stage.textContent = lb("thinking2")), 2200), setTimeout(() => (stage.textContent = lb("thinking3")), 9000)];
    let live = null;  // the bubble that shows the answer while it is being written
    const onDelta = (partial) => {
      if (partial === null) { live?.remove(); live = null; return; }  // engine switched: start over quietly
      if (!partial.trim()) return;
      if (!live) {
        timers.forEach(clearTimeout); typing.remove();
        live = h("div", { class: "msg bot streaming" }, h("div", { class: "body" }));
        msgsEl.append(live); scrollToMsg(live);
      }
      live.firstChild.innerHTML = md(partial).replace(/\[(?:p\.|page |slide |part )\s*(\d+)[^\]]*\]/gi, (_m, n) => `<span class="pg">${unitName(S.doc?.unit_label)} ${n}</span>`) + '<span class="caret"></span>';
      const nearBottom = msgsEl.scrollHeight - msgsEl.scrollTop - msgsEl.clientHeight < 140;
      if (nearBottom && live.offsetHeight < msgsEl.clientHeight * 0.8) msgsEl.scrollTop = msgsEl.scrollHeight;
    };
    try {
      const forced = replySel.value !== "auto";
      const r = await api.chatStream(S.doc.id, text, forced ? replySel.value : getLang(), S.focus, !forced, onDelta);
      timers.forEach(clearTimeout); typing.remove();
      if (live) { const el = botMsg(r, false); live.replaceWith(el); live = null; } else botMsg(r);
      if (r.focus_node && r.grounded) { S.focus = r.focus_node; refreshFocus(); suggestions(); }
      emit("progress", { touched: r.touched, doubt: r.new_doubt });
      if (r.touched?.length) emit("xp", { amount: 5 * r.touched.length }); // server: 5 XP per concept asked about
    } catch (e) { timers.forEach(clearTimeout); typing.remove(); live?.remove(); toast(e.message, true); }
    busy = false; input.focus();
  }
  async function clear() { if (!S.doc) return; await api.clearChat(S.doc.id); load(); }

  // ------------------------------------------------------------------------ speech in
  function micLocale() {
    const v = replySel?.value && replySel.value !== "auto" ? replySel.value : getLang();
    return SPEECH_LOCALE[v] || "en-IN";
  }
  function setMic(on) { micBtn.classList.toggle("on", on); input.placeholder = on ? lb("listening") : t("placeholder"); }
  async function toggleMic() {
    if (rec) { rec.stop(); return; }
    // The server recogniser (Sarvam Saaras) hears Telugu, Hindi, English AND the mix, and knows this
    // document's terms. The browser's recogniser only hears ONE preset language, so a Telugu question
    // with the app in English came out as gibberish. Use it only when the server has no recogniser.
    if (S.info?.speech?.server_asr && navigator.mediaDevices?.getUserMedia && window.MediaRecorder) { recordToServer(); return; }
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SR) {
      const r = new SR(); rec = r;
      r.lang = micLocale(); r.interimResults = true; r.continuous = false;
      let finalText = "";
      r.onresult = (e) => {
        let interim = "";
        for (const res of e.results) (res.isFinal ? (finalText = res[0].transcript) : (interim += res[0].transcript));
        input.value = finalText || interim;
      };
      r.onerror = (e) => { if (e.error === "not-allowed") toast(lb("no_mic"), true); };
      r.onend = () => { rec = null; setMic(false); if (finalText.trim()) send(finalText); };
      setMic(true); r.start();
      return;
    }
    recordToServer();
  }
  async function recordToServer() {
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) { toast(lb("no_mic"), true); return; }
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } }); }
    catch (e) { toast(lb("no_mic"), true); return; }
    const mr = new MediaRecorder(stream); const parts = [];
    // Stop by itself ~1.6 s after the student stops talking (no need to hunt for the button).
    let ctx = null, raf = 0, spoke = false, quietSince = 0;
    try {
      ctx = new (window.AudioContext || window.webkitAudioContext)();
      const an = ctx.createAnalyser(); an.fftSize = 1024; ctx.createMediaStreamSource(stream).connect(an);
      const buf = new Uint8Array(an.fftSize);
      const t0 = performance.now();
      const tick = () => {
        an.getByteTimeDomainData(buf);
        let sum = 0; for (const v of buf) sum += (v - 128) * (v - 128);
        const rms = Math.sqrt(sum / buf.length), now = performance.now();
        if (rms > 6) { spoke = true; quietSince = 0; } else if (spoke) { quietSince ||= now; if (now - quietSince > 1600) { rec?.stop(); return; } }
        if (!spoke && now - t0 > 8000) { rec?.stop(); return; }  // nothing said at all
        raf = requestAnimationFrame(tick);
      };
      raf = requestAnimationFrame(tick);
    } catch (e) { /* no Web Audio: tap again to stop */ }
    const stopAll = () => { cancelAnimationFrame(raf); ctx?.close?.(); stream.getTracks().forEach((tr) => tr.stop()); };
    rec = { stop: () => mr.state !== "inactive" && mr.stop() };
    mr.ondataavailable = (e) => e.data.size && parts.push(e.data);
    mr.onstop = async () => {
      stopAll(); rec = null; setMic(false);
      const blob = new Blob(parts, { type: mr.mimeType || "audio/webm" });
      if (blob.size < 2000 || !spoke && ctx) return;
      input.placeholder = lb("transcribing"); input.disabled = true; micBtn.classList.add("busy");
      try {
        const r = await api.asr(blob, micLocale().split("-")[0], S.doc?.id || "");
        if (r.text) send(r.text);
      } catch (e) { toast(e.message, true); }
      input.disabled = false; micBtn.classList.remove("busy"); input.placeholder = t("placeholder");
    };
    setMic(true); mr.start();
    setTimeout(() => rec?.stop(), 29000); // the recogniser takes up to 30 s per question
  }

  function refreshFocus() {
    if (!focusEl) return; focusEl.innerHTML = "";
    const n = S.focus && S.nodes.get(S.focus);
    if (n) focusEl.append(`${t("asking_about")}:`, h("span", { class: "chip gold" }, n.label), h("button", { class: "linkbtn", onclick: () => { S.focus = null; refreshFocus(); suggestions(); } }, "✕"));
  }
  function suggestions() {
    if (!sugEl) return; sugEl.innerHTML = "";
    const n = S.focus && S.nodes.get(S.focus);
    const items = [n ? t("sg_explain", { t: n.label }) : null, t("sg_example"), t("sg_summary"), t("sg_quiz"), t("sg_next")].filter(Boolean);
    for (const s of items) sugEl.append(h("button", { onclick: () => send(s) }, s));
  }

  on("ask", ({ text, focus }) => { if (!S.tutorOpen) { S.tutorOpen = true; emit("layout"); } if (focus) { S.focus = focus; refreshFocus(); suggestions(); } if (text) send(text); }, "tutor");
  on("focus", () => { refreshFocus(); suggestions(); }, "tutor");
  on("doc", () => { if (msgsEl && S.doc && loadedDoc !== S.doc.id) { refreshFocus(); suggestions(); load(); } }, "tutor");
  on("lang", mount, "tutor");
  mount();
  return { mount, send };
}

function micIcon() {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg"); svg.setAttribute("viewBox", "0 0 24 24"); svg.setAttribute("width", "20"); svg.setAttribute("height", "20"); svg.setAttribute("aria-hidden", "true");
  svg.innerHTML = '<path fill="currentColor" d="M12 15a3 3 0 0 0 3-3V6a3 3 0 1 0-6 0v6a3 3 0 0 0 3 3Zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-2.08A7 7 0 0 0 19 12h-2Z"/>';
  return svg;
}
