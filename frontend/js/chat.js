import { api } from "./api.js";
import { t, getLang, LANGS, unitName } from "./i18n.js";
import { h, md, toast } from "./util.js";
import { S, emit, on, clearOwner } from "./state.js";
import { renderQuiz } from "./quizcard.js";

export function createTutor(root) {
  clearOwner("tutor");
  let msgsEl, input, focusEl, sugEl, replySel, busy = false, loadedDoc = null;

  function mount() {
    root.innerHTML = "";
    replySel = h("select", { class: "field", style: { width: "auto", padding: "4px 8px", fontSize: "12px" }, title: t("reply_in") },
      h("option", { value: "auto" }, t("auto")), ...LANGS.map((l) => h("option", { value: l.code }, l.name)));
    const head = h("header", {}, h("div", {}, h("div", { class: "small-caps" }, t("tutor_sub")), h("h3", {}, t("tutor").split(" ")[0] + " ", h("em", {}, t("tutor").split(" ").slice(1).join(" ")))),
      h("div", { style: { display: "grid", gap: "6px", justifyItems: "end" } }, replySel, h("button", { class: "linkbtn", onclick: clear }, t("clear_chat"))));
    msgsEl = h("div", { class: "msgs", "aria-live": "polite" });
    sugEl = h("div", { class: "suggest" });
    focusEl = h("div", { class: "focusline" });
    input = h("textarea", { rows: "1", placeholder: t("placeholder"), maxlength: "1000", "aria-label": t("placeholder") });
    input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
    input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 120) + "px"; });
    const form = h("form", { onsubmit: (e) => { e.preventDefault(); send(); } }, input, h("button", { class: "btn", type: "submit" }, t("send")));
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
    const g = { en: `Hi! I'm your study tutor for “${S.doc.title}”. Ask me anything — in English, తెలుగు, हिन्दी or संस्कृतम्. I can explain, give examples, summarise and quiz you.`,
      te: `నమస్కారం! “${S.doc.title}” కోసం నేను మీ స్టడీ ట్యూటర్‌ని. ఏదైనా అడగండి — తెలుగు, English, हिन्दी లేదా संस्कृतम्‌లో.`,
      hi: `नमस्ते! मैं “${S.doc.title}” के लिए आपका स्टडी ट्यूटर हूँ। कुछ भी पूछें — हिन्दी, English, తెలుగు या संस्कृतम् में।`,
      sa: `नमस्ते! “${S.doc.title}” विषये अहं भवतः अध्ययनसहायकः। किमपि पृच्छतु — संस्कृते, हिन्दीभाषायां, तेलुगुभाषायाम् आङ्ग्लभाषायां वा।` }[getLang()];
    botMsg({ text: g }, false);
  }

  function userMsg(text) { msgsEl.append(h("div", { class: "msg user" }, text)); scroll(); }
  function botMsg(r, animate = true) {
    const el = h("div", { class: "msg bot" });
    el.innerHTML = md(r.text || "");
    if (r.citations?.length) {
      const c = h("div", { class: "cites" });
      const seen = new Set();
      for (const ci of r.citations) { const key = ci.page + ci.section_id; if (seen.has(key)) continue; seen.add(key);
        c.append(h("button", { title: ci.snippet, onclick: () => emit("goto", { page: ci.page, snippet: ci.snippet, node: ci.node_id }) }, `${unitName(S.doc.unit_label)} ${ci.page}`)); }
      el.append(c);
    }
    if (r.new_doubt) el.append(h("div", { class: "delta doubt" }, "◌ " + ({ en: "Added to your map as a doubt", te: "మీ మ్యాప్‌లో సందేహంగా చేర్చాను", hi: "आपके मैप में शंका के रूप में जोड़ा गया", sa: "मानचित्रे संशयरूपेण योजितम्" }[r.lang || getLang()] || "")));
    else if (r.touched?.length) {
      const n = S.nodes.get(r.touched[0]);
      el.append(h("div", { class: "delta" }, "✓ " + t("thanks_marked").replace(/—.*$/, "").trim() + (n ? ` · ${n.label}` : "")));
    }
    if (r.quiz) el.append(renderQuiz(r.quiz));
    if (r.intent === "recommend" && r.recommendations) { /* text already lists them */ }
    msgsEl.append(el); if (animate) scroll();
    return el;
  }
  const scroll = () => { msgsEl.scrollTop = msgsEl.scrollHeight; };

  async function send(text) {
    text = (text ?? input.value).trim();
    if (!text || busy || !S.doc) return;
    busy = true; input.value = ""; input.style.height = "auto"; userMsg(text);
    const typing = h("div", { class: "msg bot" }, h("div", { class: "typing" }, h("i"), h("i"), h("i"))); msgsEl.append(typing); scroll();
    try {
      const forced = replySel.value !== "auto";
      const r = await api.chat(S.doc.id, text, forced ? replySel.value : getLang(), S.focus, !forced);
      typing.remove(); botMsg(r);
      if (r.focus_node && r.grounded) { S.focus = r.focus_node; refreshFocus(); suggestions(); }
      emit("progress", { touched: r.touched, doubt: r.new_doubt });
    } catch (e) { typing.remove(); toast(e.message, true); }
    busy = false; input.focus();
  }
  async function clear() { if (!S.doc) return; await api.clearChat(S.doc.id); load(); }

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

  on("ask", ({ text, focus }) => { if (focus) { S.focus = focus; refreshFocus(); suggestions(); } if (text) send(text); }, "tutor");
  on("focus", () => { refreshFocus(); suggestions(); }, "tutor");
  on("doc", () => { if (msgsEl && S.doc && loadedDoc !== S.doc.id) { refreshFocus(); suggestions(); load(); } }, "tutor");
  on("lang", mount, "tutor");
  mount();
  return { mount, send };
}
