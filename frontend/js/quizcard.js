import { api } from "./api.js";
import { t, getLang } from "./i18n.js";
import { h, toast, mathify } from "./util.js";
import { emit } from "./state.js";

// Renders an interactive quiz (used inside the tutor chat and on the Quiz page).
export function renderQuiz(quiz, { onDone } = {}) {
  const box = h("div", { class: "quizbox" });
  const qs = quiz.questions; let i = 0; let score = 0;
  const log = []; // what the student chose, for the review at the end
  const show = () => {
    box.innerHTML = "";
    if (i >= qs.length) {
      const wrong = log.filter((x) => !x.ok).length;
      const card = h("div", { class: "qcard qdone" }, h("b", {}, `${t("score")}: ${score} / ${qs.length}`), h("div", { class: "small-caps", style: { marginTop: "6px" } }, quiz.scope_label || ""));
      if (log.length) {
        const rev = h("details", { class: "qreview", open: wrong > 0 }, h("summary", {}, t("review")),
          ...log.map((x, k) => h("div", { class: "qrev " + (x.ok ? "ok" : "bad") }, h("div", { class: "stem" }, `${k + 1}. ${x.q.stem || x.q.prompt}`),
            h("div", {}, h("b", {}, (x.ok ? "✓ " : "✗ ") + t("your_ans") + ": "), x.q.options[x.choice]),
            x.ok ? null : h("div", {}, h("b", {}, t("right_ans") + ": "), x.q.options[x.answer]),
            x.why ? h("div", { class: "small" }, x.why) : null)));
        card.append(rev);
        mathify(rev);
      }
      if (wrong > 0) card.append(h("button", { class: "btn sm", style: { marginTop: "12px" }, onclick: async () => {
        try { const r = await api.quizRetry(quiz.quiz_id); box.replaceWith(renderQuiz(r, { onDone })); } catch (e) { toast(e.message, true); }
      } }, t("retry_wrong", { n: wrong })));
      else card.append(h("div", { style: { marginTop: "8px" } }, t("all_right")));
      box.append(card);
      onDone?.(score, qs.length); return;
    }
    const q = qs[i]; let locked = false;
    const card = h("div", { class: "qcard" });
    card.append(h("div", { class: "qn" }, h("span", { class: "small-caps" }, t("q_of", { i: i + 1, n: qs.length })), h("span", { class: "chip gold" }, q.qtype === "tf" ? "T / F" : q.qtype === "cloze" ? "___" : "MCQ")));
    card.append(h("div", { class: "small-caps" }, q.prompt));
    if (q.stem) card.append(h("div", { class: "stem" }, q.stem));
    const fb = h("div");
    const btns = q.options.map((o, idx) => h("button", { class: "opt", onclick: async () => {
      if (locked) return; locked = true;
      btns.forEach((b) => (b.disabled = true));
      try {
        const r = await api.answer(quiz.quiz_id, q.id, idx, getLang());
        if (r.correct) score++;
        log.push({ q, choice: idx, answer: r.answer, ok: r.correct, why: r.explanation ? `${r.explanation.label}: ${r.explanation.definition}` : "" });
        btns[idx].classList.add(r.correct ? "right" : "wrong"); btns[r.answer].classList.add("right");
        fb.append(h("div", { class: "explain" }, h("b", {}, r.verdict + " "), `${r.explanation.label}: ${r.explanation.definition}`));
        const next = h("button", { class: "btn sm", style: { marginTop: "10px" }, onclick: () => { i++; show(); } }, i + 1 < qs.length ? t("next_q") : t("finish"));
        fb.append(next);
        mathify(fb);
        emit("progress", { node: r.node_id, state: r.state });
        emit("xp", { amount: r.correct ? 10 : 3 });
      } catch (e) { toast(e.message, true); locked = false; btns.forEach((b) => (b.disabled = false)); }
    } }, o));
    card.append(...btns, fb);
    box.append(card);
    mathify(card);
  };
  show();
  return box;
}
