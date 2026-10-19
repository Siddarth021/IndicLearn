// Study streak + XP + level, in the sidebar. Everything is computed on the server from the
// evidence log (questions asked, quiz answers, "I understand this"), so it cannot be gamed by
// refreshing the page and always matches the student's real activity.
import { api } from "./api.js";
import { getLang } from "./i18n.js";
import { h } from "./util.js";
import { S, on, clearOwner } from "./state.js";

const L = {
  en: { lvl: "Level", streak: "day streak", start: "Start a streak today", xp: "XP", mastered: "Mastered" },
  te: { lvl: "లెవెల్", streak: "రోజుల స్ట్రీక్", start: "ఈరోజే స్ట్రీక్ మొదలుపెట్టండి", xp: "XP", mastered: "పట్టు సాధించారు" },
  hi: { lvl: "लेवल", streak: "दिन की स्ट्रीक", start: "आज स्ट्रीक शुरू करें", xp: "XP", mastered: "महारत हासिल" },
  sa: { lvl: "स्तरः", streak: "दिनानां क्रमः", start: "अद्य आरभताम्", xp: "XP", mastered: "आयत्तम्" },
};
const lb = (k) => (L[getLang()] || L.en)[k];

export function createHud() {
  clearOwner("hud");
  const el = h("div", { class: "hud", "aria-live": "polite" });
  let timer = null;
  async function refresh() {
    try {
      const s = await api.stats();
      const pct = Math.max(4, Math.round(100 * (s.xp - s.level_floor) / Math.max(1, s.next_level_xp - s.level_floor)));
      el.innerHTML = "";
      el.append(
        h("div", { class: "top" }, h("span", { class: "lvl" }, `${lb("lvl")} ${s.level}`), h("span", {}, `${s.xp} ${lb("xp")}`)),
        h("div", { class: "xpbar", title: `${s.xp} / ${s.next_level_xp} XP` }, h("i", { style: { width: pct + "%" } })),
        h("div", { class: "streak" }, s.streak ? `🔥 ${s.streak} ${lb("streak")}` : "✨ " + lb("start")),
      );
    } catch (e) { el.hidden = true; }
  }
  const soon = () => { clearTimeout(timer); timer = setTimeout(refresh, 600); };
  on("xp", ({ amount }) => { floatXp(el, amount); soon(); }, "hud");
  on("progress", (d) => {
    soon();
    if (d?.state?.status === "mastered") celebrate(S.nodes.get(d.node)?.label);
  }, "hud");
  refresh();
  return el;
}

function floatXp(anchor, amount) {
  if (!amount) return;
  const r = anchor.getBoundingClientRect();
  const x = r.width ? r.left + r.width - 60 : window.innerWidth / 2;
  const y = r.width ? r.top : window.innerHeight - 120;
  const f = h("div", { class: "xpfloat", style: { left: x + "px", top: y + "px" } }, `+${amount} XP`);
  document.body.append(f); setTimeout(() => f.remove(), 1200);
}

export function celebrate(label) {
  const c = h("div", { class: "celebrate", role: "status" }, `🎉 ${lb("mastered")}${label ? ": " + label : ""}`);
  document.body.append(c); setTimeout(() => c.remove(), 2600);
}
