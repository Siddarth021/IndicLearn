// Thin fetch wrapper: bearer token, JSON, friendly errors.
let token = null;
try { token = localStorage.getItem("il_token"); } catch (e) { /* storage may be blocked */ }
export const getToken = () => token;
export function setToken(t) { token = t; try { t ? localStorage.setItem("il_token", t) : localStorage.removeItem("il_token"); } catch (e) { /* ignore */ } }

export class ApiError extends Error { constructor(msg, status) { super(msg); this.status = status; } }

async function req(method, url, body, isForm) {
  const headers = {};
  if (token) headers.Authorization = "Bearer " + token;
  if (body && !isForm) headers["Content-Type"] = "application/json";
  const r = await fetch("/api" + url, { method, headers, body: body ? (isForm ? body : JSON.stringify(body)) : undefined });
  let data = null;
  try { data = await r.json(); } catch (e) { /* no body */ }
  if (!r.ok) {
    const msg = typeof data?.detail === "string" ? data.detail : (data?.detail?.[0]?.msg || r.statusText);
    throw new ApiError(msg, r.status);
  }
  return data;
}
export const api = {
  info: () => req("GET", "/info"),
  courses: () => req("GET", "/courses"),
  quizRetry: (quizId) => req("POST", `/quiz/${quizId}/retry`),
  lessons: () => req("GET", "/courses/os/lessons"),
  lesson: (topic, lang = "en", seed = null) => req("GET", `/courses/os/lessons/${topic}?lang=${lang}${seed !== null ? "&seed=" + seed : ""}`),
  lessonMark: (topic, status) => req("POST", `/courses/os/lessons/${topic}/mark?status=${status}`),
  papers: () => req("GET", "/papers"),
  paperNew: (blueprint, doc_ids = null) => req("POST", "/papers", { blueprint, doc_ids }),
  paperPattern: (file) => { const f = new FormData(); f.append("file", file); return req("POST", "/papers/pattern", f, true); },
  paper: (id, answers = false) => req("GET", `/papers/${id}?answers=${answers}`),
  paperDel: (id) => req("DELETE", `/papers/${id}`),
  guest: (lang) => req("POST", "/auth/guest", { lang }),
  login: (username, password) => req("POST", "/auth/login", { username, password }),
  register: (username, password, display_name, lang) => req("POST", "/auth/register", { username, password, display_name, lang }),
  me: () => req("GET", "/me"),
  logout: () => req("POST", "/auth/logout"),
  docs: () => req("GET", "/docs"),
  doc: (id) => req("GET", `/docs/${id}`),
  upload: (file) => { const f = new FormData(); f.append("file", file); return req("POST", "/docs", f, true); },
  sample: (key) => req("POST", `/docs/sample/${key}`),
  del: (id) => req("DELETE", `/docs/${id}`),
  rename: (id, title) => req("PUT", `/docs/${id}`, { title }),
  reader: (id) => req("GET", `/docs/${id}/reader`),
  graph: (id, mode) => req("GET", `/docs/${id}/graph?mode=${mode}`),
  progress: (id) => req("GET", `/docs/${id}/progress`),
  history: (id) => req("GET", `/docs/${id}/chat`),
  clearChat: (id) => req("DELETE", `/docs/${id}/chat`),
  chat: (id, message, lang, focus_node, auto_lang = true) => req("POST", `/docs/${id}/chat`, { message, lang, focus_node, auto_lang }),
  // Streams the answer while it is written: onDelta(textSoFar | null=reset). Resolves with the full reply.
  // Falls back to the plain endpoint if streaming is not possible (old proxy, old browser).
  chatStream: async (id, message, lang, focus_node, auto_lang, onDelta) => {
    const headers = { "Content-Type": "application/json" };
    if (token) headers.Authorization = "Bearer " + token;
    let r;
    try { r = await fetch(`/api/docs/${id}/chat/stream`, { method: "POST", headers, body: JSON.stringify({ message, lang, focus_node, auto_lang }) }); }
    catch (e) { return api.chat(id, message, lang, focus_node, auto_lang); }
    if (!r.ok || !r.body) {
      if (r.status === 404 || r.status === 405) return api.chat(id, message, lang, focus_node, auto_lang);
      let data = null; try { data = await r.json(); } catch (e) { /* ignore */ }
      throw new ApiError(typeof data?.detail === "string" ? data.detail : r.statusText, r.status);
    }
    const reader = r.body.getReader(); const dec = new TextDecoder(); let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let k;
      while ((k = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, k); buf = buf.slice(k + 2);
        const ev = /^event: (\w+)/m.exec(block)?.[1]; const dl = /^data: (.*)$/m.exec(block)?.[1];
        if (!ev || dl === undefined) continue;
        const data = JSON.parse(dl);
        if (ev === "delta") onDelta?.(data.text);
        else if (ev === "reset") onDelta?.(null);
        else if (ev === "done") return data;
        else if (ev === "error") throw new ApiError(data.detail, data.status || 500);
      }
    }
    throw new ApiError("The answer was cut off - please ask again.", 0);
  },
  quiz: (id, scope, n, lang) => req("POST", `/docs/${id}/quiz`, { scope, n, lang }),
  answer: (quizId, question_id, choice, lang) => req("POST", `/quiz/${quizId}/answer`, { question_id, choice, lang }),
  mark: (id, node_id, kind) => req("POST", `/docs/${id}/mark`, { node_id, kind }),
  reset: (id) => req("POST", `/docs/${id}/reset`),
  languages: () => req("GET", "/languages"),
  stats: () => req("GET", "/me/stats"),
  translate: (body) => req("POST", "/translate", body),
  asr: (blob, lang = "", docId = "") => { const f = new FormData(); f.append("file", blob, blob.type.includes("ogg") ? "clip.ogg" : blob.type.includes("mp4") ? "clip.m4a" : blob.type.includes("wav") ? "clip.wav" : "clip.webm"); f.append("lang", lang); f.append("doc_id", docId || ""); return req("POST", "/asr", f, true); },
  tts: async (text, lang) => {
    const r = await fetch("/api/tts", { method: "POST", headers: { Authorization: "Bearer " + token, "Content-Type": "application/json" }, body: JSON.stringify({ text, lang }) });
    if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m; } catch (e) { /* ignore */ } throw new ApiError(m, r.status); }
    return r.blob();
  },
  fileBlobUrl: async (id) => {
    const r = await fetch(`/api/docs/${id}/file`, { headers: { Authorization: "Bearer " + token } });
    if (!r.ok) throw new ApiError("Could not load the original file", r.status);
    return URL.createObjectURL(await r.blob());
  },
};
