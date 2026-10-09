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
  reader: (id) => req("GET", `/docs/${id}/reader`),
  graph: (id, mode) => req("GET", `/docs/${id}/graph?mode=${mode}`),
  progress: (id) => req("GET", `/docs/${id}/progress`),
  history: (id) => req("GET", `/docs/${id}/chat`),
  clearChat: (id) => req("DELETE", `/docs/${id}/chat`),
  chat: (id, message, lang, focus_node, auto_lang = true) => req("POST", `/docs/${id}/chat`, { message, lang, focus_node, auto_lang }),
  quiz: (id, scope, n, lang) => req("POST", `/docs/${id}/quiz`, { scope, n, lang }),
  answer: (quizId, question_id, choice, lang) => req("POST", `/quiz/${quizId}/answer`, { question_id, choice, lang }),
  mark: (id, node_id, kind) => req("POST", `/docs/${id}/mark`, { node_id, kind }),
  reset: (id) => req("POST", `/docs/${id}/reset`),
  fileBlobUrl: async (id) => {
    const r = await fetch(`/api/docs/${id}/file`, { headers: { Authorization: "Bearer " + token } });
    if (!r.ok) throw new ApiError("Could not load the original file", r.status);
    return URL.createObjectURL(await r.blob());
  },
};
