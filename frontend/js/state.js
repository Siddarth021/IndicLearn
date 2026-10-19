// Shared client state + a tiny event bus (handlers can be owned so a re-render replaces, not stacks, them).
export const S = { user: null, info: null, docs: [], doc: null, graph: null, nodes: new Map(), view: "library", focus: null, tutorOpen: !(typeof window !== "undefined" && window.matchMedia && window.matchMedia("(max-width: 980px)").matches) };
let handlers = [];
export const on = (ev, fn, owner = "app") => { handlers.push({ ev, fn, owner }); };
export const clearOwner = (owner) => { handlers = handlers.filter((h) => h.owner !== owner); };
export const emit = (ev, data) => handlers.filter((h) => h.ev === ev).forEach((h) => h.fn(data));
