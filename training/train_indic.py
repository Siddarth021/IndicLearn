"""Multilingual students: Telugu definitions (M2-te) and Telugu quizzes (M4-te).

Same teacher-student recipe as train.py, but on multilingual bases that can write Telugu:
    google/mt5-base        (580M, 101 languages incl. Telugu)
    ai4bharat/IndicBARTSS  (244M, 11 Indic languages + English, native scripts)
English notes in, Telugu out (technical terms stay in English letters, as lecturers say them).

Baseline: the old route "English answer -> machine translation" with NLLB-200 (600M), applied to the
teacher's *gold* English targets - an upper bound for any English-model + translator pipeline.

    python training/train_indic.py --data training/datasets --out models/trained [--bases mt5,indicbart] [--tasks m2_te,m4_te] [--smoke]
Writes reports/<task>__<base>.json, reports/<task>__nllb_baseline.json and the best model per task to <out>/<task>/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 13
BASES = {"mt5": "google/mt5-base", "indicbart": "ai4bharat/IndicBARTSS"}
TE = re.compile(r"[\u0C00-\u0C7F]")
MCQ = re.compile(r"ప్రశ్న:\s*(?P<q>.+?)\s+A\)\s*(?P<a>.+?)\s+B\)\s*(?P<b>.+?)\s+C\)\s*(?P<c>.+?)\s+D\)\s*(?P<d>.+?)\s+జవాబు:\s*(?P<ans>[A-D])", re.S)


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def te_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    return sum(1 for c in letters if TE.match(c)) / max(1, len(letters))


def chrf(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (chrF, Popovic 2015) - the standard metric for Indic generation."""
    h, r = hyp.replace(" ", ""), ref.replace(" ", "")
    ps, rs = [], []
    for k in range(1, n + 1):
        hc, rc = Counter(h[i:i + k] for i in range(len(h) - k + 1)), Counter(r[i:i + k] for i in range(len(r) - k + 1))
        if not hc or not rc:
            continue
        m = sum((hc & rc).values())
        ps.append(m / sum(hc.values()))
        rs.append(m / sum(rc.values()))
    if not ps:
        return 0.0
    p, rr = sum(ps) / len(ps), sum(rs) / len(rs)
    return 0.0 if p + rr == 0 else 100 * (1 + beta ** 2) * p * rr / (beta ** 2 * p + rr)


def parse_mcq(t: str):
    m = MCQ.search(t or "")
    if not m:
        return None
    opts = [m.group(x).strip() for x in "abcd"]
    if len({o for o in opts}) < 4 or len(m.group("q")) < 8:
        return None
    return m.group("q").strip(), opts, "ABCD".index(m.group("ans"))


class Codec:
    """Model-specific input/target formatting (IndicBART needs language tags)."""

    def __init__(self, base: str):
        from transformers import AutoTokenizer

        self.ib = "IndicBART" in base
        if self.ib:
            from transformers import AlbertTokenizer  # IndicBART ships a sentencepiece model (needs `sentencepiece`)

            self.tok = AlbertTokenizer.from_pretrained(base, do_lower_case=False, keep_accents=True)
            self.bos, self.eos, self.pad, self.te_tag = self.tok.convert_tokens_to_ids(["<s>", "</s>", "<pad>", "<2te>"])
        else:
            self.tok = AutoTokenizer.from_pretrained(base)
            self.pad = self.tok.pad_token_id

    def src(self, xs: list[str]) -> list[str]:
        return [x + " </s> <2en>" for x in xs] if self.ib else xs

    def enc(self, xs, max_len):
        return self.tok(self.src(xs), truncation=True, max_length=max_len, padding=True, return_tensors="pt", add_special_tokens=not self.ib)

    def labels(self, ys, max_len):
        if self.ib:
            ids = self.tok([y + " </s>" for y in ys], truncation=True, max_length=max_len - 1, padding=False, add_special_tokens=False).input_ids
            ids = [x[: max_len - 1] for x in ids]
            n = max(len(x) for x in ids)
            dec = torch.tensor([[self.te_tag] + x + [self.pad] * (n - len(x)) for x in ids])  # decoder input starts with <2te>
            lab = torch.tensor([x + [-100] * (n + 1 - len(x)) for x in ids])
            return dec, lab
        lab = self.tok(ys, truncation=True, max_length=max_len, padding=True, return_tensors="pt").input_ids
        lab[lab == self.pad] = -100
        return None, lab

    def gen_kwargs(self):
        if self.ib:
            return {"decoder_start_token_id": self.te_tag, "pad_token_id": self.pad, "bos_token_id": self.bos, "eos_token_id": self.eos}
        return {}

    def decode(self, g):
        out = self.tok.batch_decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [re.sub(r"\s+", " ", re.sub(r"</?s>|<pad>|<2[a-z]{2}>", " ", x)).strip() for x in out]


def train_one(task: str, bkey: str, ds: Path, out: Path, smoke: bool) -> dict:
    from transformers import AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
    from transformers.optimization import Adafactor

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    base = BASES[bkey]
    tr, va, te = (jl(ds / task / f"{s}.jsonl") for s in ("train", "val", "test"))
    if smoke:
        tr, va, te = tr[:16], va[:8], te[:8]
    c = Codec(base)
    model = AutoModelForSeq2SeqLM.from_pretrained(base, torch_dtype=torch.float32).to(DEV)
    if DEV == "cuda" and bkey == "mt5":
        model.gradient_checkpointing_enable()
    max_in, max_out = {"m2_te": (192, 96), "m4_te": (384, 256), "te_style": (256, 256), "te_plus": (256, 256)}[task]
    epochs = 1 if smoke else {"m2_te": 10, "m4_te": 6, "te_style": 8, "te_plus": 6}[task]
    bs, accum = (4, 4) if task == "m4_te" else (8, 2)
    if bkey == "mt5":  # mT5 trains best with Adafactor at a high constant-ish rate
        opt = Adafactor(model.parameters(), lr=5e-4, relative_step=False, scale_parameter=False, warmup_init=False)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    steps = epochs * math.ceil(len(tr) / (bs * accum))
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)

    def batches(rows, shuffle):
        idx = list(range(len(rows)))
        if shuffle:
            random.shuffle(idx)
        for i in range(0, len(idx), bs):
            j = [rows[k] for k in idx[i:i + bs]]
            e = c.enc([r["input"] for r in j], max_in)
            dec, lab = c.labels([r["target"] for r in j], max_out)
            kw = {"input_ids": e["input_ids"].to(DEV), "attention_mask": e["attention_mask"].to(DEV), "labels": lab.to(DEV)}
            if dec is not None:
                kw["decoder_input_ids"] = dec.to(DEV)
            yield kw

    def val_loss():
        model.eval(); tot = n = 0
        with torch.no_grad():
            for kw in batches(va, False):
                tot += float(model(**kw).loss); n += 1
        model.train()
        return tot / max(1, n)

    best, best_state, log, bad = 1e9, None, [], 0
    model.train()
    for ep in range(epochs):
        t0, tot, k = time.time(), 0.0, 0
        for i, kw in enumerate(batches(tr, True)):
            loss = model(**kw).loss
            if not torch.isfinite(loss):
                opt.zero_grad(); continue
            (loss / accum).backward(); tot += float(loss); k += 1
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad()
        vl = val_loss()
        log.append({"epoch": ep + 1, "train_loss": round(tot / max(1, k), 4), "val_loss": round(vl, 4), "secs": round(time.time() - t0)})
        print(task, bkey, log[-1], flush=True)
        if vl < best:
            best, bad, best_state = vl, 0, {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 2:
                break
    model.load_state_dict(best_state); model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(te), 16):
            e = c.enc([r["input"] for r in te[i:i + 16]], max_in)
            g = model.generate(input_ids=e["input_ids"].to(DEV), attention_mask=e["attention_mask"].to(DEV), max_new_tokens=max_out,
                               num_beams=4, no_repeat_ngram_size=4, early_stopping=True, **c.gen_kwargs())
            preds += c.decode(g)
    rep = score(task, te, preds)
    rep.update({"task": task, "base": base, "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6), "train_rows": len(tr),
                "best_val_loss": round(best, 4), "log": log})
    folder = out / f"{task}__{bkey}"
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder, safe_serialization=True)
    c.tok.save_pretrained(folder)
    (folder / "indiclearn.json").write_text(json.dumps({"task": task, "base": base, "trained": time.strftime("%Y-%m-%d"), "train_rows": len(tr),
                                                        "prompt": "define in Telugu: <term> | context: <english sentence>" if task == "m2_te"
                                                        else "quiz in Telugu: <concept> | passage: <english text>", "log": log}, indent=2), encoding="utf-8")
    del model; torch.cuda.empty_cache()
    return rep


def score(task: str, rows: list[dict], preds: list[str]) -> dict:
    rep: dict = {"n_test": len(rows)}
    rep["chrF"] = round(float(np.mean([chrf(p, r["target"]) for p, r in zip(preds, rows)])), 2)
    rep["telugu_script_ratio"] = round(float(np.mean([te_ratio(p) for p in preds])), 3)
    rep["empty_or_short"] = round(float(np.mean([len(p) < 15 for p in preds])), 3)
    mrows = [(r, p) for r, p in zip(rows, preds) if "జవాబు:" in r["target"]]
    if mrows:
        parsed = [parse_mcq(p) for _, p in mrows]
        rep["well_formed_mcq"] = round(float(np.mean([x is not None for x in parsed])), 3)
        gold = [parse_mcq(r["target"]) for r, _ in mrows]
        rep["answer_option_matches_gold"] = round(float(np.mean([bool(x and g and chrf(x[1][x[2]], g[1][g[2]]) > 60) for x, g in zip(parsed, gold)])), 3)
    rep["samples"] = [{"input": r["input"][:300], "gold_te": r["target"], "gold_en": r.get("target_en", ""), "model": p} for r, p in zip(rows, preds)]
    return rep


def nllb_baseline(task: str, ds: Path, smoke: bool) -> dict:
    """English gold answer -> NLLB-200 -> Telugu (the translate-afterwards route)."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    rows = jl(ds / task / "test.jsonl")[: 8 if smoke else None]
    name = "facebook/nllb-200-distilled-600M"
    tok = AutoTokenizer.from_pretrained(name, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(name).to(DEV).eval()

    def tr(xs):
        out = []
        with torch.no_grad():
            for i in range(0, len(xs), 16):
                e = tok(xs[i:i + 16], return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEV)
                g = model.generate(**e, forced_bos_token_id=tok.convert_tokens_to_ids("tel_Telu"), max_new_tokens=256, num_beams=4)
                out += tok.batch_decode(g, skip_special_tokens=True)
        return out

    if task == "m2_te":
        preds = tr([r["target_en"] for r in rows])
    else:  # translate question and options separately, keep the structure
        pat = re.compile(r"Q:\s*(.*?)\s+A\)\s*(.*?)\s+B\)\s*(.*?)\s+C\)\s*(.*?)\s+D\)\s*(.*?)\s+ANSWER:\s*([ABCD])", re.S)
        parts = [pat.search(r["target_en"]).groups() for r in rows]
        flat = tr([x for p in parts for x in p[:5]])
        preds = [f"ప్రశ్న: {flat[5*i]} A) {flat[5*i+1]} B) {flat[5*i+2]} C) {flat[5*i+3]} D) {flat[5*i+4]} జవాబు: {p[5]}" for i, p in enumerate(parts)]
    rep = score(task, rows, preds)
    rep.update({"task": task, "base": name + " (gold English -> Telugu)"})
    del model; torch.cuda.empty_cache()
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent / "datasets"))
    ap.add_argument("--out", default="models/trained")
    ap.add_argument("--bases", default="mt5,indicbart")
    ap.add_argument("--tasks", default="m2_te,m4_te")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    ds, out = Path(a.data), Path(a.out)
    rdir = out / "reports"
    rdir.mkdir(parents=True, exist_ok=True)
    for task in a.tasks.split(","):
        reps = {}
        for b in a.bases.split(","):
            t0 = time.time()
            try:
                rep = train_one(task, b, ds, out, a.smoke)
            except Exception as e:  # one base failing must not lose the other
                import traceback
                traceback.print_exc()
                rep = {"task": task, "base": BASES[b], "error": str(e)[:500]}
            rep["minutes"] = round((time.time() - t0) / 60, 1)
            reps[b] = rep
            (rdir / f"{task}__{b}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
            print("==", task, b, {k: v for k, v in rep.items() if k not in ("samples", "log")}, flush=True)
        try:
            if task in ("te_style", "te_plus"):
                raise RuntimeError("baseline = the m2_te / m4_te NLLB runs (same test items)")
            bl = nllb_baseline(task, ds, a.smoke)
            (rdir / f"{task}__nllb_baseline.json").write_text(json.dumps(bl, indent=1, ensure_ascii=False), encoding="utf-8")
            print("== baseline", task, {k: v for k, v in bl.items() if k != "samples"}, flush=True)
        except Exception as e:
            print("baseline failed", e, flush=True)
        # keep only the best model per task (by chrF) to save space
        ok = {b: r for b, r in reps.items() if "chrF" in r}
        if ok:
            best = max(ok, key=lambda b: ok[b]["chrF"])
            for b in reps:
                f = out / f"{task}__{b}"
                if b == best and f.exists():
                    if (out / task).exists():
                        shutil.rmtree(out / task)
                    f.rename(out / task)
                elif f.exists():
                    shutil.rmtree(f)
            print("best", task, best, flush=True)


if __name__ == "__main__":
    main()
