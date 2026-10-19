"""Step 5 - fine-tune M1-M4 (runs on a Kaggle GPU via kaggle_run.py, or locally if you have a GPU).

    python training/train.py --model m1          # concept filter       MuRIL  (sequence pair classifier)
    python training/train.py --model m3          # relation classifier  MuRIL  (prereq / related / none)
    python training/train.py --model m2          # description writer   Flan-T5-base
    python training/train.py --model m4          # quiz generator       Flan-T5-base
    python training/train.py --model all

Writes   <out>/m*/          the final model only (no checkpoints) + indiclearn.json (metadata, threshold)
         <out>/reports/m*.json   test metrics + sample outputs, compared with the rule-based baseline

Lessons from the first attempt (reports/phase2): mT5 in fp16 collapsed to "<extra_id_0>" (T5 overflows in
fp16 -> train T5 in fp32); M3 learnt "always none" from 1:140 imbalanced data (balanced data + class
weights here); M4's newline format could never be produced by a T5 tokenizer (one-line format now).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SEED = 13
# MuRIL was tried first: its [CLS]/pooler output is almost identical for every input in current transformers, so a
# classifier on top could not learn (M1/M3 collapsed to one class). XLM-RoBERTa (100 languages incl. Telugu) trains fine.
BASES = {"m1": "FacebookAI/xlm-roberta-base", "m3": "FacebookAI/xlm-roberta-base", "m2": "google/flan-t5-base", "m4": "google/flan-t5-base"}
BASES = {k: os.environ.get(f"TRAIN_BASE_{k.upper()}", v) for k, v in BASES.items()}  # override for experiments / smoke tests
M3_LABELS = ["none", "prereq", "related"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = torch.float16 if DEV == "cuda" else torch.bfloat16


def seed_all(s: int = SEED) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").split("\n") if x.strip()]


def data(ds: Path, m: str) -> tuple[list[dict], list[dict], list[dict]]:
    return tuple(jl(ds / m / f"{s}.jsonl") for s in ("train", "val", "test"))  # type: ignore[return-value]


# ============================================================================ classifiers (M1, M3)
def train_classifier(m: str, ds: Path, out: Path, epochs: int, bs: int, lr: float) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    tr, va, te = data(ds, m)
    labels = ["0", "1"] if m == "m1" else M3_LABELS
    lid = {l: i for i, l in enumerate(labels)}

    def enc_rows(rows):
        if m == "m1":
            return [(r["span"], r["context"][:400] or r["span"]) for r in rows], [lid[str(r["label"])] for r in rows]
        return [(f"{r['a']} [SEP] {r['b']}", r["context"][:300] or "-") for r in rows], [lid[r["label"]] for r in rows]

    tok = AutoTokenizer.from_pretrained(BASES[m])
    model = AutoModelForSequenceClassification.from_pretrained(BASES[m], num_labels=len(labels),
                                                               id2label=dict(enumerate(labels)), label2id=lid, ignore_mismatched_sizes=True).to(DEV)
    xtr, ytr = enc_rows(tr)
    counts = np.bincount(ytr, minlength=len(labels)).astype(float)
    weights = torch.tensor((counts.sum() / (len(labels) * np.maximum(counts, 1))) ** 0.5, dtype=torch.float).to(DEV)  # softened inverse frequency
    lossf = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(xtr) / bs)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda")

    def batches(x, y, shuffle):
        idx = list(range(len(x)))
        if shuffle:
            random.shuffle(idx)
        for i in range(0, len(idx), bs):
            j = idx[i : i + bs]
            e = tok([x[k][0] for k in j], [x[k][1] for k in j], truncation=True, max_length=128, padding=True, return_tensors="pt")
            yield {k: v.to(DEV) for k, v in e.items()}, torch.tensor([y[k] for k in j]).to(DEV)

    def predict(rows):
        x, y = enc_rows(rows)
        model.eval()
        ps = []
        with torch.no_grad():
            for e, _ in batches(x, y, False):
                with torch.autocast(DEV, dtype=AMP, enabled=DEV == "cuda"):
                    ps.append(torch.softmax(model(**e).logits.float(), -1).cpu().numpy())
        model.train()
        return (np.concatenate(ps) if ps else np.zeros((0, len(labels)))), np.array(y)

    best, best_state, log = -1.0, None, []
    for ep in range(epochs):
        t0, tot = time.time(), 0.0
        for e, y in batches(xtr, ytr, True):
            with torch.autocast(DEV, dtype=AMP, enabled=DEV == "cuda"):
                loss = lossf(model(**e).logits.float(), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            tot += float(loss)
        p, y = predict(va)
        f1 = macro_f1(y, p.argmax(-1), len(labels))
        log.append({"epoch": ep + 1, "train_loss": round(tot / max(1, math.ceil(len(xtr) / bs)), 4), "val_macro_f1": round(f1, 4), "secs": round(time.time() - t0)})
        print(m, log[-1], flush=True)
        if f1 > best:
            best, best_state = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    meta: dict = {"model": m, "base": BASES[m], "labels": labels, "trained": time.strftime("%Y-%m-%d"), "train_rows": len(tr), "log": log}
    report: dict = {"model": m, "n_test": len(te)}
    if m == "m1":
        pv, yv = predict(va)
        # tuned on val (never on test) for F0.5: a junk node on the map annoys more than a missing one
        th = max(np.arange(0.2, 0.86, 0.05), key=lambda t: fbeta(yv, pv[:, 1] >= t, 0.5))
        meta["threshold"] = round(float(th), 2)
        pt, yt = predict(te)
        report["model_metrics"] = prf(yt, pt[:, 1] >= th)
        report["baseline_rules_keep_all"] = prf(yt, np.ones_like(yt, dtype=bool))
        report["threshold"] = meta["threshold"]
        report["samples"] = [{"span": r["span"], "gold": r["label"], "p_concept": round(float(s), 3)} for r, s in list(zip(te, pt[:, 1]))[:40]]
    else:
        pt, yt = predict(te)
        pred = pt.argmax(-1)
        report["model_macro_f1"] = round(macro_f1(yt, pred, len(labels)), 4)
        report["per_class"] = {labels[c]: prf(yt == c, pred == c) for c in range(len(labels))}
        maj = np.bincount(yt, minlength=len(labels)).argmax() if len(yt) else 0
        report["baseline_majority_macro_f1"] = round(macro_f1(yt, np.full_like(yt, maj), len(labels)), 4)
        # the app's old rule: every co-occurring pair is "related"
        report["baseline_cooccurrence_macro_f1"] = round(macro_f1(yt, np.full_like(yt, labels.index("related")), len(labels)), 4)
    save(model, tok, out / m, meta)
    return report


def macro_f1(y, p, k) -> float:
    return float(np.mean([f1_bin(y == c, p == c) for c in range(k)])) if len(y) else 0.0


def f1_bin(y, p) -> float:
    y, p = np.asarray(y).astype(bool), np.asarray(p).astype(bool)
    tp = (y & p).sum()
    return float(2 * tp / max(1, (2 * tp + (~y & p).sum() + (y & ~p).sum())))


def fbeta(y, p, b: float) -> float:
    y, p = np.asarray(y).astype(bool), np.asarray(p).astype(bool)
    tp = (y & p).sum()
    pr, rc = tp / max(1, p.sum()), tp / max(1, y.sum())
    return float((1 + b * b) * pr * rc / max(1e-9, b * b * pr + rc))


def prf(y, p) -> dict:
    y, p = np.asarray(y).astype(bool), np.asarray(p).astype(bool)
    tp, fp, fn = (y & p).sum(), (~y & p).sum(), (y & ~p).sum()
    pr, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {"precision": round(float(pr), 4), "recall": round(float(rc), 4), "f1": round(float(2 * pr * rc / max(1e-9, pr + rc)), 4),
            "accuracy": round(float((y == p).mean()), 4) if len(y) else 0.0}


# ============================================================================ seq2seq (M2, M4)
def train_seq2seq(m: str, ds: Path, out: Path, epochs: int, bs: int, lr: float) -> dict:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

    tr, va, te = data(ds, m)
    tok = AutoTokenizer.from_pretrained(BASES[m])
    model = AutoModelForSeq2SeqLM.from_pretrained(BASES[m], torch_dtype=torch.float32).to(DEV)  # fp32: T5 overflows in fp16
    max_in, max_out = (192, 64) if m == "m2" else (384, 160)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    accum = 2
    steps = epochs * math.ceil(len(tr) / (bs * accum))
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)

    def batches(rows, shuffle):
        idx = list(range(len(rows)))
        if shuffle:
            random.shuffle(idx)
        for i in range(0, len(idx), bs):
            j = [rows[k] for k in idx[i : i + bs]]
            e = tok([r["input"] for r in j], truncation=True, max_length=max_in, padding=True, return_tensors="pt")
            lab = tok([r["target"] for r in j], truncation=True, max_length=max_out, padding=True, return_tensors="pt").input_ids
            lab[lab == tok.pad_token_id] = -100
            yield {k: v.to(DEV) for k, v in e.items()}, lab.to(DEV)

    def val_loss() -> float:
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for e, lab in batches(va[:1500], False):
                tot += float(model(**e, labels=lab).loss)
                n += 1
        model.train()
        return tot / max(1, n)

    best, best_state, log = 1e9, None, []
    model.train()
    for ep in range(epochs):
        t0, tot, k = time.time(), 0.0, 0
        for i, (e, lab) in enumerate(batches(tr, True)):
            loss = model(**e, labels=lab).loss / accum
            loss.backward()
            tot += float(loss) * accum
            k += 1
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sch.step()
                opt.zero_grad()
        vl = val_loss()
        log.append({"epoch": ep + 1, "train_loss": round(tot / max(1, k), 4), "val_loss": round(vl, 4), "secs": round(time.time() - t0)})
        print(m, log[-1], flush=True)
        if vl < best:
            best, best_state = vl, {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()

    def gen(rows):
        outs = []
        with torch.no_grad():
            for i in range(0, len(rows), 16):
                e = tok([r["input"] for r in rows[i : i + 16]], truncation=True, max_length=max_in, padding=True, return_tensors="pt")
                g = model.generate(**{k: v.to(DEV) for k, v in e.items()}, max_new_tokens=max_out, num_beams=4, no_repeat_ngram_size=3)
                outs += tok.batch_decode(g, skip_special_tokens=True)
        return outs

    sample = te[:300] if m == "m2" else [r for r in te if r["src"] != "sciq"][:150] + [r for r in te if r["src"] == "sciq"][:150]
    preds = gen(sample)
    report: dict = {"model": m, "n_test_scored": len(sample)}
    if m == "m2":
        report["rougeL_f"] = round(float(np.mean([rouge_l(p, r["target"]) for p, r in zip(preds, sample)])), 4)
        report["echo_rate"] = round(float(np.mean([len(p) < 25 for p in preds])), 4)  # too short to be a description
        report["baseline_copy_context_rougeL_f"] = round(float(np.mean([rouge_l(r["input"].split("| context:", 1)[-1], r["target"]) for r in sample])), 4)
    else:
        parsed = [parse_mcq(p) for p in preds]
        report["well_formed"] = round(float(np.mean([x is not None for x in parsed])), 4)
        report["answer_matches_gold"] = round(float(np.mean([bool(x and parse_mcq(r["target"]) and x[1][x[2]].lower() == parse_mcq(r["target"])[1][parse_mcq(r["target"])[2]].lower())
                                                             for x, r in zip(parsed, sample)])), 4)
    report["samples"] = [{"input": r["input"][:200], "gold": r["target"], "model": p} for r, p in list(zip(sample, preds))[:40]]
    save(model, tok, out / m, {"model": m, "base": BASES[m], "trained": time.strftime("%Y-%m-%d"), "train_rows": len(tr), "log": log,
                               "prompt": "define: <term> | context: <sentence>" if m == "m2" else "question: <concept> | passage: <text>"})
    return report


_MCQ = re.compile(r"Q:\s*(?P<q>.+?)\s+A\)\s*(?P<a>.+?)\s+B\)\s*(?P<b>.+?)\s+C\)\s*(?P<c>.+?)\s+D\)\s*(?P<d>.+?)\s+ANSWER:\s*(?P<ans>[A-D])", re.S | re.I)


def parse_mcq(t: str):
    m = _MCQ.search(t or "")
    if not m:
        return None
    opts = [m.group(x).strip() for x in "abcd"]
    if len({o.lower() for o in opts}) < 4 or len(m.group("q")) < 10:
        return None
    return m.group("q").strip(), opts, "ABCD".index(m.group("ans").upper())


def rouge_l(a: str, b: str) -> float:
    x, y = a.lower().split(), b.lower().split()
    if not x or not y:
        return 0.0
    dp = [[0] * (len(y) + 1) for _ in range(len(x) + 1)]
    for i in range(len(x)):
        for j in range(len(y)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if x[i] == y[j] else max(dp[i][j + 1], dp[i + 1][j])
    l = dp[-1][-1]
    p, r = l / len(x), l / len(y)
    return 2 * p * r / max(1e-9, p + r)


def save(model, tok, folder: Path, meta: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder, safe_serialization=True)
    tok.save_pretrained(folder)
    (folder / "indiclearn.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", help="all, or a comma list such as m1,m3")
    ap.add_argument("--data", default=str(HERE / "datasets"))
    ap.add_argument("--out", default=os.environ.get("TRAIN_OUT", str(HERE.parent / "models" / "trained")))
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run to check the code")
    a = ap.parse_args()
    ds, out = Path(a.data), Path(a.out)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    plan = {"m1": (train_classifier, 5, 32, 2e-5), "m3": (train_classifier, 5, 32, 2e-5),
            "m2": (train_seq2seq, 6, 16, 3e-4), "m4": (train_seq2seq, 3, 8, 3e-4)}
    for m in (["m1", "m3", "m2", "m4"] if a.model == "all" else [x.strip() for x in a.model.split(",")]):
        if not (ds / m / "train.jsonl").exists():
            print("skip", m, "(no data)")
            continue
        seed_all()
        fn, ep, bs, lr = plan[m]
        if a.smoke:
            ep = 1
        t0 = time.time()
        rep = fn(m, ds, out, ep, bs, lr)
        rep["minutes"] = round((time.time() - t0) / 60, 1)
        (out / "reports" / f"{m}.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
        print("==", m, {k: v for k, v in rep.items() if k != "samples"}, flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
