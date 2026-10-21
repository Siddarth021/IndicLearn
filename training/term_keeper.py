"""Contribution 3: Telugu technical-term retention model.
Learns WHICH English words a Telugu lecturer keeps in English vs translates.
Dataset is built automatically from the te_style corpus: an English source word is labelled KEEP
if it appears verbatim (Latin script) in the code-mixed Telugu target, else TRANSLATE.
    python training/term_keeper.py
Outputs: training/datasets/term_keeper/{train,test}.jsonl, models/trained/te_term_keeper/model.joblib,
glossary.json, training/reports/term_keeper.json"""
import json, re, joblib
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import classification_report, f1_score
ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "training" / "datasets"
STOP = set("a an the of to in on for and or is are was be by with as at from that this it its if then than into not can".split())

def rows(split):
    out = []
    for t in ("te_style",):
        f = D / t / f"{split}.jsonl"
        for l in f.open(encoding="utf-8"):
            r = json.loads(l); en = r.get("target_en") or r["input"].replace("to Telugu: ", "")
            kept = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]+", r["target"])}
            toks = re.findall(r"[A-Za-z][A-Za-z\-]+", en)
            for i, w in enumerate(toks):
                if w.lower() in STOP or len(w) < 3: continue
                ctx = " ".join(toks[max(0, i-2):i+3]).lower()
                out.append({"word": w, "ctx": ctx, "label": int(w.lower() in kept)})
    return out

tr, te = rows("train") + rows("val"), rows("test")
(D / "term_keeper").mkdir(exist_ok=True)
for n, rs in (("train", tr), ("test", te)):
    (D / "term_keeper" / f"{n}.jsonl").write_text("\n".join(json.dumps(r) for r in rs), encoding="utf-8")
X = lambda rs: [f"{r['word']} || {r['ctx']}" for r in rs]
feat = make_union(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2),
                  TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2))
clf = make_pipeline(feat, LogisticRegression(max_iter=2000, C=4, class_weight="balanced"))
clf.fit(X(tr), [r["label"] for r in tr])
pred = clf.predict(X(te)); gold = [r["label"] for r in te]
# baselines: always-keep, and "keep if capitalised/acronym"
f_model = f1_score(gold, pred, average="macro")
f_keep = f1_score(gold, [1]*len(gold), average="macro")
f_cap = f1_score(gold, [int(r["word"][0].isupper() or r["word"].isupper()) for r in te], average="macro")
c = Counter(); k = Counter()
for r in tr + te: c[r["word"].lower()] += 1; k[r["word"].lower()] += r["label"]
gloss = {w: round(k[w]/c[w], 2) for w in c if c[w] >= 3}
out = ROOT / "models" / "trained" / "te_term_keeper"; out.mkdir(parents=True, exist_ok=True)
joblib.dump(clf, out / "model.joblib")
(out / "glossary.json").write_text(json.dumps(dict(sorted(gloss.items(), key=lambda x: -c[x[0]])), indent=1), encoding="utf-8")
rep = {"task": "te_term_keeper", "train_words": len(tr), "test_words": len(te),
       "keep_rate": round(sum(gold)/len(gold), 3), "macroF1_model": round(f_model, 3),
       "macroF1_always_keep": round(f_keep, 3), "macroF1_capitalised_rule": round(f_cap, 3),
       "glossary_terms": len(gloss), "report": classification_report(gold, pred, output_dict=True)}
(ROOT / "training" / "reports" / "term_keeper.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
print({k: v for k, v in rep.items() if k != "report"})
for w in ["stack", "process", "memory", "deadlock", "true", "rule", "function", "values"]:
    print(f"  {w:10} keep-prob {clf.predict_proba([f'{w} || {w}'])[0][1]:.2f}")
