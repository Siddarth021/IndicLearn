"""QLoRA fine-tune of Gemma 4 E2B (the offline tutor) on datasets/gemma_sft - runs on a Kaggle GPU.

    python gemma_finetune.py --src <dataset folder> --out /kaggle/working

Steps: 4-bit QLoRA (Unsloth, r=16, 2 epochs, answers-only loss) -> merge -> GGUF q4_k_m for Ollama ->
re-run the 56-doubt benchmark with the tuned model (gemma_ft_e2b.json) so it can be judged against the base.
Output: gemma_te_tutor.Q4_K_M.gguf + Modelfile (install locally with: ollama create indiclearn-tutor -f Modelfile)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def sh(cmd: str) -> None:
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="unsloth/gemma-4-E2B-it")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    sh(f"{sys.executable} -m pip install -q -U unsloth")

    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only

    import torch
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()  # T4 (Kaggle) has no bf16 -> fp16
    model, tok = FastModel.from_pretrained(a.base, max_seq_length=3072, load_in_4bit=True, full_finetuning=False,
                                           dtype=torch.bfloat16 if bf16 else torch.float16)
    model = FastModel.get_peft_model(model, finetune_vision_layers=False, finetune_language_layers=True, finetune_attention_modules=True,
                                     finetune_mlp_modules=True, r=16, lora_alpha=16, lora_dropout=0.0, bias="none", random_state=13)
    for name in ("gemma-4", "gemma4", "gemma-3"):
        try:
            tok = get_chat_template(tok, chat_template=name)
            break
        except Exception as e:  # keep the model's own template
            print("chat template", name, "not available:", e)

    def render(msgs: list[dict]) -> str:
        try:
            return tok.apply_chat_template(msgs, tokenize=False)
        except Exception:  # multimodal processors want content parts
            return tok.apply_chat_template([{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in msgs], tokenize=False)

    def load(name: str) -> Dataset:
        rows = [json.loads(x) for x in (src / "datasets" / "gemma_sft" / f"{name}.jsonl").read_text(encoding="utf-8").split("\n") if x.strip()]
        return Dataset.from_list([{"text": render(r["messages"])} for r in rows])

    # the turn markers, read from the template itself (Gemma 3 and 4 differ)
    probe = render([{"role": "user", "content": "@@U@@"}, {"role": "assistant", "content": "@@A@@"}])
    inst = probe[: probe.index("@@U@@")]
    inst = inst[inst.rfind("<"):]
    resp = probe[probe.index("@@U@@") + 5 : probe.index("@@A@@")]
    resp = resp[resp.rfind("<"):]
    print("turn markers:", repr(inst), repr(resp), flush=True)

    tr, va = load("train"), load("val")
    trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=tr, eval_dataset=va,
                         args=SFTConfig(dataset_text_field="text", per_device_train_batch_size=2, gradient_accumulation_steps=8, num_train_epochs=2,
                                        learning_rate=1e-4, warmup_steps=4, lr_scheduler_type="cosine", logging_steps=10, eval_strategy="steps",
                                        eval_steps=50, optim="adamw_8bit", weight_decay=0.01, bf16=bf16, fp16=not bf16, seed=13, report_to="none", output_dir=str(out / "ckpt"),
                                        save_strategy="no"))
    try:
        trainer = train_on_responses_only(trainer, instruction_part=inst, response_part=resp)
    except Exception as e:  # never lose a run over the masking helper: train on the whole text instead
        print("train_on_responses_only unavailable:", e, flush=True)
    if not bf16:  # T4: the fp16 grad scaler cannot unscale bf16 LoRA weights -> keep the trainable weights in fp32
        for prm in model.parameters():
            if prm.requires_grad and prm.dtype != torch.float32:
                prm.data = prm.data.float()
    try:
        stats = trainer.train()
    except NotImplementedError as e:  # still a bf16 tensor in the scaler: train without the grad scaler
        print("fp16 AMP failed, retrying without it:", e, flush=True)
        trainer.args.fp16 = False
        trainer.accelerator.scaler = None
        stats = trainer.train()
    (out / "gemma_ft_log.json").write_text(json.dumps({"train": stats.metrics, "eval": trainer.evaluate(), "history": trainer.state.log_history}, indent=1))
    model.save_pretrained(str(out / "gemma_lora"))  # the adapter alone, in case the GGUF export fails
    tok.save_pretrained(str(out / "gemma_lora"))
    # the export needs ~21 GB scratch (16-bit merge + f16 GGUF + q4); /kaggle/working is capped at ~20 GB -> use the biggest disk
    import shutil
    scratch = max((Path(d) for d in ("/tmp", "/kaggle/tmp", str(out)) if Path(d).exists()), key=lambda d: shutil.disk_usage(d).free) / "gguf_export"
    print("GGUF scratch:", scratch, round(shutil.disk_usage(scratch.parent).free / 1e9, 1), "GB free", flush=True)
    try:
        model.save_pretrained_gguf(str(scratch), tok, quantization_method="q4_k_m")
    except TypeError:  # newer Unsloth: keyword name changed
        model.save_pretrained_gguf(str(scratch), tokenizer=tok, quantization_method="q4_k_m")
    # Unsloth writes into <dir> or a sibling <dir>_gguf depending on version; skip the vision projector (mmproj)
    found = [f for d in (scratch, Path(str(scratch) + "_gguf")) if d.exists() for f in d.rglob("*.gguf") if "mmproj" not in f.name.lower()]
    print("GGUF files:", found, flush=True)
    gguf = max(found, key=lambda f: ("q4" in f.name.lower(), f.stat().st_size))
    shutil.move(str(gguf), str(out / "gemma_te_tutor.Q4_K_M.gguf"))
    shutil.rmtree(scratch, ignore_errors=True)
    shutil.rmtree(str(scratch) + "_gguf", ignore_errors=True)
    sh(f"rm -rf {out / 'ckpt'}")
    # Ollama: copy the chat template / renderer / stop tokens of the official gemma4:e2b so the tuned model is prompted identically
    sh("apt-get update -qq && apt-get install -y -qq zstd > /dev/null && curl -fsSL https://ollama.com/install.sh | sh")
    subprocess.Popen("ollama serve > /tmp/ollama.log 2>&1", shell=True)
    sh("sleep 8 && ollama pull gemma4:e2b")
    base_mf = subprocess.run("ollama show gemma4:e2b --modelfile", shell=True, capture_output=True, text=True).stdout
    keep, skip = [], False
    for line in base_mf.splitlines():
        head = line.split(" ", 1)[0].upper()
        if head in ("FROM", "LICENSE") or line.startswith("#"):
            skip = head == "LICENSE" and line.count('"""') == 1  # skip a multi-line licence block
            continue
        if skip:
            skip = '"""' not in line
            continue
        if head == "PARAMETER" and line.split()[1] in ("temperature", "num_ctx"):
            continue
        keep.append(line)
    mf = "FROM ./gemma_te_tutor.Q4_K_M.gguf\n" + "\n".join(keep).strip() + "\nPARAMETER temperature 0.3\nPARAMETER num_ctx 8192\n"
    (out / "Modelfile").write_text(mf)
    print("Modelfile:\n" + mf[:1500], flush=True)
    # benchmark the tuned model with exactly the same harness as the base model (never fails the run: the GGUF is already saved)
    r = subprocess.run(f"cd {out} && ollama create indiclearn-tutor -f Modelfile", shell=True)
    if r.returncode == 0:
        subprocess.run(f"{sys.executable} {src}/gemma_bench.py --src {src} --out {out} --models indiclearn-tutor --no-install", shell=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
