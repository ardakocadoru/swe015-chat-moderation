"""Step 4 - fine-tune BERTurk (dbmdz/bert-base-turkish-cased) with a
classification head on the training split. The checkpoint with the best
validation macro-F1 is kept; its decision threshold is tuned on validation.
Runs on a GPU if available (about 5 minutes on an RTX 4070)."""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moderation.config import (BASE_BERT, BERT_FP32_DIR, MAX_LEN,  # noqa: E402
                               RESULTS_DIR, SEED)
from moderation.data import load_split  # noqa: E402
from moderation.metrics import best_threshold, classification_metrics  # noqa: E402
from moderation.normalize import normalize  # noqa: E402

EPOCHS, BATCH, LR, WARMUP, WEIGHT_DECAY = 3, 32, 2e-5, 0.1, 0.01
EVALS_PER_EPOCH = 2


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val = load_split("train"), load_split("val")
    tok = AutoTokenizer.from_pretrained(BASE_BERT)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_BERT, num_labels=2, id2label={0: "NOT", 1: "OFF"}, label2id={"NOT": 0, "OFF": 1}).to(device)

    def collate(batch):
        texts, labels = zip(*batch)
        enc = tok([normalize(t, "light") for t in texts], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(labels)
        return enc

    tr_loader = DataLoader(list(zip(train.text, train.label)), batch_size=BATCH, shuffle=True,
                           collate_fn=collate, generator=torch.Generator().manual_seed(SEED))
    va_loader = DataLoader(list(zip(val.text, val.label)), batch_size=128, collate_fn=collate)

    no_decay = ["bias", "LayerNorm.weight"]
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=LR)
    steps = EPOCHS * len(tr_loader)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    eval_every = math.ceil(len(tr_loader) / EVALS_PER_EPOCH)

    @torch.no_grad()
    def evaluate():
        model.eval()
        probs = []
        for b in va_loader:
            b = {k: v.to(device) for k, v in b.items() if k != "labels"}
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(**b).logits
            probs.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
        model.train()
        return np.concatenate(probs)

    history, best, step, t0 = [], None, 0, time.time()
    model.train()
    for epoch in range(1, EPOCHS + 1):
        for batch in tr_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % eval_every == 0 or step == steps:
                p = evaluate()
                f1 = classification_metrics(val.label, p, 0.5)["macro_f1"]
                history.append({"step": step, "epoch": round(step / len(tr_loader), 2),
                                "train_loss": float(loss.item()), "val_macro_f1@0.5": f1})
                print(f"step {step}/{steps} epoch {step / len(tr_loader):.2f} loss {loss.item():.4f} "
                      f"val macro-F1 {f1:.4f} ({time.time() - t0:.0f}s)", flush=True)
                if best is None or f1 > best[0]:
                    best = (f1, step)
                    model.save_pretrained(BERT_FP32_DIR)
                    tok.save_pretrained(BERT_FP32_DIR)

    # Reload the best checkpoint and tune the decision threshold on validation.
    model = AutoModelForSequenceClassification.from_pretrained(BERT_FP32_DIR).to(device)
    p = evaluate()
    thr = best_threshold(val.label, p)
    val_m = classification_metrics(val.label, p, thr)
    (BERT_FP32_DIR / "meta.json").write_text(json.dumps({"threshold": thr, "max_len": MAX_LEN}))
    report = {"base_model": BASE_BERT, "best_step": best[1], "threshold": thr,
              "val_macro_f1": val_m["macro_f1"], "train_minutes": round((time.time() - t0) / 60, 1),
              "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
              "hyperparameters": {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "warmup": WARMUP,
                                  "weight_decay": WEIGHT_DECAY, "max_len": MAX_LEN,
                                  "optimizer": "AdamW", "precision": "bf16 autocast"},
              "history": history}
    (RESULTS_DIR / "train_bert.json").write_text(json.dumps(report, indent=2))
    print(f"best step {best[1]}, threshold {thr:.2f}, val macro-F1 {val_m['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
