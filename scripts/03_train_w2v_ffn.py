"""Step 3 - train word2vec on the training tweets, then a feedforward network
with SGD + backpropagation. Early stopping and the decision threshold use the
validation split only."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moderation.config import RESULTS_DIR, SEED, W2V_DIR  # noqa: E402
from moderation.data import load_split  # noqa: E402
from moderation.metrics import best_threshold, classification_metrics  # noqa: E402
from moderation.w2v_ffn import (FeedForward, W2VFFNModel, embed,  # noqa: E402
                                train_word2vec)

EPOCHS, BATCH, LR, MOMENTUM, PATIENCE = 80, 64, 0.02, 0.9, 10


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train, val = load_split("train"), load_split("val")
    tr_tok, va_tok = W2VFFNModel.tokens(train.text), W2VFFNModel.tokens(val.text)

    kv = train_word2vec(tr_tok, seed=SEED)
    print(f"word2vec vocabulary: {len(kv.key_to_index)} words, dim {kv.vector_size}")
    for probe in ["salak", "aptal", "guzel"]:
        if probe in kv.key_to_index:
            print(f"  nearest to '{probe}':", [w for w, _ in kv.most_similar(probe, topn=6)])

    x_tr, x_va = torch.from_numpy(embed(tr_tok, kv)), torch.from_numpy(embed(va_tok, kv))
    y_tr = torch.tensor(train.label.values)

    net = FeedForward()
    net.mu.copy_(x_tr.mean(0))
    net.sd.copy_(x_tr.std(0) + 1e-6)
    # class weights counter the 80/20 imbalance
    weights = torch.tensor([1.0, (y_tr == 0).sum().item() / (y_tr == 1).sum().item()])
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)

    history, best, bad = [], None, 0
    for epoch in range(1, EPOCHS + 1):
        net.train()
        perm = torch.randperm(len(x_tr))
        total = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(net(x_tr[idx]), y_tr[idx])
            loss.backward()          # backpropagation
            opt.step()               # SGD update
            total += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            p_va = torch.softmax(net(x_va), -1)[:, 1].numpy()
        f1 = classification_metrics(val.label, p_va, 0.5)["macro_f1"]
        history.append({"epoch": epoch, "train_loss": total / len(x_tr), "val_macro_f1@0.5": f1})
        print(f"epoch {epoch:2d} loss {total / len(x_tr):.4f} val macro-F1 {f1:.4f}")
        if best is None or f1 > best[0]:
            best, bad = (f1, {k: v.clone() for k, v in net.state_dict().items()}, epoch), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print("early stopping")
                break

    net.load_state_dict(best[1])
    net.eval()
    with torch.no_grad():
        p_va = torch.softmax(net(x_va), -1)[:, 1].numpy()
    thr = best_threshold(val.label, p_va)
    val_m = classification_metrics(val.label, p_va, thr)
    W2VFFNModel(kv, net, thr).save(W2V_DIR)
    report = {"best_epoch": best[2], "threshold": thr, "val_macro_f1": val_m["macro_f1"],
              "vocab": len(kv.key_to_index), "hyperparameters": {
                  "epochs_max": EPOCHS, "batch": BATCH, "lr": LR, "momentum": MOMENTUM,
                  "patience": PATIENCE, "optimizer": "SGD"}, "history": history}
    (RESULTS_DIR / "train_w2v_ffn.json").write_text(json.dumps(report, indent=2))
    print(f"best epoch {best[2]}, threshold {thr:.2f}, val macro-F1 {val_m['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
