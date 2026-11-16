"""Step 3b - 1D CNN over our own word2vec vectors (from step 3), trained with
SGD + backpropagation. Early stopping and threshold use the validation split."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moderation.config import RESULTS_DIR, SEED, W2V_CNN_DIR, W2V_DIR  # noqa: E402
from moderation.data import load_split  # noqa: E402
from moderation.metrics import best_threshold, classification_metrics  # noqa: E402
from moderation.w2v_cnn import (TextCNN, W2VCNNModel, build_vocab,  # noqa: E402
                                encode)
from moderation.w2v_ffn import W2VFFNModel  # noqa: E402

EPOCHS, BATCH, LR, MOMENTUM, PATIENCE = 30, 64, 0.02, 0.9, 6


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val = load_split("train"), load_split("val")
    kv = W2VFFNModel.load(W2V_DIR).kv                     # our own word2vec from step 3
    vocab, matrix = build_vocab(kv)
    x_tr = encode(W2VCNNModel.tokens(train.text), vocab)
    x_va = encode(W2VCNNModel.tokens(val.text), vocab)
    y_tr = torch.tensor(train.label.values)

    net = TextCNN(len(vocab) + 1)
    net.emb.weight.data.copy_(torch.from_numpy(matrix))   # initialise with word2vec, then fine-tune
    net.to(device)
    weights = torch.tensor([1.0, (y_tr == 0).sum().item() / (y_tr == 1).sum().item()]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)

    def val_probs():
        net.eval()
        with torch.no_grad():
            return torch.softmax(net(x_va.to(device)), -1)[:, 1].cpu().numpy()

    history, best, bad = [], None, 0
    for epoch in range(1, EPOCHS + 1):
        net.train()
        perm = torch.randperm(len(x_tr))
        total = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(net(x_tr[idx].to(device)), y_tr[idx].to(device))
            loss.backward()          # backpropagation
            opt.step()               # SGD update
            total += loss.item() * len(idx)
        f1 = classification_metrics(val.label, val_probs(), 0.5)["macro_f1"]
        history.append({"epoch": epoch, "train_loss": total / len(x_tr), "val_macro_f1@0.5": f1})
        print(f"epoch {epoch:2d} loss {total / len(x_tr):.4f} val macro-F1 {f1:.4f}", flush=True)
        if best is None or f1 > best[0]:
            best, bad = (f1, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, epoch), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print("early stopping")
                break

    net.load_state_dict(best[1])
    p_va = val_probs()
    thr = best_threshold(val.label, p_va)
    val_m = classification_metrics(val.label, p_va, thr)
    W2VCNNModel(vocab, net.cpu(), thr).save(W2V_CNN_DIR)
    report = {"best_epoch": best[2], "threshold": thr, "val_macro_f1": val_m["macro_f1"],
              "hyperparameters": {"filters": 128, "kernel_sizes": [2, 3, 4], "dropout": 0.5,
                                  "epochs_max": EPOCHS, "batch": BATCH, "lr": LR, "momentum": MOMENTUM,
                                  "patience": PATIENCE, "optimizer": "SGD",
                                  "embeddings": "own word2vec, fine-tuned"},
              "history": history}
    (RESULTS_DIR / "train_w2v_cnn.json").write_text(json.dumps(report, indent=2))
    print(f"best epoch {best[2]}, threshold {thr:.2f}, val macro-F1 {val_m['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
