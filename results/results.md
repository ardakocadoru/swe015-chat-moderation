# Results - OffensEval-TR 2020 official test set (n = 3,528)

Hardware: AMD Ryzen 7 5700X 8-Core Processor, 8 cores, 31.9 GB RAM, Windows 11; CPU only (GPU not used for inference). PyTorch 2.15.0+cu132, ONNX Runtime 1.32.0. Measured 2026-11-16T12:27; lexicon re-measured 2026-11-17T22:00.

Latency: batch size 1, end-to-end (normalisation + tokenisation + model), mean of 300 runs after 20 warm-up calls. Budget: mean <= 50 ms per message on 1 CPU thread.

| Ladder step | Model | Precision / runtime | Macro-F1 | Offensive F1 | Size (MB) | Latency 1 thread mean (ms) | p95 (ms) | Latency 4 threads mean (ms) | Budget |
|---|---|---|---|---|---|---|---|---|---|
| Rule baseline (extra) | Rule: edit-distance lexicon | rules / Python | 0.654 | 0.439 | 0.003 | 0.53 | 1.33 | 0.50 | yes |
| 1 · Classical baseline | tf-idf + Logistic Regression | FP64 / scikit-learn | 0.769 | 0.624 | 2.2 | 1.68 | 2.18 | 1.67 | yes |
| 2a · Neural network (ours, extra) | word2vec + Feedforward NN | FP32 / PyTorch | 0.689 | 0.506 | 11.8 | 0.30 | 0.44 | 0.26 | yes |
| 2 · Neural network (ours) | word2vec + 1D CNN | FP32 / PyTorch | 0.728 | 0.556 | 11.8 | 0.57 | 0.71 | 0.55 | yes |
| 3 · Transformer | BERTurk fine-tuned | FP32 / PyTorch | 0.814 | 0.700 | 422.0 | 81.65 | 136.75 | 50.41 | no |
| 3b · Transformer, ONNX graph | BERTurk fine-tuned (ONNX) | FP32 / ONNX Runtime | 0.814 | 0.700 | 422.2 | 54.35 | 108.23 | 28.96 | no |
| 4 · Quantized Transformer | BERTurk fine-tuned, quantized | INT8 (dynamic) / ONNX Runtime | 0.815 | 0.700 | 106.2 | 27.85 | 58.49 | 15.63 | yes |
