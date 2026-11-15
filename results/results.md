# Results

- Test set: 4000 messages (491 offensive), the same file for every model
- Latency: one message at a time, mean of 200 runs after 10 warm-up runs, normalization and tokenization included
- Hardware: Apple M2, 8 logical cores, Darwin 27.0.0, CPU only
- Latency budget: 50 ms per message

| Model | Precision | Macro-F1 | Accuracy | Offensive recall | Size (MB) | Latency (ms) | p95 (ms) | Within budget |
|---|---|---|---|---|---|---|---|---|
| Rule: edit-distance lexicon | rules | 0.8234 | 0.9350 | 0.5703 | 0.09 | 0.18 | 0.59 | yes |
| Classical: tf-idf + logistic regression | FP64 | 0.8722 | 0.9435 | 0.8004 | 2.42 | 0.26 | 0.30 | yes |
| Neural: word2vec + feedforward network | FP32 | 0.8382 | 0.9267 | 0.7617 | 7.72 | 0.07 | 0.10 | yes |
| Transformer: BERT (FP32, ONNX Runtime) | FP32 | 0.9171 | 0.9625 | 0.9063 | 438.18 | 20.38 | 37.40 | yes |
| Quantized Transformer: BERT (INT8, ONNX Runtime) | INT8 | 0.9207 | 0.9650 | 0.8859 | 110.29 | 7.49 | 13.34 | yes |
