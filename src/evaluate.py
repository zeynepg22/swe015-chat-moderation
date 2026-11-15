import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .config import LATENCY_BUDGET_MS, RESULTS_DIR, THRESHOLD
from .data import load_split
from .models import load_available


def hardware() -> str:
    cpu = platform.processor() or platform.machine()
    try:
        if sys.platform == "darwin":
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        elif sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                cpu = next((l.split(":", 1)[1].strip() for l in f if l.startswith("model name")), cpu)
    except Exception:
        pass
    return f"{cpu}, {os.cpu_count()} logical cores, {platform.system()} {platform.release()}, CPU only"


def measure_latency(model, texts, runs: int) -> tuple[float, float]:
    for t in texts[:10]:
        model.proba(t)
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        model.proba(texts[i % len(texts)])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.percentile(times, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency-runs", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N test messages (debugging)")
    args = ap.parse_args()
    assert args.latency_runs >= 100, "the homework requires at least 100 latency runs"

    texts, labels = load_split("test")
    if args.limit:
        texts, labels = texts[:args.limit], labels[:args.limit]
    models = load_available()
    rows, probs = [], {}
    for key, model in models.items():
        p = np.array([model.proba(t) for t in texts])
        pred = (p >= THRESHOLD).astype(int)
        mean_ms, p95_ms = measure_latency(model, texts, args.latency_runs)
        probs[key] = p
        rows.append({
            "key": key, "model": model.name, "precision_type": model.precision,
            "macro_f1": f1_score(labels, pred, average="macro"),
            "accuracy": accuracy_score(labels, pred),
            "offensive_precision": precision_score(labels, pred, zero_division=0),
            "offensive_recall": recall_score(labels, pred, zero_division=0),
            "size_mb": model.size_mb, "latency_ms": mean_ms, "latency_p95_ms": p95_ms,
            "within_budget": mean_ms <= LATENCY_BUDGET_MS,
        })
        print(f"{model.name:50} macro-F1 {rows[-1]['macro_f1']:.4f} | {model.size_mb:8.2f} MB | {mean_ms:7.2f} ms")

    hw = hardware()
    RESULTS_DIR.mkdir(exist_ok=True)
    meta = {"hardware": hw, "test_size": len(texts), "offensive_in_test": int(sum(labels)),
            "latency_runs": args.latency_runs, "latency_budget_ms": LATENCY_BUDGET_MS, "threshold": THRESHOLD}
    (RESULTS_DIR / "results.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2), encoding="utf-8")

    lines = [
        "# Results", "",
        f"- Test set: {len(texts)} messages ({sum(labels)} offensive), the same file for every model",
        f"- Latency: one message at a time, mean of {args.latency_runs} runs after 10 warm-up runs, normalization and tokenization included",
        f"- Hardware: {hw}",
        f"- Latency budget: {LATENCY_BUDGET_MS:.0f} ms per message", "",
        "| Model | Precision | Macro-F1 | Accuracy | Offensive recall | Size (MB) | Latency (ms) | p95 (ms) | Within budget |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['model']} | {r['precision_type']} | {r['macro_f1']:.4f} | {r['accuracy']:.4f} | "
                     f"{r['offensive_recall']:.4f} | {r['size_mb']:.2f} | {r['latency_ms']:.2f} | "
                     f"{r['latency_p95_ms']:.2f} | {'yes' if r['within_budget'] else 'NO'} |")
    (RESULTS_DIR / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    key = "bert_int8" if "bert_int8" in probs else list(probs)[-1]
    df = pd.DataFrame({"text": texts, "true": labels, "p_offensive": probs[key]})
    df["pred"] = (df.p_offensive >= THRESHOLD).astype(int)
    wrong = df[df.pred != df.true].copy()
    wrong["error"] = np.where(wrong.true == 1, "false negative (missed)", "false positive (wrongly flagged)")
    wrong["confidence"] = np.where(wrong.pred == 1, wrong.p_offensive, 1 - wrong.p_offensive)
    wrong.sort_values("confidence", ascending=False).head(40).to_csv(RESULTS_DIR / "failures.csv", index=False, encoding="utf-8")
    print(f"\n{len(wrong)} errors of '{key}' -> results/failures.csv (top 40 by confidence)")


if __name__ == "__main__":
    main()
