import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
matplotlib.rcParams["text.parse_math"] = False
matplotlib.rcParams["font.family"] = ["Avenir Next", "Helvetica Neue", "Arial", "DejaVu Sans"]

res = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
rows, meta = res["rows"], res["meta"]

SHORT = {"rule": "Rule\nlexicon", "tfidf": "tf-idf\n+ LR", "w2v": "word2vec\n+ FFN",
         "bert_fp32": "BERT\nFP32", "bert_int8": "BERT\nINT8"}
GREY = "#9a9a9a"
ACCENT = "#c8501e"
LINE = "#5c5c5c"
COLOR = {"rule": GREY, "tfidf": GREY, "w2v": GREY, "bert_fp32": GREY, "bert_int8": ACCENT}
names = [SHORT[r["key"]] for r in rows]
colors = [COLOR[r["key"]] for r in rows]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=160)
panels = [("macro_f1", "Macro-F1 on the test set", "{:.4f}", (0, 1.0)),
          ("size_mb", "Model size on disk (MB)", "{:.2f}", None),
          ("latency_ms", "Latency per message (ms, mean)", "{:.2f}", None)]
for ax, (key, title, fmt, ylim) in zip(axes, panels):
    vals = [r[key] for r in rows]
    bars = ax.bar(names, vals, color=colors, width=0.62, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=8.5)
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=10)
    if ylim:
        ax.set_ylim(*ylim)
    elif key == "latency_ms":
        ax.set_ylim(0, max(max(vals), meta["latency_budget_ms"]) * 1.18)
    else:
        ax.set_ylim(0, max(vals) * 1.18)
    if key == "latency_ms":
        ax.axhline(meta["latency_budget_ms"], color=LINE, linestyle="--", linewidth=1)
        ax.text(len(vals) - 0.5, meta["latency_budget_ms"], f" {meta['latency_budget_ms']:.0f} ms budget",
                color=LINE, fontsize=8, va="bottom", ha="right")
    ax.tick_params(axis="x", labelsize=8.5)
    ax.tick_params(axis="y", labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

fig.suptitle(f"Model ladder on the same {meta['test_size']} test messages  ·  {meta['hardware'].split(',')[0]}, CPU only",
             fontsize=11, y=1.02)
fig.tight_layout()
out = ROOT / "docs" / "results.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("saved", out)
