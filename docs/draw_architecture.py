from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.rcParams["text.parse_math"] = False
matplotlib.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

OUT = Path(__file__).resolve().parent / "architecture.png"

INK = "#222222"
MUTED = "#8a8a8a"
LINE = "#5c5c5c"
FAINT = "#c9c9c9"
ACCENT = "#c8501e"
ACCENT_FILL = "#fdf1ea"

fig, ax = plt.subplots(figsize=(16, 8.06), dpi=170)
ax.set_xlim(0, 16)
ax.set_ylim(0.49, 8.55)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, title, file="", note="", tag="", faint=False, accent=False):
    edge_c = ACCENT if accent else (FAINT if faint else LINE)
    fill = ACCENT_FILL if accent else ("#fbfbfb" if faint else "white")
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.1",
                                facecolor=fill, edgecolor=edge_c, linewidth=1.8 if accent else 1.1, zorder=2))
    color = MUTED if faint else INK
    rows = []
    rows.append((title, 8.6 if faint else 9.2, "normal" if faint else "bold", ACCENT if accent else color, None))
    for line in note.split("\n") if note else []:
        rows.append((line, 7.3, "normal", INK if not faint else MUTED, None))
    if file:
        rows.append((file, 7.2, "normal", MUTED, "Menlo"))
    step = 0.24
    top = y + h / 2 + step * (len(rows) - 1) / 2
    for i, (text, size, weight, col, fam) in enumerate(rows):
        ax.text(x + w / 2, top - i * step, text, ha="center", va="center", fontsize=size,
                fontweight=weight, color=col, family=fam, zorder=3)
    if tag:
        ax.text(x + w - 0.07, y + 0.06, "".join(f"[{t}]" for t in tag), ha="right", va="bottom", fontsize=7, color=MUTED, zorder=3)
    return (x, y, w, h)


def edge(b, side):
    x, y, w, h = b
    return {"l": (x, y + h / 2), "r": (x + w, y + h / 2), "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[side]


def arrow(p, q, dashed=False, faint=False, shrink=3):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, linewidth=1.1,
                                 color=FAINT if faint else LINE, linestyle="--" if dashed else "-",
                                 shrinkA=shrink, shrinkB=shrink, zorder=1))


ax.text(0.4, 8.3, "Offline training, done once", fontsize=8.6, color=MUTED, va="center")
sw, sh, sy = 2.1, 0.72, 7.4
s1 = box(0.4, sy, sw, sh, "Jigsaw Toxic Comment", file="data/raw/train.csv", faint=True)
s2 = box(2.95, sy, sw, sh, "Build train / val / test", file="prepare_data.py", faint=True)
s3 = box(5.5, sy - 0.15, 4.3, sh + 0.3, "Train the models", note="BERT fine-tuned on Google Colab (T4 GPU)",
         file="train_baseline.py · train_w2v_nn.py · train_bert.py", faint=True)
s4 = box(10.3, sy, 2.5, sh, "Quantize BERT: FP32 → INT8", file="quantize.py", faint=True)
s5 = box(13.3, sy, 2.3, sh, "Hugging Face Hub", file="model weights, not in git", faint=True)
for a, b in ((s1, s2), (s2, s3), (s3, s4), (s4, s5)):
    arrow(edge(a, "r"), edge(b, "l"), faint=True)

fx, fy, fw, fh = 0.3, 0.55, 15.4, 6.2
fb = 0.94
ax.add_patch(FancyBboxPatch((fx, fb), fw, fy + fh - fb, boxstyle="round,pad=0,rounding_size=0.15",
                            facecolor="white", edgecolor=LINE, linewidth=1.0, zorder=0))
ax.text(fx + 0.25, fy + fh - 0.28, "Runs locally on CPU (Apple M2)  ·  python app.py",
        fontsize=9.2, color=INK, va="center", fontweight="bold")

cy = fy + fh / 2
widths = [1.5, 1.4, 1.8, 1.55, 2.0, 1.95, 1.6]
gaps = [0.22, 0.22, 0.22, 1.1, 1.1, 0.28]
xs = [0.55]
for i in range(6):
    xs.append(xs[-1] + widths[i] + gaps[i])
bh = 1.05
y = cy - bh / 2

user = box(xs[0], y, widths[0], bh, "User message", note="typed in\nthe chat box")
ui = box(xs[1], y, widths[1], bh, "Gradio interface", file="app.py")
pre = box(xs[2], y - 0.35, widths[2], bh + 0.7, "Pre-processing",
          note="regex rules:\nleetspeak · symbols\nrepeats · Turkish chars", file="normalize.py", tag="B")
sel = box(xs[3], y, widths[3], bh, "Model selector", note="dropdown in\nthe side panel")

mh, mgap = 0.8, 0.18
total = 5 * mh + 4 * mgap
my = [cy + total / 2 - mh - i * (mh + mgap) for i in range(5)]
mw = widths[4]
models = [
    box(xs[4], my[0], mw, mh, "Rule lexicon", note="rules · edit distance ≤ 1", file="lexicon.py", tag="B"),
    box(xs[4], my[1], mw, mh, "tf-idf + LR", note="FP64 · logistic regression", file="models/tfidf", tag="B"),
    box(xs[4], my[2], mw, mh, "word2vec + FFN", note="FP32 · own embeddings", file="models/w2v_ffn", tag="A"),
    box(xs[4], my[3], mw, mh, "BERT ONNX", note="FP32 · fine-tuned", file="model_fp32.onnx", tag="C"),
    box(xs[4], my[4], mw, mh, "BERT ONNX", note="INT8 · quantized, default", file="model_int8.onnx",
        tag="D", accent=True),
]

post = box(xs[5], cy - 1.0, widths[5], 2.0, "Post-processing",
           note="P(offensive) ≥ 0.5 →\nlabel + confidence\nmask offending words\nlatency vs 50 ms budget",
           file="app.py · lexicon.py", tag="BD")
out = box(xs[6], y - 0.35, widths[6], bh + 0.7, "Output in the UI",
          note="masked chat\nlatency log\nmodels side by side")

arrow(edge(user, "r"), edge(ui, "l"))
arrow(edge(ui, "r"), edge(pre, "l"))
arrow(edge(pre, "r"), edge(sel, "l"))
arrow(edge(post, "r"), edge(out, "l"))

bus1 = xs[4] - 0.3
ax.plot([edge(sel, "r")[0], bus1], [cy, cy], color=LINE, linewidth=1.1, zorder=1)
ax.plot([bus1, bus1], [my[0] + mh / 2, my[4] + mh / 2], color=LINE, linewidth=1.1, zorder=1)
for m in models:
    ym = m[1] + m[3] / 2
    arrow((bus1, ym), (m[0], ym), shrink=0)

bus2 = xs[4] + mw + 0.3
for m in models:
    ym = m[1] + m[3] / 2
    ax.plot([m[0] + m[2], bus2], [ym, ym], color=LINE, linewidth=1.1, zorder=1)
ax.plot([bus2, bus2], [my[0] + mh / 2, my[4] + mh / 2], color=LINE, linewidth=1.1, zorder=1)
arrow((bus2, cy), (xs[5], cy), shrink=0)

ax.text((edge(sel, "r")[0] + bus1) / 2, cy + 0.1, "normalized\ntext", ha="center", va="bottom",
        fontsize=6.8, color=MUTED, linespacing=1.1)
ax.text((bus2 + xs[5]) / 2, cy + 0.1, "P(offensive)", ha="center", va="bottom", fontsize=6.8, color=MUTED)

hx, hy = edge(s5, "b")
sx, sy_top = edge(sel, "t")
ax.plot([hx, hx, sx], [hy, 7.05, 7.05], color=MUTED, linestyle="--", linewidth=1.0, zorder=1)
ax.plot([sx, sx], [7.05, sy_top + 0.01], color=MUTED, linestyle="--", linewidth=1.0, zorder=1)
arrow((sx, sy_top + 0.3), (sx, sy_top), dashed=True, shrink=0)
ax.text(sx + 0.12, 6.92, "weights downloaded on first run  (download.py)", fontsize=7.4, color=MUTED, va="center")

ax.text(fx + fw, fb - 0.22, "[A] neural network   [B] classical NLP   [C] Transformer   [D] efficient model",
        fontsize=7.4, color=MUTED, ha="right", va="center")

fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("saved", OUT)
