import time

import gradio as gr

from src.config import LATENCY_BUDGET_MS
from src.download import ensure_models
from src.lexicon import LexiconMatcher
from src.models import load_available

ensure_models()
MODELS = load_available()
MATCHER = LexiconMatcher()
DEFAULT = "bert_int8" if "bert_int8" in MODELS else list(MODELS)[-1]
LOG_HEADERS = ["#", "model", "label", "conf.", "ms", f"≤ {LATENCY_BUDGET_MS:.0f} ms"]
LOG_TYPES = ["number", "str", "html", "str", "html", "html"]
CMP_HEADERS = ["model", "label", "conf.", "ms"]
CMP_TYPES = ["html", "html", "html", "html"]

ACCENT = "#c8501e"

CSS = """
:root {
  --clean-fg: #1d6b34;   --clean-bg: #e3f3e7;
  --off-fg:   #a8231a;   --off-bg:   #fbe4e2;
  --over-fg:  #b42318;
  --accent:   #c8501e;   --accent-fg: #a8420f;  --accent-bg: #fdf1ea;
  --muted:    #6b6b6b;
}
.dark {
  --clean-fg: #8fdba5;   --clean-bg: #173b23;
  --off-fg:   #ffb3ab;   --off-bg:   #4a1d19;
  --over-fg:  #ff8a80;
  --accent:   #c8501e;   --accent-fg: #f2a57e;  --accent-bg: #3a2418;
  --muted:    #a3a3a3;
}
.badge {
  display: inline-block; padding: 1px 9px; border-radius: 999px;
  font-size: 0.85em; font-weight: 600; line-height: 1.5; white-space: nowrap;
}
.badge.clean     { color: var(--clean-fg); background: var(--clean-bg); }
.badge.offensive { color: var(--off-fg);   background: var(--off-bg); }
.over { color: var(--over-fg); font-weight: 600; }
.def  { color: var(--accent-fg); font-weight: 600; }
.virtual-row:has(.def) .body-cell { background: var(--accent-bg) !important; }
.virtual-row:has(.def) .body-cell.first-column { box-shadow: inset 3px 0 0 var(--accent); }
.subtitle { color: var(--muted); font-size: 0.95em; margin-top: -6px; }
"""

ENGLISH_ONLY = gr.I18n(tr={
    "errors.use_via_api": "Use via API",
    "common.built_with_gradio": "Built with Gradio",
    "chatbot.clear": "Clear",
})

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.neutral,
    font=[gr.themes.Font("Helvetica Neue"), gr.themes.Font("Arial"), gr.themes.Font("sans-serif")],
).set(
    color_accent=ACCENT,
    color_accent_soft="#fdf1ea",
    color_accent_soft_dark="#3a2418",
    button_primary_background_fill=ACCENT,
    button_primary_background_fill_dark=ACCENT,
    button_primary_text_color="white",
    input_border_color_focus=ACCENT,
    input_border_color_focus_dark=ACCENT,
    loader_color=ACCENT,
)


def badge(label):
    return f'<span class="badge {label}">{label}</span>'


def fmt_ms(ms):
    text = f"{ms:.2f}"
    return f'<span class="over">{text}</span>' if ms > LATENCY_BUDGET_MS else text


def fmt_budget(ms):
    return "yes" if ms <= LATENCY_BUDGET_MS else '<span class="over">NO</span>'


def render_log(log):
    return [[n, short, badge(label), f"{conf:.2f}", fmt_ms(ms), fmt_budget(ms)]
            for n, short, label, conf, ms, _ in log]


def render_compare(rows):
    out = []
    for key, short, label, conf, ms in rows:
        cells = [short, badge(label), f"{conf:.2f}", fmt_ms(ms)]
        if key == DEFAULT:
            cells[0] = f"{short} · default"
            cells = [f'<span class="def">{c}</span>' for c in cells]
        out.append(cells)
    return out


def timed_predict(model, message):
    t0 = time.perf_counter()
    label, conf = model.predict(message)
    return label, conf, (time.perf_counter() - t0) * 1000


def moderate(message, model_key, history, log):
    message = (message or "").strip()
    if not message:
        return "", history, log, render_log(log), []
    model = MODELS[model_key]
    label, conf, ms = timed_predict(model, message)

    shown = message
    if label == "offensive":
        shown, hits = MATCHER.mask(message)
        if hits == 0:
            shown = "[message hidden by the moderator]"
    budget = "within budget" if ms <= LATENCY_BUDGET_MS else "OVER budget"
    verdict = f"{badge(label)} · confidence {conf:.2f} · {ms:.2f} ms ({budget}) · {model.name}"
    history = history + [{"role": "user", "content": shown}, {"role": "assistant", "content": verdict}]

    log = [[len(log) + 1, model.short, label, round(conf, 3), round(ms, 2),
            "yes" if ms <= LATENCY_BUDGET_MS else "NO"]] + log
    compare = []
    for key, m in MODELS.items():
        l, c, t = timed_predict(m, message)
        compare.append([key, m.short, l, c, t])
    return "", history, log, render_log(log), render_compare(compare)


with gr.Blocks(title="Real-Time Chat Moderation") as demo:
    gr.Markdown("# Real-Time Chat Moderation")
    gr.HTML('<div class="subtitle">SWE015 · Use Case 4 · default model: BERT INT8 (ONNX Runtime, CPU)</div>')
    gr.Markdown("Type a chat message. It is labelled live as **clean** or **offensive**; offensive messages are masked. "
                f"Latency budget: **{LATENCY_BUDGET_MS:.0f} ms** per message on CPU.")
    log_state = gr.State([])
    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.Chatbot(label="Game chat", height=520)
            box = gr.Textbox(label="Message", placeholder="Type a message and press Enter", autofocus=True)
            gr.Examples(["gg well played everyone", "you are an id10t", "shiiiit that was close",
                         "nice shot, how did you do that?", "uninstall the game you worthless n00b"], inputs=box)
        with gr.Column(scale=2):
            model_dd = gr.Dropdown(choices=[(m.name, k) for k, m in MODELS.items()], value=DEFAULT,
                                   label="Model used for moderation")
            log_table = gr.Dataframe(headers=LOG_HEADERS, datatype=LOG_TYPES,
                                     label="Response-time log (newest first)", interactive=False, wrap=True,
                                     column_widths=["7%", "26%", "24%", "14%", "12%", "17%"])
            cmp_table = gr.Dataframe(headers=CMP_HEADERS, datatype=CMP_TYPES,
                                     label="Last message: all models side by side", interactive=False, wrap=True,
                                     column_widths=["40%", "22%", "21%", "17%"])
    box.submit(moderate, [box, model_dd, chat, log_state], [box, chat, log_state, log_table, cmp_table])

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, i18n=ENGLISH_ONLY, footer_links=["api", "gradio"])
