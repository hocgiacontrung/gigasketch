"""
GigaSketch web UI (Gradio): upload or draw a sketch, describe it, pick the settings, generate.

Run from the repo root:
    python src/app.py
then open http://127.0.0.1:7860 in a browser.

All the generation logic lives in generate.py; this file only builds the interface around it.
"""

import random
from pathlib import Path

import gradio as gr
import torch

from generate import (DEFAULT_MODEL, MODELS, STYLE_CN_END, STYLES, build_prompt, generate, load_pipeline,
                      prepare_sketch)

SKETCH_DIR = Path(__file__).resolve().parent.parent / "sketches"

# The prompts used for the results in the README.
EXAMPLES = [
    ("m1.jpg", "two men close together, one man holding the other man's cheek with his hand"),
    ("m3.png", "two men, one sitting on the ground resting his cheek on his hand looking bored, "
               "the other leaning toward him laughing excitedly with fists clenched"),
    ("m4.png", "two men fighting, one throwing a punch to the face of the other man who recoils in pain"),
    ("m5.jpg", "two men, one jumping and holding a basketball over his head as if about to dunk on the other "
               "man, with an excited grin, the other standing with arms crossed looking at him"),
    ("m6.png", "three men and one woman striking cool confident poses, a man with spiky hair and arms crossed, "
               "a woman with a ponytail, a man adjusting his glasses, a man crouching in front"),
    ("m2.jpg", "man slapping another man"),
]

# Only one base model fits comfortably in 8 GB of VRAM next to ControlNet, so keep a single one loaded
# and swap it when the user picks another model.
_loaded = {"name": None, "pipe": None}


def get_pipeline(model: str):
    if _loaded["name"] != model:
        _loaded["pipe"] = None
        torch.cuda.empty_cache()
        _loaded["pipe"] = load_pipeline(model)
        _loaded["name"] = model
    return _loaded["pipe"]


def store_drawing(drawing):
    """Keeps the latest drawing. Reading the sketchpad only here (when the user draws) avoids Gradio failing on an
    empty canvas when the Generate button is pressed with the Upload tab."""
    return (drawing or {}).get("composite")


def is_blank(img) -> bool:
    """True for an empty canvas: fully transparent, or (almost) all white."""
    if img.mode == "RGBA" and img.getchannel("A").getextrema()[1] == 0:
        return True
    return img.convert("L").getextrema()[0] >= 250


def run(source, upload, drawn, prompt, style, model, cn_end, scale, guidance, steps, seed, random_seed):
    sketch = upload if source == "upload" else drawn
    if sketch is None or is_blank(sketch):
        raise gr.Error("Please upload or draw a sketch first.")
    if not prompt.strip():
        raise gr.Error("Please describe what is in the sketch (subjects + action).")
    if random_seed:
        seed = random.randint(0, 2**31 - 1)

    full_prompt, negative = build_prompt(prompt, style)
    _, control = prepare_sketch(sketch)
    pipe = get_pipeline(model)
    # Sliders send whole numbers as int (e.g. 1 instead of 1.0); diffusers requires floats.
    image = generate(pipe, control, full_prompt, negative, scale=float(scale), cn_end=float(cn_end),
                     guidance=float(guidance), steps=int(steps), seed=int(seed))

    info = (f"**Prompt:** {full_prompt}\n\n**Negative:** {negative}\n\n"
            f"**Settings:** model `{model}`, style `{style}`, cn-end {cn_end}, scale {scale}, "
            f"guidance {guidance}, steps {int(steps)}, seed {int(seed)}")
    return image, info, int(seed)


with gr.Blocks(title="GigaSketch") as demo:
    gr.Markdown("# GigaSketch\nSketch-to-image with Stable Diffusion 1.5 + ControlNet Scribble. "
                "Upload or draw a sketch, describe **what is in it** (the style is added for you), and generate.")
    source = gr.State("upload")
    drawn = gr.State(None)

    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("Upload") as tab_upload:
                    upload = gr.Image(label="Sketch", type="pil", sources=["upload", "clipboard"], height=400)
                with gr.Tab("Draw") as tab_draw:
                    drawing = gr.Sketchpad(label="Draw here", type="pil", canvas_size=(512, 512),
                                           brush=gr.Brush(colors=["#000000"], default_size=4))
            prompt = gr.Textbox(label="What is in the sketch?", lines=2,
                                placeholder="e.g. two men fighting, one throwing a punch to the face of the other")
            with gr.Row():
                style = gr.Dropdown(list(STYLES), value="gigachad", label="Style")
                model = gr.Dropdown(list(MODELS), value=DEFAULT_MODEL, label="Base model")
            cn_end = gr.Slider(0.1, 1.0, value=STYLE_CN_END["gigachad"], step=0.05, label="ControlNet end (cn-end)",
                               info="Pose vs. faces: higher = follows the pose more closely, but round cartoon "
                                    "heads can turn into objects; lower = more natural faces, looser pose. "
                                    "Set to the recommended value when you change the style.")
            with gr.Accordion("Advanced settings", open=False):
                scale = gr.Slider(0.3, 1.5, value=1.0, step=0.05, label="ControlNet scale",
                                  info="How strictly the sketch lines are followed.")
                guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance (CFG)",
                                     info="How strongly the prompt is followed.")
                steps = gr.Slider(10, 50, value=20, step=1, label="Steps")
                with gr.Row():
                    seed = gr.Number(value=42, precision=0, label="Seed")
                    random_seed = gr.Checkbox(value=False, label="Random seed")
            button = gr.Button("Generate", variant="primary")

        with gr.Column():
            result = gr.Image(label="Result", type="pil", height=600)
            info = gr.Markdown()

    gr.Examples(
        examples=[[str(SKETCH_DIR / f), p] for f, p in EXAMPLES],
        inputs=[upload, prompt],
        label="Example sketches (click to load)",
    )

    tab_upload.select(lambda: "upload", outputs=source)
    tab_draw.select(lambda: "draw", outputs=source)
    style.change(lambda s: STYLE_CN_END[s], inputs=style, outputs=cn_end, show_progress="hidden")
    drawing.change(store_drawing, inputs=drawing, outputs=drawn, show_progress="hidden")
    button.click(
        run,
        inputs=[source, upload, drawn, prompt, style, model, cn_end, scale, guidance, steps, seed, random_seed],
        outputs=[result, info, seed],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1")
