"""
GigaSketch - turn a rough sketch into an image with Stable Diffusion 1.5 + ControlNet (Scribble).

Usage (from the repo root):
    python src/generate.py --sketch sketches/m2.jpg --prompt "man slapping another man" --style gigachad

The final prompt is built from a template: "{your content}, {style preset}, {quality tags}".
--prompt only needs to say WHAT is in the sketch (subjects + action); the style comes from --style.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_scribble"

# ---------------------------------------------------------------------------
# 0. Prompt template: user content + style preset + quality tags
# ---------------------------------------------------------------------------
QUALITY_TAGS = "highly detailed, sharp focus, best quality"

BASE_NEGATIVE = (
    "blurry, low quality, deformed, disfigured, bad anatomy, extra limbs, "
    "extra fingers, mutated hands, watermark, text, "
    # Big empty round cartoon heads tend to be rendered as objects covering the face.
    "helmet, mask, hat, covered face"
)

# Each preset = (style description, extra negative terms).
# Keep them short: CLIP truncates prompts after 77 tokens.
STYLES = {
    "gigachad": (
        "extremely muscular man, chiseled jawline, sharp cheekbones, confident expression, "
        "black and white photo, dramatic studio lighting, high contrast",
        "skinny, chubby, cartoon, drawing",
    ),
    "photo": (
        "realistic photo, natural lighting, 35mm, depth of field",
        "cartoon, drawing, painting, anime",
    ),
    "anime": (
        "anime style, cel shading, vibrant colors, clean lineart",
        "photo, realistic, 3d render",
    ),
    "comic": (
        "comic book style, bold ink outlines, halftone shading, dynamic",
        "photo, realistic, 3d render",
    ),
    "none": ("", ""),  # raw prompt, only quality tags added
}


def build_prompt(content: str, style: str) -> tuple[str, str]:
    """Returns (prompt, negative_prompt) for the given user content and style preset."""
    style_text, style_negative = STYLES[style]
    prompt = ", ".join(part for part in (content.strip(), style_text, QUALITY_TAGS) if part)
    negative = ", ".join(part for part in (BASE_NEGATIVE, style_negative) if part)
    return prompt, negative


# ---------------------------------------------------------------------------
# 1. Preprocessing: messy photo of a sketch -> clean white-lines-on-black map
# ---------------------------------------------------------------------------
def prepare_sketch(path: Path, max_side: int = 512) -> tuple[Image.Image, Image.Image]:
    """Returns (resized original, control image for ControlNet)."""
    img = Image.open(path).convert("RGB")

    # Resize so the longest side is `max_side`; SD needs width/height divisible by 8.
    scale = max_side / max(img.size)
    w = int(img.width * scale) // 8 * 8
    h = int(img.height * scale) // 8 * 8
    img = img.resize((w, h), Image.LANCZOS)

    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)  # soften JPEG noise / paper texture

    # Adaptive threshold handles uneven lighting (gray paper, shadows).
    # THRESH_BINARY_INV -> lines become white, background black (what Scribble expects).
    # blockSize must be larger than the stroke width, otherwise thick strokes get hollowed
    # out into double outlines (15 did that on m1; 51 keeps strokes solid).
    lines = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=51, C=15
    )
    lines = cv2.morphologyEx(lines, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))  # remove specks
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8), iterations=1)          # thicken strokes

    control = Image.fromarray(lines).convert("RGB")
    return img, control


# ---------------------------------------------------------------------------
# 2. Model loading
# ---------------------------------------------------------------------------
def load_pipeline() -> StableDiffusionControlNetPipeline:
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD_MODEL,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,  # avoids false positives (e.g. shirtless people -> black image)
    )
    # UniPC gives good results in ~20 steps (default scheduler needs ~50).
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe.to("cuda")


# ---------------------------------------------------------------------------
# 3. Generation + saving
# ---------------------------------------------------------------------------
def side_by_side(*images: Image.Image, labels: list[str] | None = None) -> Image.Image:
    """Puts images in one row (all resized to the first one's size), with an optional label on each panel."""
    w, h = images[0].size
    canvas = Image.new("RGB", (w * len(images), h), "white")
    for i, im in enumerate(images):
        canvas.paste(im.resize((w, h)), (i * w, 0))
    if labels:
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default(size=max(16, h // 20))
        for i, text in enumerate(labels):
            if text:
                x0, y0, x1, y1 = draw.textbbox((i * w + 8, 8), text, font=font)
                draw.rectangle((x0 - 4, y0 - 4, x1 + 4, y1 + 4), fill="black")
                draw.text((i * w + 8, 8), text, fill="white", font=font)
    return canvas


def generate(pipe: StableDiffusionControlNetPipeline, control: Image.Image, prompt: str, negative: str,
             scale: float = 1.0, cn_end: float = 0.4, guidance: float = 7.5, steps: int = 20,
             seed: int = 42) -> Image.Image:
    """One image from a control map. `cn_end` = fraction of the steps during which ControlNet is active."""
    generator = torch.Generator("cuda").manual_seed(seed)
    return pipe(
        prompt=prompt,
        negative_prompt=negative,
        image=control,
        controlnet_conditioning_scale=scale,
        control_guidance_end=cn_end,
        guidance_scale=guidance,
        num_inference_steps=steps,
        generator=generator,
        width=control.width,
        height=control.height,
    ).images[0]


def main():
    p = argparse.ArgumentParser(description="Sketch -> image with SD 1.5 + ControlNet Scribble")
    p.add_argument("--sketch", required=True, type=Path)
    p.add_argument("--prompt", required=True, help="what is in the sketch, e.g. 'man slapping another man'")
    p.add_argument("--style", default="gigachad", choices=STYLES, help="style preset added to the prompt")
    p.add_argument("--negative", default=None, help="override the negative prompt entirely")
    p.add_argument("--scale", type=float, default=1.0, help="ControlNet strength (sketch adherence)")
    # cn-end is a trade-off between pose and face preservation:
    # high values follow the pose closely but round cartoon heads can become helmets/bowls;
    # below 0.5 heads become real faces, but the pose is followed less closely. 0.4 = default.
    p.add_argument("--cn-end", type=float, default=0.4,
                   help="fraction of steps where ControlNet is active (1.0 = whole run; 0.4 = pose fixed "
                        "early, free details at the end)")
    p.add_argument("--guidance", type=float, default=7.5, help="CFG scale (prompt adherence)")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seeds", "--seed", type=int, nargs="+", default=[42],
                   help="one or more seeds, e.g. --seeds 42 43 44 (model is loaded once)")
    p.add_argument("--out", type=Path, default=Path("outputs"), help="results go to <out>/<sketch name>/")
    args = p.parse_args()

    prompt, negative = build_prompt(args.prompt, args.style)
    if args.negative is not None:
        negative = args.negative
    print(f"Prompt:   {prompt}")
    print(f"Negative: {negative}")

    original, control = prepare_sketch(args.sketch)
    pipe = load_pipeline()

    out_dir = args.out / args.sketch.stem  # one folder per sketch, e.g. outputs/m1/
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{args.sketch.stem}_{args.style}_s{args.scale}_e{args.cn_end}_g{args.guidance}"

    results = []
    for seed in args.seeds:
        result = generate(pipe, control, prompt, negative, scale=args.scale, cn_end=args.cn_end,
                          guidance=args.guidance, steps=args.steps, seed=seed)
        results.append(result)

        name = f"{base}_seed{seed}"
        result.save(out_dir / f"{name}.png")
        side_by_side(original, control, result).save(out_dir / f"{name}_compare.png")
        print(f"Saved -> {out_dir / name}.png (+ _compare.png)")

    if len(results) > 1:  # sketch | control map | one result per seed, handy for the report
        seeds_tag = "-".join(map(str, args.seeds))
        path = out_dir / f"{base}_seeds{seeds_tag}_grid.png"
        side_by_side(original, control, *results,
                     labels=["sketch", "control", *(f"seed {s}" for s in args.seeds)]).save(path)
        print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
