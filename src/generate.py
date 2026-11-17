"""
GigaSketch - turn a rough sketch into an image with Stable Diffusion 1.5 + ControlNet (Scribble).

Usage (from the repo root):
    python src/generate.py --sketch sketches/meme2.jpg --prompt "two men in a photo studio"
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_scribble"

DEFAULT_NEGATIVE = (
    "blurry, low quality, deformed, disfigured, bad anatomy, extra limbs, "
    "extra fingers, mutated hands, watermark, text"
)


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
    # out into double outlines (15 did that on meme1; 51 keeps strokes solid).
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
def side_by_side(*images: Image.Image) -> Image.Image:
    w, h = images[0].size
    canvas = Image.new("RGB", (w * len(images), h), "white")
    for i, im in enumerate(images):
        canvas.paste(im.resize((w, h)), (i * w, 0))
    return canvas


def main():
    p = argparse.ArgumentParser(description="Sketch -> image with SD 1.5 + ControlNet Scribble")
    p.add_argument("--sketch", required=True, type=Path)
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative", default=DEFAULT_NEGATIVE)
    p.add_argument("--scale", type=float, default=1.0, help="ControlNet strength (sketch adherence)")
    p.add_argument("--guidance", type=float, default=7.5, help="CFG scale (prompt adherence)")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("outputs"))
    args = p.parse_args()

    original, control = prepare_sketch(args.sketch)
    pipe = load_pipeline()

    generator = torch.Generator("cuda").manual_seed(args.seed)
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative,
        image=control,
        controlnet_conditioning_scale=args.scale,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        generator=generator,
        width=control.width,
        height=control.height,
    ).images[0]

    args.out.mkdir(parents=True, exist_ok=True)
    name = f"{args.sketch.stem}_s{args.scale}_g{args.guidance}_seed{args.seed}"
    result.save(args.out / f"{name}.png")
    side_by_side(original, control, result).save(args.out / f"{name}_compare.png")
    print(f"Saved -> {args.out / name}.png (+ _compare.png)")


if __name__ == "__main__":
    main()
