# Sketch-to-Image with Stable Diffusion + ControlNet

Turn rough hand-drawn sketches into polished images while preserving their structure.

## Setup

```bash
conda create -n sketch2img python=3.11 -y
conda activate sketch2img
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Project structure

```
sketches/   input sketches
outputs/    generated images
src/        source code
```

## Credits

- `sketches/meme_cheek_hold.jpg` — internet meme, source: TODO
- `sketches/meme_slap.jpg` — internet meme, source: TODO
