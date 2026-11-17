# GigaSketch: Sketch-to-Image with Stable Diffusion + ControlNet

Turn rough hand-drawn sketches into polished images while preserving their structure.

## Setup

```bash
conda create -n gigasketch python=3.11 -y
conda activate gigasketch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Project structure

```
sketches/   input sketches
outputs/    generated images
src/        source code
```

## Credits

- `sketches/meme1.jpg` — internet meme, source: TODO
- `sketches/meme2.jpg` — internet meme, source: TODO
