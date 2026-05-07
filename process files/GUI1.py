"""
PixelForge GUI — Gradio interface for Conditional WGAN-GP sprite generator
Run in Google Colab:
    !pip install gradio
    python pixelforge_gui.py
or just call demo.launch(share=True) at the bottom of a notebook cell.

Model checkpoint expected at:
    /content/drive/MyDrive/dataset_output_final/pixelforge_final.pt
Keys: epoch, G_state_dict, D_state_dict, class_to_idx,
      img_channels, noise_dim, num_classes, img_size
"""

import math
import random

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# ─── paths ────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "/content/drive/MyDrive/dataset_output_final/pixelforge_final.pt"

# ─── device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── model definitions (must match train_v2 exactly) ──────────────────────────

class Generator(nn.Module):
    def __init__(self, noise_dim: int, num_classes: int, img_channels: int = 3, img_size: int = 64):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.img_size = img_size

        self.input_dim = noise_dim + num_classes

        self.project = nn.Sequential(
            nn.Linear(self.input_dim, 512 * 4 * 4, bias=False),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        self.conv_blocks = nn.Sequential(
            # 4 → 8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 8 → 16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 16 → 32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 32 → 64
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, noise: torch.Tensor, labels_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([noise, labels_onehot], dim=1)
        x = self.project(x)
        x = x.view(x.size(0), 512, 4, 4)
        return self.conv_blocks(x)


class Discriminator(nn.Module):
    """Included so the checkpoint loads cleanly; not used at inference time."""

    def __init__(self, num_classes: int, img_channels: int = 3, img_size: int = 64):
        super().__init__()
        self.img_size = img_size
        self.embed = nn.Embedding(num_classes, img_size * img_size)
        in_channels = img_channels + 1

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=True),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=True),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=True),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1),
        )

    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_map = self.embed(labels).view(labels.size(0), 1, self.img_size, self.img_size)
        x = torch.cat([img, label_map], dim=1)
        return self.classifier(self.conv_blocks(x))


# ─── load checkpoint ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_model(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    noise_dim   = ckpt.get("noise_dim", 100)
    num_classes = ckpt.get("num_classes", 4)
    img_channels= ckpt.get("img_channels", 3)
    img_size    = ckpt.get("img_size", 64)
    class_to_idx= ckpt.get("class_to_idx", {"back": 0, "front": 1, "left": 2, "right": 3})

    G = Generator(noise_dim, num_classes, img_channels, img_size).to(DEVICE)
    G.load_state_dict(ckpt["G_state_dict"])
    G.eval()

    meta = {
        "noise_dim":    noise_dim,
        "num_classes":  num_classes,
        "img_channels": img_channels,
        "img_size":     img_size,
        "class_to_idx": class_to_idx,
        "epoch":        ckpt.get("epoch", "?"),
    }
    return G, meta


G_MODEL, META = load_model(CHECKPOINT_PATH)

# class label helpers
IDX_TO_CLASS = {v: k for k, v in META["class_to_idx"].items()}
ANGLE_LABELS = ["↑ Front", "↓ Back", "← Left", "→ Right"]
ANGLE_TO_IDX = {
    "↑ Front": META["class_to_idx"]["front"],
    "↓ Back":  META["class_to_idx"]["back"],
    "← Left":  META["class_to_idx"]["left"],
    "→ Right": META["class_to_idx"]["right"],
}


# ─── inference helpers ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def to_onehot(labels, num_classes, device):
    onehot = torch.zeros(labels.size(0), num_classes, device=device)
    onehot.scatter_(1, labels.unsqueeze(1), 1.0)
    return onehot

def seed_to_noise(seed: int, noise_dim: int, temperature: float, generator: torch.Generator) -> torch.Tensor:
    generator.manual_seed(seed)
    z = torch.randn(1, noise_dim, generator=generator, device=DEVICE)
    return z * temperature


def slerp(z1: torch.Tensor, z2: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between two noise vectors."""
    z1_flat = z1.view(-1)
    z2_flat = z2.view(-1)
    z1_n = z1_flat / (z1_flat.norm() + 1e-8)
    z2_n = z2_flat / (z2_flat.norm() + 1e-8)
    dot = torch.clamp((z1_n * z2_n).sum(), -1.0, 1.0)
    omega = torch.acos(dot).item()
    if abs(omega) < 1e-6:
        return ((1 - t) * z1 + t * z2).view(1, -1)
    return (math.sin((1 - t) * omega) / math.sin(omega) * z1_flat
            + math.sin(t * omega)       / math.sin(omega) * z2_flat).view(1, -1)


def tensor_to_pil(t: torch.Tensor, upscale: int = 4) -> Image.Image:
    """Convert (C, H, W) float tensor in [0,1] to a PIL image, upscaled with nearest-neighbour."""
    arr = (t.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    w, h = img.size
    return img.resize((w * upscale, h * upscale), Image.NEAREST)


def make_grid_image(images: list[Image.Image]) -> Image.Image:
    """Tile a list of same-size PIL images into the tightest square grid."""
    n = len(images)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w + (cols - 1) * 4, rows * h + (rows - 1) * 4), (20, 20, 40))
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        grid.paste(img, (c * (w + 4), r * (h + 4)))
    return grid


# ─── generate sprites ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_sprites(angle_label: str, seed: int, batch_size: int, temperature: float):
    class_idx = ANGLE_TO_IDX[angle_label]
    gen = torch.Generator(device=DEVICE)

    images = []
    with torch.no_grad():
        for i in range(batch_size):
            z = seed_to_noise(seed + i, META["noise_dim"], temperature, gen)
            label_tensor = torch.tensor([class_idx], device=DEVICE)
            onehot = to_onehot(label_tensor, META["num_classes"], DEVICE)
            out = G_MODEL(z, onehot)
            images.append(tensor_to_pil(out.squeeze(0)))

    return make_grid_image(images)


# ─── latent space interpolation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def interpolate_sprites(angle_label: str, seed_a: int, seed_b: int, steps: int):
    class_idx = ANGLE_TO_IDX[angle_label]
    gen = torch.Generator(device=DEVICE)

    z_a = seed_to_noise(seed_a, META["noise_dim"], 1.0, gen)
    z_b = seed_to_noise(seed_b, META["noise_dim"], 1.0, gen)

    images = []
    with torch.no_grad():
        for i in range(steps):
            t = i / (steps - 1)
            z = slerp(z_a, z_b, t).to(DEVICE)
            label_tensor = torch.tensor([class_idx], device=DEVICE)
            onehot = to_onehot(label_tensor, META["num_classes"], DEVICE)
            out = G_MODEL(z, onehot)
            images.append(tensor_to_pil(out.squeeze(0)))

    # stitch horizontally
    w, h = images[0].size
    strip = Image.new("RGB", (len(images) * w + (len(images) - 1) * 4, h), (20, 20, 40))
    for i, img in enumerate(images):
        strip.paste(img, (i * (w + 4), 0))
    return strip


# ─── gradio UI ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323:wght@400&display=swap');

body, .gradio-container {
    background: #1a1a2e !important;
    font-family: 'VT323', monospace !important;
    color: #e0e0e0 !important;
}

h1, h2, h3, .gr-panel > label, .gr-block-label, .label-wrap span {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 9px !important;
    color: #f5c400 !important;
    letter-spacing: 0.5px !important;
}

.gr-button-primary {
    background: #f5c400 !important;
    color: #1a1a2e !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 10px !important;
    border: none !important;
    border-radius: 6px !important;
}

.gr-button-primary:hover { background: #ffd700 !important; }

.gr-button-secondary {
    background: transparent !important;
    border: 1px solid #2a2a5a !important;
    color: #a0a0c0 !important;
    font-family: 'VT323', monospace !important;
    font-size: 16px !important;
}

.gr-box, .gr-panel, .gr-form, .gr-block {
    background: #16213e !important;
    border: 1px solid #2a2a4a !important;
    border-radius: 8px !important;
}

input[type=number], input[type=text], .gr-input {
    background: #0f3460 !important;
    border: 1px solid #2a2a5a !important;
    color: #e0e0e0 !important;
    font-family: 'VT323', monospace !important;
    font-size: 18px !important;
    border-radius: 6px !important;
}

.gr-radio label, .gr-radio-label {
    font-family: 'VT323', monospace !important;
    font-size: 18px !important;
    color: #a0a0c0 !important;
}

.gr-image img {
    image-rendering: pixelated !important;
    image-rendering: crisp-edges !important;
    border-radius: 6px !important;
}

footer { display: none !important; }
"""

TITLE_HTML = """
<div style="text-align:center; padding: 20px 0 8px;">
  <h1 style="font-family:'Press Start 2P',monospace; font-size:16px; color:#f5c400; margin:0; letter-spacing:2px;">
    PIXEL FORGE
  </h1>
  <p style="font-family:'VT323',monospace; font-size:18px; color:#666; margin:6px 0 0;">
    conditional GAN  pixel art sprite generator  400 epochs
  </p>
</div>
"""

MODEL_INFO_HTML = f"""
<div style="font-family:'VT323',monospace; font-size:15px; color:#4a6a8a;
            background:#0a1628; border:1px solid #1a2a4a; border-radius:8px; padding:12px 16px; line-height:1.8;">
  <span style="color:#f5c400;">MODEL</span> &nbsp; Conditional WGAN-GP &nbsp;|&nbsp; {META['epoch']} epochs<br>
  <span style="color:#f5c400;">ARCH</span>  &nbsp;&nbsp; noise_dim={META['noise_dim']}, {META['num_classes']} classes, {META['img_size']}{META['img_size']} RGB<br>
  <span style="color:#f5c400;">DATA</span>  &nbsp;&nbsp; TinyHero — 3,648 sprites (back / front / left / right)<br>
  <span style="color:#f5c400;">DEVICE</span> &nbsp; {DEVICE}
</div>
"""


def randomize_seed():
    return random.randint(0, 999999)


with gr.Blocks(css=CUSTOM_CSS, title="PixelForge") as demo:

    gr.HTML(TITLE_HTML)

    with gr.Row():
        # ── left column: controls ━━━━━━━━━━━━━━━━━━━━━━━━━━━
        with gr.Column(scale=1):

            angle = gr.Radio(
                choices=ANGLE_LABELS,
                value="↑ Front",
                label="VIEW ANGLE",
            )

            with gr.Row():
                seed = gr.Number(
                    value=42,
                    label="SEED",
                    minimum=0,
                    maximum=999999,
                    precision=0,
                    step=1,
                )
                rand_btn = gr.Button("", elem_classes=["gr-button-secondary"])

            batch_size = gr.Radio(
                choices=[1, 4, 9],
                value=4,
                label="BATCH SIZE",
            )

            temperature = gr.Slider(
                minimum=0.5,
                maximum=2.0,
                step=0.1,
                value=1.0,
                label="CREATIVITY (temperature)",
            )

            generate_btn = gr.Button("[ GENERATE ]", variant="primary")

        # ── right column: output ━━━━━━━━━━━━━━━━━━━━━━━━━━
        with gr.Column(scale=1):
            output_image = gr.Image(
                label="GENERATED SPRITES",
                type="pil",
                show_download_button=True,
            )

    # ── latent space explorer (collapsible) ━━━━━━━━━━━━━━━━━
    with gr.Accordion("  LATENT SPACE EXPLORER  (SLERP interpolation)", open=False):
        with gr.Row():
            seed_a = gr.Number(value=42,  label="SEED A", minimum=0, maximum=999999, precision=0, step=1)
            seed_b = gr.Number(value=999, label="SEED B", minimum=0, maximum=999999, precision=0, step=1)

        with gr.Row():
            interp_angle = gr.Radio(
                choices=ANGLE_LABELS,
                value="↑ Front",
                label="ANGLE FOR INTERPOLATION",
            )
            interp_steps = gr.Slider(
                minimum=3, maximum=12, step=1, value=7,
                label="STEPS",
            )

        interp_btn = gr.Button("[ INTERPOLATE → SLERP ]", variant="primary")
        interp_output = gr.Image(label="SLERP STRIP  (seed A ──→ seed B)", type="pil", show_download_button=True)

    # ── model info ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    gr.HTML(MODEL_INFO_HTML)

    # ── event wiring ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    rand_btn.click(fn=randomize_seed, inputs=[], outputs=[seed])

    generate_btn.click(
        fn=generate_sprites,
        inputs=[angle, seed, batch_size, temperature],
        outputs=[output_image],
    )

    interp_btn.click(
        fn=interpolate_sprites,
        inputs=[interp_angle, seed_a, seed_b, interp_steps],
        outputs=[interp_output],
    )


if __name__ == "__main__":
    demo.launch(share=True)
