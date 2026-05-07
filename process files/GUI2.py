"""
PixelForge GUI — Character Angle Explorer
Generates a character at all 4 view angles and lets you SLERP between them
with a rotation slider.
"""

import math
import os
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# ─── config ───
CHECKPOINT_PATH = "/content/drive/MyDrive/dataset_output_final/pixelforge_final.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ANGLE_ORDER = ["front", "left", "back", "right"]
ANGLE_LABELS = {
    "front": "FRONT",
    "left":  "LEFT",
    "back":  "BACK",
    "right": "RIGHT",
}

# ─── model definition ───
class Generator(nn.Module):
    def __init__(self, noise_dim, num_classes, img_channels=3, img_size=64):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.input_dim = noise_dim + num_classes
        self.project = nn.Sequential(
            nn.Linear(self.input_dim, 512 * 4 * 4, bias=False),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, noise, labels_onehot):
        x = torch.cat([noise, labels_onehot], dim=1)
        x = self.project(x).view(-1, 512, 4, 4)
        return self.conv_blocks(x)

def load_model(path):
    ckpt = torch.load(path, map_location=DEVICE)
    meta = {
        "noise_dim": ckpt.get("noise_dim", 100),
        "num_classes": ckpt.get("num_classes", 4),
        "img_channels": ckpt.get("img_channels", 3),
        "img_size": ckpt.get("img_size", 64),
        "class_to_idx": ckpt.get("class_to_idx", {'back': 0, 'front': 1, 'left': 2, 'right': 3}),
    }
    G = Generator(meta["noise_dim"], meta["num_classes"], meta["img_channels"], meta["img_size"]).to(DEVICE)
    G.load_state_dict(ckpt["G_state_dict"])
    G.eval()
    return G, meta

G_MODEL, META = load_model(CHECKPOINT_PATH)
CLASS_TO_IDX = META["class_to_idx"]

def to_onehot(idx, num_classes):
    onehot = torch.zeros(1, num_classes, device=DEVICE)
    onehot[0, idx] = 1.0
    return onehot

def make_noise(seed, noise_dim):
    gen = torch.Generator(device='cpu')
    gen.manual_seed(int(seed))
    return torch.randn(1, noise_dim, generator=gen).to(DEVICE)

def to_pil(tensor, upscale=4):
    arr = (tensor.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    w, h = img.size
    return img.resize((w * upscale, h * upscale), Image.NEAREST)

def get_thumbs(char_num):
    char_num = int(char_num)
    nd = META["noise_dim"]
    z = make_noise(char_num, nd)
    thumbs = []
    for angle in ANGLE_ORDER:
        oh = to_onehot(CLASS_TO_IDX[angle], META["num_classes"])
        with torch.no_grad():
            out = G_MODEL(z, oh).squeeze(0)
        thumbs.append(to_pil(out, upscale=3))
    return thumbs

def get_main_image(char_num, slider_val):
    char_num = int(char_num)
    slider_val = float(slider_val)
    nd = META["noise_dim"]
    z = make_noise(char_num, nd)
    
    # Interpolate the condition vectors (one-hot labels) for smooth rotation,
    # keeping the latent noise (character identity) identical.
    seg = min(int(slider_val // 100), 2)
    t = (slider_val % 100) / 100.0
    a1, a2 = ANGLE_ORDER[seg], ANGLE_ORDER[seg+1]
    
    oh1 = to_onehot(CLASS_TO_IDX[a1], META["num_classes"])
    oh2 = to_onehot(CLASS_TO_IDX[a2], META["num_classes"])
    oh_interp = (1 - t) * oh1 + t * oh2
    
    with torch.no_grad():
        main_out = G_MODEL(z, oh_interp).squeeze(0)
    
    main_img = to_pil(main_out, upscale=5)
    target_angle = a1 if t < 0.5 else a2
    cur_label = ANGLE_LABELS[target_angle]
    return main_img, cur_label

def update_all(char_num, slider_val):
    thumbs = get_thumbs(char_num)
    main_img, cur_label = get_main_image(char_num, slider_val)
    return main_img, thumbs[0], thumbs[1], thumbs[2], thumbs[3], cur_label

CSS = """body, .gradio-container { background:#1a1a2e !important; color:#e0e0e0 !important; } .gr-image img { image-rendering:pixelated !important; }"""

with gr.Blocks(css=CSS) as demo:
    gr.HTML("<h1 style='text-align:center;color:#f5c400'>PIXEL FORGE EXPLORER</h1>")
    with gr.Row():
        char_num = gr.Number(label="CHARACTER SEED", value=0, precision=0)
        load_btn = gr.Button("GENERATE", variant="primary")
    angle_label = gr.Textbox(label="CURRENT VIEW", interactive=False)
    main_image = gr.Image(label="SPRITE", type="pil", height=350)
    rotation_slider = gr.Slider(minimum=0, maximum=300, step=1, value=0, label="ROTATION (Front -> Left -> Back -> Right)")
    with gr.Row():
        t1 = gr.Image(label="FRONT", type="pil", height=150)
        t2 = gr.Image(label="LEFT", type="pil", height=150)
        t3 = gr.Image(label="BACK", type="pil", height=150)
        t4 = gr.Image(label="RIGHT", type="pil", height=150)

    # show_progress=False prevents the loading screen overlay from blocking the UI during slider drags
    load_btn.click(update_all, inputs=[char_num, rotation_slider], outputs=[main_image, t1, t2, t3, t4, angle_label], show_progress=False)
    rotation_slider.change(get_main_image, inputs=[char_num, rotation_slider], outputs=[main_image, angle_label], show_progress=False)

if __name__ == '__main__':
    demo.launch(share=True)
