"""
PixelForge — Conditional GAN for Pixel Art Sprite Generation
=============================================================
Generates 64x64 pixel art character sprites conditioned on view angle.
Classes: front, back, left, right

Architecture:
  Generator  : noise (NOISE_DIM,) + one-hot (NUM_CLASSES,) → 64x64x4 RGBA sprite
  Discriminator: 64x64x4 image + class embedding → real/fake scalar

Loss: Binary Cross Entropy (BCE). WGAN-GP enabled via LOSS_MODE = "wgan-gp"

Requirements:
    pip install torch torchvision pillow tqdm

Usage:
    python train.py
    (edit the CONFIG section below before running)
"""

import os
import json
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════
#  CONFIG  ← only section you need to edit
# ═══════════════════════════════════════════════════════════════════

DATA_DIR    = "E:\genai dataset2\data"   # parent folder containing front/back/left/right
OUTPUT_DIR  = "E:\output"        # where checkpoints and sample grids are saved

# Training
NUM_EPOCHS  = 200
BATCH_SIZE  = 32
LR_G        = 0.0002          # generator learning rate
LR_D        = 0.0002          # discriminator learning rate
BETA1       = 0.5             # Adam β1 (0.5 is standard for GANs)
BETA2       = 0.999

# Architecture
NOISE_DIM   = 100             # length of the input noise vector
NUM_CLASSES = 4               # front / back / left / right
IMG_SIZE    = 64              # sprites are 64x64
IMG_CHANNELS = 4              # RGBA

# Loss mode: "bce" or "wgan-gp"
# Start with "bce". Switch to "wgan-gp" if you see mode collapse.
LOSS_MODE   = "bce"
GP_LAMBDA   = 10              # gradient penalty weight (only used for wgan-gp)
D_STEPS     = 1               # discriminator updates per generator update
                              # (increase to 2-5 if using wgan-gp)

# Logging
SAVE_EVERY  = 10              # save sample grid every N epochs
CKPT_EVERY  = 25              # save model checkpoint every N epochs

# Reproducibility
SEED        = 42

# ═══════════════════════════════════════════════════════════════════


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ───────────────────────────────────────────────────────────────────
#  DATASET
# ───────────────────────────────────────────────────────────────────

def get_dataloader(data_dir, batch_size, img_size):
    """
    Loads images from a folder structure:
        data_dir/
            front/  *.png
            back/   *.png
            left/   *.png
            right/  *.png

    ImageFolder automatically assigns integer labels 0-3 in alphabetical order:
        back=0, front=1, left=2, right=3
    Labels are remapped to a fixed order below.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),           # → [0, 1] float, shape (C, H, W)
        # No normalization to [-1, 1] because we use Sigmoid output, not Tanh.
        # Sigmoid naturally outputs [0, 1], matching the loaded tensor range.
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Print the class → index mapping so you can verify
    print(f"Class mapping: {dataset.class_to_idx}")
    print(f"Total images : {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,    # keeps batch size consistent for BatchNorm
    )
    return loader, dataset.class_to_idx


# ───────────────────────────────────────────────────────────────────
#  GENERATOR
# ───────────────────────────────────────────────────────────────────
#
#  Input  : noise (B, NOISE_DIM) concatenated with class one-hot (B, NUM_CLASSES)
#           → combined vector (B, NOISE_DIM + NUM_CLASSES)
#  Output : image (B, IMG_CHANNELS, 64, 64)  values in [0, 1]
#
#  Architecture (transposed convolutions, doubling spatial dims each layer):
#    Linear → reshape to (B, 512, 4, 4)
#    ConvT  → (B, 256, 8, 8)
#    ConvT  → (B, 128, 16, 16)
#    ConvT  → (B, 64,  32, 32)
#    ConvT  → (B, 4,   64, 64)  → Sigmoid
# ───────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    def __init__(self, noise_dim, num_classes, img_channels):
        super().__init__()
        self.input_dim = noise_dim + num_classes

        # Project and reshape to spatial feature map
        self.project = nn.Sequential(
            nn.Linear(self.input_dim, 512 * 4 * 4, bias=False),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        # Upsample: 4→8→16→32→64
        self.conv_blocks = nn.Sequential(
            # 4x4 → 8x8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 8x8 → 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 16x16 → 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 32x32 → 64x64
            nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),   # output in [0, 1] — correct for pixel art hard edges
        )

    def forward(self, noise, labels_onehot):
        # Concatenate noise and class embedding
        x = torch.cat([noise, labels_onehot], dim=1)   # (B, noise_dim + num_classes)
        x = self.project(x)                             # (B, 512*4*4)
        x = x.view(x.size(0), 512, 4, 4)               # (B, 512, 4, 4)
        x = self.conv_blocks(x)                         # (B, C, 64, 64)
        return x


# ───────────────────────────────────────────────────────────────────
#  DISCRIMINATOR
# ───────────────────────────────────────────────────────────────────
#
#  Input  : image (B, IMG_CHANNELS, 64, 64) + class label
#  Class conditioning: embed label → (B, NUM_CLASSES, 64, 64) and
#                      concatenate as extra channels to the image.
#  Output : scalar per image (real/fake probability)
#
#  Architecture (strided convolutions, halving spatial dims each layer):
#    (B, C+NUM_CLASSES, 64, 64)
#    Conv → (B, 64,  32, 32)
#    Conv → (B, 128, 16, 16)
#    Conv → (B, 256, 8,  8)
#    Conv → (B, 512, 4,  4)
#    Flatten → Linear(1)
# ───────────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    def __init__(self, num_classes, img_channels, img_size):
        super().__init__()
        self.img_size = img_size
        # Embed class label as a spatial map concatenated to image
        self.embed = nn.Embedding(num_classes, img_size * img_size)

        in_channels = img_channels + 1   # image channels + 1 class channel

        self.conv_blocks = nn.Sequential(
            # 64x64 → 32x32  (no BN on first layer — standard practice)
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 → 16x16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 → 8x8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 → 4x4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Final classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1),
            # No Sigmoid here when using WGAN-GP.
            # For BCE we apply Sigmoid inside the loss (BCEWithLogitsLoss).
        )

    def forward(self, img, labels):
        # Embed labels and reshape to spatial map (B, 1, H, W)
        label_map = self.embed(labels)                          # (B, H*W)
        label_map = label_map.view(labels.size(0), 1,
                                   self.img_size, self.img_size)  # (B, 1, H, W)
        # Concatenate along channel dim
        x = torch.cat([img, label_map], dim=1)                 # (B, C+1, H, W)
        x = self.conv_blocks(x)
        x = self.classifier(x)
        return x


# ───────────────────────────────────────────────────────────────────
#  WEIGHT INITIALIZATION
# ───────────────────────────────────────────────────────────────────

def weights_init(m):
    """
    Apply the DCGAN paper weight initialization:
    Conv and ConvTranspose weights ~ N(0, 0.02)
    BatchNorm weights ~ N(1, 0.02), biases = 0
    """
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ───────────────────────────────────────────────────────────────────
#  LOSSES
# ───────────────────────────────────────────────────────────────────

def bce_loss_real(logits):
    """Discriminator loss on real images: log D(x)"""
    return nn.functional.binary_cross_entropy_with_logits(
        logits, torch.ones_like(logits)
    )


def bce_loss_fake(logits):
    """Discriminator loss on fake images: log(1 - D(G(z)))"""
    return nn.functional.binary_cross_entropy_with_logits(
        logits, torch.zeros_like(logits)
    )


def bce_loss_generator(logits):
    """Generator loss: log D(G(z))  — non-saturating formulation"""
    return nn.functional.binary_cross_entropy_with_logits(
        logits, torch.ones_like(logits)
    )


def gradient_penalty(D, real_imgs, fake_imgs, labels, device):
    """
    WGAN-GP gradient penalty.
    Enforces 1-Lipschitz constraint by penalizing |∇D(x̂)| deviating from 1.
    """
    B = real_imgs.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)
    interpolated = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)

    d_interpolated = D(interpolated, labels)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(B, -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty


# ───────────────────────────────────────────────────────────────────
#  ONE-HOT ENCODING
# ───────────────────────────────────────────────────────────────────

def to_onehot(labels, num_classes, device):
    """Convert integer label tensor to one-hot float tensor."""
    onehot = torch.zeros(labels.size(0), num_classes, device=device)
    onehot.scatter_(1, labels.unsqueeze(1), 1.0)
    return onehot


# ───────────────────────────────────────────────────────────────────
#  FIXED NOISE FOR SAMPLE GRIDS
# ───────────────────────────────────────────────────────────────────

def make_fixed_noise(noise_dim, num_classes, n_per_class, device):
    """
    Create fixed noise + labels used to generate consistent sample grids
    across epochs so you can visually track training progress.
    Layout: n_per_class columns per class, num_classes rows.
    """
    noise_list, label_list = [], []
    for cls in range(num_classes):
        noise_list.append(torch.randn(n_per_class, noise_dim, device=device))
        label_list.append(torch.full((n_per_class,), cls, dtype=torch.long, device=device))
    return torch.cat(noise_list), torch.cat(label_list)


# ───────────────────────────────────────────────────────────────────
#  TRAINING LOOP
# ───────────────────────────────────────────────────────────────────

def train():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Output directories
    out = Path(OUTPUT_DIR)
    samples_dir = out / "samples"
    ckpt_dir    = out / "checkpoints"
    samples_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──
    loader, class_to_idx = get_dataloader(DATA_DIR, BATCH_SIZE, IMG_SIZE)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # ── Models ──
    G = Generator(NOISE_DIM, NUM_CLASSES, IMG_CHANNELS).to(device)
    D = Discriminator(NUM_CLASSES, IMG_CHANNELS, IMG_SIZE).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    print(f"\nGenerator     params: {sum(p.numel() for p in G.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in D.parameters()):,}")

    # ── Optimizers ──
    opt_G = optim.Adam(G.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    opt_D = optim.Adam(D.parameters(), lr=LR_D, betas=(BETA1, BETA2))

    # ── Fixed noise for sample grids ──
    N_PER_CLASS = 8
    fixed_noise, fixed_labels = make_fixed_noise(NOISE_DIM, NUM_CLASSES, N_PER_CLASS, device)
    fixed_onehot = to_onehot(fixed_labels, NUM_CLASSES, device)

    # ── Training log ──
    log = {"epoch": [], "loss_D": [], "loss_G": [], "elapsed_s": []}
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        G.train()
        D.train()
        epoch_loss_D, epoch_loss_G = 0.0, 0.0
        n_batches = 0

        for real_imgs, labels in tqdm(loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=False):
            real_imgs = real_imgs.to(device)
            labels    = labels.to(device)
            B         = real_imgs.size(0)

            # ── Train Discriminator ──────────────────────────────
            for _ in range(D_STEPS):
                opt_D.zero_grad()

                # Real images
                real_logits = D(real_imgs, labels)

                # Fake images
                noise       = torch.randn(B, NOISE_DIM, device=device)
                fake_labels = torch.randint(0, NUM_CLASSES, (B,), device=device)
                fake_onehot = to_onehot(fake_labels, NUM_CLASSES, device)
                fake_imgs   = G(noise, fake_onehot).detach()
                fake_logits = D(fake_imgs, fake_labels)

                if LOSS_MODE == "bce":
                    loss_D = bce_loss_real(real_logits) + bce_loss_fake(fake_logits)

                elif LOSS_MODE == "wgan-gp":
                    gp     = gradient_penalty(D, real_imgs, fake_imgs, labels, device)
                    loss_D = fake_logits.mean() - real_logits.mean() + GP_LAMBDA * gp

                loss_D.backward()
                opt_D.step()

            # ── Train Generator ──────────────────────────────────
            opt_G.zero_grad()

            noise       = torch.randn(B, NOISE_DIM, device=device)
            fake_labels = torch.randint(0, NUM_CLASSES, (B,), device=device)
            fake_onehot = to_onehot(fake_labels, NUM_CLASSES, device)
            fake_imgs   = G(noise, fake_onehot)
            fake_logits = D(fake_imgs, fake_labels)

            if LOSS_MODE == "bce":
                loss_G = bce_loss_generator(fake_logits)
            elif LOSS_MODE == "wgan-gp":
                loss_G = -fake_logits.mean()

            loss_G.backward()
            opt_G.step()

            epoch_loss_D += loss_D.item()
            epoch_loss_G += loss_G.item()
            n_batches    += 1

        avg_loss_D = epoch_loss_D / n_batches
        avg_loss_G = epoch_loss_G / n_batches
        elapsed    = time.time() - start_time

        print(f"Epoch {epoch:>4}/{NUM_EPOCHS}  |  "
              f"Loss D: {avg_loss_D:.4f}  |  "
              f"Loss G: {avg_loss_G:.4f}  |  "
              f"Elapsed: {elapsed/60:.1f}m")

        # ── Save sample grid ──
        if epoch % SAVE_EVERY == 0 or epoch == 1:
            G.eval()
            with torch.no_grad():
                samples = G(fixed_noise, fixed_onehot)
            # samples shape: (NUM_CLASSES * N_PER_CLASS, C, H, W)
            # save_image arranges them in a grid; nrow = N_PER_CLASS gives one row per class
            save_image(
                samples,
                samples_dir / f"epoch_{epoch:04d}.png",
                nrow=N_PER_CLASS,
                normalize=False,   # already in [0,1], no normalization needed
            )
            G.train()

        # ── Save checkpoint ──
        if epoch % CKPT_EVERY == 0:
            torch.save({
                "epoch":         epoch,
                "G_state_dict":  G.state_dict(),
                "D_state_dict":  D.state_dict(),
                "opt_G":         opt_G.state_dict(),
                "opt_D":         opt_D.state_dict(),
                "loss_D":        avg_loss_D,
                "loss_G":        avg_loss_G,
                "class_to_idx":  class_to_idx,
            }, ckpt_dir / f"ckpt_epoch_{epoch:04d}.pt")

        # ── Log ──
        log["epoch"].append(epoch)
        log["loss_D"].append(avg_loss_D)
        log["loss_G"].append(avg_loss_G)
        log["elapsed_s"].append(elapsed)

    # Save final model and log
    torch.save({
        "epoch":        NUM_EPOCHS,
        "G_state_dict": G.state_dict(),
        "D_state_dict": D.state_dict(),
        "class_to_idx": class_to_idx,
    }, out / "pixelforge_final.pt")

    with open(out / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nTraining complete. Model saved to {out / 'pixelforge_final.pt'}")
    print(f"Sample grids saved to {samples_dir}")


if __name__ == "__main__":
    train()