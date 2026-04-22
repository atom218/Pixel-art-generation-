"""
PixelForge — Conditional GAN for Pixel Art Sprite Generation
=============================================================
v2 — WGAN-GP edition, fixes discriminator collapse seen in v1.

Key changes from v1:
  - LOSS_MODE switched to "wgan-gp" by default
  - D_STEPS increased to 5 (standard for WGAN-GP)
  - Discriminator: BatchNorm replaced with InstanceNorm (required for WGAN-GP)
  - Discriminator: no final Sigmoid — outputs raw critic score
  - Adam betas changed to (0.0, 0.9) per WGAN-GP paper
  - IMG_CHANNELS auto-detected from dataset
  - LR scheduler halves LR at halfway point
  - Gradient clipping on generator
  - SAVE_EVERY = 5 for more granular visual tracking
  - Wasserstein distance logged — should decrease as training improves

Requirements:
    pip install torch torchvision pillow tqdm

Usage:
    python train.py
"""

import os
import json
import time
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

DATA_DIR    = "/content/drive/MyDrive/dataset"   # parent folder containing front/back/left/right
OUTPUT_DIR  = "/content/drive/MyDrive/dataset_output_v4"        # where samples and checkpoints are saved

# Training
NUM_EPOCHS  = 400          # more epochs — WGAN-GP trains stably for longer
BATCH_SIZE  = 64
LR_G        = 0.0001       # WGAN-GP paper recommends 1e-4
LR_D        = 0.0001
BETA1       = 0.0          # WGAN-GP paper: β1=0 (not 0.5)
BETA2       = 0.9          # WGAN-GP paper: β2=0.9 (not 0.999)

# Architecture
NOISE_DIM   = 100
NUM_CLASSES = 4            # front / back / left / right
IMG_SIZE    = 64

# WGAN-GP
GP_LAMBDA   = 10           # gradient penalty weight
D_STEPS     = 5            # critic updates per generator update

# LR schedule: decay LR by this factor at the halfway epoch
LR_DECAY_FACTOR = 0.5

# Logging
SAVE_EVERY  = 5            # save sample grid every N epochs
CKPT_EVERY  = 25           # save checkpoint every N epochs

SEED        = 42

# ═══════════════════════════════════════════════════════════════════


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ───────────────────────────────────────────────────────────────────
#  DATASET
# ───────────────────────────────────────────────────────────────────

def get_dataloader(data_dir, batch_size, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    print(f"Class mapping : {dataset.class_to_idx}")
    print(f"Total images  : {len(dataset)}")

    # Auto-detect channels from first image so script works for both RGB and RGBA
    sample_img, _ = dataset[0]
    img_channels = sample_img.shape[0]
    print(f"Image channels: {img_channels}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    return loader, dataset.class_to_idx, img_channels


# ───────────────────────────────────────────────────────────────────
#  GENERATOR
# ───────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    def __init__(self, noise_dim, num_classes, img_channels):
        super().__init__()
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
            nn.Sigmoid(),   # [0,1] output — correct for pixel art
        )

    def forward(self, noise, labels_onehot):
        x = torch.cat([noise, labels_onehot], dim=1)
        x = self.project(x)
        x = x.view(x.size(0), 512, 4, 4)
        return self.conv_blocks(x)


# ───────────────────────────────────────────────────────────────────
#  DISCRIMINATOR (critic)
# ───────────────────────────────────────────────────────────────────
# BatchNorm is REMOVED. It creates batch-level correlations that break
# the WGAN-GP gradient penalty. InstanceNorm is used instead — it
# normalizes per-image, not per-batch, so the penalty stays valid.
# ───────────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    def __init__(self, num_classes, img_channels, img_size):
        super().__init__()
        self.img_size = img_size
        self.embed = nn.Embedding(num_classes, img_size * img_size)
        in_channels = img_channels + 1

        self.conv_blocks = nn.Sequential(
            # 64 → 32  (no norm on first layer — standard practice)
            nn.Conv2d(in_channels, 64, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 32 → 16
            nn.Conv2d(64, 128, 4, 2, 1, bias=True),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 16 → 8
            nn.Conv2d(128, 256, 4, 2, 1, bias=True),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 8 → 4
            nn.Conv2d(256, 512, 4, 2, 1, bias=True),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Raw score output — no Sigmoid for WGAN-GP
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1),
        )

    def forward(self, img, labels):
        label_map = self.embed(labels).view(
            labels.size(0), 1, self.img_size, self.img_size
        )
        x = torch.cat([img, label_map], dim=1)
        return self.classifier(self.conv_blocks(x))


# ───────────────────────────────────────────────────────────────────
#  WEIGHT INIT
# ───────────────────────────────────────────────────────────────────

def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ───────────────────────────────────────────────────────────────────
#  GRADIENT PENALTY
# ───────────────────────────────────────────────────────────────────

def gradient_penalty(D, real_imgs, fake_imgs, labels, device):
    B     = real_imgs.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)
    interpolated = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)
    d_interp = D(interpolated, labels)
    gradients = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(B, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


# ───────────────────────────────────────────────────────────────────
#  HELPERS
# ───────────────────────────────────────────────────────────────────

def to_onehot(labels, num_classes, device):
    onehot = torch.zeros(labels.size(0), num_classes, device=device)
    onehot.scatter_(1, labels.unsqueeze(1), 1.0)
    return onehot


def make_fixed_noise(noise_dim, num_classes, n_per_class, device):
    noise_list, label_list = [], []
    for cls in range(num_classes):
        noise_list.append(torch.randn(n_per_class, noise_dim, device=device))
        label_list.append(
            torch.full((n_per_class,), cls, dtype=torch.long, device=device)
        )
    return torch.cat(noise_list), torch.cat(label_list)


# ───────────────────────────────────────────────────────────────────
#  TRAINING LOOP
# ───────────────────────────────────────────────────────────────────

def train():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out = Path(OUTPUT_DIR)
    samples_dir = out / "samples"
    ckpt_dir    = out / "checkpoints"
    samples_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──
    loader, class_to_idx, img_channels = get_dataloader(DATA_DIR, BATCH_SIZE, IMG_SIZE)

    # ── Models ──
    G = Generator(NOISE_DIM, NUM_CLASSES, img_channels).to(device)
    D = Discriminator(NUM_CLASSES, img_channels, IMG_SIZE).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    print(f"Generator     params: {sum(p.numel() for p in G.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in D.parameters()):,}")

    # ── Optimizers ──
    opt_G = optim.Adam(G.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    opt_D = optim.Adam(D.parameters(), lr=LR_D, betas=(BETA1, BETA2))

    # LR halved at the halfway point
    half = NUM_EPOCHS // 2
    sched_G = optim.lr_scheduler.MultiStepLR(opt_G, milestones=[half], gamma=LR_DECAY_FACTOR)
    sched_D = optim.lr_scheduler.MultiStepLR(opt_D, milestones=[half], gamma=LR_DECAY_FACTOR)

    # ── Fixed samples for visual tracking ──
    N_PER_CLASS  = 8
    fixed_noise, fixed_labels = make_fixed_noise(NOISE_DIM, NUM_CLASSES, N_PER_CLASS, device)
    fixed_onehot = to_onehot(fixed_labels, NUM_CLASSES, device)

    log = {"epoch": [], "critic_loss": [], "gen_loss": [], "w_distance": []}
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        G.train(); D.train()
        epoch_loss_D = epoch_loss_G = epoch_w_dist = 0.0
        n_batches = 0

        for real_imgs, labels in tqdm(loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=False):
            real_imgs = real_imgs.to(device)
            labels    = labels.to(device)
            B         = real_imgs.size(0)

            # ── Critic (D) steps ─────────────────────────────────
            for _ in range(D_STEPS):
                opt_D.zero_grad()

                noise       = torch.randn(B, NOISE_DIM, device=device)
                fake_labels = torch.randint(0, NUM_CLASSES, (B,), device=device)
                fake_onehot = to_onehot(fake_labels, NUM_CLASSES, device)

                with torch.no_grad():
                    fake_imgs = G(noise, fake_onehot)

                real_scores = D(real_imgs, labels)
                fake_scores = D(fake_imgs.detach(), fake_labels)
                gp          = gradient_penalty(D, real_imgs, fake_imgs.detach(),
                                               labels, device)

                loss_D = fake_scores.mean() - real_scores.mean() + GP_LAMBDA * gp
                loss_D.backward()
                opt_D.step()

            w_dist = (real_scores.mean() - fake_scores.mean()).item()

            # ── Generator step ───────────────────────────────────
            opt_G.zero_grad()

            noise       = torch.randn(B, NOISE_DIM, device=device)
            fake_labels = torch.randint(0, NUM_CLASSES, (B,), device=device)
            fake_onehot = to_onehot(fake_labels, NUM_CLASSES, device)
            fake_imgs   = G(noise, fake_onehot)
            fake_scores = D(fake_imgs, fake_labels)

            loss_G = -fake_scores.mean()
            loss_G.backward()
            nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            opt_G.step()

            epoch_loss_D += loss_D.item()
            epoch_loss_G += loss_G.item()
            epoch_w_dist += w_dist
            n_batches    += 1

        sched_G.step()
        sched_D.step()

        avg_loss_D = epoch_loss_D / n_batches
        avg_loss_G = epoch_loss_G / n_batches
        avg_w_dist = epoch_w_dist / n_batches
        elapsed    = time.time() - start_time

        # Wasserstein distance should trend negative then slowly improve toward 0
        print(f"Epoch {epoch:>4}/{NUM_EPOCHS}  |  "
              f"Critic: {avg_loss_D:+.4f}  |  "
              f"Gen: {avg_loss_G:+.4f}  |  "
              f"W-dist: {avg_w_dist:+.4f}  |  "
              f"Elapsed: {elapsed/60:.1f}m")

        # ── Sample grid ──
        if epoch % SAVE_EVERY == 0 or epoch == 1:
            G.eval()
            with torch.no_grad():
                samples = G(fixed_noise, fixed_onehot)
            save_image(
                samples,
                samples_dir / f"epoch_{epoch:04d}.png",
                nrow=N_PER_CLASS,
                normalize=False,
            )
            G.train()

        # ── Checkpoint ──
        if epoch % CKPT_EVERY == 0:
            torch.save({
                "epoch":        epoch,
                "G_state_dict": G.state_dict(),
                "D_state_dict": D.state_dict(),
                "opt_G":        opt_G.state_dict(),
                "opt_D":        opt_D.state_dict(),
                "critic_loss":  avg_loss_D,
                "gen_loss":     avg_loss_G,
                "class_to_idx": class_to_idx,
                "img_channels": img_channels,
                "noise_dim":    NOISE_DIM,
            }, ckpt_dir / f"ckpt_epoch_{epoch:04d}.pt")

        log["epoch"].append(epoch)
        log["critic_loss"].append(avg_loss_D)
        log["gen_loss"].append(avg_loss_G)
        log["w_distance"].append(avg_w_dist)

    # ── Final save ──
    torch.save({
        "epoch":        NUM_EPOCHS,
        "G_state_dict": G.state_dict(),
        "D_state_dict": D.state_dict(),
        "class_to_idx": class_to_idx,
        "img_channels": img_channels,
        "noise_dim":    NOISE_DIM,
        "num_classes":  NUM_CLASSES,
        "img_size":     IMG_SIZE,
    }, out / "pixelforge_final.pt")

    with open(out / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nTraining complete.")
    print(f"Model   → {out / 'pixelforge_final.pt'}")
    print(f"Samples → {samples_dir}")


if __name__ == "__main__":
    train()