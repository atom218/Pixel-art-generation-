"""
PixelForge — Hyperparameter Sweep
===================================
Runs every combination of hyperparameters, trains a mini version of the model
for each, scores it with FID, and saves a ranked results table.

The best config is automatically saved to best_config.json which train.py
can load directly.

Sweep covers ~12 configurations as planned in the project proposal.

Requirements:
    pip install torch torchvision pillow tqdm scipy

Usage:
    python sweep.py

Edit only the CONFIG and SWEEP_GRID sections below.
"""

import os
import json
import time
import shutil
import itertools
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from scipy import linalg
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════
#  CONFIG  ← set your paths here
# ═══════════════════════════════════════════════════════════════════

DATA_DIR    = "E:\genai dataset2\data"   # parent folder containing front/back/left/right
OUTPUT_DIR  = "E:\output1"  # where sweep results are saved

# Fixed across all runs (not swept)
NUM_CLASSES   = 4
IMG_SIZE      = 64
IMG_CHANNELS  = 4
LOSS_MODE     = "bce"       # "bce" or "wgan-gp"
GP_LAMBDA     = 10
SEED          = 42

# How many epochs to train each config during the sweep.
# Keep this low (20-30) so the sweep finishes in reasonable time.
# The winner gets retrained for full epochs in train.py.
SWEEP_EPOCHS  = 25

# ═══════════════════════════════════════════════════════════════════
#  SWEEP GRID  ← these are the values that get combined
# ═══════════════════════════════════════════════════════════════════
#
# Total combinations = product of all list lengths.
# Current grid = 3 x 2 x 2 = 12 configurations (matches project proposal).
# Add or remove values freely — the script handles any grid size.

SWEEP_GRID = {
    "lr":         [0.0002, 0.0001, 0.00005],   # learning rate (applied to both G and D)
    "batch_size": [32, 64],                     # batch size
    "noise_dim":  [100, 200],                   # noise vector length
}

# ═══════════════════════════════════════════════════════════════════


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ───────────────────────────────────────────────────────────────────
#  MODEL DEFINITIONS (same as train.py, self-contained here)
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
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, noise, labels_onehot):
        x = torch.cat([noise, labels_onehot], dim=1)
        x = self.project(x)
        x = x.view(x.size(0), 512, 4, 4)
        return self.conv_blocks(x)


class Discriminator(nn.Module):
    def __init__(self, num_classes, img_channels, img_size):
        super().__init__()
        self.img_size = img_size
        self.embed = nn.Embedding(num_classes, img_size * img_size)
        in_channels = img_channels + 1
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1),
        )

    def forward(self, img, labels):
        label_map = self.embed(labels).view(labels.size(0), 1, self.img_size, self.img_size)
        x = torch.cat([img, label_map], dim=1)
        return self.classifier(self.conv_blocks(x))


def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def to_onehot(labels, num_classes, device):
    onehot = torch.zeros(labels.size(0), num_classes, device=device)
    onehot.scatter_(1, labels.unsqueeze(1), 1.0)
    return onehot


# ───────────────────────────────────────────────────────────────────
#  FID SCORE
# ───────────────────────────────────────────────────────────────────
#
# FID (Fréchet Inception Distance) measures how similar the distribution
# of generated images is to real images. Lower = better.
#
# Full FID uses InceptionV3 features. For small 64x64 pixel art sprites,
# we use a lightweight approximation: extract features from the discriminator's
# penultimate layer instead of Inception. This is faster, avoids a large
# pretrained model dependency, and is still meaningful for comparison
# across configs since the same feature extractor is used for all runs.

def extract_features(model_conv, imgs, device):
    """Run images through discriminator conv layers to get feature vectors."""
    with torch.no_grad():
        feats = model_conv(imgs)                    # (B, 512, 4, 4)
        feats = feats.view(feats.size(0), -1)       # (B, 512*4*4)
    return feats.cpu().numpy()


def compute_fid(real_feats, fake_feats):
    """
    Compute FID between real and fake feature distributions.
    FID = ||μ_r - μ_f||² + Tr(Σ_r + Σ_f - 2(Σ_r Σ_f)^(1/2))
    """
    mu_r, sigma_r = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu_f, sigma_f = fake_feats.mean(axis=0), np.cov(fake_feats, rowvar=False)

    diff = mu_r - mu_f
    # Numerically stable square root of matrix product
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean)
    return float(fid)


def gather_real_features(loader, D_conv, device, max_batches=20):
    """Collect feature vectors from real images."""
    all_feats = []
    for i, (imgs, _) in enumerate(loader):
        if i >= max_batches:
            break
        imgs = imgs.to(device)
        all_feats.append(extract_features(D_conv, imgs, device))
    return np.concatenate(all_feats, axis=0)


def gather_fake_features(G, D_conv, noise_dim, num_classes, n_samples, device):
    """Generate fake images and collect their feature vectors."""
    all_feats = []
    batch = 64
    generated = 0
    while generated < n_samples:
        current_batch = min(batch, n_samples - generated)
        noise = torch.randn(current_batch, noise_dim, device=device)
        labels = torch.randint(0, num_classes, (current_batch,), device=device)
        onehot = to_onehot(labels, num_classes, device)
        with torch.no_grad():
            fake = G(noise, onehot)
        all_feats.append(extract_features(D_conv, fake, device))
        generated += current_batch
    return np.concatenate(all_feats, axis=0)


# ───────────────────────────────────────────────────────────────────
#  SINGLE RUN
# ───────────────────────────────────────────────────────────────────

def run_config(config, run_dir, device):
    """
    Train one config for SWEEP_EPOCHS and return its FID score.
    config = {"lr": ..., "batch_size": ..., "noise_dim": ...}
    """
    set_seed(SEED)

    lr         = config["lr"]
    batch_size = config["batch_size"]
    noise_dim  = config["noise_dim"]

    # ── Data ──
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # ── Models ──
    G = Generator(noise_dim, NUM_CLASSES, IMG_CHANNELS).to(device)
    D = Discriminator(NUM_CLASSES, IMG_CHANNELS, IMG_SIZE).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    loss_log = []

    # ── Training ──
    for epoch in range(1, SWEEP_EPOCHS + 1):
        G.train(); D.train()
        epoch_loss_D = epoch_loss_G = 0.0
        n = 0

        for real_imgs, labels in loader:
            real_imgs = real_imgs.to(device)
            labels    = labels.to(device)
            B         = real_imgs.size(0)

            # Discriminator step
            opt_D.zero_grad()
            real_logits = D(real_imgs, labels)
            noise       = torch.randn(B, noise_dim, device=device)
            fake_labels = torch.randint(0, NUM_CLASSES, (B,), device=device)
            fake_onehot = to_onehot(fake_labels, NUM_CLASSES, device)
            fake_imgs   = G(noise, fake_onehot).detach()
            fake_logits = D(fake_imgs, fake_labels)
            loss_D = (
                nn.functional.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)) +
                nn.functional.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            )
            loss_D.backward()
            opt_D.step()

            # Generator step
            opt_G.zero_grad()
            noise       = torch.randn(B, noise_dim, device=device)
            fake_labels = torch.randint(0, NUM_CLASSES, (B,), device=device)
            fake_onehot = to_onehot(fake_labels, NUM_CLASSES, device)
            fake_imgs   = G(noise, fake_onehot)
            fake_logits = D(fake_imgs, fake_labels)
            loss_G = nn.functional.binary_cross_entropy_with_logits(
                fake_logits, torch.ones_like(fake_logits)
            )
            loss_G.backward()
            opt_G.step()

            epoch_loss_D += loss_D.item()
            epoch_loss_G += loss_G.item()
            n += 1

        loss_log.append({
            "epoch":  epoch,
            "loss_D": epoch_loss_D / n,
            "loss_G": epoch_loss_G / n,
        })

    # ── FID Score ──
    G.eval(); D.eval()
    # Use discriminator conv layers as feature extractor
    D_conv = D.conv_blocks

    real_feats = gather_real_features(loader, D_conv, device, max_batches=30)
    fake_feats = gather_fake_features(G, D_conv, noise_dim, NUM_CLASSES,
                                      n_samples=len(real_feats), device=device)
    fid = compute_fid(real_feats, fake_feats)

    # ── Save sample grid for this run ──
    run_dir.mkdir(parents=True, exist_ok=True)
    noise_sample  = torch.randn(NUM_CLASSES * 8, noise_dim, device=device)
    labels_sample = torch.cat([torch.full((8,), c, dtype=torch.long) for c in range(NUM_CLASSES)]).to(device)
    onehot_sample = to_onehot(labels_sample, NUM_CLASSES, device)
    with torch.no_grad():
        samples = G(noise_sample, onehot_sample)
    save_image(samples, run_dir / "sample_grid.png", nrow=8, normalize=False)

    # Save loss log
    with open(run_dir / "loss_log.json", "w") as f:
        json.dump(loss_log, f, indent=2)

    # Save model
    torch.save(G.state_dict(), run_dir / "G.pt")

    return fid, loss_log[-1]["loss_D"], loss_log[-1]["loss_G"]


# ───────────────────────────────────────────────────────────────────
#  SWEEP RUNNER
# ───────────────────────────────────────────────────────────────────

def build_configs(grid):
    """Generate all combinations from the sweep grid."""
    keys   = list(grid.keys())
    values = list(grid.values())
    combos = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combos]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    configs = build_configs(SWEEP_GRID)
    total   = len(configs)
    print(f"\nTotal configurations to sweep: {total}")
    print(f"Epochs per config            : {SWEEP_EPOCHS}")
    print(f"Estimated configs: {total}\n")

    # Print the full grid so you can see exactly what will run
    print("Configurations:")
    for i, cfg in enumerate(configs):
        print(f"  [{i+1:>2}/{total}]  {cfg}")
    print()

    results = []
    sweep_start = time.time()

    for i, config in enumerate(configs):
        run_name = f"run_{i+1:02d}_lr{config['lr']}_bs{config['batch_size']}_nd{config['noise_dim']}"
        run_dir  = out / run_name

        print(f"[{i+1}/{total}] Starting: {run_name}")
        run_start = time.time()

        try:
            fid, final_loss_D, final_loss_G = run_config(config, run_dir, device)
            elapsed = time.time() - run_start

            result = {
                "rank":        None,         # filled in after sorting
                "run":         run_name,
                "fid":         round(fid, 4),
                "final_loss_D": round(final_loss_D, 4),
                "final_loss_G": round(final_loss_G, 4),
                "elapsed_min": round(elapsed / 60, 1),
                **config,
            }
            results.append(result)
            print(f"  ✓ FID: {fid:.4f}  |  Loss D: {final_loss_D:.4f}  |  "
                  f"Loss G: {final_loss_G:.4f}  |  Time: {elapsed/60:.1f}m\n")

        except Exception as e:
            print(f"  ✗ Run failed: {e}\n")
            results.append({"run": run_name, "fid": float("inf"), "error": str(e), **config})

    # ── Rank results by FID (lower is better) ──
    results.sort(key=lambda x: x.get("fid", float("inf")))
    for rank, r in enumerate(results, 1):
        r["rank"] = rank

    # ── Save results table ──
    results_path = out / "sweep_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── Save best config separately so train.py can load it ──
    best = results[0]
    best_config = {k: best[k] for k in SWEEP_GRID.keys()}
    with open(out / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)

    # ── Print final leaderboard ──
    total_elapsed = time.time() - sweep_start
    print("\n" + "═" * 65)
    print(f"SWEEP COMPLETE  ({total_elapsed/60:.1f} min total)")
    print("═" * 65)
    print(f"{'Rank':<5} {'FID':>8} {'LR':>8} {'Batch':>6} {'Noise':>6}  Run")
    print("-" * 65)
    for r in results:
        marker = " ◄ BEST" if r["rank"] == 1 else ""
        print(f"  {r['rank']:<4} {r.get('fid', 'ERR'):>8.4f} "
              f"{r.get('lr', '?'):>8} "
              f"{r.get('batch_size', '?'):>6} "
              f"{r.get('noise_dim', '?'):>6}  "
              f"{r['run']}{marker}")
    print("═" * 65)
    print(f"\nBest config: {best_config}")
    print(f"Saved to   : {out / 'best_config.json'}")
    print(f"Full table : {results_path}")
    print("\nNext step: copy the best config values into train.py CONFIG and run full training.")


if __name__ == "__main__":
    main()
