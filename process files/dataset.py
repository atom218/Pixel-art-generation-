"""
PixelForge — Dataset Download Pipeline
=======================================
Downloads pixel art character sprites from the nyuuzyou/OpenGameArt-CC0
HuggingFace dataset mirror, slices sprite sheets into individual frames,
filters to a consistent resolution, and organizes into labeled class folders.

Requirements:
    pip install datasets pillow requests tqdm

Usage:
    1. Set OUTPUT_DIR below to your desired folder path
    2. Run: python download_dataset.py
"""

import os
import io
import json
import zipfile
import requests
import traceback
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

# ============================================================
# >>> SET YOUR OUTPUT FOLDER PATH HERE <<<
OUTPUT_DIR = "E:\genai dataset"
# ============================================================

# --- Config --------------------------------------------------
TARGET_SIZE      = (16, 32)   # Final sprite resolution (w, h)
MAX_SRC_SIZE     = 128        # Skip sprites larger than this in either dimension
MIN_SRC_SIZE     = 8          # Skip sprites smaller than this (likely icons/dots)
REQUEST_TIMEOUT  = 15         # Seconds before a download times out
MAX_PACKS        = None       # Set to an int (e.g. 50) to do a dry run; None = download all

# Class label mapping: folder name -> tags that trigger it
# Order matters — first match wins
CLASS_MAP = {
    "warrior": ["warrior", "knight", "fighter", "soldier", "swordsman", "swordsmen"],
    "mage":    ["mage", "wizard", "sorcerer", "witch", "magician", "mages"],
    "archer":  ["archer", "ranger", "bow", "bowman", "hunters"],
    "rogue":   ["rogue", "thief", "assassin", "ninja", "stealth"],
    "monster": ["monster", "goblin", "slime", "zombie", "skeleton", "enemy",
                "creature", "orc", "demon", "beast", "dragon"],
    "healer":  ["healer", "priest", "cleric", "paladin", "shaman"],
    "misc":    [],   # catch-all — filled at labeling stage
}

# Tags that must appear in a pack for it to be downloaded at all
REQUIRED_TAGS = {
    "sprite", "sprites", "character", "characters",
    "rpg", "warrior", "mage", "archer", "knight",
    "enemy", "monster", "goblin", "rogue", "healer",
    "pixel", "pixelart", "pixel art", "animated",
    "16x16", "16x32", "32x32", "16x24", "32x48",
}

# Tags that immediately disqualify a pack (non-character content)
EXCLUDE_TAGS = {
    "tileset", "tile", "tiles", "background", "terrain",
    "ui", "hud", "icon", "icons", "font", "fonts",
    "item", "weapon", "weapons", "effect", "effects",
    "music", "sound", "3d", "isometric",
}
# -------------------------------------------------------------


def assign_class(tags: list[str]) -> str:
    """Return the best-matching class label for a tag list."""
    tags_lower = {t.lower() for t in tags}
    for cls, keywords in CLASS_MAP.items():
        if cls == "misc":
            continue
        if tags_lower & set(keywords):
            return cls
    return "misc"


def is_relevant_pack(tags: list[str]) -> bool:
    """Return True if the pack should be downloaded."""
    tags_lower = {t.lower() for t in tags}
    if tags_lower & EXCLUDE_TAGS:
        return False
    return bool(tags_lower & REQUIRED_TAGS)


def download_bytes(url: str) -> bytes | None:
    """Download a URL and return raw bytes, or None on failure."""
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True)
        r.raise_for_status()
        return r.content
    except Exception:
        return None


def is_blank_frame(img: Image.Image) -> bool:
    """Return True if the frame is fully transparent or a single solid color."""
    if img.mode == "RGBA":
        alpha = img.split()[3]
        if max(alpha.getdata()) == 0:   # fully transparent
            return True
    # Treat single-color frames as blank (common padding in sprite sheets)
    colors = img.getcolors(maxcolors=4)
    return colors is not None and len(colors) == 1


def slice_sheet(img: Image.Image, frame_w: int, frame_h: int) -> list[Image.Image]:
    """Slice a sprite sheet into individual frame images."""
    W, H = img.size
    frames = []
    for row in range(H // frame_h):
        for col in range(W // frame_w):
            box = (col * frame_w, row * frame_h,
                   (col + 1) * frame_w, (row + 1) * frame_h)
            frame = img.crop(box).convert("RGBA")
            if not is_blank_frame(frame):
                frames.append(frame)
    return frames


def resize_to_target(img: Image.Image, target: tuple[int, int]) -> Image.Image:
    """Resize using NEAREST to preserve pixel art crispness."""
    return img.resize(target, Image.NEAREST)


def process_image_file(raw: bytes, pack_name: str, class_dir: Path,
                        counter: list[int]) -> int:
    """
    Given raw PNG/GIF bytes, try to extract individual sprites.
    Returns the number of sprites saved.
    """
    saved = 0
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception:
        return 0

    W, H = img.size

    # Skip images that are already the right size — treat as a single sprite
    if (W, H) == TARGET_SIZE:
        if not is_blank_frame(img):
            out_path = class_dir / f"{pack_name}_{counter[0]:05d}.png"
            img.save(out_path)
            counter[0] += 1
            saved += 1
        return saved

    # Skip clearly non-sprite images (too large or too small)
    if W > 1024 or H > 1024:
        return 0
    if W < MIN_SRC_SIZE or H < MIN_SRC_SIZE:
        return 0

    # Try common sprite sheet frame sizes, picking the one that fits evenly
    candidates = [
        TARGET_SIZE,
        (16, 16), (16, 24), (24, 24),
        (32, 32), (32, 48), (48, 48),
        (64, 64),
    ]

    sliced_any = False
    for fw, fh in candidates:
        if W % fw == 0 and H % fh == 0 and fw <= W and fh <= H:
            frames = slice_sheet(img, fw, fh)
            if not frames:
                continue
            sliced_any = True
            for frame in frames:
                if frame.size != TARGET_SIZE:
                    # Only downscale; skip upscaling small frames aggressively
                    if frame.size[0] > TARGET_SIZE[0] * 4 or frame.size[1] > TARGET_SIZE[1] * 4:
                        continue
                    frame = resize_to_target(frame, TARGET_SIZE)
                out_path = class_dir / f"{pack_name}_{counter[0]:05d}.png"
                frame.save(out_path)
                counter[0] += 1
                saved += 1
            break   # Use the first frame size that works

    # If no frame size fit evenly, treat the whole image as one sprite and resize
    if not sliced_any:
        if (MIN_SRC_SIZE <= W <= MAX_SRC_SIZE and
                MIN_SRC_SIZE <= H <= MAX_SRC_SIZE and
                not is_blank_frame(img)):
            frame = resize_to_target(img, TARGET_SIZE)
            out_path = class_dir / f"{pack_name}_{counter[0]:05d}.png"
            frame.save(out_path)
            counter[0] += 1
            saved += 1

    return saved


def process_zip(zip_bytes: bytes, pack_name: str, class_dir: Path,
                counter: list[int]) -> int:
    """Extract PNGs from a zip archive and process each one."""
    saved = 0
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            png_names = [n for n in zf.namelist()
                         if n.lower().endswith((".png", ".gif"))
                         and not n.startswith("__MACOSX")]
            for name in png_names:
                raw = zf.read(name)
                saved += process_image_file(raw, pack_name, class_dir, counter)
    except zipfile.BadZipFile:
        pass
    return saved


def safe_pack_name(title: str) -> str:
    """Convert a pack title to a safe directory/file prefix."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in title)[:40]


def main():
    out = Path(OUTPUT_DIR)
    if not OUTPUT_DIR or OUTPUT_DIR == "/path/to/your/dataset/folder":
        raise ValueError("Please set OUTPUT_DIR at the top of this script.")

    # Create class directories
    for cls in CLASS_MAP:
        (out / cls).mkdir(parents=True, exist_ok=True)

    log_path = out / "download_log.jsonl"

    print("Loading HuggingFace dataset (this may take a minute on first run)...")
    ds = load_dataset("nyuuzyou/OpenGameArt-CC0", split="2d_art")
    print(f"Total 2d_art entries: {len(ds)}")

    # Filter to relevant packs
    relevant = [row for row in ds if is_relevant_pack(row["tags"])]
    print(f"Relevant packs after tag filtering: {len(relevant)}")

    if MAX_PACKS is not None:
        relevant = relevant[:MAX_PACKS]
        print(f"Dry-run cap applied: processing {len(relevant)} packs")

    total_sprites = 0
    class_counts = {cls: 0 for cls in CLASS_MAP}
    log_entries = []

    with open(log_path, "w") as log_file:
        for row in tqdm(relevant, desc="Downloading packs"):
            tags      = row["tags"]
            title     = row["title"]
            files     = row["files"]          # list of {"url": ..., "name": ..., "size": ...}
            previews  = row["preview_images"] # list of image URLs

            cls       = assign_class(tags)
            class_dir = out / cls
            pack_name = safe_pack_name(title)
            counter   = [class_counts[cls]]   # mutable ref for helpers

            pack_saved = 0

            # --- Try downloadable files first (zips & direct PNGs) ---
            for file_entry in files:
                url  = file_entry.get("url", "")
                name = file_entry.get("name", "").lower()

                if not url:
                    continue

                if name.endswith(".zip"):
                    data = download_bytes(url)
                    if data:
                        pack_saved += process_zip(data, pack_name, class_dir, counter)

                elif name.endswith((".png", ".gif")):
                    data = download_bytes(url)
                    if data:
                        pack_saved += process_image_file(data, pack_name, class_dir, counter)

            # --- Fall back to preview images if no sprites extracted yet ---
            if pack_saved == 0:
                for url in previews:
                    data = download_bytes(url)
                    if data:
                        pack_saved += process_image_file(data, pack_name, class_dir, counter)

            class_counts[cls] = counter[0]
            total_sprites += pack_saved

            entry = {
                "title":   title,
                "class":   cls,
                "tags":    tags,
                "sprites": pack_saved,
                "source":  row["url"],
            }
            log_file.write(json.dumps(entry) + "\n")
            log_entries.append(entry)

    # --- Summary ---
    print("\n" + "=" * 50)
    print("Download complete.")
    print(f"Total sprites saved : {total_sprites}")
    print("\nPer-class breakdown:")
    for cls, count in class_counts.items():
        print(f"  {cls:<10} {count:>5} sprites")
    print(f"\nFull log written to: {log_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
