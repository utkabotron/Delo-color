"""Background tasks: mask generation and recoloring via recolor.py."""

import asyncio
import os
import sys

import cv2
import numpy as np

# Add parent dir so we can import recolor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from recolor import (
    generate_mask, load_mask, recolor_hsl, apply_exclusion_zones,
    load_ral_library, hex_to_rgb,
)

from web.db import update_image_status, add_result, get_images
from web.ai_service import is_ai_available, analyze_exclusions, recolor_with_gemini

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "projects")


def _project_dir(project_id: str) -> str:
    return os.path.join(DATA_DIR, project_id)


def _generate_mask_sync(project_id: str, image_id: str, filename: str, use_ai: bool = False):
    """Generate mask for a single image (runs in thread)."""
    pdir = _project_dir(project_id)
    image_path = os.path.join(pdir, "originals", filename)
    mask_path = os.path.join(pdir, "masks", f"{image_id}.png")
    os.makedirs(os.path.join(pdir, "masks"), exist_ok=True)

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return False

    # AI: find exclusion zones before mask generation
    exclusions = []
    if use_ai and is_ai_available():
        print(f"  AI: анализ зон исключения для {filename}...")
        exclusions = analyze_exclusions(image_path)

    # Standard mask generation
    mask = generate_mask(image_path, image_bgr, mask_path)

    # Apply AI exclusion zones
    if exclusions:
        print(f"  AI: применение {len(exclusions)} зон исключения...")
        mask = apply_exclusion_zones(mask, exclusions, image_bgr.shape)
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

    return True


async def generate_masks_for_project(project_id: str, use_ai: bool = False):
    """Generate masks for all images without masks."""
    images = await get_images(project_id)
    for img in images:
        if img["has_mask"]:
            continue
        await update_image_status(img["id"], "masking")
        try:
            ok = await asyncio.to_thread(
                _generate_mask_sync, project_id, img["id"], img["filename"], use_ai
            )
            if ok:
                await update_image_status(img["id"], "masked", has_mask=True)
            else:
                await update_image_status(img["id"], "error")
        except Exception as e:
            print(f"Mask error for {img['filename']}: {e}")
            await update_image_status(img["id"], "error")



async def recolor_project(project_id: str, ral_codes: list[str], mode: str = "hsl"):
    """Recolor all images. mode: 'hsl' (needs masks), 'ai' (Gemini, no masks needed)."""
    images = await get_images(project_id)
    all_colors = load_ral_library()

    for img in images:
        # HSL mode requires mask; AI mode works on any uploaded image
        if mode == "hsl" and not img["has_mask"]:
            continue
        if mode == "ai" and img["status"] == "error":
            continue

        await update_image_status(img["id"], "recoloring")
        pdir = os.path.join(DATA_DIR, img["project_id"])
        image_path = os.path.join(pdir, "originals", img["filename"])
        results_dir = os.path.join(pdir, "results")
        os.makedirs(results_dir, exist_ok=True)
        base = os.path.splitext(img["filename"])[0]

        # Load mask once for HSL mode
        mask = None
        if mode == "hsl":
            mask_path = os.path.join(pdir, "masks", f"{img['id']}.png")
            try:
                mask = await asyncio.to_thread(load_mask, mask_path)
                image_bgr = cv2.imread(image_path)
                if image_bgr is not None:
                    h, w = image_bgr.shape[:2]
                    if mask.shape[:2] != (h, w):
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            except Exception as e:
                print(f"Mask load error: {e}")
                await update_image_status(img["id"], "error")
                continue

        done_count = 0
        for code in ral_codes:
            info = all_colors.get(code)
            if not info:
                continue

            result_filename = f"{base}_{code}.jpg"
            result_path = os.path.join(results_dir, result_filename)

            try:
                if mode == "ai" and is_ai_available():
                    color_name = info.get("name_ru", code)
                    # Use calibrated photo_hex if available, fallback to hex
                    color_hex = info.get("photo_hex", info["hex"])
                    # Check for reference photo
                    ref_path = None
                    ref_file = info.get("reference")
                    if ref_file:
                        ref_candidate = os.path.join(os.path.dirname(DATA_DIR), "references", ref_file)
                        if os.path.exists(ref_candidate):
                            ref_path = ref_candidate
                            print(f"  AI: using reference {ref_file}")
                    color_desc = info.get("color_description")
                    print(f"  AI recolor: {img['filename']} → {code} ({color_name}, hex={color_hex})...")
                    ai_bytes = await asyncio.to_thread(
                        recolor_with_gemini, image_path, color_hex, color_name, ref_path, color_desc
                    )
                    if ai_bytes:
                        with open(result_path, "wb") as f:
                            f.write(ai_bytes)
                        print(f"  AI recolor: OK")
                    else:
                        print(f"  AI recolor: FAILED for {code}")
                        continue
                else:
                    if mask is None:
                        continue
                    image_bgr = cv2.imread(image_path)
                    result_bgr = recolor_hsl(image_bgr, mask, info["hex"])
                    cv2.imwrite(result_path, result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # Add result to DB IMMEDIATELY so UI shows it
                await add_result(img["id"], code, result_filename)
                done_count += 1

            except Exception as e:
                print(f"  Recolor error {code}: {e}")

        if done_count > 0:
            await update_image_status(img["id"], "done")
        else:
            await update_image_status(img["id"], "error")
