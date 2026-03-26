"""AI services: Gemini Vision (mask exclusions) + Gemini Image (recolor)."""

import base64
import io
import json
import os
import re

from dotenv import load_dotenv
from PIL import Image

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

_API_KEY = os.getenv("GOOGLE_API_KEY", "")
_AVAILABLE = False
_client = None

if _API_KEY:
    try:
        from google import genai
        _client = genai.Client(api_key=_API_KEY)
        _AVAILABLE = True
    except ImportError:
        pass


def is_ai_available() -> bool:
    return _AVAILABLE


# ─── Mask exclusions via Gemini Vision ────────────────────────────────────

_EXCLUSION_PROMPT = """Ты анализируешь фото мебели для каталога. Мебель будет перекрашена в другой цвет.

Определи элементы на фото, которые НЕ должны менять цвет при перекраске:
- Чёрные пластиковые наконечники ножек
- Резиновые заглушки и накладки
- Колёсики (ролики)
- Механизмы (петли, замки, газлифты, пружины)
- Текстиль (подушки, обивка, сиденья из ткани)
- Стеклянные элементы
- Деревянные элементы (если мебель металлическая)

Для каждого найденного элемента укажи ограничивающий прямоугольник (bounding box).

Ответ — ТОЛЬКО валидный JSON массив, без markdown:
[{"label": "описание элемента", "bbox": [x1, y1, x2, y2], "confidence": 0.9}]

Координаты bbox — нормализованные от 0.0 до 1.0 (доля от ширины/высоты изображения).
x1,y1 — верхний левый угол. x2,y2 — нижний правый угол.

Если таких элементов нет, верни пустой массив: []
"""


def analyze_exclusions(image_path: str) -> list[dict]:
    """Analyze furniture photo to find non-paintable zones."""
    if not _AVAILABLE:
        return []

    try:
        img = Image.open(image_path)
        response = _client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[_EXCLUSION_PROMPT, img],
            config={
                "temperature": 0.2,
                "max_output_tokens": 2048,
            },
        )
        text = response.text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        zones = json.loads(text)
        if not isinstance(zones, list):
            return []
        valid = []
        for z in zones:
            bbox = z.get("bbox", [])
            if (len(bbox) == 4
                    and all(isinstance(v, (int, float)) for v in bbox)
                    and 0 <= bbox[0] < bbox[2] <= 1.0
                    and 0 <= bbox[1] < bbox[3] <= 1.0):
                valid.append(z)
        print(f"  AI: найдено {len(valid)} зон исключения")
        return valid
    except Exception as e:
        print(f"  AI exclusion error: {e}")
        return []


# ─── Color reference analysis ────────────────────────────────────────────

def analyze_color_reference(image_path: str, ral_code: str, color_name: str) -> dict | None:
    """Analyze a reference photo to extract calibrated color values."""
    if not _AVAILABLE:
        return None

    try:
        img = Image.open(image_path)

        response = _client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                img,
                f"This furniture is RAL {ral_code} ({color_name}). "
                f"What are the average hex colors (#RRGGBB) of: 1) metal frame 2) fabric upholstery 3) overall? "
                f"Also describe the color visually in Russian (5-8 words). "
                f'Reply as JSON: {{"hex_metal":"#...","hex_fabric":"#...","hex_average":"#...","description":"..."}}',
            ],
            config={"temperature": 0.1},
        )
        text = response.text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        result = json.loads(text)

        if "hex_average" not in result or not result["hex_average"].startswith("#"):
            return None
        print(f"  AI color analysis: {ral_code} → {result}")
        return result
    except Exception as e:
        print(f"  AI color analysis error: {e}")
        return None


# ─── Direct Gemini recolor ────────────────────────────────────────────────

_RECOLOR_SYSTEM = """You edit product photos for a furniture catalog. You receive a photo of a chair/table/shelf and must output the same item in a different color. The output must look like a real photo of this exact furniture model manufactured in the requested color — all painted surfaces and all upholstered surfaces in the new color."""


def recolor_with_gemini(
    original_path: str,
    target_hex: str,
    color_name: str,
    reference_path: str | None = None,
    color_description: str | None = None,
) -> bytes | None:
    """Recolor furniture using Gemini directly. Returns image bytes or None."""
    if not _AVAILABLE:
        return None

    try:
        img = Image.open(original_path)

        # Detect dark color: R+G+B < 200
        r = int(target_hex[1:3], 16)
        g = int(target_hex[3:5], 16)
        b = int(target_hex[5:7], 16)
        is_dark = (r + g + b) < 200

        dark_instruction = ""
        if is_dark:
            dark_instruction = (
                " This is a DARK color — pay EXTRA attention to preserving visible fabric texture "
                "and metal surface detail. Dark fabrics still show weave patterns through subtle "
                "light/shadow variation on each thread. Do NOT flatten the texture into a uniform dark surface. "
                "Boost micro-contrast on fabric so individual threads remain visible."
            )

        # Build contents: original image + optional reference + prompt
        contents = [img]

        ref_instruction = ""
        if reference_path and os.path.exists(reference_path):
            ref_img = Image.open(reference_path)
            contents.append(ref_img)
            ref_instruction = (
                " The second image is a COLOR REFERENCE — it shows the exact target color on a different piece of furniture. "
                "Use it ONLY to match the color (hue, saturation, warmth). Ignore its shape — recolor the FIRST image's furniture."
            )

        desc_instruction = ""
        if color_description:
            desc_instruction = f" The target color should look like: {color_description}."

        contents.append(
            f"Show me this exact furniture as if it was manufactured in {color_name} ({target_hex}). "
            f"The metal frame is powder-coated in {target_hex}. "
            f"The seat and backrest fabric is dyed in {target_hex}. "
            f"Every surface that had the original color now has {target_hex}. "
            f"Only the tiny black rubber caps on leg tips stay black. "
            f"Same photo, same angle, same lighting, same texture."
            f"{dark_instruction}{ref_instruction}{desc_instruction}"
        )

        response = _client.models.generate_content(
            model="gemini-3.1-flash-image-preview",
            contents=contents,
            config={
                "temperature": 0.2,
                "response_modalities": ["IMAGE", "TEXT"],
                "system_instruction": _RECOLOR_SYSTEM,
            },
        )

        # Extract image from response
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    print(f"  AI: got image ({len(part.inline_data.data)} bytes)")
                    return part.inline_data.data
        print(f"  AI: no image in response")
        return None
    except Exception as e:
        print(f"  AI recolor error: {e}")
        return None
