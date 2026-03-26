#!/usr/bin/env python3
"""
Пакетная перекраска изделий в цвета RAL.

Гибридная маска (rembg + color distance + guided filter)
+ HSL "Color" blend mode для перекраски с сохранением текстуры.

Использование:
    python recolor.py --input photo.jpg
    python recolor.py --input photo.jpg --mask mask.png --colors RAL9003,RAL5013
    python recolor.py --input photo.jpg --threshold 12 --sharpness 0.6 --debug
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

# ─── Библиотека RAL-цветов ────────────────────────────────────────────────

_RAL_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ral_colors.json")


def load_ral_library(path: str = _RAL_JSON) -> dict:
    """Загрузка полной библиотеки RAL из JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["colors"]


def get_catalog_colors(tag: str = "delo-design") -> dict:
    """Возвращает только каталожные цвета (помеченные тегом)."""
    all_colors = load_ral_library()
    return {
        code: {"hex": info["hex"], "name": info["name_ru"]}
        for code, info in all_colors.items()
        if tag in info.get("catalog", [])
    }


def get_colors_by_codes(codes: list[str]) -> dict:
    """Возвращает цвета по списку кодов из полной библиотеки."""
    all_colors = load_ral_library()
    result = {}
    for code in codes:
        if code in all_colors:
            info = all_colors[code]
            result[code] = {"hex": info["hex"], "name": info["name_ru"]}
    return result


# Обратная совместимость: RAL_COLORS = 26 каталожных цветов delo-design
RAL_COLORS = get_catalog_colors()


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """#RRGGBB -> (R, G, B)."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# ─── Зоны исключения (AI) ──────────────────────────────────────────────────

def apply_exclusion_zones(
    mask: np.ndarray,
    exclusion_zones: list[dict],
    image_shape: tuple,
    fade_px: int = 8,
) -> np.ndarray:
    """
    Обнуляет маску в зонах исключения (bbox от Gemini Vision).
    Внутри bbox маска = 0, по краям — плавный переход за fade_px пикселей.
    """
    h, w = image_shape[:2]
    mask = mask.copy()

    for zone in exclusion_zones:
        bbox = zone.get("bbox", [])
        if len(bbox) != 4:
            continue

        # Core zone (exact bbox)
        cx1 = int(bbox[0] * w)
        cy1 = int(bbox[1] * h)
        cx2 = int(bbox[2] * w)
        cy2 = int(bbox[3] * h)

        # Expanded zone (with fade margin)
        x1 = max(0, cx1 - fade_px)
        y1 = max(0, cy1 - fade_px)
        x2 = min(w, cx2 + fade_px)
        y2 = min(h, cy2 + fade_px)

        if x2 <= x1 or y2 <= y1:
            continue

        # Zero out the core zone completely
        mask[cy1:cy2, cx1:cx2] = 0

        # Fade margins: gradual transition from 0 (at core edge) to 1 (at fade edge)
        # Top margin
        if cy1 > y1:
            for i in range(cy1 - y1):
                factor = (i + 1) / (cy1 - y1 + 1)
                row = y1 + i
                mask[row, cx1:cx2] *= factor
        # Bottom margin
        if y2 > cy2:
            for i in range(y2 - cy2):
                factor = (i + 1) / (y2 - cy2 + 1)
                row = y2 - 1 - i
                mask[row, cx1:cx2] *= factor
        # Left margin
        if cx1 > x1:
            for j in range(cx1 - x1):
                factor = (j + 1) / (cx1 - x1 + 1)
                col = x1 + j
                mask[y1:y2, col] *= factor
        # Right margin
        if x2 > cx2:
            for j in range(x2 - cx2):
                factor = (j + 1) / (x2 - cx2 + 1)
                col = x2 - 1 - j
                mask[y1:y2, col] *= factor

    return mask


# ─── Маскирование ──────────────────────────────────────────────────────────

def detect_background_color(image_bgr: np.ndarray) -> np.ndarray:
    """
    Определяет цвет фона по краям изображения.
    Возвращает медианный LAB-цвет фона как (L, a, b) float64.
    """
    h, w = image_bgr.shape[:2]
    margin = max(int(min(h, w) * 0.05), 5)

    # Собираем пиксели с 4 краёв
    top = image_bgr[:margin, :, :]
    bottom = image_bgr[-margin:, :, :]
    left = image_bgr[:, :margin, :]
    right = image_bgr[:, -margin:, :]

    border_pixels = np.concatenate([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3),
    ], axis=0)

    # Конвертируем в LAB
    border_lab = cv2.cvtColor(
        border_pixels.reshape(1, -1, 3), cv2.COLOR_BGR2LAB
    ).reshape(-1, 3).astype(np.float64)

    return np.median(border_lab, axis=0)


def compute_color_distance(image_lab: np.ndarray, bg_lab: np.ndarray) -> np.ndarray:
    """
    Попиксельная Delta E (CIE76) от цвета фона.
    Возвращает float32 (H, W).
    """
    diff = image_lab.astype(np.float64) - bg_lab.reshape(1, 1, 3)
    return np.sqrt(np.sum(diff ** 2, axis=2)).astype(np.float32)


def create_color_mask(delta_e: np.ndarray, threshold: float, sharpness: float) -> np.ndarray:
    """
    Sigmoid-маска на основе цветового расстояния.
    threshold — Delta E при которой маска = 0.5
    sharpness — крутизна перехода (больше = резче)
    """
    return (1.0 / (1.0 + np.exp(-sharpness * (delta_e - threshold)))).astype(np.float32)


def generate_mask(
    image_path: str,
    image_bgr: np.ndarray,
    mask_save_path: str,
    threshold: float = 15.0,
    sharpness: float = 0.5,
    debug_dir: str | None = None,
) -> np.ndarray:
    """
    Гибридная маска: rembg (пространственная граница) + color distance (точность).
    """
    try:
        from rembg import remove
    except ImportError:
        raise RuntimeError("rembg не установлен. Установите: pip install 'rembg[cpu]'")

    h, w = image_bgr.shape[:2]

    # Шаг 1: rembg для грубого силуэта
    print("  Шаг 1/4: rembg (грубый силуэт)...")
    img_pil = Image.open(image_path).convert("RGBA")
    result = remove(img_pil)
    rembg_alpha = np.array(result)[:, :, 3].astype(np.float32) / 255.0

    # Бинаризуем: точная граница (без теней) + расширенная зона интереса
    rembg_binary = (rembg_alpha > 0.5).astype(np.uint8)
    kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    rembg_expanded = cv2.dilate(rembg_binary, kernel_expand, iterations=1).astype(np.float32)

    # Шаг 2: Color distance маска
    print("  Шаг 2/4: Color distance...")
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    bg_lab = detect_background_color(image_bgr)
    delta_e = compute_color_distance(image_lab, bg_lab)
    color_mask = create_color_mask(delta_e, threshold, sharpness)

    # Шаг 3: Комбинация
    # color_mask отвечает за точность (щели между прутьями)
    # rembg_expanded ограничивает зону (отсекает стрелки и прочий UI)
    # rembg_binary (без расширения) убирает тени под ножками
    print("  Шаг 3/4: Комбинация масок...")
    combined = color_mask * rembg_expanded
    # Тени: пиксели вне силуэта rembg, но с малым deltaE — это тени, убираем
    shadow_zone = (rembg_binary == 0).astype(np.float32)
    combined = combined * (1 - shadow_zone)

    # Заполнение дыр: если rembg уверен что это объект, а color_mask низкий
    # (светлый объект на белом фоне), гарантируем минимальное значение маски
    combined = np.maximum(combined, rembg_binary.astype(np.float32) * 0.85)

    # Шаг 4: Guided filter для чётких краёв
    print("  Шаг 4/4: Guided filter (уточнение краёв)...")
    guide = image_bgr.astype(np.float32) / 255.0
    mask = cv2.ximgproc.guidedFilter(
        guide=guide,
        src=combined,
        radius=4,
        eps=0.01,
    )
    # Минимальное сглаживание (1px) для антиалиасинга краёв прутков
    mask = cv2.GaussianBlur(mask, (3, 3), 0.7)
    mask = np.clip(mask, 0, 1)

    # Сохраняем маску
    cv2.imwrite(mask_save_path, (mask * 255).astype(np.uint8))
    print(f"  Маска сохранена: {mask_save_path}")

    # Debug: промежуточные маски
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(debug_dir, "debug_1_rembg_raw.png"),
            (rembg_alpha * 255).astype(np.uint8),
        )
        cv2.imwrite(
            os.path.join(debug_dir, "debug_2_rembg_expanded.png"),
            (rembg_expanded * 255).astype(np.uint8),
        )
        cv2.imwrite(
            os.path.join(debug_dir, "debug_3_color_distance.png"),
            np.clip(delta_e * 5, 0, 255).astype(np.uint8),
        )
        cv2.imwrite(
            os.path.join(debug_dir, "debug_4_color_mask.png"),
            (color_mask * 255).astype(np.uint8),
        )
        cv2.imwrite(
            os.path.join(debug_dir, "debug_5_combined.png"),
            (combined * 255).astype(np.uint8),
        )
        cv2.imwrite(
            os.path.join(debug_dir, "debug_6_guided.png"),
            (mask * 255).astype(np.uint8),
        )
        print(f"  Debug маски сохранены в: {debug_dir}")

    return mask


def load_mask(mask_path: str) -> np.ndarray:
    """Загружает готовую маску из файла PNG."""
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        print(f"ОШИБКА: не удалось загрузить маску: {mask_path}")
        sys.exit(1)
    return mask_img.astype(np.float32) / 255.0


# ─── Перекраска ─────────────────────────────────────────────────────────────

def recolor_hsl(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    target_hex: str,
) -> np.ndarray:
    """
    Перекраска интерполяцией в HSL-пространстве.
    Маска модулирует степень цветового сдвига (не alpha blending в RGB).
    """
    r, g, b = hex_to_rgb(target_hex)
    target_bgr = np.uint8([[[b, g, r]]])
    target_hls = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HLS)
    t_h, t_l, t_s = target_hls[0, 0].astype(np.float64)

    # Конвертируем изображение в HLS
    image_hls = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HLS).astype(np.float64)
    orig_h = image_hls[:, :, 0]
    orig_l = image_hls[:, :, 1]
    orig_s = image_hls[:, :, 2]

    # Маска как float для плавной интерполяции
    m = mask.astype(np.float64)

    binary_mask = mask > 0.5
    if not np.any(binary_mask):
        return image_bgr.copy()

    # Статистики L по пикселям изделия
    masked_l = orig_l[binary_mask]
    l_source_median = np.median(masked_l)
    if l_source_median < 1.0:
        l_source_median = 1.0
    l_5 = np.percentile(masked_l, 5)
    l_95 = np.percentile(masked_l, 95)
    src_range = max(l_95 - l_5, 1.0)

    # ─── L: ремап с гарантированным контрастом ───
    tgt_range = src_range * (t_l / l_source_median)
    tgt_range = np.clip(tgt_range, 55, src_range * 1.2)

    l_remapped = t_l + (orig_l - l_source_median) * (tgt_range / src_range)
    if t_l < 50:
        l_remapped = l_remapped + (50 - t_l) * 0.4
    l_remapped = np.clip(l_remapped, 0, 255)

    # ─── S: целевая с модуляцией по яркости, 80% интенсивности ───
    t_s_soft = t_s * 0.80
    if t_s_soft > 5 and l_source_median > 1:
        s_factor = 0.7 + 0.3 * (orig_l / l_source_median)
        s_modulated = t_s_soft * np.clip(s_factor, 0.3, 1.3)
    else:
        s_modulated = np.full_like(orig_s, t_s_soft)

    # ─── Полная замена H/S/L внутри бинарной маски ───
    new_h = np.where(binary_mask, t_h, orig_h)
    new_s = np.where(binary_mask, np.clip(s_modulated, 0, 255), orig_s)
    new_l = np.where(binary_mask, np.clip(l_remapped, 0, 255), orig_l)

    # Защита чёрных наконечников (L < 12%, S < 10%)
    dark_parts = binary_mask & (orig_l < 30) & (orig_s < 25)
    new_h[dark_parts] = orig_h[dark_parts]
    new_s[dark_parts] = orig_s[dark_parts]
    new_l[dark_parts] = orig_l[dark_parts]

    # Собираем HLS → BGR
    result_hls = np.stack([
        np.clip(new_h, 0, 179),
        np.clip(new_l, 0, 255),
        np.clip(new_s, 0, 255),
    ], axis=2).astype(np.uint8)

    result_bgr = cv2.cvtColor(result_hls, cv2.COLOR_HLS2BGR)

    # Alpha blending с ОБЕСЦВЕЧЕННЫМ оригиналом (не бежевым)
    # Так на краях: серый→цвет, а не бежевый→цвет
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    base_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float64)

    mask_3d = m[:, :, np.newaxis]
    blended = base_bgr * (1 - mask_3d) + result_bgr.astype(np.float64) * mask_3d

    return np.clip(blended, 0, 255).astype(np.uint8)


# ─── Пакетная обработка ────────────────────────────────────────────────────

def process_batch(
    image_path: str,
    colors: dict,
    output_dir: str,
    mask_path: str | None = None,
    threshold: float = 15.0,
    sharpness: float = 0.5,
    debug: bool = False,
):
    """Перекрашивает изделие во все указанные цвета."""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"ОШИБКА: не удалось загрузить изображение: {image_path}")
        sys.exit(1)

    h, w = image_bgr.shape[:2]
    print(f"Изображение: {image_path} ({w}x{h})")

    # Маска
    mask_save = os.path.join(output_dir, "mask.png")
    if mask_path and os.path.exists(mask_path):
        print(f"Загрузка маски: {mask_path}")
        mask = load_mask(mask_path)
        if mask.shape[:2] != (h, w):
            print(f"  Масштабирование маски {mask.shape[:2]} -> ({h}, {w})")
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        debug_dir = os.path.join(output_dir, "debug") if debug else None
        mask = generate_mask(
            image_path, image_bgr, mask_save,
            threshold=threshold,
            sharpness=sharpness,
            debug_dir=debug_dir,
        )

    # Пакетная перекраска
    total = len(colors)
    print(f"\nПерекраска в {total} цветов...")

    for i, (code, info) in enumerate(colors.items(), 1):
        name = info["name"]
        hex_color = info["hex"]

        result = recolor_hsl(image_bgr, mask, hex_color)

        filename = f"{code}_{name}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, result, [cv2.IMWRITE_JPEG_QUALITY, 95])

        print(f"  [{i:2d}/{total}] {code} {name} ({hex_color}) -> {filename}")

    print(f"\nГотово! Результаты в: {output_dir}")


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Пакетная перекраска изделий в цвета RAL"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Путь к фото изделия"
    )
    parser.add_argument(
        "--output", "-o", default="output",
        help="Папка для результатов (по умолчанию: output/)"
    )
    parser.add_argument(
        "--mask", "-m", default=None,
        help="Путь к готовой маске PNG (если не указан — генерируется автоматически)"
    )
    parser.add_argument(
        "--colors", "-c", default="catalog",
        help="'catalog' (26 каталожных delo-design), 'all' (все 213 RAL) "
             "или коды через запятую: RAL9003,RAL5013 (по умолчанию: catalog)"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=15.0,
        help="Порог Delta E для цветовой маски (по умолчанию: 15.0)"
    )
    parser.add_argument(
        "--sharpness", "-s", type=float, default=0.5,
        help="Крутизна сигмоиды маски (по умолчанию: 0.5)"
    )
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Сохранять промежуточные маски для отладки"
    )

    args = parser.parse_args()

    # Определяем набор цветов
    choice = args.colors.lower().strip()
    if choice == "catalog":
        colors = get_catalog_colors()
    elif choice == "all":
        all_lib = load_ral_library()
        colors = {
            code: {"hex": info["hex"], "name": info["name_ru"]}
            for code, info in all_lib.items()
        }
    else:
        codes = [c.strip().upper() for c in args.colors.split(",")]
        colors = get_colors_by_codes(codes)
        unknown = [c for c in codes if c.upper() not in colors]
        for code in unknown:
            print(f"Предупреждение: неизв��стный код {code}, пропускаю")
        if not colors:
            print("ОШИБКА: не найдено ни одного валидного RAL-кода")
            sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    process_batch(
        args.input, colors, args.output, args.mask,
        threshold=args.threshold,
        sharpness=args.sharpness,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
