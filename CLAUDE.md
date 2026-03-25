# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Язык общения

Отвечай на **русском языке**.

## Назначение

Инструмент пакетной перекраски продуктовых фото (мебель delo-design.com) в 26 цветов RAL. Единственный скрипт `recolor.py` — без фреймворков, без сервера.

## Команды

```bash
# Установка (один раз)
python3 -m venv .venv
.venv/bin/pip install "rembg[cpu]" Pillow numpy opencv-contrib-python-headless

# Все 26 цветов
.venv/bin/python recolor.py --input photo.jpg

# Конкретные цвета
.venv/bin/python recolor.py --input photo.jpg --colors RAL9005,RAL9003

# Своя маска + другая папка
.venv/bin/python recolor.py --input photo.jpg --mask mask.png --output results/

# Отладка маски (сохраняет 6 промежуточных масок в output/debug/)
.venv/bin/python recolor.py --input photo.jpg --debug

# Тюнинг маски
.venv/bin/python recolor.py --input photo.jpg --threshold 12 --sharpness 0.6
```

## Архитектура (recolor.py)

Два этапа: **маскирование** → **перекраска**.

### Маскирование (generate_mask)
Гибридная маска из 4 шагов — решает проблему тонких прутьев с щелями:
1. **rembg** — грубый силуэт объекта (пространственная граница)
2. **Color distance** — Delta E каждого пикселя от цвета фона (определяется автоматически по краям фото). Sigmoid-функция с настраиваемыми threshold/sharpness
3. **Комбинация** — `color_mask × rembg_expanded`, тени под ножками вырезаются через `rembg_binary`
4. **Guided filter** — `cv2.ximgproc.guidedFilter` выравнивает маску по реальным краям + лёгкий GaussianBlur (1px) для антиалиасинга

### Перекраска (recolor_hsl)
HSL-based с тремя ключевыми решениями:
- **H**: бинарная замена (не интерполяция — иначе синий идёт через зелёный)
- **S**: 80% от целевого, модулируется по яркости (натуральнее для металла)
- **L**: ремап с percentile-based контрастом (min 55 единиц), тёмные цвета сдвигаются вверх для читаемости
- **Blending**: alpha blend с **обесцвеченным** оригиналом (не бежевым) — убирает цветной ореол на краях

### Ключевые параметры для тюнинга
- `threshold` (default 15.0) — порог Delta E. Меньше = больше пикселей в маске
- `sharpness` (default 0.5) — крутизна sigmoid. Больше = резче границы маски
- `t_s * 0.80` в recolor_hsl — коэффициент смягчения насыщенности
- `tgt_range` clip 55 — минимальный контраст L для тёмных цветов
- `(50 - t_l) * 0.4` — сдвиг L вверх для тёмных цветов

## Известные ограничения

- Белый на белом фоне — объективно мал контраст, улучшить можно только с лучшим исходником
- Чёрный — прутки на сиденье слабо различимы (в оригинале мало контраста в этой зоне)
- Маска рассчитана на **светлый/белый фон** — для тёмного фона нужна другая логика detect_background_color
