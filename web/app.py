"""FastAPI web application for Delo-Color batch recoloring."""

import asyncio
import os
import shutil
import tempfile
import zipfile

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web.db import (
    init_db, create_project, list_projects, get_project, delete_project,
    add_image, get_images, update_image_status, delete_image, get_results, clear_results,
)
from web.tasks import generate_masks_for_project, recolor_project, DATA_DIR

# Add parent so recolor.py is importable
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from recolor import RAL_COLORS, load_ral_library
from web.ai_service import is_ai_available, analyze_color_reference

BASE_DIR = os.path.dirname(__file__)

app = FastAPI(title="Delo-Color")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
app.mount("/data", StaticFiles(directory=os.path.join(BASE_DIR, "..", "data")), name="data")

# Track running background tasks
_running_tasks: dict[str, asyncio.Task] = {}


@app.on_event("startup")
async def startup():
    await init_db()
    os.makedirs(DATA_DIR, exist_ok=True)


# ─── Pages ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def page_projects(request: Request):
    projects = await list_projects()
    return templates.TemplateResponse("projects.html", {
        "request": request, "projects": projects,
    })


@app.get("/projects/{project_id}", response_class=HTMLResponse)
async def page_project(request: Request, project_id: str):
    project = await get_project(project_id)
    if not project:
        return HTMLResponse("Проект не найден", status_code=404)
    images = await get_images(project_id)
    results = await get_results(project_id)
    # Group results by image_id
    results_by_image = {}
    for r in results:
        results_by_image.setdefault(r["image_id"], []).append(r)
    return templates.TemplateResponse("project.html", {
        "request": request,
        "project": project,
        "images": images,
        "results_by_image": results_by_image,
        "ral_library": load_ral_library(),
        "ai_available": is_ai_available(),
        "busy": project_id in _running_tasks and not _running_tasks[project_id].done(),
    })


# ─── API: Projects ───────────────────────────────────────────────────────

@app.post("/api/projects")
async def api_create_project(request: Request):
    data = await request.json()
    name = data.get("name", "").strip()
    if not name:
        return JSONResponse({"error": "Имя проекта обязательно"}, status_code=400)
    project = await create_project(name)
    # Create directory structure
    pdir = os.path.join(DATA_DIR, project["id"])
    for sub in ("originals", "masks", "results"):
        os.makedirs(os.path.join(pdir, sub), exist_ok=True)
    return JSONResponse(project)


@app.delete("/api/projects/{project_id}")
async def api_delete_project(project_id: str):
    await delete_project(project_id)
    pdir = os.path.join(DATA_DIR, project_id)
    if os.path.exists(pdir):
        shutil.rmtree(pdir)
    return JSONResponse({"ok": True})


# ─── API: Images ─────────────────────────────────────────────────────────

@app.post("/api/projects/{project_id}/upload")
async def api_upload(project_id: str, files: list[UploadFile] = File(...)):
    project = await get_project(project_id)
    if not project:
        return JSONResponse({"error": "Проект не найден"}, status_code=404)

    pdir = os.path.join(DATA_DIR, project_id, "originals")
    os.makedirs(pdir, exist_ok=True)

    uploaded = []
    for f in files:
        content = await f.read()
        filepath = os.path.join(pdir, f.filename)
        with open(filepath, "wb") as out:
            out.write(content)
        img = await add_image(project_id, f.filename)
        uploaded.append(img)

    return JSONResponse({"uploaded": uploaded})


@app.delete("/api/images/{image_id}")
async def api_delete_image(image_id: str):
    img = await delete_image(image_id)
    if img:
        pdir = os.path.join(DATA_DIR, img["project_id"])
        # Remove original
        orig = os.path.join(pdir, "originals", img["filename"])
        if os.path.exists(orig):
            os.remove(orig)
        # Remove mask
        mask = os.path.join(pdir, "masks", f"{image_id}.png")
        if os.path.exists(mask):
            os.remove(mask)
    return JSONResponse({"ok": True})


# ─── API: Color References ───────────────────────────────────────────────

_RAL_JSON = os.path.join(BASE_DIR, "..", "ral_colors.json")
_REF_DIR = os.path.join(BASE_DIR, "..", "data", "references")


@app.post("/api/colors/{code}/reference")
async def api_upload_reference(code: str, file: UploadFile = File(...)):
    """Upload a reference photo for a RAL color and analyze it."""
    lib = load_ral_library()
    if code not in lib:
        return JSONResponse({"error": "Цвет не найден"}, status_code=404)

    os.makedirs(_REF_DIR, exist_ok=True)
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    ref_filename = f"{code}{ext}"
    ref_path = os.path.join(_REF_DIR, ref_filename)

    content = await file.read()
    with open(ref_path, "wb") as f:
        f.write(content)

    # Analyze with Gemini
    info = lib[code]
    analysis = None
    if is_ai_available():
        analysis = analyze_color_reference(ref_path, code, info.get("name_ru", code))

    # Update ral_colors.json
    import json
    with open(_RAL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    color_data = data["colors"][code]
    color_data["reference"] = ref_filename
    if analysis:
        color_data["photo_hex"] = analysis.get("hex_average", color_data.get("photo_hex", info["hex"]))
        color_data["color_description"] = analysis.get("description", "")
        color_data["hex_metal"] = analysis.get("hex_metal")
        color_data["hex_fabric"] = analysis.get("hex_fabric")

    with open(_RAL_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return JSONResponse({
        "code": code,
        "reference": ref_filename,
        "analysis": analysis,
        "photo_hex": color_data.get("photo_hex"),
        "color_description": color_data.get("color_description"),
    })


@app.delete("/api/colors/{code}/reference")
async def api_delete_reference(code: str):
    """Remove reference photo for a RAL color."""
    import json
    with open(_RAL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    if code not in data["colors"]:
        return JSONResponse({"error": "Цвет не найден"}, status_code=404)

    color_data = data["colors"][code]
    ref_file = color_data.pop("reference", None)
    color_data.pop("color_description", None)
    color_data.pop("hex_metal", None)
    color_data.pop("hex_fabric", None)

    if ref_file:
        ref_path = os.path.join(_REF_DIR, ref_file)
        if os.path.exists(ref_path):
            os.remove(ref_path)

    with open(_RAL_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return JSONResponse({"ok": True})


# ─── API: Processing ─────────────────────────────────────────────────────

@app.post("/api/projects/{project_id}/generate-masks")
async def api_generate_masks(project_id: str, request: Request):
    if project_id in _running_tasks and not _running_tasks[project_id].done():
        return JSONResponse({"error": "Задача уже выполняется"}, status_code=409)
    try:
        data = await request.json()
    except Exception:
        data = {}
    use_ai = data.get("use_ai", False)
    task = asyncio.create_task(generate_masks_for_project(project_id, use_ai=use_ai))
    _running_tasks[project_id] = task
    return JSONResponse({"status": "started"})


@app.post("/api/projects/{project_id}/recolor")
async def api_recolor(project_id: str, request: Request):
    if project_id in _running_tasks and not _running_tasks[project_id].done():
        return JSONResponse({"error": "Задача уже выполняется"}, status_code=409)
    data = await request.json()
    colors = data.get("colors", [])
    mode = data.get("mode", "hsl")  # "hsl" or "ai"
    if not colors:
        return JSONResponse({"error": "Выберите хотя бы один цвет"}, status_code=400)
    # Clear previous results and reset image statuses
    await clear_results(project_id)
    results_dir = os.path.join(DATA_DIR, project_id, "results")
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    # Reset all images so they get reprocessed
    images = await get_images(project_id)
    for img in images:
        if img["status"] in ("done", "error", "recoloring"):
            new_status = "masked" if img["has_mask"] else "uploaded"
            await update_image_status(img["id"], new_status)

    task = asyncio.create_task(recolor_project(project_id, colors, mode=mode))
    _running_tasks[project_id] = task
    return JSONResponse({"status": "started"})


@app.get("/api/projects/{project_id}/status")
async def api_status(project_id: str):
    images = await get_images(project_id)
    results = await get_results(project_id)
    busy = project_id in _running_tasks and not _running_tasks[project_id].done()
    return JSONResponse({
        "busy": busy,
        "images": [dict(i) for i in images],
        "results": [dict(r) for r in results],
    })


# ─── API: Download ───────────────────────────────────────────────────────

@app.get("/api/projects/{project_id}/download")
async def api_download_zip(project_id: str):
    results_dir = os.path.join(DATA_DIR, project_id, "results")
    if not os.path.exists(results_dir) or not os.listdir(results_dir):
        return JSONResponse({"error": "Нет результатов"}, status_code=404)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(results_dir)):
            zf.write(os.path.join(results_dir, fname), fname)

    project = await get_project(project_id)
    download_name = f"{project['name']}_results.zip" if project else "results.zip"
    return FileResponse(
        tmp.name,
        media_type="application/zip",
        filename=download_name,
    )


# ─── Run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)
