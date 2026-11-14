"""Stage 5 - Output: format the JSON response, attach the debug trace, render the PDF."""
import base64
import io
import time
from dataclasses import asdict
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from PIL import Image
from weasyprint import HTML

from src.agent import PageResult
from src.input import PageImage

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
CROP_PADDING = 0.03
LETTERS = "ABCDEFGH"

_env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=select_autoescape(["html", "j2"]))


def _data_uri(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def crop_visual(page: PageImage, bbox: dict) -> bytes:
    """Cut the original pixels out of the slide; nothing is redrawn."""
    with Image.open(io.BytesIO(page.png_bytes)) as image:
        width, height = image.size
        left = max(0.0, bbox["x"] - CROP_PADDING)
        top = max(0.0, bbox["y"] - CROP_PADDING)
        right = min(1.0, bbox["x"] + bbox["width"] + CROP_PADDING)
        bottom = min(1.0, bbox["y"] + bbox["height"] + CROP_PADDING)
        box = (int(left * width), int(top * height), max(int(right * width), int(left * width) + 2),
               max(int(bottom * height), int(top * height) + 2))
        buffer = io.BytesIO()
        image.crop(box).save(buffer, format="PNG")
        return buffer.getvalue()


def build_response(results: list[PageResult], elapsed: float | None = None) -> dict:
    """Clean JSON response with a minimal per-page debug trace."""
    pages = []
    for result in results:
        data = asdict(result)
        trace = data.pop("trace")
        trace.append({"stage": "output", "tool_used": "build_response", "latency": "0.0s"})
        pages.append({**data, "trace": trace})
    response = {"pages": pages}
    if elapsed is not None:
        response["total_latency"] = f"{elapsed:.1f}s"
    return response


def _slide_context(result: PageResult, page: PageImage) -> dict:
    visuals = []
    if result.action != "preserve_only":
        visuals = [{"label": v["label"], "src": _data_uri(crop_visual(page, v["bbox"]))} for v in result.visuals]
    text_length = sum(len(p) for p in result.paragraphs)
    return {
        "number": result.page_number,
        "title": result.title,
        "paragraphs": result.paragraphs,
        "layout": result.layout,
        "visuals": visuals,
        "full_page": _data_uri(page.png_bytes) if result.action == "preserve_only" else None,
        "size_class": "text-lg" if text_length < 220 else "text-md" if text_length < 420 else "text-sm",
        "questions": [
            {
                "before": q["sentence"].split("[MASK]")[0],
                "after": q["sentence"].split("[MASK]")[1] if "[MASK]" in q["sentence"] else "",
                "options": list(zip(LETTERS, q["options"])),
                "answer": LETTERS[q["options"].index(q["answer"])] if q["answer"] in q["options"] else "?",
                "answer_text": q["answer"],
            }
            for q in result.masks
        ],
    }


def render_pdf(results: list[PageResult], pages: list[PageImage]) -> bytes | None:
    """Render accepted pages to a PDF; returns None when every page was refused."""
    by_number = {p.page_number: p for p in pages}
    slides = [_slide_context(r, by_number[r.page_number]) for r in results if r.action != "refuse"]
    if not slides:
        return None
    html = _env.get_template("slide.html.j2").render(slides=slides)
    return HTML(string=html).write_pdf()


def timed_render(results: list[PageResult], pages: list[PageImage]) -> tuple[bytes | None, float]:
    start = time.perf_counter()
    pdf = render_pdf(results, pages)
    return pdf, time.perf_counter() - start
