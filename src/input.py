"""Stage 1 - Input: validate uploads and queries, rasterize PDFs, encode images."""
import base64
import io
import re
from dataclasses import dataclass

import pymupdf as fitz
from PIL import Image, UnidentifiedImageError

MAX_PAGES = 10
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_QUERY_CHARS = 500
MAX_IMAGE_SIDE = 1800
PDF_ZOOM = 2.0
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

_INJECTION = re.compile(
    r"(ignore|disregard|forget)\s+(all\s+|any\s+)?(previous|prior|above|your)\s+(instructions|rules)"
    r"|system prompt|reveal your (prompt|instructions)",
    re.IGNORECASE,
)


class InputError(ValueError):
    """Raised for uploads or queries that fail validation."""


@dataclass
class PageImage:
    page_number: int
    png_bytes: bytes

    @property
    def base64(self) -> str:
        return base64.b64encode(self.png_bytes).decode("ascii")

    @property
    def size(self) -> tuple[int, int]:
        with Image.open(io.BytesIO(self.png_bytes)) as im:
            return im.size


def normalize_query(query: str | None) -> str:
    """Trim, collapse whitespace and cap length. Injection attempts are kept so the agent can refuse them."""
    query = re.sub(r"\s+", " ", (query or "")).strip()
    return query[:MAX_QUERY_CHARS]


def looks_like_injection(query: str) -> bool:
    return bool(_INJECTION.search(query))


def _to_png(image: Image.Image) -> bytes:
    image = image.convert("RGB")
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def load_pages(filename: str, data: bytes) -> list[PageImage]:
    """Validate the upload and return one PNG per page (PDF) or a single page (image)."""
    if not data:
        raise InputError("The uploaded file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise InputError(f"File is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB.")

    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if extension == "pdf" or data[:5] == b"%PDF-":
        try:
            document = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise InputError("The PDF could not be read.") from exc
        if document.needs_pass:
            raise InputError("Password-protected PDFs are not supported.")
        if document.page_count == 0:
            raise InputError("The PDF has no pages.")
        if document.page_count > MAX_PAGES:
            raise InputError(f"The PDF has {document.page_count} pages; the limit is {MAX_PAGES}.")
        pages = []
        for index, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(matrix=fitz.Matrix(PDF_ZOOM, PDF_ZOOM), alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            pages.append(PageImage(index, _to_png(image)))
        return pages

    if extension in IMAGE_EXTENSIONS:
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                return [PageImage(1, _to_png(image))]
        except (UnidentifiedImageError, OSError) as exc:
            raise InputError("The image could not be read.") from exc

    raise InputError("Unsupported file type. Upload a PDF, PNG, JPG or WEBP.")
