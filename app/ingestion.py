"""Load the NCERT PDF, detect chapters from the real text, chunk with page + chapter metadata, build the FAISS index."""
import bisect
import hashlib
import json
import re
import shutil
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader, apply_configuration

from app.config import get_settings

META_FILE = "index_meta.json"

# Chapter title block on the opening page: "<title lines>\n<N>CHAPTER", optionally preceded
# by a running header ("Science58") and/or a stray "Activity 13.1" line.
_TITLE_RE = re.compile(
    r"^(?:Science\d+\n)?(?:[^\n]*(?:Activity|Figure)\s[\d.]+[^\n]*\n)?((?:[^\n]+\n){1,3}?)(\d{1,2})CHAPTER"
)
# Running header on odd pages: "<Chapter name> <page number>".
_HEADER_RE = re.compile(r"^(?!Science\d)(.+?)\s+(\d{1,3})\n")


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=get_settings().embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


def pdf_fingerprint(pdf_path: Path) -> str:
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract_pages(pdf_path: Path) -> list[str]:
    """Raw text of every page, in order. Fails loudly if any page cannot be read."""
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    # Some pages decompress past pypdf's default 75 MB safety cap; the PDF is a trusted local file.
    with apply_configuration(zlib_maximum_output_length=0):
        for i, page in enumerate(reader.pages, start=1):
            try:
                pages.append(page.extract_text() or "")
            except Exception as exc:
                raise RuntimeError(f"Could not extract page {i} of {pdf_path}: {exc}") from exc
    if not any(pages):
        raise RuntimeError(f"No text extracted from {pdf_path}")
    return pages


def detect_chapters(raw_pages: list[str]) -> list[tuple[int, str]]:
    """Return [(chapter_number, chapter_name)] per page, detected from headings in the PDF text."""
    n = len(raw_pages)
    names: list[tuple[int, str] | None] = [None] * n

    # 1) Explicit chapter-opening pages ("Heading ... 7CHAPTER").
    titled: set[int] = set()
    for i, text in enumerate(raw_pages):
        m = _TITLE_RE.match(text[:300])
        if m:
            names[i] = (int(m.group(2)), " ".join(m.group(1).split()))
            titled.add(i)

    # 2) Running headers ("Heredity 129") name chapters whose opening page has no title block.
    known = {name for _, name in filter(None, names)}
    for i, text in enumerate(raw_pages):
        if names[i]:
            continue
        m = _HEADER_RE.match(text)
        if m and int(m.group(2)) == i + 1:
            names[i] = (0, " ".join(m.group(1).split()))
    # Chapters named only by running headers are numbered as the one after the previous chapter;
    # explicit numbers from title blocks must be strictly increasing.
    numbering: dict[str, int] = {}
    last = 0
    for item in names:
        if not item or item[1] in numbering:
            continue
        number = item[0] or last + 1
        if number <= last:
            raise RuntimeError(f"Chapter numbering mismatch at {item[1]!r}")
        numbering[item[1]] = last = number
    for i, item in enumerate(names):
        if item:
            names[i] = (numbering[item[1]], item[1])

    # 3) Pages with no header (even pages) inherit the previous page's chapter, except a chapter's
    #    opening page that has no title block (e.g. Heredity, p.128), which takes the next header's chapter.
    result: list[tuple[int, str] | None] = list(names)
    for i in range(n):
        if result[i]:
            continue
        prev = result[i - 1] if i else None
        nxt = names[i + 1] if i + 1 < n else None
        result[i] = nxt if (nxt and nxt != prev and i + 1 not in titled) else prev
    if any(r is None for r in result):
        raise RuntimeError("Could not assign a chapter to every page")
    return result  # type: ignore[return-value]


def clean_page(text: str, page_no: int) -> str:
    """Strip running headers/footers and extraction artefacts, then reflow lines into paragraphs."""
    text = text.replace("/square6", "• ").replace(" ", " ")
    text = re.sub(r"Reprint 2026-27", "", text)
    lines = text.split("\n")
    # Drop the running header line ("Science58" / "Heredity 129").
    if lines and (re.fullmatch(r"Science\s?\d+", lines[0].strip()) or re.fullmatch(r".+\s%d" % page_no, lines[0].strip())):
        lines = lines[1:]
    text = "\n".join(lines)
    # Drop-cap artefact on chapter pages: "I\nn the last Chapter" -> "In the last Chapter".
    text = re.sub(r"(?m)^([A-Za-z])\n(?=[a-z])", r"\1", text)
    text = re.sub(r"\d{1,2}CHAPTER\n", "", text)
    # Bold headings are drawn several times: "Activity 1.2Activity 1.2Activity 1.2" -> "Activity 1.2".
    text = re.sub(r"(.{4,80}?)\1{2,}", r"\1", text)
    # Reflow: a newline not after sentence-ending punctuation / before a bullet is a soft wrap.
    text = re.sub(r"(?<![.?!:;)\]])\n(?![•\d(]|[A-Z][a-z]* \d)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def chunk_pages(raw_pages: list[str], source: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Chunk chapter by chapter so explanations spanning a page break stay together;
    each chunk keeps the page it starts on and its chapter."""
    chapters = detect_chapters(raw_pages)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n", ". ", " ", ""],
        add_start_index=True,
    )
    docs: list[Document] = []
    i = 0
    while i < len(raw_pages):
        j = i
        while j + 1 < len(raw_pages) and chapters[j + 1] == chapters[i]:
            j += 1
        text, starts = "", []
        for p in range(i, j + 1):
            starts.append((len(text), p + 1))
            text += clean_page(raw_pages[p], p + 1) + "\n"
        offsets = [s for s, _ in starts]
        number, name = chapters[i]
        for piece in splitter.create_documents([text]):
            idx = piece.metadata["start_index"]
            page = starts[bisect.bisect_right(offsets, idx) - 1][1]
            if len(piece.page_content.strip()) < 40:
                continue
            docs.append(Document(
                page_content=piece.page_content.strip(),
                metadata={"source": source, "page": page, "chapter": name, "chapter_number": number},
            ))
        i = j + 1
    return docs


def build_index(pdf_path: Path | None = None, index_dir: Path | None = None) -> dict:
    """Rebuild the FAISS index from scratch from the NCERT PDF (any old index is deleted first)."""
    s = get_settings()
    pdf_path = Path(pdf_path or s.pdf_path)
    index_dir = Path(index_dir or s.index_dir)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    if index_dir.exists():
        for child in index_dir.iterdir():
            if child.name == ".gitkeep":
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    index_dir.mkdir(parents=True, exist_ok=True)

    raw_pages = extract_pages(pdf_path)
    docs = chunk_pages(raw_pages, pdf_path.name, s.chunk_size, s.chunk_overlap)
    store = FAISS.from_documents(docs, get_embeddings(), distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    store.save_local(str(index_dir))

    meta = {
        "source": pdf_path.name,
        "pdf_sha256": pdf_fingerprint(pdf_path),
        "pages": len(raw_pages),
        "chunks": len(docs),
        "chapters": sorted({(d.metadata["chapter_number"], d.metadata["chapter"]) for d in docs}),
        "embedding_model": s.embedding_model,
        "chunk_size": s.chunk_size,
        "chunk_overlap": s.chunk_overlap,
    }
    (index_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_index(pdf_path: Path | None = None, index_dir: Path | None = None) -> tuple[FAISS, dict]:
    """Load the FAISS index, refusing one that was not built from the current PDF."""
    s = get_settings()
    pdf_path = Path(pdf_path or s.pdf_path)
    index_dir = Path(index_dir or s.index_dir)
    meta_path = index_dir / META_FILE
    if not meta_path.exists():
        raise RuntimeError("No FAISS index found. Run: python scripts/build_index.py")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("pdf_sha256") != pdf_fingerprint(pdf_path):
        raise RuntimeError("FAISS index was built from a different PDF. Run: python scripts/build_index.py")
    store = FAISS.load_local(
        str(index_dir), get_embeddings(), allow_dangerous_deserialization=True,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    return store, meta
