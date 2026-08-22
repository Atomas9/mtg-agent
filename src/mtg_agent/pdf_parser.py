import argparse
import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF_PATH = PROJECT_ROOT / "data" / "raw" / "MagicCompRules 20260417.pdf"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "rule_chunks.jsonl"

PARSER_VERSION = "1.0"
CHAPTER_INDENT = 90.0
CHAPTER_FONT_SIZE = 12.0
SECTION_INDENT = 90.0
RULE_INDENT = 105.1
SUBRULE_INDENT = 120.2
INDENT_TOLERANCE = 3.0
FONT_SIZE_TOLERANCE = 1.0

# ----------
# RE
# ----------
CHAPTER_PATTERN = re.compile(r"^(?P<number>[1-9])\.\s+(?P<title>.+)$")
SECTION_PATTERN = re.compile(r"^(?P<number>\d{3})\.\s+(?P<title>.+)$")
RULE_PATTERN = re.compile(r"^(?P<number>\d{3}\.\d+)\.\s+(?P<text>.+)$")
SUBRULE_PATTERN = re.compile(
    r"^(?P<base_rule>\d{3}\.\d+)(?P<letter>[a-z])\s+(?P<text>.+)$"
)


# ----------
# CLASSES
# ----------
class RuleChunk(BaseModel):
    chunk_id: str
    chapter_number: int
    chapter_title: str
    section_number: int
    section_title: str
    rule_number: str
    subrule_numbers: list[str] = Field(default_factory=list)
    page_start: int
    page_end: int
    source_file: str
    source_sha256: str
    parser_version: str = PARSER_VERSION
    raw_text: str
    chunk_text: str


@dataclass(frozen=True)
class PdfLine:
    text: str
    page_number: int
    x0: float
    font_size: float


@dataclass
class PendingRule:
    chapter_number: int
    chapter_title: str
    section_number: int
    section_title: str
    rule_number: str
    page_start: int
    page_end: int
    paragraphs: list[str]
    subrule_numbers: list[str]


def extract_pdf_lines(pdf_path: Path) -> Iterator[PdfLine]:
    """Extract non-empty lines and their position from the PDF."""
    with pymupdf.open(pdf_path) as document:
        for page_index, page in enumerate(document, start=1):
            page_dict = page.get_text("dict", sort=True)

            for block in page_dict["blocks"]:
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    text = "".join(span["text"] for span in line["spans"])
                    text = re.sub(r"\s+", " ", text).strip()

                    if not text:
                        continue

                    x0, _, _, _ = line["bbox"]
                    yield PdfLine(
                        text=text,
                        page_number=page_index,
                        x0=x0,
                        font_size=line["spans"][0]["size"],
                    )


def _has_indent(line: PdfLine, expected_indent: float) -> bool:
    return abs(line.x0 - expected_indent) <= INDENT_TOLERANCE


def identify_chapter(line: PdfLine) -> re.Match[str] | None:
    has_chapter_style = (
        _has_indent(line, CHAPTER_INDENT)
        and abs(line.font_size - CHAPTER_FONT_SIZE) <= FONT_SIZE_TOLERANCE
    )
    if not has_chapter_style:
        return None
    return CHAPTER_PATTERN.fullmatch(line.text)


def identify_section(line: PdfLine) -> re.Match[str] | None:
    if not _has_indent(line, SECTION_INDENT):
        return None
    return SECTION_PATTERN.fullmatch(line.text)


def identify_rule(line: PdfLine) -> re.Match[str] | None:
    if not _has_indent(line, RULE_INDENT):
        return None
    return RULE_PATTERN.fullmatch(line.text)


def identify_subrule(line: PdfLine) -> re.Match[str] | None:
    if not _has_indent(line, SUBRULE_INDENT):
        return None
    return SUBRULE_PATTERN.fullmatch(line.text)


def _belongs_to_section(rule_number: str, section_number: int) -> bool:
    return rule_number.startswith(f"{section_number}.")


def _append_continuation(pending_rule: PendingRule, text: str) -> None:
    if text.startswith(("Example:", "Note:", "•")):
        pending_rule.paragraphs.append(text)
    else:
        separator = "" if pending_rule.paragraphs[-1].endswith("-") else " "
        pending_rule.paragraphs[-1] += f"{separator}{text}"


def _make_chunk(
    pending_rule: PendingRule,
    source_file: str,
    source_sha256: str,
) -> RuleChunk:
    raw_text = "\n".join(pending_rule.paragraphs)
    chunk_text = (
        f"Chapter {pending_rule.chapter_number}: {pending_rule.chapter_title}\n"
        f"Section {pending_rule.section_number}: {pending_rule.section_title}\n"
        f"Rule {pending_rule.rule_number}\n\n"
        f"{raw_text}"
    )
    chunk_hash = hashlib.sha256(
        f"{source_sha256}:{pending_rule.rule_number}:{raw_text}".encode()
    ).hexdigest()[:12]

    return RuleChunk(
        chunk_id=f"rule-{pending_rule.rule_number.replace('.', '-')}-{chunk_hash}",
        chapter_number=pending_rule.chapter_number,
        chapter_title=pending_rule.chapter_title,
        section_number=pending_rule.section_number,
        section_title=pending_rule.section_title,
        rule_number=pending_rule.rule_number,
        subrule_numbers=pending_rule.subrule_numbers,
        page_start=pending_rule.page_start,
        page_end=pending_rule.page_end,
        source_file=source_file,
        source_sha256=source_sha256,
        raw_text=raw_text,
        chunk_text=chunk_text,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_rule_chunks(pdf_path: Path) -> list[RuleChunk]:
    source_sha256 = _sha256_file(pdf_path)
    chunks: list[RuleChunk] = []
    pending_rule: PendingRule | None = None
    chapter_number: int | None = None
    chapter_title: str | None = None
    section_number: int | None = None
    section_title: str | None = None

    for line in extract_pdf_lines(pdf_path):
        if (
            chapter_number == 9
            and line.text == "Glossary"
            and _has_indent(line, CHAPTER_INDENT)
        ):
            break

        chapter_match = identify_chapter(line)
        if chapter_match:
            if pending_rule:
                chunks.append(_make_chunk(pending_rule, pdf_path.name, source_sha256))
                pending_rule = None

            chapter_number = int(chapter_match["number"])
            chapter_title = chapter_match["title"]
            section_number = None
            section_title = None
            continue

        section_match = identify_section(line)
        if section_match and chapter_number is not None:
            candidate_section = int(section_match["number"])
            if candidate_section // 100 == chapter_number:
                if pending_rule:
                    chunks.append(
                        _make_chunk(pending_rule, pdf_path.name, source_sha256)
                    )
                    pending_rule = None

                section_number = candidate_section
                section_title = section_match["title"]
                continue

        rule_match = identify_rule(line)
        if (
            rule_match
            and chapter_number is not None
            and chapter_title is not None
            and section_number is not None
            and section_title is not None
            and _belongs_to_section(rule_match["number"], section_number)
        ):
            if pending_rule:
                chunks.append(_make_chunk(pending_rule, pdf_path.name, source_sha256))

            pending_rule = PendingRule(
                chapter_number=chapter_number,
                chapter_title=chapter_title,
                section_number=section_number,
                section_title=section_title,
                rule_number=rule_match["number"],
                page_start=line.page_number,
                page_end=line.page_number,
                paragraphs=[line.text],
                subrule_numbers=[],
            )
            continue

        subrule_match = identify_subrule(line)
        if (
            subrule_match
            and pending_rule is not None
            and subrule_match["base_rule"] == pending_rule.rule_number
        ):
            subrule_number = f"{subrule_match['base_rule']}{subrule_match['letter']}"
            pending_rule.subrule_numbers.append(subrule_number)
            pending_rule.paragraphs.append(line.text)
            pending_rule.page_end = line.page_number
            continue

        if pending_rule is not None:
            _append_continuation(pending_rule, line.text)
            pending_rule.page_end = line.page_number

    if pending_rule:
        chunks.append(_make_chunk(pending_rule, pdf_path.name, source_sha256))

    validate_chunks(chunks)
    return chunks


def validate_chunks(chunks: list[RuleChunk]) -> None:
    if not chunks:
        raise ValueError("No rules were found in the PDF.")

    rule_numbers = [chunk.rule_number for chunk in chunks]
    duplicate_rules = sorted(
        rule for rule in set(rule_numbers) if rule_numbers.count(rule) > 1
    )
    if duplicate_rules:
        raise ValueError(f"Duplicate rules found: {duplicate_rules}")

    invalid_sections = [
        chunk.rule_number
        for chunk in chunks
        if not _belongs_to_section(chunk.rule_number, chunk.section_number)
    ]
    if invalid_sections:
        raise ValueError(f"Rules outside their section found: {invalid_sections}")

    invalid_content = [
        chunk.rule_number
        for chunk in chunks
        if not chunk.raw_text.startswith(f"{chunk.rule_number}.")
    ]
    if invalid_content:
        raise ValueError(f"Rules with invalid content found: {invalid_content}")

    invalid_subrules = [
        chunk.rule_number
        for chunk in chunks
        if len(chunk.subrule_numbers) != len(set(chunk.subrule_numbers))
        or any(
            not subrule.startswith(chunk.rule_number)
            for subrule in chunk.subrule_numbers
        )
    ]
    if invalid_subrules:
        raise ValueError(f"Invalid subrules found: {invalid_subrules}")

    invalid_pages = [
        chunk.rule_number for chunk in chunks if chunk.page_start > chunk.page_end
    ]
    if invalid_pages:
        raise ValueError(f"Invalid page ranges found: {invalid_pages}")


def save_rule_chunks(chunks: list[RuleChunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(chunk.model_dump_json() + "\n")


def parse_pdf(
    pdf_path: Path = DEFAULT_PDF_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> list[RuleChunk]:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    chunks = build_rule_chunks(pdf_path)
    save_rule_chunks(chunks, output_path)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse the Magic Comprehensive Rules into rule chunks."
    )
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    chunks = parse_pdf(args.pdf, args.output)
    print(f"Created {len(chunks)} rule chunks in {args.output}")


if __name__ == "__main__":
    main()
