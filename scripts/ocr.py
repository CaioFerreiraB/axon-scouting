"""Transcrição dos documentos digitalizados do corpus (ADR-050). Ferramenta de quem cura o corpus, fora do pack.

A ingestão lê a transcrição de um PDF sem camada de texto em `corpus/ocr/<caminho do original>.txt`, uma página por
bloco, separadas por quebra de página (U+000C). Este script a produz e a confere:

    uv run python scripts/ocr.py tesseract corpus/assembleias/atas/regional/sao-paulo/1994_Ata….pdf …
        Tesseract 5 (português) pelo contêiner `jbarlow83/ocrmypdf`, a 300 dpi; página que já tem camada de texto
        (200 caracteres ou mais) fica com o texto do PDF. Grava o `.txt`; é a transcrição de partida, gratuita.

    uv run python scripts/ocr.py pages <pdf>…
        Prepara a revisão por outro transcritor (o Codex, pelo prompt em `scripts/ocr-codex-prompt.md`): para cada
        documento, `.cache/ocr/<id>/p001.png`… e `p001.tesseract.txt`… e um `manifest.json` com o destino.

    uv run python scripts/ocr.py assemble
        Junta os `pNNN.codex.txt` de cada documento preparado no `.txt` do corpus (falta uma página → o documento fica).

    uv run python scripts/ocr.py check [--limiar 0.6]
        Concordância por página entre a transcrição do corpus e o Tesseract (palavras de 3 letras ou mais, F1 sobre os
        conjuntos): grava `.cache/ocr/concordancia.json` e lista as páginas abaixo do limiar, para conferir na imagem.

O Tesseract roda num contêiner (não há binário na máquina): `docker pull jbarlow83/ocrmypdf` uma vez.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
WORK = ROOT / ".cache" / "ocr"
IMAGE = "jbarlow83/ocrmypdf:latest"
DPI = 300
TEXT_LAYER = 200
PAGE_BREAK = "\f"
_WORD = re.compile(r"[^\W\d_]{3,}", re.UNICODE)


def transcription_path(pdf: Path) -> Path:
    """`corpus/<caminho>.pdf` → `corpus/ocr/<caminho>.txt` (o `ocr/` espelha o caminho do original)."""
    relative = pdf.resolve().relative_to(CORPUS)
    if relative.parts[:1] == ("ocr",):
        relative = Path(*relative.parts[1:])
    return CORPUS / "ocr" / relative.with_suffix(".txt")


def render(pdf: Path, directory: Path) -> list[tuple[Path, str]]:
    """As páginas em PNG (tons de cinza, 300 dpi) e o texto da camada de cada uma."""
    directory.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(pdf)
    pages: list[tuple[Path, str]] = []
    try:
        for index in range(len(document)):
            page = document[index]
            text = page.get_textpage().get_text_range()
            image = directory / f"p{index + 1:03d}.png"
            page.render(scale=DPI / 72).to_pil().convert("L").save(image)
            pages.append((image, text))
    finally:
        document.close()
    return pages


def tesseract(directory: Path) -> None:
    """Roda o Tesseract em todos os PNG do diretório, num contêiner só; grava `pNNN.tesseract.txt`."""
    script = 'for f in /w/*.png; do tesseract "$f" "${f%.png}.tesseract" -l por --psm 1 >/dev/null 2>&1; done'
    subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "sh", "-v", f"{directory}:/w", IMAGE, "-c", script],
        check=True,
    )


def command_tesseract(pdfs: list[Path]) -> None:
    for pdf in pdfs:
        target = transcription_path(pdf)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            pages = render(pdf, directory)
            if any(len(text.strip()) < TEXT_LAYER for _, text in pages):
                tesseract(directory)
            texts = [
                text if len(text.strip()) >= TEXT_LAYER else _read(image.with_suffix(".tesseract.txt"))
                for image, text in pages
            ]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(PAGE_BREAK.join(_tidy(t) for t in texts) + PAGE_BREAK, encoding="utf-8")
        print(f"{target.relative_to(ROOT)}: {len(texts)} páginas")


def command_pages(pdfs: list[Path]) -> None:
    for pdf in pdfs:
        target = transcription_path(pdf)
        directory = WORK / target.stem
        pages = render(pdf, directory)
        tesseract(directory)
        manifest = {
            "pdf": str(pdf.resolve().relative_to(ROOT)),
            "transcription": str(target.relative_to(ROOT)),
            "pages": [image.name for image, _ in pages],
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), "utf-8")
        print(f"{directory.relative_to(ROOT)}: {len(pages)} páginas")


def command_assemble() -> None:
    for manifest_path in sorted(WORK.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text("utf-8"))
        directory = manifest_path.parent
        texts = [directory / Path(name).with_suffix(".codex.txt").name for name in manifest["pages"]]
        missing = [path.name for path in texts if not path.exists()]
        if missing:
            print(f"{directory.name}: faltam {len(missing)} páginas ({', '.join(missing[:3])}…); fica como está")
            continue
        target = ROOT / manifest["transcription"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(PAGE_BREAK.join(_tidy(_read(p)) for p in texts) + PAGE_BREAK, encoding="utf-8")
        print(f"{target.relative_to(ROOT)}: {len(texts)} páginas montadas")


def command_check(threshold: float) -> int:
    report: dict[str, list[float]] = {}
    flagged: list[str] = []
    for manifest_path in sorted(WORK.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text("utf-8"))
        target = ROOT / manifest["transcription"]
        if not target.exists():
            continue
        pages = target.read_text("utf-8").split(PAGE_BREAK)
        scores = []
        for number, name in enumerate(manifest["pages"], start=1):
            reference = _read(manifest_path.parent / Path(name).with_suffix(".tesseract.txt"))
            text = pages[number - 1] if number <= len(pages) else ""
            score = _agreement(text, reference)
            scores.append(round(score, 3))
            if score < threshold:
                flagged.append(f"{manifest['transcription']} p. {number}: {score:.2f}")
        report[manifest["transcription"]] = scores
    (WORK / "concordancia.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
    print(f"{sum(len(v) for v in report.values())} páginas conferidas; {len(flagged)} abaixo de {threshold}")
    for line in flagged:
        print("  ", line)
    return 0


def _agreement(a: str, b: str) -> float:
    """F1 entre os conjuntos de palavras (3 letras ou mais, sem caixa nem acento); duas páginas vazias concordam."""
    x, y = _words(a), _words(b)
    if not x and not y:
        return 1.0
    if not x or not y:
        return 0.0
    common = len(x & y)
    return 2 * common / (len(x) + len(y))


def _words(text: str) -> set[str]:
    folded = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return set(_WORD.findall(plain))


def _read(path: Path) -> str:
    return path.read_text("utf-8") if path.exists() else ""


def _tidy(text: str) -> str:
    """Sem a quebra de página que o Tesseract põe no fim e sem linhas em branco em excesso."""
    text = text.replace(PAGE_BREAK, "").replace("\r\n", "\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("tesseract", "pages"):
        sub.add_parser(name).add_argument("pdfs", nargs="+", type=Path)
    sub.add_parser("assemble")
    check = sub.add_parser("check")
    check.add_argument("--limiar", type=float, default=0.6)
    args = parser.parse_args()
    if args.command in ("tesseract", "pages") and shutil.which("docker") is None:
        print("erro: o Tesseract roda num contêiner e o docker não está no PATH", file=sys.stderr)
        return 1
    if args.command == "tesseract":
        command_tesseract(args.pdfs)
    elif args.command == "pages":
        command_pages(args.pdfs)
    elif args.command == "assemble":
        command_assemble()
    else:
        return command_check(args.limiar)
    return 0


if __name__ == "__main__":
    sys.exit(main())
