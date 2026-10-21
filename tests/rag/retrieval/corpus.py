"""Corpus sintético dos testes do retrieval (spec 103), montado com o kit do índice da 102."""

from __future__ import annotations

from datetime import date

from axon_scouting.rag.index.adapters.bm25 import Bm25Encoder
from axon_scouting.rag.index.adapters.memory import HashingEmbedder, MemoryIndex, MemoryStorage
from axon_scouting.rag.index.api import DocumentType, IndexedChunk, ValidityStatus
from axon_scouting.rag.index.ports import DocumentFields, Embedder

from ..index.kit import Index, campos, codificar, registro, trecho, versao

POR = campos("por", "2025", document_title="POR", document_type=DocumentType.POR, effective_date=date(2025, 4, 1))
MANUAL_2023 = campos(
    "manual-x",
    "2023",
    document_title="Manual X",
    effective_date=date(2023, 1, 1),
    expiry_date=date(2026, 2, 28),
    superseded_by="manual-x-2026",
    status=ValidityStatus.SUBSTITUIDO,
)
MANUAL_2026 = campos("manual-x", "2026", document_title="Manual X", effective_date=date(2026, 3, 1))
RASCUNHO = campos(
    "manual-x", "2027", document_title="Manual X", effective_date=date(2026, 3, 1), status=ValidityStatus.RASCUNHO
)
PIONEIRO = campos(
    "manual-pio", "1", document_title="Manual Pioneiro", branch="pioneiro", effective_date=date(2020, 1, 1)
)
ESCOTEIRO = campos(
    "manual-esc", "1", document_title="Manual Escoteiro", branch="escoteiro", effective_date=date(2020, 1, 1)
)

DOCUMENTOS: list[tuple[DocumentFields, list[IndexedChunk]]] = [
    (
        POR,
        [
            trecho(
                POR,
                "regra33",
                38,
                "[POR · Capítulo 5]\nREGRA 033 NOME DO GRUPO\nO nome do grupo é escolhido.",
                article="Regra 033",
            ),
            trecho(
                POR,
                "regra5",
                15,
                "[POR · Capítulo 1]\nREGRA 005 PROMESSA DO LOBINHO\nA promessa do lobinho.",
                article="Regra 005",
            ),
            trecho(
                POR,
                "regra49",
                40,
                "[POR · Capítulo 6]\nREGRA 049 UNIFORME\nO uniforme é usado em cerimônias.",
                article="Regra 049",
            ),
        ],
    ),
    (MANUAL_2023, [trecho(MANUAL_2023, "4.2", 10, "[Manual X · 4.2 Acampamento]\nO acampamento antigo da tropa.")]),
    (
        MANUAL_2026,
        [trecho(MANUAL_2026, "4.2", 12, "[Manual X · 4.2 Acampamento]\nO acampamento da tropa exige cardápio.")],
    ),
    (RASCUNHO, [trecho(RASCUNHO, "4.2", 12, "[Manual X · 4.2 Acampamento]\nO acampamento em revisão da tropa.")]),
    (PIONEIRO, [trecho(PIONEIRO, "s1", 3, "[Manual Pioneiro]\nO acampamento do clã pioneiro e a tropa.")]),
    (ESCOTEIRO, [trecho(ESCOTEIRO, "s1", 3, "[Manual Escoteiro]\nO acampamento da tropa escoteira.")]),
]


async def indice(embedder: Embedder | None = None) -> Index:
    """Um `MemoryIndex` publicado com o corpus sintético."""
    embedder = embedder or HashingEmbedder()
    sparse = Bm25Encoder()
    index = MemoryIndex("scouting", embedder, sparse, storage=MemoryStorage())
    await publicar(index, embedder, sparse)
    return index


async def publicar(index: Index, embedder: Embedder, sparse: Bm25Encoder) -> None:
    build = await index.create(versao(embedder, sparse))
    for fields, chunks in DOCUMENTOS:
        await index.write(build.name, registro(fields, chunks, ["p"] * 40), await codificar(chunks, embedder, sparse))
    await index.publish(build.name)


def filtros(dia: str = "2026-10-07", **extra: object) -> dict[str, object]:
    """Filtros de um dia no formato da 101 (contracts/retrieval-filters.md), com chaves extras."""
    return {"effective_from": dia, "effective_until": dia, **extra}
