"""`IngestionService`: o pipeline offline da ingestão (spec 102 research R13).

Cada execução é tudo ou nada: valida o catálogo inteiro, deriva as linhagens, lê todos os PDFs e gera todos os
chunks antes de escrever qualquer coisa. Sem problema, monta um build novo — copiando os vetores do que não mudou e
codificando o resto — e o publica por troca de alias; com problema, devolve o relatório sem tocar o índice.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from axon_scouting.rag.index.api import NORMATIVE_TYPES, IndexedChunk, IndexVersion
from axon_scouting.rag.index.ports import (
    Build,
    DocumentFields,
    DocumentRecord,
    DocumentStore,
    Embedder,
    EncodedChunk,
    IndexExport,
    IndexWriter,
    Original,
    SparseEncoder,
    StoredDocument,
)
from axon_scouting.rag.ingestion.domain.catalog import CatalogEntry, validate_catalog
from axon_scouting.rag.ingestion.domain.chunking import ChunkDraft, ChunkingPolicy, estimate_tokens, narrative_chunks
from axon_scouting.rag.ingestion.domain.ids import chunk_ids, content_key
from axon_scouting.rag.ingestion.domain.lineage import Validity, derive_lineages
from axon_scouting.rag.ingestion.domain.report import DocumentOutcome, IngestionReport
from axon_scouting.rag.ingestion.domain.structure import UNIT_BY_TYPE, normative_chunks
from axon_scouting.rag.ingestion.domain.text import Line, clean_pages
from axon_scouting.rag.ingestion.ports import Clock, DocumentParser

DENSE_LIMIT: Final = 512
"""Tokens que o embedding denso lê (o e5 trunca); unidade maior ganha aviso — o esparso cobre o texto inteiro."""


@dataclass(frozen=True, slots=True, kw_only=True)
class IndexStatus:
    build: Build | None
    documents: tuple[StoredDocument, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Prepared:
    record: DocumentRecord
    chunks: tuple[IndexedChunk, ...]
    normative_units: int
    warnings: tuple[str, ...]


class IngestionService:
    def __init__(
        self,
        *,
        parser: DocumentParser,
        embedder: Embedder,
        sparse: SparseEncoder,
        writer: IndexWriter,
        store: DocumentStore,
        clock: Clock,
        policy: ChunkingPolicy | None = None,
    ) -> None:
        self._parser = parser
        self._embedder = embedder
        self._sparse = sparse
        self._writer = writer
        self._store = store
        self._clock = clock
        self._policy = policy or ChunkingPolicy()

    async def ingest(self, catalog: object, base_dir: Path) -> IngestionReport:
        """Publica um build que espelha o catálogo, reaproveitando os vetores do que não mudou."""
        return await self._run(catalog, base_dir, reuse=True)

    async def reindex(self, catalog: object, base_dir: Path) -> IngestionReport:
        """Recodifica tudo com a configuração atual e publica — o caminho para trocar embedder ou chunking (US5)."""
        return await self._run(catalog, base_dir, reuse=False)

    async def original(self, chunk_id: str) -> Original | None:
        """O texto original das páginas de um chunk do build publicado (FR-009)."""
        return await self._store.original(chunk_id)

    async def status(self) -> IndexStatus:
        build = await self._writer.published()
        documents = () if build is None else tuple(await self._writer.documents(build.name))
        return IndexStatus(build=build, documents=documents)

    # --- pipeline -----------------------------------------------------------------------------------------------

    async def _run(self, catalog: object, base_dir: Path, *, reuse: bool) -> IngestionReport:
        prepared, problems = self._prepare(catalog, base_dir)
        version = IndexVersion(
            chunking=self._policy.identity(),
            embedding=self._embedder.model(),
            sparse=self._sparse.identity(),
            built_at=self._clock.now(),
        )
        published = await self._writer.published()
        if reuse and published is not None and not published.version.compatible_with(version):
            problems.append(
                f"a versão do índice publicado ({published.version.describe()}) difere da configurada "
                f"({version.describe()}); rode `axon-scouting-corpus reindex`"
            )
        if problems:
            return IngestionReport(problems=tuple(problems))

        previous: dict[str, StoredDocument] = {}
        if published is not None:
            previous = {d.doc_key: d for d in await self._writer.documents(published.name)}

        build = await self._writer.create(version)
        outcomes: list[DocumentOutcome] = []
        try:
            for item in prepared:
                record = item.record
                old = previous.get(record.doc_key)
                reused = (
                    reuse
                    and published is not None
                    and old is not None
                    and old.content_key == record.content_key
                    and old.chunk_ids == record.chunk_ids
                )
                if reused:
                    assert published is not None
                    await self._writer.copy(published.name, build.name, record)
                else:
                    await self._writer.write(build.name, record, await self._encode(item.chunks))
                outcomes.append(
                    DocumentOutcome(
                        doc_key=record.doc_key,
                        status=record.fields.status,
                        chunks=len(item.chunks),
                        reused=reused,
                        normative_units=item.normative_units,
                        warnings=item.warnings,
                    )
                )
            await self._writer.publish(build.name)
        except BaseException:
            await self._writer.discard(build.name)
            raise
        current = {item.record.doc_key for item in prepared}
        return IngestionReport(
            documents=tuple(outcomes),
            removed=tuple(sorted(set(previous) - current)),
            version=version,
            build=build.name,
        )

    def _prepare(self, catalog: object, base_dir: Path) -> tuple[list[_Prepared], list[str]]:
        validated = validate_catalog(catalog, base_dir)
        problems = [p.message for p in validated.problems]
        entries = [e for e in validated.valid if e.document_id not in validated.invalid_documents]
        lineages = derive_lineages(entries)
        problems += [f"linhagem {p.document_id}: {p.message}" for p in lineages.problems]
        prepared: list[_Prepared] = []
        for entry in sorted(entries, key=lambda e: e.doc_key):
            try:
                item = self._prepare_document(entry, lineages.validities.get(entry.doc_key))
            except _Rejected as rejected:
                problems.append(f"entrada {entry.entry} ({entry.label}): {rejected}")
                continue
            if item is not None:
                prepared.append(item)
        return prepared, problems

    def _prepare_document(self, entry: CatalogEntry, validity: Validity | None) -> _Prepared | None:
        try:
            parsed = self._parser.parse(Path(entry.path))
        except FileNotFoundError:
            raise _Rejected(f"arquivo não encontrado: {entry.file}") from None
        except Exception as exc:  # PDF corrompido ou ilegível: rejeição, nunca falha silenciosa
            raise _Rejected(f"não foi possível ler {entry.file}: {exc}") from exc
        if not any(page.strip() for page in parsed.pages):
            raise _Rejected(f"{entry.file} sem camada de texto (OCR fora do escopo)")
        warnings = [f"página {n} sem texto" for n, page in enumerate(parsed.pages, start=1) if not page.strip()]

        drafts, units, chunk_warnings = self._drafts(entry, clean_pages(parsed.pages))
        if not drafts:
            raise _Rejected(f"{entry.file} sem texto aproveitável depois da limpeza")
        warnings.extend(chunk_warnings)
        if validity is None:  # a linhagem tem problema: já relatado; o documento não segue
            return None

        fields = _fields(entry, validity)
        ids = chunk_ids(fields.doc_key, [(d.section_key, d.page_start) for d in drafts])
        chunks = tuple(_chunk(chunk_id, draft, fields) for chunk_id, draft in zip(ids, drafts, strict=True))
        record = DocumentRecord(
            fields=fields,
            content_key=content_key(parsed.source_sha256, entry.document_type.value, entry.title, parsed.parser),
            source_file=entry.file,
            source_url=entry.source_url,
            source_sha256=parsed.source_sha256,
            chunk_ids=tuple(ids),
            pages=parsed.pages,
        )
        return _Prepared(record=record, chunks=chunks, normative_units=units, warnings=tuple(warnings))

    def _drafts(self, entry: CatalogEntry, lines: Sequence[Line]) -> tuple[list[ChunkDraft], int, list[str]]:
        """Os chunks do documento, o número de unidades normativas e os avisos da estratégia (FR-002)."""
        if entry.document_type not in NORMATIVE_TYPES:
            return narrative_chunks(lines, entry.title, self._policy), 0, []
        result = normative_chunks(lines, entry.title, self._policy, unit=UNIT_BY_TYPE[entry.document_type])
        if result.units == 0:  # clarificação 3: normativo sem unidade vira janelas, com aviso
            return (
                narrative_chunks(lines, entry.title, self._policy),
                0,
                ["normativo sem unidade reconhecida: dividido em janelas"],
            )
        warnings = [
            f"unidade {d.section_key} com ~{tokens} tokens estimados (o embedding denso trunca em {DENSE_LIMIT})"
            for d in result.drafts
            if d.normative and (tokens := estimate_tokens(d.body)) > DENSE_LIMIT
        ]
        return result.drafts, result.units, warnings

    async def _encode(self, chunks: Sequence[IndexedChunk]) -> list[EncodedChunk]:
        dense = await self._embedder.embed_passages([c.text for c in chunks])
        return [
            EncodedChunk(chunk=c, dense=d, sparse=self._sparse.encode_passage(c.text))
            for c, d in zip(chunks, dense, strict=True)
        ]


class _Rejected(Exception):
    pass


async def copy_index(source: IndexExport, writer: IndexWriter) -> Build:
    """Publica em `writer` um build igual ao publicado em `source` — mesma versão, documentos e vetores (ADR-043, item
    4: o Qdrant carregado a partir do artefato). Tudo ou nada: falha descarta o build novo."""
    version, documents = await source.export()
    build = await writer.create(version)
    try:
        for record, chunks in documents:
            await writer.write(build.name, record, chunks)
        await writer.publish(build.name)
    except BaseException:
        await writer.discard(build.name)
        raise
    return build


def _fields(entry: CatalogEntry, validity: Validity) -> DocumentFields:
    return DocumentFields(
        document_id=entry.document_id,
        version=entry.version,
        document_type=entry.document_type,
        document_title=entry.title,
        jurisdiction=entry.jurisdiction,
        branch=entry.branch,
        publication_date=entry.publication_date,
        effective_date=validity.effective_date,
        expiry_date=validity.expiry_date,
        status=validity.status,
        superseded_by=validity.superseded_by,
        successor_expiry_date=validity.successor_expiry_date,
    )


def _chunk(chunk_id: str, draft: ChunkDraft, fields: DocumentFields) -> IndexedChunk:
    document = {f.name: getattr(fields, f.name) for f in dataclasses.fields(fields)}
    return IndexedChunk(
        chunk_id=chunk_id,
        text=draft.text,
        doc_key=fields.doc_key,
        page_start=draft.page_start,
        page_end=draft.page_end,
        chapter=draft.chapter,
        section=draft.section,
        subsection=draft.subsection,
        article=draft.article,
        **document,
    )
