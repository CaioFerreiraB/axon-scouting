"""`IngestionService`: o pipeline offline da ingestão (spec 102 research R13).

Cada execução é tudo ou nada: valida o catálogo inteiro, deriva as linhagens, lê todos os PDFs e gera todos os
chunks antes de escrever qualquer coisa. Sem problema, monta um build novo — copiando os vetores do que não mudou e
codificando o resto — e o publica por troca de alias; com problema, devolve o relatório sem tocar o índice.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

from axon_scouting.rag.index.api import IndexedChunk, IndexVersion, UnitKind
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
from axon_scouting.rag.ingestion.domain import ocr_noise
from axon_scouting.rag.ingestion.domain.catalog import CatalogEntry, validate_catalog
from axon_scouting.rag.ingestion.domain.ceiling import CEILING, fit
from axon_scouting.rag.ingestion.domain.chunking import (
    UNIT_LINES,
    ChunkDraft,
    ChunkingPolicy,
    narrative_chunks,
)
from axon_scouting.rag.ingestion.domain.corpus_structure import (
    NORMATIVE_PROFILES,
    CorpusStructure,
    Resolution,
    bodies_of,
    parse_structure,
    scope_of,
    type_of,
)
from axon_scouting.rag.ingestion.domain.ids import chunk_ids, content_key
from axon_scouting.rag.ingestion.domain.lineage import Validity, derive_lineages
from axon_scouting.rag.ingestion.domain.report import DocumentOutcome, IngestionReport
from axon_scouting.rag.ingestion.domain.sources import bookmarks_from, toc_entries
from axon_scouting.rag.ingestion.domain.structure import normative_chunks
from axon_scouting.rag.ingestion.domain.text import Line, clean_pages, running_labels
from axon_scouting.rag.ingestion.domain.tree import PlacedChunk, TreeProblem, build_tree, shape_of, unit_boundary
from axon_scouting.rag.ingestion.ports import Clock, DocumentParser, EstimatedTokens, ParsedDocument, TokenCounter

MISSING_STRUCTURE: Final = (
    "sem taxonomia.yaml e estrutura.yaml ao lado do catálogo: o tipo e o perfil de cada documento vêm da taxonomia "
    "(ADR-049)"
)


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
        tokens: TokenCounter | None = None,
    ) -> None:
        self._parser = parser
        self._embedder = embedder
        self._sparse = sparse
        self._writer = writer
        self._store = store
        self._clock = clock
        self._policy = policy or ChunkingPolicy()
        self._tokens = tokens or EstimatedTokens()

    async def ingest(
        self, catalog: object, base_dir: Path, structure: tuple[object, object] | None = None
    ) -> IngestionReport:
        """Publica um build que espelha o catálogo, reaproveitando os vetores do que não mudou. `structure` são a
        taxonomia e a estrutura carregadas (ADR-048); sem elas, o perfil vem do `type` do catálogo."""
        return await self._run(catalog, base_dir, structure, reuse=True)

    async def reindex(
        self, catalog: object, base_dir: Path, structure: tuple[object, object] | None = None
    ) -> IngestionReport:
        """Recodifica tudo com a configuração atual e publica — o caminho para trocar embedder ou chunking (US5)."""
        return await self._run(catalog, base_dir, structure, reuse=False)

    async def original(self, chunk_id: str) -> Original | None:
        """O texto original das páginas de um chunk do build publicado (FR-009)."""
        return await self._store.original(chunk_id)

    async def status(self) -> IndexStatus:
        build = await self._writer.published()
        documents = () if build is None else tuple(await self._writer.documents(build.name))
        return IndexStatus(build=build, documents=documents)

    # --- pipeline -----------------------------------------------------------------------------------------------

    async def _run(
        self, catalog: object, base_dir: Path, raw_structure: tuple[object, object] | None, *, reuse: bool
    ) -> IngestionReport:
        structure: CorpusStructure | None = None
        problems: list[str] = []
        if raw_structure is None:
            return IngestionReport(problems=(MISSING_STRUCTURE,))
        structure, problems = parse_structure(*raw_structure)
        if structure is None:
            return IngestionReport(problems=tuple(problems))
        prepared, found = self._prepare(catalog, base_dir, structure)
        problems += found
        version = IndexVersion(
            chunking=f"{self._policy.identity()};teto={CEILING}:{self._tokens.identity()}",
            embedding=self._embedder.model(),
            sparse=self._sparse.identity(),
            built_at=self._clock.now(),
            unit_kinds=tuple(
                UnitKind(name=k.name, label=k.label, plurals=k.plurals) for k in structure.unit_kinds.values()
            ),
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

    def _prepare(
        self, catalog: object, base_dir: Path, structure: CorpusStructure
    ) -> tuple[list[_Prepared], list[str]]:
        validated = validate_catalog(catalog, base_dir)
        problems = [p.message for p in validated.problems]
        entries = [e for e in validated.valid if e.document_id not in validated.invalid_documents]
        lineages = derive_lineages(entries)
        problems += [f"linhagem {p.document_id}: {p.message}" for p in lineages.problems]
        parsed: list[tuple[CatalogEntry, Resolution, ParsedDocument]] = []
        for entry in sorted(entries, key=lambda e: e.doc_key):
            try:
                parsed.append((entry, self._resolve(entry, structure), self._parse(entry)))
            except _Rejected as rejected:
                problems.append(f"entrada {entry.entry} ({entry.label}): {rejected}")
        # O vocabulário do filtro de OCR sai dos documentos nascidos digitais da mesma execução (ADR-050).
        known = ocr_noise.vocabulary(
            [line.text for line in clean_pages(document.for_chunking)]
            for entry, _, document in parsed
            if not _is_ocr(entry)
        )
        vocabulary_key = hashlib.sha256("\n".join(sorted(known)).encode()).hexdigest()[:16]
        prepared: list[_Prepared] = []
        for entry, resolution, document in parsed:
            try:
                item = self._prepare_document(
                    entry,
                    lineages.validities.get(entry.doc_key),
                    resolution,
                    document,
                    known if _is_ocr(entry) else None,
                    vocabulary_key,
                )
            except _Rejected as rejected:
                problems.append(f"entrada {entry.entry} ({entry.label}): {rejected}")
                continue
            if item is not None:
                prepared.append(item)
        return prepared, problems

    @staticmethod
    def _resolve(entry: CatalogEntry, structure: CorpusStructure) -> Resolution:
        classification = structure.classify(entry.file)
        if classification is None:
            raise _Rejected(f"pasta fora da taxonomia: {Path(entry.file).parent}")
        return structure.resolve(entry.document_id, classification)

    def _parse(self, entry: CatalogEntry) -> ParsedDocument:
        try:
            parsed = self._parser.parse(Path(entry.path))
        except FileNotFoundError:
            raise _Rejected(f"arquivo não encontrado: {entry.file}") from None
        except Exception as exc:  # PDF corrompido ou ilegível: rejeição, nunca falha silenciosa
            raise _Rejected(f"não foi possível ler {entry.file}: {exc}") from exc
        if not any(page.strip() for page in parsed.pages):
            raise _Rejected(
                f"{entry.file} sem camada de texto: transcreva em ocr/<caminho>.txt e aponte o catálogo para ela"
            )
        return parsed

    def _prepare_document(
        self,
        entry: CatalogEntry,
        validity: Validity | None,
        resolution: Resolution,
        parsed: ParsedDocument,
        known: frozenset[str] | None,
        vocabulary_key: str,
    ) -> _Prepared | None:
        warnings = [f"página {n} sem texto" for n, page in enumerate(parsed.pages, start=1) if not page.strip()]
        lines = clean_pages(parsed.for_chunking)
        if known is not None:
            lines = ocr_noise.clean(lines, known)
        typography = {(page, text): level for page, text, level in parsed.headings}
        toc = toc_entries(parsed.for_chunking) if "sumário" in resolution.profile.sources else ()
        boundary = unit_boundary(resolution, toc)
        drafts, units, chunk_warnings = self._drafts(entry, lines, resolution, typography, boundary)
        if not drafts:
            raise _Rejected(f"{entry.file} sem texto aproveitável depois da limpeza")
        warnings.extend(chunk_warnings)
        if validity is None:  # a linhagem tem problema: já relatado; o documento não segue
            return None

        fields = _fields(entry, validity, resolution)
        sources = set(resolution.profile.sources)
        pages = parsed.for_chunking
        try:
            tree = build_tree(
                drafts,
                resolution,
                doc_key=fields.doc_key,
                title=entry.title,
                branch=entry.branch,
                toc=toc,
                bookmarks=bookmarks_from(parsed.bookmarks, len(pages)) if "bookmarks" in sources else (),
                running=running_labels(pages) if "rodapé" in sources else (),
            )
        except TreeProblem as problem:
            raise _Rejected(str(problem)) from None
        placed_chunks = [
            dataclasses.replace(placed, draft=part, shape=placed.shape if part is placed.draft else shape_of(part.body))
            for placed in tree.chunks
            for part in fit(placed.draft, self._tokens.count)
        ]
        ids = chunk_ids(fields.doc_key, [(p.draft.section_key, p.draft.page_start) for p in placed_chunks])
        chunks = tuple(
            _chunk(chunk_id, placed, fields, entry.source_url)
            for chunk_id, placed in zip(ids, placed_chunks, strict=True)
        )
        structure_key = resolution.fingerprint()
        record = DocumentRecord(
            fields=fields,
            content_key=content_key(
                parsed.source_sha256,
                type_of(resolution),
                entry.title,
                parsed.parser,
                structure=structure_key + (f";vocabulario={vocabulary_key}" if known is not None else ""),
            ),
            source_file=entry.file,
            source_url=entry.source_url,
            source_sha256=parsed.source_sha256,
            chunk_ids=tuple(ids),
            pages=parsed.pages,
            units=tree.units,
        )
        return _Prepared(record=record, chunks=chunks, normative_units=units, warnings=tuple(warnings))

    def _drafts(
        self,
        entry: CatalogEntry,
        lines: Sequence[Line],
        resolution: Resolution,
        typography: Mapping[tuple[int, str], int] | None = None,
        boundary: Callable[[Sequence[str], int], bool] | None = None,
    ) -> tuple[list[ChunkDraft], int, list[str]]:
        """Os chunks do documento, o número de unidades normativas e os avisos da estratégia (FR-002). O perfil decide
        a estratégia: `regras` e `artigos` são normativos mesmo num documento catalogado como `manual` (ADR-048)."""
        unit = NORMATIVE_PROFILES.get(resolution.profile.name)
        if unit is None:
            units = UNIT_LINES.get(resolution.profile.name)
            drafts = narrative_chunks(
                lines, entry.title, self._policy, units=units, typography=typography, boundary=boundary
            )
            return drafts, 0, []
        result = normative_chunks(lines, entry.title, self._policy, unit=unit)
        if result.units == 0:  # clarificação 3: normativo sem unidade vira janelas, com aviso
            return (
                narrative_chunks(lines, entry.title, self._policy, typography=typography),
                0,
                ["normativo sem unidade reconhecida: dividido em janelas"],
            )
        return result.drafts, result.units, []

    async def _encode(self, chunks: Sequence[IndexedChunk]) -> list[EncodedChunk]:
        dense = await self._embedder.embed_passages([c.text for c in chunks])
        return [
            EncodedChunk(chunk=c, dense=d, sparse=self._sparse.encode_passage(c.text))
            for c, d in zip(chunks, dense, strict=True)
        ]


class _Rejected(Exception):
    pass


def _is_ocr(entry: CatalogEntry) -> bool:
    """O documento vem de uma cópia OCR ou de uma transcrição (`ocr/…`): passa pelo filtro de ruído (ADR-050)."""
    return PurePosixPath(entry.file).parts[:1] == ("ocr",)


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


def _fields(entry: CatalogEntry, validity: Validity, resolution: Resolution) -> DocumentFields:
    taxonomy = resolution.classification
    return DocumentFields(
        document_id=entry.document_id,
        version=entry.version,
        document_type=type_of(resolution),
        document_title=entry.title,
        jurisdiction=entry.jurisdiction,
        branch=entry.branch,
        publication_date=entry.publication_date,
        effective_date=validity.effective_date,
        expiry_date=validity.expiry_date,
        status=validity.status,
        superseded_by=validity.superseded_by,
        successor_expiry_date=validity.successor_expiry_date,
        category=None if taxonomy is None else taxonomy.category,
        subcategory=None if taxonomy is None else taxonomy.subcategory,
        level=None if taxonomy is None else taxonomy.level,
        semantic_type=None if taxonomy is None else taxonomy.semantic_type,
        bodies=bodies_of(resolution),
        scope=scope_of(resolution, entry.jurisdiction),
    )


def _chunk(chunk_id: str, placed: PlacedChunk, fields: DocumentFields, source_url: str | None) -> IndexedChunk:
    document = {f.name: getattr(fields, f.name) for f in dataclasses.fields(fields)}
    draft = placed.draft
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
        path=placed.path,
        unit_kind=placed.unit_kind,
        shape=placed.shape,
        branches=placed.branches if placed.branches != ((fields.branch,) if fields.branch else ()) else (),
        source_url=source_url,
        **document,
    )
