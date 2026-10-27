"""O artefato do índice em disco (spec 102; ADR-043): arquivos, conjuntos de vetores por modelo e integridade.

O comportamento de índice (busca, filtros, versões, originais) é da suíte de conformidade, que roda também contra o
`ArtifactIndex`; aqui fica o que é só do artefato.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path

import pytest

from axon_scouting.rag.index.adapters.artifact import ArtifactIndex
from axon_scouting.rag.index.adapters.bm25 import Bm25Encoder
from axon_scouting.rag.index.adapters.memory import HashingEmbedder
from axon_scouting.rag.index.api import IndexMismatch
from axon_scouting.rag.index.domain.artifact import pack_vectors, unpack_vectors, vectors_file

from .kit import INSTANTE, campos, codificar, registro, trecho, versao
from .test_conformance import DEFAULT

DOC = campos()
TRECHOS = [
    trecho(DOC, "4.1", 10, "[Manual X · 4.1 Acampamento]\nO acampamento de patrulha exige planejamento."),
    trecho(DOC, "4.2", 11, "[Manual X · 4.2 Uniforme]\nO uniforme escoteiro é usado em cerimônias."),
]
E5_FALSO = HashingEmbedder(dimensions=16, fingerprint="e5-falso")
PPLX_FALSO = HashingEmbedder(dimensions=24, fingerprint="pplx-falso")


async def _publicar(root: Path, embedder: HashingEmbedder, chunks: list = TRECHOS, hour: int = 12) -> ArtifactIndex:  # type: ignore[type-arg]
    index = ArtifactIndex(root, embedder, Bm25Encoder())
    build = await index.create(versao(embedder, built_at=INSTANTE.replace(hour=hour)))
    await index.write(build.name, registro(DOC, chunks, ("p",) * 11), await codificar(chunks, embedder, Bm25Encoder()))
    await index.publish(build.name)
    return index


def test_arquivos_e_manifesto(tmp_path: Path) -> None:
    root = tmp_path / "index"
    asyncio.run(_publicar(root, E5_FALSO))
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()) == [
        "chunks.jsonl",
        "documents.jsonl",
        "manifest.json",
        vectors_file(E5_FALSO.model()),
    ]
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["format"] == 1 and manifest["index"] == "scouting"
    assert [m["fingerprint"] for m in manifest["models"]] == ["e5-falso"]
    assert len((root / "chunks.jsonl").read_text().splitlines()) == 2
    assert not list(tmp_path.glob(".index.*")), "o diretório de troca não sobra"


def test_outra_instancia_le_o_artefato(tmp_path: Path) -> None:
    root = tmp_path / "index"
    asyncio.run(_publicar(root, E5_FALSO))
    leitora = ArtifactIndex(root, E5_FALSO, Bm25Encoder())
    achados = asyncio.run(leitora.search("uniforme escoteiro", filters=DEFAULT, limit=2))
    assert achados[0].chunk_id == TRECHOS[1].chunk_id
    assert achados[0].similarity is not None


def test_separador_unicode_no_texto_nao_quebra_a_leitura(tmp_path: Path) -> None:
    """Texto de PDF com fonte corrompida traz U+0085 e U+2028, que o JSON sem `ensure_ascii` grava como estão: o
    leitor separa as linhas só em `\\n` (102 research R29)."""
    root = tmp_path / "index"
    estranho = [trecho(DOC, "4.1", 10, "[Manual X · 4.1]\nTexto com \x85 e \u2028 no meio.")]

    async def publicar() -> None:
        index = ArtifactIndex(root, E5_FALSO, Bm25Encoder())
        build = await index.create(versao(E5_FALSO, built_at=INSTANTE))
        pages = ("página com \x85 corrompida",) * 11
        await index.write(
            build.name, registro(DOC, estranho, pages), await codificar(estranho, E5_FALSO, Bm25Encoder())
        )
        await index.publish(build.name)

    asyncio.run(publicar())
    leitora = ArtifactIndex(root, E5_FALSO, Bm25Encoder())
    (achado,) = asyncio.run(leitora.get([estranho[0].chunk_id]))
    assert "\x85" in achado.text and "\u2028" in achado.text


def test_dois_modelos_no_mesmo_artefato(tmp_path: Path) -> None:
    """ADR-043, item 2: um build por modelo, os trechos iguais; cada leitor usa os próprios vetores."""
    root = tmp_path / "index"
    asyncio.run(_publicar(root, E5_FALSO))
    asyncio.run(_publicar(root, PPLX_FALSO, hour=13))
    manifest = json.loads((root / "manifest.json").read_text())
    assert [m["fingerprint"] for m in manifest["models"]] == ["pplx-falso", "e5-falso"]
    for embedder in (E5_FALSO, PPLX_FALSO):
        leitor = ArtifactIndex(root, embedder, Bm25Encoder())
        assert asyncio.run(leitor.version()).embedding == embedder.model()
        assert asyncio.run(leitor.search("uniforme", filters=DEFAULT, limit=2))


def test_texto_mudado_tira_o_outro_modelo(tmp_path: Path) -> None:
    """Um conjunto que deixaria trecho sem vetor sai do artefato: o modelo dele precisa de `reindex`."""
    root = tmp_path / "index"
    asyncio.run(_publicar(root, E5_FALSO))
    asyncio.run(_publicar(root, PPLX_FALSO, hour=13))
    mudado = [TRECHOS[0], dataclasses.replace(TRECHOS[1], text=TRECHOS[1].text + " Novo parágrafo.")]
    asyncio.run(_publicar(root, E5_FALSO, mudado, hour=14))
    manifest = json.loads((root / "manifest.json").read_text())
    assert [m["fingerprint"] for m in manifest["models"]] == ["e5-falso"]
    with pytest.raises(IndexMismatch):
        asyncio.run(ArtifactIndex(root, PPLX_FALSO, Bm25Encoder()).search("uniforme", filters=DEFAULT, limit=2))


def test_arquivo_adulterado_e_recusado(tmp_path: Path) -> None:
    root = tmp_path / "index"
    asyncio.run(_publicar(root, E5_FALSO))
    vetores = root / vectors_file(E5_FALSO.model())
    vetores.write_bytes(bytes(len(vetores.read_bytes())))
    with pytest.raises(ValueError, match="corrompido"):
        asyncio.run(ArtifactIndex(root, E5_FALSO, Bm25Encoder()).version())


def test_formato_desconhecido(tmp_path: Path) -> None:
    root = tmp_path / "index"
    asyncio.run(_publicar(root, E5_FALSO))
    manifest = json.loads((root / "manifest.json").read_text())
    (root / "manifest.json").write_text(json.dumps({**manifest, "format": 99}))
    with pytest.raises(ValueError, match="formato"):
        asyncio.run(ArtifactIndex(root, E5_FALSO, Bm25Encoder()).version())


def test_exportar_para_outro_indice(tmp_path: Path) -> None:
    index = asyncio.run(_publicar(tmp_path / "index", E5_FALSO))
    version, documents = asyncio.run(index.export())
    assert version.embedding == E5_FALSO.model()
    ((record, chunks),) = documents
    assert record.doc_key == DOC.doc_key and [c.chunk.chunk_id for c in chunks] == sorted(t.chunk_id for t in TRECHOS)
    assert all(len(c.dense) == 16 for c in chunks)


def test_vetores_ida_e_volta() -> None:
    data = pack_vectors([(0.5, -0.25), (1.0, 0.0)], 2)
    assert unpack_vectors(data, 2, 2) == [(0.5, -0.25), (1.0, 0.0)]
    with pytest.raises(ValueError, match="dimensões"):
        pack_vectors([(1.0,)], 2)
    with pytest.raises(ValueError, match="esperado"):
        unpack_vectors(data, 2, 3)
