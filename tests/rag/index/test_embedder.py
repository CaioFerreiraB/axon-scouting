"""Suíte da porta `Embedder` (spec 102 research R10, R16; T018 e T027)."""

from __future__ import annotations

import asyncio
import io
import math
import os
from pathlib import Path

import pytest

from axon_scouting.rag.index.adapters.memory import HashingEmbedder
from axon_scouting.rag.index.ports import Embedder


def _targets() -> list[object]:
    targets: list[object] = [pytest.param(lambda: HashingEmbedder(), id="hashing")]
    directory = os.environ.get("AXON_SCOUTING_EMBEDDING_DIR")
    try:
        from axon_scouting.rag.index.adapters.e5 import E5Embedder
    except ImportError:  # pragma: no cover - o adapter chega na T032
        return targets
    targets.append(
        pytest.param(
            lambda: E5Embedder(Path(directory or ".")),
            id="e5",
            marks=pytest.mark.skipif(not directory, reason="AXON_SCOUTING_EMBEDDING_DIR ausente"),
        )
    )
    return targets


@pytest.fixture(params=_targets())
def embedder(request: pytest.FixtureRequest) -> Embedder:
    factory = request.param
    return factory()  # type: ignore[no-any-return]


def test_dimensao_norma_e_determinismo(embedder: Embedder) -> None:
    textos = ["O uniforme escoteiro é usado em cerimônias.", "Acampamento de patrulha no fim de semana."]
    vetores = asyncio.run(embedder.embed_passages(textos))
    assert len(vetores) == 2
    for vetor in vetores:
        assert len(vetor) == embedder.model().dimensions
        assert math.isclose(math.sqrt(sum(x * x for x in vetor)), 1.0, abs_tol=1e-6)
    assert asyncio.run(embedder.embed_passages(textos)) == vetores


def test_pergunta_e_passagem_diferem(embedder: Embedder) -> None:
    texto = "uniforme escoteiro"
    assert asyncio.run(embedder.embed_query(texto)) != asyncio.run(embedder.embed_passages([texto]))[0]


def test_lista_vazia(embedder: Embedder) -> None:
    assert asyncio.run(embedder.embed_passages([])) == []


def test_hashing_parametrizavel() -> None:
    modelo = HashingEmbedder(dimensions=32, fingerprint="x").model()
    assert (modelo.name, modelo.fingerprint, modelo.dimensions) == ("hashing-test", "x", 32)


# --- e5: download fixado, sem rede (T027) ------------------------------------------------------------------------


def _arquivo(conteudo: bytes) -> object:
    import hashlib

    from axon_scouting.rag.index.adapters.e5 import ModelFile

    return ModelFile("onnx/x.bin", "x.bin", hashlib.sha256(conteudo).hexdigest())


def test_fetch_nao_baixa_arquivo_integro(tmp_path: Path) -> None:
    from axon_scouting.rag.index.adapters.e5 import fetch

    (tmp_path / "x.bin").write_bytes(b"pesos")
    chamadas: list[str] = []

    def opener(url: str) -> io.BytesIO:
        chamadas.append(url)
        return io.BytesIO(b"pesos")

    resultado = fetch(tmp_path, files=[_arquivo(b"pesos")], opener=opener)  # type: ignore[list-item]
    assert chamadas == []
    assert resultado == [(tmp_path / "x.bin", _arquivo(b"pesos").sha256)]  # type: ignore[attr-defined]


def test_fetch_baixa_e_confere(tmp_path: Path) -> None:
    from axon_scouting.rag.index.adapters.e5 import fetch

    urls: list[str] = []

    def opener(url: str) -> io.BytesIO:
        urls.append(url)
        return io.BytesIO(b"pesos")

    fetch(tmp_path, files=[_arquivo(b"pesos")], base_url="https://exemplo/rev", opener=opener)  # type: ignore[list-item]
    assert urls == ["https://exemplo/rev/onnx/x.bin"]
    assert (tmp_path / "x.bin").read_bytes() == b"pesos"


def test_fetch_recusa_hash_errado(tmp_path: Path) -> None:
    from axon_scouting.rag.index.adapters.e5 import fetch

    with pytest.raises(ValueError, match=r"x\.bin"):
        fetch(tmp_path, files=[_arquivo(b"pesos")], opener=lambda url: io.BytesIO(b"outros"))  # type: ignore[list-item]
    assert list(tmp_path.iterdir()) == []


def test_revisao_fixada() -> None:
    from axon_scouting.rag.index.adapters.e5 import FILES, REVISION

    assert REVISION == "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    assert [(f.remote, f.sha256[:8]) for f in FILES] == [
        ("onnx/model.onnx", "ca456c06"),
        ("onnx/tokenizer.json", "0b44a9d7"),
    ]


def test_e5_aproxima_parafrase() -> None:
    directory = os.environ.get("AXON_SCOUTING_EMBEDDING_DIR")
    if not directory:
        pytest.skip("AXON_SCOUTING_EMBEDDING_DIR ausente")
    from axon_scouting.rag.index.adapters.e5 import E5Embedder

    embedder = E5Embedder(Path(directory))
    modelo = embedder.model()
    assert (modelo.name, modelo.dimensions) == ("intfloat/multilingual-e5-small", 384)
    assert modelo.fingerprint.startswith("ca456c06")
    trecho, outro = asyncio.run(
        embedder.embed_passages(
            [
                "O acampamento de patrulha exige cardápio, lista de material e autorização dos responsáveis.",
                "O uniforme escoteiro é usado nas cerimônias de promessa.",
            ]
        )
    )
    pergunta = asyncio.run(embedder.embed_query("o que preciso para acampar com a patrulha?"))

    def produto(a: tuple[float, ...], b: tuple[float, ...]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert produto(pergunta, trecho) > produto(pergunta, outro)
