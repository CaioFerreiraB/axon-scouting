"""Configuração do pack na Fase 5 (spec 101 FR-003, FR-012; research R18; data-model "Configuração do pack"; T030).

Pura de rede: `read_config` aplica os defaults (o core valida o esquema, mas não aplica defaults) e recusa, nomeando o
campo, cada combinação que o esquema sozinho não pega. A chave do Qdrant nunca passa por aqui.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_scouting.pack import COLLECTION, SCOUTING_CONFIG_SCHEMA, ScoutingConfig, read_config

SENTINELA = "chave-sentinela-que-nunca-aparece"


def modelo(tmp_path: Path, *files: str) -> str:
    directory = tmp_path / "e5"
    directory.mkdir()
    for name in files:
        (directory / name).write_bytes(b"")
    return str(directory)


def test_colecao_logica_fixa() -> None:
    assert COLLECTION == "scouting"


def test_defaults_com_o_embedder_de_teste() -> None:
    assert read_config({"embedder": "hashing"}) == ScoutingConfig(
        qdrant_url="http://localhost:6333",
        qdrant_path=None,
        index="scouting",
        embedder="hashing",
        embedding_dir=None,
        default_limit=5,
        max_limit=30,
    )


def test_qdrant_embutido_sem_url() -> None:
    config = read_config({"qdrant_path": "/tmp/q", "embedder": "hashing", "index": "outro", "default_limit": 3})
    assert (config.qdrant_url, config.qdrant_path, config.index, config.default_limit) == (None, "/tmp/q", "outro", 3)


def test_e5_por_default_exige_o_diretorio() -> None:
    with pytest.raises(ValueError, match="embedding_dir"):
        read_config({})


def test_url_e_caminho_sao_exclusivos() -> None:
    with pytest.raises(ValueError, match="qdrant_url") as error:
        read_config({"qdrant_url": "http://q:6333", "qdrant_path": "/tmp/q", "embedder": "hashing"})
    assert "qdrant_path" in str(error.value)


def test_e5_sem_os_arquivos_do_modelo(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"model\.onnx") as error:
        read_config({"embedder": "e5", "embedding_dir": modelo(tmp_path)})
    assert "tokenizer.json" in str(error.value)


def test_e5_com_os_arquivos_do_modelo(tmp_path: Path) -> None:
    directory = modelo(tmp_path, "model.onnx", "tokenizer.json")
    config = read_config({"embedding_dir": directory})
    assert (config.embedder, config.embedding_dir) == ("e5", directory)


def test_e5_com_diretorio_inexistente(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="embedding_dir"):
        read_config({"embedding_dir": str(tmp_path / "nao-existe")})


@pytest.mark.parametrize(
    "limits",
    [{"default_limit": 31}, {"default_limit": 10, "max_limit": 5}, {"default_limit": 0}, {"max_limit": 0}],
)
def test_limites_incoerentes(limits: dict[str, int]) -> None:
    with pytest.raises(ValueError, match=r"default_limit|max_limit"):
        read_config({"embedder": "hashing", **limits})


def test_esquema() -> None:
    schema = SCOUTING_CONFIG_SCHEMA
    assert schema["type"] == "object" and schema["additionalProperties"] is False
    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert set(properties) == {
        "qdrant_url",
        "qdrant_path",
        "index",
        "embedder",
        "embedding_dir",
        "default_limit",
        "max_limit",
    }
    assert properties["qdrant_url"] == {"type": "string", "pattern": "^https?://"}
    assert properties["qdrant_path"] == {"type": "string", "minLength": 1}
    assert properties["index"] == {"type": "string", "pattern": "^[a-z0-9][a-z0-9_-]*$"}
    assert properties["embedder"] == {"enum": ["e5", "hashing"]}
    assert properties["embedding_dir"] == {"type": "string", "minLength": 1}
    for name in ("default_limit", "max_limit"):
        assert properties[name] == {"type": "integer", "minimum": 1}
    assert not [name for name in properties if "key" in name], "a chave do Qdrant fica fora do esquema (FR-012)"


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"qdrant_url": "http://q:6333", "qdrant_path": "/tmp/q", "embedder": "hashing"},
        {"embedder": "hashing", "default_limit": 31},
    ],
)
def test_a_chave_nunca_aparece_nos_erros(config: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_SCOUTING_QDRANT_API_KEY", SENTINELA)
    with pytest.raises(ValueError) as error:
        read_config(config)  # type: ignore[arg-type]
    assert SENTINELA not in str(error.value)
