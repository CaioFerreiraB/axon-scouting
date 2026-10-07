"""`citation_integrity`: só fica a citação que esteve no contexto da geração (spec 104 US1, US2, US4; ADR-007; R4, R5).

A fonte da verdade é `view.generation_context` — nem `retrieved`, nem `reranked`: trecho recuperado e cortado pelo
orçamento não foi visto pelo modelo. Citação inválida sai do texto e vira `issue` (o core a registra como `notice`);
as válidas viram `[n]`, na ordem da primeira aparição, e levam a proveniência do trecho. Sem nenhuma válida, a
resposta ganha o aviso de "não fundamentada".
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from ai_core_contracts import Chunk, Citation, TurnView, ValidationResult, ValidatorContext

from ai_scouting.validators.citations import extract, key, rewrite

UNGROUNDED_NOTICE: Final = (
    "Atenção: esta resposta não se apoia em nenhum documento do acervo consultado. "
    "Confirme a informação na fonte oficial."
)
INVALID_ISSUE: Final = "citação fora do contexto da geração: {raw}"
UNGROUNDED_ISSUE: Final = "nenhuma citação válida"


class CitationIntegrityValidator:
    """Implementa `AnswerValidator` e, estruturalmente, `StreamingValidator` (001; ADR-024). Determinístico, sem modelo
    e sem I/O (FR-006)."""

    name = "citation_integrity"

    async def validate(self, view: TurnView, ctx: ValidatorContext) -> ValidationResult:
        if view.answer is None:
            return ValidationResult(ok=True)
        return _check(view.answer, _delivered(view))

    def open_stream(self, view: TurnView, ctx: ValidatorContext) -> IntegrityStream:
        """A sessão de streaming: nenhuma citação inventada sai para o cliente, nem com a resposta em pedaços."""
        return IntegrityStream(_delivered(view))


class IntegrityStream:
    """Segura o texto só enquanto um marcador `[[` pode estar aberto — e o espaço antes dele, que sai junto com um
    marcador inválido. O resto sai como no lote: `[n]` para a citação válida, nada para a inválida (ADR-024).

    O texto liberado é sempre a reescrita do trecho estável da resposta; `finish` faz a validação em lote sobre a
    resposta inteira — por isso o resultado é o mesmo do `validate`, e o fim só acrescenta.
    """

    def __init__(self, delivered: Mapping[str, Chunk]) -> None:
        self._delivered = delivered
        self._raw = ""
        self._released = ""

    def feed(self, delta: str) -> str:
        self._raw += delta
        stable = self._raw[: _stable_end(self._raw)]
        text = rewrite(stable, _numbers(stable, self._delivered))
        released = text[len(self._released) :]
        self._released = text
        return released

    def finish(self) -> ValidationResult:
        return _check(self._raw, self._delivered)


def _delivered(view: TurnView) -> dict[str, Chunk]:
    """Os trechos entregues à geração, pela chave de comparação (o primeiro vence)."""
    delivered: dict[str, Chunk] = {}
    for item in view.generation_context:
        delivered.setdefault(key(item.chunk_id), item)
    return delivered


def _numbers(answer: str, delivered: Mapping[str, Chunk]) -> dict[str, int]:
    numbers: dict[str, int] = {}
    for raw in extract(answer):
        if key(raw) in delivered and key(raw) not in numbers:
            numbers[key(raw)] = len(numbers) + 1
    return numbers


def _stable_end(raw: str) -> int:
    """Até onde a resposta já não muda com o que ainda vai chegar: antes do primeiro `[[` sem `]]` depois dele, antes
    de um `[` final e antes do espaço que precede esse ponto."""
    last_close = raw.rfind("]]")
    pending = raw.find("[[", last_close + 2 if last_close >= 0 else 0)
    end = pending if pending >= 0 else len(raw)
    if end == len(raw) and raw.endswith("["):
        end -= 1
    while end > 0 and raw[end - 1].isspace():
        end -= 1
    return end


def _check(answer: str, delivered: Mapping[str, Chunk]) -> ValidationResult:
    numbers: dict[str, int] = {}
    cited: list[Chunk] = []
    issues: list[str] = []
    for raw in extract(answer):
        found = delivered.get(key(raw))
        if found is None:
            issues.append(INVALID_ISSUE.format(raw=raw))
        elif key(raw) not in numbers:
            numbers[key(raw)] = len(numbers) + 1
            cited.append(found)

    flags = {"invalid_citation"} if issues else set()
    rewritten = rewrite(answer, numbers)
    if not cited:
        issues.append(UNGROUNDED_ISSUE)
        flags.add("ungrounded")
        rewritten = _append(rewritten, UNGROUNDED_NOTICE)
    return ValidationResult(
        ok=not issues,
        issues=tuple(issues),
        answer=rewritten,
        citations=tuple(_citation(item) for item in cited),
        flags=frozenset(flags),
    )


def _citation(item: Chunk) -> Citation:
    """A proveniência exibível (FR-005), com o `chunk_id` canônico do trecho."""
    return Citation(
        chunk_id=item.chunk_id,
        document=item.document,
        version=item.version,
        status=item.status,
        section=item.section,
        page=item.page,
    )


def _append(answer: str, notice: str) -> str:
    body = answer.rstrip()
    return f"{body}\n\n{notice}" if body else notice
