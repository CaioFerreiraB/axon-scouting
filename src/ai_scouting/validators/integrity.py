"""`citation_integrity`: só fica a citação que esteve no contexto da geração (spec 104 US1, US2, US4; ADR-007; R4, R5).

A fonte da verdade é `view.generation_context` — nem `retrieved`, nem `reranked`: trecho recuperado e cortado pelo
orçamento não foi visto pelo modelo. Citação inválida sai do texto e vira `issue` (o core a registra como `notice`);
as válidas viram `[n]`, na ordem da primeira aparição, e levam a proveniência do trecho. Sem nenhuma válida, a
resposta ganha o aviso de "não fundamentada".
"""

from __future__ import annotations

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
    """Implementa `AnswerValidator` (001). Determinístico, sem modelo e sem I/O (FR-006)."""

    name = "citation_integrity"

    async def validate(self, view: TurnView, ctx: ValidatorContext) -> ValidationResult:
        if view.answer is None:
            return ValidationResult(ok=True)
        delivered: dict[str, Chunk] = {}
        for item in view.generation_context:
            delivered.setdefault(key(item.chunk_id), item)

        numbers: dict[str, int] = {}
        cited: list[Chunk] = []
        issues: list[str] = []
        for raw in extract(view.answer):
            found = delivered.get(key(raw))
            if found is None:
                issues.append(INVALID_ISSUE.format(raw=raw))
            elif key(raw) not in numbers:
                numbers[key(raw)] = len(numbers) + 1
                cited.append(found)

        flags = {"invalid_citation"} if issues else set()
        answer = rewrite(view.answer, numbers)
        if not cited:
            issues.append(UNGROUNDED_ISSUE)
            flags.add("ungrounded")
            answer = _append(answer, UNGROUNDED_NOTICE)
        return ValidationResult(
            ok=not issues,
            issues=tuple(issues),
            answer=answer,
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
