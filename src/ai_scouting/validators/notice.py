"""`validity_notice`: toda citação não vigente sai com aviso (spec 104 US3, FR-004; princípio IX; R6, R7).

Lê as citações já validadas pelo `citation_integrity` (por isso roda depois). Só o status `vigente` dispensa aviso;
`revogado`, `substituído` e `rascunho` avisam com documento e status; status vazio ou desconhecido avisa que a vigência
não foi confirmada (clarificação 2 — falha fechada). O aviso que o modelo já deu conta: uma oração com `[n]` e uma marca
do status, ou a linha padrão — assim reexecutar não duplica.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from ai_core_contracts import Citation, TurnView, ValidationResult, ValidatorContext

from ai_scouting.validators._text import fold

NON_CURRENT_NOTICE: Final = "Atenção: [{n}] cita {source}, que está {status} — não é a versão vigente."
UNCONFIRMED_NOTICE: Final = "Atenção: [{n}] cita {source}, cuja vigência não foi confirmada."

CURRENT: Final = "vigente"
NON_CURRENT: Final = frozenset({"revogado", "substituido", "rascunho"})
"""Os status da 102, dobrados (sem acento, minúsculos)."""

NON_CURRENT_MARKS: Final = ("revogad", "substitu", "nao vigente", "nao esta vigente", "rascunho")
UNCONFIRMED_MARKS: Final = ("vigencia nao confirmada",)

_CLAUSES: Final = re.compile(r"[^.;!?\n]+")


class ValidityNoticeValidator:
    """Implementa `AnswerValidator` (001). Determinístico, sem modelo e sem I/O (FR-006)."""

    name = "validity_notice"

    async def validate(self, view: TurnView, ctx: ValidatorContext) -> ValidationResult:
        if view.answer is None:
            return ValidationResult(ok=True)
        notices: list[str] = []
        non_current = False
        for n, citation in enumerate(view.citations, start=1):
            status = fold(citation.get("status") or "")
            if status == CURRENT:
                continue
            non_current = True
            if status in NON_CURRENT:
                line = NON_CURRENT_NOTICE.format(n=n, source=_source(citation), status=citation.get("status"))
                marks: Sequence[str] = NON_CURRENT_MARKS
            else:
                line = UNCONFIRMED_NOTICE.format(n=n, source=_source(citation))
                marks = UNCONFIRMED_MARKS
            if not _already_warned(view.answer, n, line, marks):
                notices.append(line)

        flags = frozenset({"non_current_source"}) if non_current else frozenset()
        if not notices:
            return ValidationResult(ok=True, flags=flags)
        body = view.answer.rstrip()
        block = "\n".join(notices)
        return ValidationResult(ok=True, answer=f"{body}\n\n{block}" if body else block, flags=flags)


def _source(citation: Citation) -> str:
    version = citation.get("version")
    document = citation.get("document") or citation["chunk_id"]
    return f"{document} ({version})" if version else document


def _already_warned(answer: str, n: int, line: str, marks: Sequence[str]) -> bool:
    """A linha padrão está no texto, ou alguma oração cita `[n]` e diz o status."""
    if line in answer:
        return True
    reference = f"[{n}]"
    for clause in _CLAUSES.findall(answer):
        if reference in clause:
            folded = fold(clause)
            if any(mark in folded for mark in marks):
                return True
    return False
