# ai-scouting

Pack de domínio escoteiro da plataforma Agentic AI: um pack externo, descoberto pelo entry point `ai_core.packs`,
que depende só de `ai-core-contracts` (ADR-004, ADR-014, ADR-021).

A especificação (SDD) mora no repositório da plataforma, `CaioFerreiraB/agentic-app`, nas specs `1xx`
(`specs/101-scouting-pack/` e seguintes); lá este repositório é o submódulo `ai-scouting/`.

## Corpus

`corpus/` guarda o catálogo curado (`corpus.yaml`) e os PDFs oficiais que ele cita, no git normal (spec 102,
clarificação de 2026-10-08). Cada entrada traz o link oficial do arquivo em `source_url`; o formato completo está em
`specs/102-scouting-corpus-ingestion/contracts/corpus-catalog.md`, no `agentic-app`. Para indexar num Qdrant:

```bash
uv run ai-scouting-corpus model fetch ~/.cache/ai-scouting/e5-small       # uma vez: e5-small na revisão fixada
export AI_SCOUTING_EMBEDDING_DIR=~/.cache/ai-scouting/e5-small AI_SCOUTING_QDRANT_URL=http://localhost:6333
uv run ai-scouting-corpus ingest corpus/corpus.yaml                       # tudo ou nada; reaproveita o que não mudou
uv run ai-scouting-corpus status
```

Mudou a estratégia de chunking ou o modelo de embedding: `ai-scouting-corpus reindex corpus/corpus.yaml`.
