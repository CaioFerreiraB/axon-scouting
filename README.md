# axon-scouting

Pack de domínio escoteiro da plataforma Axon: um pack externo, descoberto pelo entry point `axon.packs`,
que depende só de `axon-ai-contracts` (ADR-004, ADR-014, ADR-021).

A especificação (SDD) mora no repositório da plataforma, `CaioFerreiraB/axon`, nas specs `1xx`
(`specs/101-scouting-pack/` e seguintes); lá este repositório é o submódulo `axon-scouting/`.

## Corpus

`corpus/` guarda o catálogo curado (`corpus.yaml`) e os PDFs oficiais que ele cita, no git normal (spec 102,
clarificação de 2026-10-08). Cada entrada traz o link oficial do arquivo em `source_url`; o formato completo está em
`specs/102-scouting-corpus-ingestion/contracts/corpus-catalog.md`, no `axon`. Para indexar num Qdrant:

```bash
uv run axon-scouting-corpus model fetch ~/.cache/axon-scouting/e5-small       # uma vez: e5-small na revisão fixada
export AXON_SCOUTING_EMBEDDING_DIR=~/.cache/axon-scouting/e5-small AXON_SCOUTING_QDRANT_URL=http://localhost:6333
uv run axon-scouting-corpus ingest corpus/corpus.yaml                       # tudo ou nada; reaproveita o que não mudou
uv run axon-scouting-corpus status
```

Mudou a estratégia de chunking ou o modelo de embedding: `axon-scouting-corpus reindex corpus/corpus.yaml`.

## Avaliação

`eval/` guarda o conjunto dourado e a configuração com que ele roda pelo harness da plataforma (`axon eval run`,
spec 009), e o pack publica as métricas de domínio pelo hook `eval_metrics` (spec 105): `context_accuracy` (a
derivação de contexto contra o `extra.context` do item), `current_document_rate` (trechos vigentes entregues ao
modelo) e `validity_notice_rate` (aviso de vigência quando a resposta cita fonte não vigente). As duas primeiras são
determinísticas e, com as de recuperação, são as únicas que barram uma regressão.

O `eval/golden.yaml` é uma **semente provisória**: perguntas escritas a partir do corpus, sem curadoria, para provar o
pipeline e dar ao CI uma linha de base. Nenhuma calibração se decide sobre ela; a curadoria troca o conteúdo pelas
perguntas reais e mantém o formato (`specs/105-scouting-evaluation/contracts/golden-set.md`, validado por
`tests/eval/test_golden.py`). Para rodar como a CI, da raiz do repositório e com o corpus ingerido num Qdrant local:

```bash
uv run axon-scouting-corpus model fetch .cache/e5
AXON_SCOUTING_QDRANT_URL=http://localhost:6333 AXON_SCOUTING_EMBEDDING_DIR=.cache/e5 \
  uv run axon-scouting-corpus ingest corpus/corpus.yaml
AXON_PROFILE=test AXON_PACKS_FILE=eval/packs.yaml AXON_ASSISTANTS_FILE=eval/assistants.yaml \
  uv run --python 3.14 --group platform axon eval run --golden eval/golden.yaml --baseline eval/baseline.json
```

Saída 0 sem regressão, 1 com regressão determinística, 2 com entrada inválida. A linha de base (`eval/baseline.json`)
só muda de propósito: a mesma rodada com `--update-baseline` (com regressão, também `--force`), num commit que diz por
quê. O workflow `eval` faz isso a cada mudança de corpus, RAG, prompt, derivação, métricas, pack, `eval/` ou
dependências, com um Qdrant de serviço e o gerador simulado do perfil `test` — sem modelo pago nem juiz. Uma rodada com
modelo real usa o perfil `memory` (Ollama local) e `--judge provider/modelo` para a fidelidade.
