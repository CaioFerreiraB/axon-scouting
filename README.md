# axon-scouting

Pack de domínio escoteiro da plataforma Axon: um pack externo, descoberto pelo entry point `axon.packs`,
que depende só de `axon-ai-contracts` (ADR-004, ADR-014, ADR-021).

A especificação (SDD) mora no repositório da plataforma, `CaioFerreiraB/axon`, nas specs `1xx`
(`specs/101-scouting-pack/` e seguintes); lá este repositório é o submódulo `axon-scouting/`.

## Numa instalação

O pack traz o próprio índice: o artefato em `src/axon_scouting/data/index/` (ADR-043) vai no pacote e é carregado em
memória, sem serviço nenhum. O que a instalação escolhe no `packs.yaml` é o modelo denso da pergunta:

```yaml
packs:
  scouting:
    enabled: true
    config:
      embedder: e5     # default; ou pplx (remoto, pelo OpenRouter)
```

| `embedder` | O que a instalação precisa |
|---|---|
| `e5` (default) | nada além do build: `axon packs prepare` baixa o `multilingual-e5-small` (revisão e sha256 fixados, ~470 MB) em `<AXON_PACKS_DATA_DIR>/scouting/e5-small` (ADR-041). Em runtime, ~860 MB de memória |
| `pplx` | `AXON_SCOUTING_EMBEDDING_API_KEY` no ambiente (uma chave do OpenRouter); a pergunta vai ao `perplexity/pplx-embed-v1-0.6b`. A imagem não carrega modelo nenhum |

Nenhum serviço adicional: o Qdrant é opcional, para corpus grande (`store: qdrant`, `qdrant_url`, `index`; a chave em
`AXON_SCOUTING_QDRANT_API_KEY`), carregado a partir do artefato por `axon-scouting-corpus load-qdrant`. Configuração
completa e defaults: `read_config` em `src/axon_scouting/pack.py` e o `contracts/` da spec 101 no `axon`.

## Corpus e o artefato do índice

`corpus/` guarda o catálogo curado (`corpus.yaml`) e os PDFs oficiais que ele cita, no git normal (spec 102,
clarificação de 2026-10-08). Cada entrada traz o link oficial do arquivo em `source_url`; o formato completo está em
`specs/102-scouting-corpus-ingestion/contracts/corpus-catalog.md`, no `axon`.

A ingestão roda aqui, no desenvolvimento, e nunca na instalação: ela grava o artefato do pacote, que se commita. Os
trechos são os mesmos para qualquer modelo; cada modelo acrescenta o próprio conjunto de vetores.

```bash
uv run axon-scouting-corpus model fetch .cache/e5                              # uma vez: e5-small na revisão fixada
AXON_SCOUTING_EMBEDDING_DIR=.cache/e5 uv run axon-scouting-corpus ingest corpus/corpus.yaml
AXON_SCOUTING_EMBEDDING_API_KEY=… uv run axon-scouting-corpus ingest corpus/corpus.yaml --embedder pplx
uv run axon-scouting-corpus status
```

Tudo ou nada; o que não mudou é reaproveitado. Mudou o chunking ou o modelo: `reindex` em vez de `ingest` (para cada
modelo). Um modelo cujo conjunto deixaria trecho sem vetor sai do artefato até ser reindexado. Atualizar o corpus é
lançar uma versão nova do pack.

## Avaliação

`eval/` guarda o conjunto dourado e a configuração com que ele roda pelo harness da plataforma (`axon eval run`,
spec 009), e o pack publica as métricas de domínio pelo hook `eval_metrics` (spec 105): `context_accuracy` (a
derivação de contexto contra o `extra.context` do item), `current_document_rate` (trechos vigentes entregues ao
modelo) e `validity_notice_rate` (aviso de vigência quando a resposta cita fonte não vigente). As duas primeiras são
determinísticas e, com as de recuperação, são as únicas que barram uma regressão.

O `eval/golden.yaml` é uma **semente provisória**: perguntas escritas a partir do corpus, sem curadoria, para provar o
pipeline e dar ao CI uma linha de base. Nenhuma calibração se decide sobre ela; a curadoria troca o conteúdo pelas
perguntas reais e mantém o formato (`specs/105-scouting-evaluation/contracts/golden-set.md`, validado por
`tests/eval/test_golden.py`). Para rodar como a CI, da raiz do repositório:

```bash
uv run axon-scouting-corpus model fetch .cache/e5
AXON_PROFILE=test AXON_PACKS_FILE=eval/packs.yaml AXON_ASSISTANTS_FILE=eval/assistants.yaml \
  uv run --python 3.14 --group platform axon eval run --golden eval/golden.yaml --baseline eval/baseline.json
```

Saída 0 sem regressão, 1 com regressão determinística, 2 com entrada inválida. A linha de base (`eval/baseline.json`)
só muda de propósito: a mesma rodada com `--update-baseline` (com regressão, também `--force`), num commit que diz por
quê. O workflow `eval` faz isso a cada mudança de corpus, RAG, prompt, derivação, métricas, pack, `eval/` ou
dependências, com o artefato em memória e o gerador simulado do perfil `test` — sem modelo pago nem juiz; ele também
confere que o artefato commitado espelha o corpus.

Para comparar os modelos (ADR-043, item 7), a mesma rodada com `AXON_PACKS_FILE=eval/packs-pplx.yaml` (e
`AXON_SCOUTING_EMBEDDING_API_KEY`) e `--report`, e depois `axon eval compare` dos dois relatórios.
