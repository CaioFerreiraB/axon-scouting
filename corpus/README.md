# Corpus do pack scouting

Atualizado em **26/10/2026**, a partir da lista fornecida pelo curador e das páginas oficiais indicadas. Foram percorridas **64 páginas de categorias**, incluindo a primeira página da RNJL e toda a paginação, além das **19 atas** da pasta pública do Conselho Consultivo Regional de São Paulo.

O catálogo contém **593 documentos únicos**, com **57 cópias OCR** para leitura e **37 documentos com revogação expressa** registrada. Os **598 PDFs originais** foram preservados, inclusive cinco duplicatas binárias, que entram uma única vez no catálogo. A planilha disponível na categoria RNJL também foi baixada, totalizando **599 arquivos originais**. Os manuais disponibilizados via id@escoteiros foram aproveitados dos PDFs já fornecidos localmente, sem tentar acessar a conta do curador.

- `corpus.yaml`: catálogo compatível com o contrato atual da ingestão.
- `fontes.yaml`: URLs, páginas de origem, rótulos oficiais, caminhos, tamanho, SHA-256, duplicatas, relação entre original e OCR e notas de curadoria.
- `taxonomia.yaml`: o mapeamento semântico do acervo (categoria, subcategoria, nível e tipo), do mantenedor, com as linhas inferidas marcadas. Cada entrada é uma pasta: `<categoria>/<subcategoria>/<nível>/`, e o documento de uma região fica na subpasta dela (`…/sao-paulo/`). Desde 27/10/2026 todo original mora na pasta da sua entrada.
- `estrutura.yaml`: como cada documento se organiza por dentro (ADR-048, ADR-050) — o perfil vem do tipo da taxonomia, com as exceções.
- `ocr/`: derivados de leitura, no mesmo caminho do original. O catálogo aponta para eles quando necessário, mantendo o link da fonte original. OCR é transcrição automática e deve ser conferido contra a imagem quando a precisão da redação importar. Assinaturas digitais se referem ao PDF original, não ao derivado.

## Critérios do catálogo

Datas e versões de capa, expediente, cabeçalho, assinatura e cláusula de vigência prevalecem sobre o caminho de upload. `version: "1"`, quando nenhuma edição é informada, funciona apenas como identificador catalográfico. Textos consolidados declaram a revisão disponível, sem retroagir silenciosamente as alterações ao texto-base.

O esquema exige `effective_date` mesmo quando o documento não informa dia ou vigência. Os casos incompletos ou inferidos estão marcados **A CONFIRMAR** no comentário da entrada: primeiro dia do mês, 01/01 do ano conhecido ou data-base de upload, quando não há referência melhor. Há **90 entradas com metadados provisórios**. Em atas, a data-base é a reunião; em relatórios anuais, o exercício. Esses valores **não certificam a publicação nem a vigência jurídica**. A classificação `vigente` que o pipeline deriva para documentos não revogados/substituídos também não substitui a confirmação do curador.

Somente revogações integrais expressas foram registradas; alteração de artigo, consolidação e a cláusula genérica “revogam-se as disposições em contrário” não revogam automaticamente outro documento inteiro. `expiry_date` é inclusivo e usa a véspera da data-base do ato revogador. Uma revogação cujo ato não informa o dia mantém a ressalva de data provisória.

Desde o ADR-049 (2026-10-28) o catálogo **não tem `type`**: o tipo de cada documento é o da `taxonomia.yaml`, pela pasta do arquivo (Ata, Regulamento, Resolução, Política, Manual…), e é dele que saem o perfil de corte (`estrutura.yaml`) e o filtro de tipo da busca; a categoria (o órgão) e o nível da mesma entrada também filtram. Para pôr um documento novo, basta colocá-lo na pasta da entrada certa da taxonomia — ou criar a entrada. Documentos de vários ramos ficam sem `branch`; desde o ADR-045 (2026-10-27), o campo aceita também `filhotes`.

Documento digitalizado (sem camada de texto) entra pela transcrição em `ocr/<caminho do original>.txt`, uma página por bloco separado por quebra de página (ADR-050): `uv run python scripts/ocr.py tesseract <pdf>` faz a de partida, com o Tesseract local, e `scripts/ocr-codex-prompt.md` descreve a revisão página a página. As 57 cópias OCR em PDF de antes continuam valendo até a revisão chegar.

## Conferências e pendências

- A URL de “Façamos um Plano de Grupo” na lista apontava para “Captando e Engajando Adultos”. Foi localizada a [fonte correta](https://www.escoteiros.org.br/downloads/facamos-um-plano-de-grupo/) para o PDF já fornecido. O caminho da Política Nacional de Diversidade e Inclusão foi corrigido para `/2024/12/`.
- O arquivo publicado como [Resolução DEN 03/2015](https://www.escoteiros.org.br/wp-content/uploads/2020/07/ResDEN_003-2015.pdf) contém uma manifestação do CAN sobre o recurso da Região do Maranhão. Foi catalogado pelo conteúdo, com a divergência registrada. Esse download não fornece o texto da resolução mencionada.
- O card “Resolução DEN 06/21” entrega a [Resolução DEN 01/2022](https://www.escoteiros.org.br/wp-content/uploads/2022/04/Resolucao-DEN-01-2022_Refitica-a-Resolucao-DEN-06-2021-medalha-e-o-pin-da-Perseveranca.pdf), que a revoga. Foi usado o número do PDF; a versão anterior não está contida nesse arquivo.
- A [Resolução Regional sobre o GE Louveira](https://www.escoteirossp.org.br/wp/wp-content/uploads/2026/09/R_DRSP_2026_16_Instaura_Intervencao_GE_Louveira.pdf) traz divergência entre a numeração 2026.16 da fonte e 2026.015 no cabeçalho impresso, com correção manuscrita. A entrada conserva a numeração da fonte e a ressalva.
- A planilha `nacional/rede-nacional-de-jovens-lideres/MODELO-NRS-Plano-de-Acao-e-Previsao-Orcamentaria.xlsx` foi preservada, mas não entra em `corpus.yaml`: o parser existente lê PDFs. Sua proveniência está em `fontes.yaml`.
- A entrada do antigo `Escotistas_em_acao_ramo_lobinho.pdf` foi retirada porque o arquivo já estava removido da pasta ao iniciar este ajuste. O novo Manual do Escotista — Ramo Lobinho foi catalogado com sua própria identidade, sem inventar uma relação de substituição.

## Distribuição dos documentos únicos

| Pasta (taxonomia) | Documentos |
|---|---:|
| `assembleias/atas/nacional` | 39 |
| `assembleias/regulamentos/nacional` | 1 |
| `comissao-de-etica/regulamentos/regional/sao-paulo` | 1 |
| `comissao-fiscal/atas/nacional` | 2 |
| `comissao-fiscal/oficios/regional/sao-paulo` | 1 |
| `comissao-fiscal/pareceres/regional/sao-paulo` | 3 |
| `comissao-fiscal/relatorios/nacional` | 1 |
| `conselho-consultivo/atas/nacional` | 14 |
| `conselho-consultivo/atas/regional/sao-paulo` | 19 |
| `conselho-de-administracao-nacional/atas` | 104 |
| `conselho-de-administracao-nacional/manifestacoes` | 1 |
| `conselho-de-administracao-nacional/regulamentos` | 1 |
| `conselho-de-administracao-nacional/resolucoes` | 149 |
| `conselho-de-velhos-lobos/regulamentos/regional/sao-paulo` | 1 |
| `conselho-jovem/regulamentos/regional/sao-paulo` | 1 |
| `diretoria-executiva-nacional/oficios` | 5 |
| `diretoria-executiva-nacional/resolucoes` | 43 |
| `diretoria-regional/instrucoes-normativas/sao-paulo` | 10 |
| `diretoria-regional/oficios/sao-paulo` | 1 |
| `diretoria-regional/resolucoes/sao-paulo` | 71 |
| `espacos-seguros/politicas/interamericano` | 1 |
| `espacos-seguros/politicas/nacional` | 2 |
| `gestao-de-adultos/manuais` | 11 |
| `gestao-de-adultos/politicas/mundial` | 1 |
| `gestao-de-adultos/politicas/nacional` | 1 |
| `gestao-institucional-e-governanca/estatutos/nacional` | 1 |
| `gestao-institucional-e-governanca/manuais` | 5 |
| `gestao-institucional-e-governanca/politicas/nacional` | 6 |
| `gestao-institucional-e-governanca/por` | 1 |
| `gestao-institucional-e-governanca/regulamentos/regional/sao-paulo` | 1 |
| `gestao-institucional-e-governanca/relatorios/regional/sao-paulo` | 13 |
| `programa-educativo/guias-de-jovens` | 6 |
| `programa-educativo/manuais` | 6 |
| `programa-educativo/manuais-de-escotistas` | 5 |
| `programa-educativo/politicas/nacional` | 1 |
| `rede-de-jovens-lideres/apresentacoes/nacional` | 4 |
| `rede-de-jovens-lideres/atas/nacional` | 14 |
| `rede-de-jovens-lideres/atas/regional/sao-paulo` | 19 |
| `rede-de-jovens-lideres/boletins/nacional` | 4 |
| `rede-de-jovens-lideres/editais/nacional` | 5 |
| `rede-de-jovens-lideres/manuais/nacional` | 4 |
| `rede-de-jovens-lideres/planejamento/nacional` | 1 |
| `rede-de-jovens-lideres/posicionamentos/nacional` | 5 |
| `rede-de-jovens-lideres/recomendacoes/nacional` | 4 |
| `rede-de-jovens-lideres/regulamentos/nacional` | 1 |
| `relacoes-institucionais/politicas/nacional` | 3 |

## Validação

Catálogo, linhagens e preparação da ingestão (leitura de todos os PDFs pelo PDFium, limpeza, divisão e identificadores) conferidos sem gerar embeddings nem escrever no índice. Os derivados OCR foram comparados aos originais quanto a quantidade e tamanho de páginas, com renderização de amostras. Os arquivos de código, specs, configurações, testes, conjunto dourado, baseline e artefato do índice não foram alterados. O índice empacotado ainda corresponde ao corpus anterior e precisará de uma ingestão posterior, fora do escopo deste ajuste restrito ao corpus.
