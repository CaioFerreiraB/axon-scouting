# Corpus do pack scouting

Atualizado em **26/10/2026**, a partir da lista fornecida pelo curador e das páginas oficiais indicadas. Foram percorridas **64 páginas de categorias**, incluindo a primeira página da RNJL e toda a paginação, além das **19 atas** da pasta pública do Conselho Consultivo Regional de São Paulo.

O catálogo contém **593 documentos únicos**, com **57 cópias OCR** para leitura e **37 documentos com revogação expressa** registrada. Os **598 PDFs originais** foram preservados, inclusive cinco duplicatas binárias, que entram uma única vez no catálogo. A planilha disponível na categoria RNJL também foi baixada, totalizando **599 arquivos originais**. Os manuais disponibilizados via id@escoteiros foram aproveitados dos PDFs já fornecidos localmente, sem tentar acessar a conta do curador.

- `corpus.yaml`: catálogo compatível com o contrato atual da ingestão.
- `fontes.yaml`: URLs, páginas de origem, rótulos oficiais, caminhos, tamanho, SHA-256, duplicatas, relação entre original e OCR e notas de curadoria.
- `nacional/` e `sao-paulo/`: novos downloads organizados por órgão/categoria; os arquivos previamente fornecidos mantêm seus caminhos.
- `ocr/`: derivados de leitura. O catálogo aponta para eles quando necessário, mantendo o link da fonte original. OCR é transcrição automática e deve ser conferido contra a imagem quando a precisão da redação importar. Assinaturas digitais se referem ao PDF original, não ao derivado.

## Critérios do catálogo

Datas e versões de capa, expediente, cabeçalho, assinatura e cláusula de vigência prevalecem sobre o caminho de upload. `version: "1"`, quando nenhuma edição é informada, funciona apenas como identificador catalográfico. Textos consolidados declaram a revisão disponível, sem retroagir silenciosamente as alterações ao texto-base.

O esquema exige `effective_date` mesmo quando o documento não informa dia ou vigência. Os casos incompletos ou inferidos estão marcados **A CONFIRMAR** no comentário da entrada: primeiro dia do mês, 01/01 do ano conhecido ou data-base de upload, quando não há referência melhor. Há **90 entradas com metadados provisórios**. Em atas, a data-base é a reunião; em relatórios anuais, o exercício. Esses valores **não certificam a publicação nem a vigência jurídica**. A classificação `vigente` que o pipeline deriva para documentos não revogados/substituídos também não substitui a confirmação do curador.

Somente revogações integrais expressas foram registradas; alteração de artigo, consolidação e a cláusula genérica “revogam-se as disposições em contrário” não revogam automaticamente outro documento inteiro. `expiry_date` é inclusivo e usa a véspera da data-base do ato revogador. Uma revogação cujo ato não informa o dia mantém a ressalva de data provisória.

O contrato atual oferece apenas `por`, `estatuto`, `resolucao`, `manual`, `programa_educativo` e `guia`. Atas, relatórios, políticas, regulamentos, ofícios e outros textos institucionais usam `manual`, que preserva sua leitura em janelas de texto. Instruções normativas usam `resolucao`. Título, caminho e proveniência mantêm a categoria editorial original. Documentos de vários ramos ficam sem `branch`. Desde o ADR-045 (2026-10-27), o campo aceita também `filhotes`, e os três documentos do Ramo Filhotes o declaram. Não houve mudança de código ou contrato para acomodar essas categorias.

## Conferências e pendências

- A URL de “Façamos um Plano de Grupo” na lista apontava para “Captando e Engajando Adultos”. Foi localizada a [fonte correta](https://www.escoteiros.org.br/downloads/facamos-um-plano-de-grupo/) para o PDF já fornecido. O caminho da Política Nacional de Diversidade e Inclusão foi corrigido para `/2024/12/`.
- O arquivo publicado como [Resolução DEN 03/2015](https://www.escoteiros.org.br/wp-content/uploads/2020/07/ResDEN_003-2015.pdf) contém uma manifestação do CAN sobre o recurso da Região do Maranhão. Foi catalogado pelo conteúdo, com a divergência registrada. Esse download não fornece o texto da resolução mencionada.
- O card “Resolução DEN 06/21” entrega a [Resolução DEN 01/2022](https://www.escoteiros.org.br/wp-content/uploads/2022/04/Resolucao-DEN-01-2022_Refitica-a-Resolucao-DEN-06-2021-medalha-e-o-pin-da-Perseveranca.pdf), que a revoga. Foi usado o número do PDF; a versão anterior não está contida nesse arquivo.
- A [Resolução Regional sobre o GE Louveira](https://www.escoteirossp.org.br/wp/wp-content/uploads/2026/09/R_DRSP_2026_16_Instaura_Intervencao_GE_Louveira.pdf) traz divergência entre a numeração 2026.16 da fonte e 2026.015 no cabeçalho impresso, com correção manuscrita. A entrada conserva a numeração da fonte e a ressalva.
- A planilha `nacional/rede-nacional-de-jovens-lideres/MODELO-NRS-Plano-de-Acao-e-Previsao-Orcamentaria.xlsx` foi preservada, mas não entra em `corpus.yaml`: o parser existente lê PDFs. Sua proveniência está em `fontes.yaml`.
- A entrada do antigo `Escotistas_em_acao_ramo_lobinho.pdf` foi retirada porque o arquivo já estava removido da pasta ao iniciar este ajuste. O novo Manual do Escotista — Ramo Lobinho foi catalogado com sua própria identidade, sem inventar uma relação de substituição.

## Distribuição dos documentos únicos

| Categoria/pasta | Documentos |
|---|---:|
| `documentos institucionais` | 4 |
| `estatuto` | 1 |
| `manuais` | 35 |
| `nacional/atas-assembleia-nacional` | 39 |
| `nacional/atas-can` | 104 |
| `nacional/atas-comissao-fiscal` | 2 |
| `nacional/atas-conselho-consultivo` | 14 |
| `nacional/oficios-den` | 5 |
| `nacional/rede-nacional-de-jovens-lideres` | 41 |
| `nacional/relatorios-comissao-fiscal` | 1 |
| `nacional/resolucoes-can` | 147 |
| `nacional/resolucoes-den` | 45 |
| `politicas` | 12 |
| `por` | 1 |
| `resolucoes` | 1 |
| `sao-paulo/atas-conselho-consultivo` | 19 |
| `sao-paulo/atas-frjl-sp` | 19 |
| `sao-paulo/instrucoes-normativas` | 10 |
| `sao-paulo/regulamentos` | 4 |
| `sao-paulo/relatorios-anuais` | 13 |
| `sao-paulo/resolucoes-regionais` | 71 |
| `sao-paulo/transparencia-financeiro` | 5 |

## Validação

Catálogo, linhagens e preparação da ingestão (leitura de todos os PDFs pelo PDFium, limpeza, divisão e identificadores) conferidos sem gerar embeddings nem escrever no índice. Os derivados OCR foram comparados aos originais quanto a quantidade e tamanho de páginas, com renderização de amostras. Os arquivos de código, specs, configurações, testes, conjunto dourado, baseline e artefato do índice não foram alterados. O índice empacotado ainda corresponde ao corpus anterior e precisará de uma ingestão posterior, fora do escopo deste ajuste restrito ao corpus.
