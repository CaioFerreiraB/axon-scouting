# Transcrição dos documentos digitalizados do corpus (para o Codex)

Você vai transcrever, página por página, os 91 documentos digitalizados do corpus do pack `axon-scouting` (atas de
assembleias, conselhos e fóruns, relatórios antigos). Hoje esses documentos são lidos por OCR automático (Tesseract),
que erra onde há carimbo de cartório, rubrica ou datilografia fraca. A sua transcrição substitui a do OCR na ingestão
do índice de busca, então ela precisa ser **fiel ao que está escrito**: é melhor marcar `[ilegível]` do que adivinhar.

Trabalhe na raiz do repositório `axon-scouting`. Rode os comandos com `XDG_DATA_HOME=$HOME/.local/share` no ambiente.

## 1. Preparar as páginas

```bash
docker pull jbarlow83/ocrmypdf:latest   # uma vez; o Tesseract roda nesse contêiner
XDG_DATA_HOME=$HOME/.local/share uv run python scripts/ocr.py pages $(cat scripts/ocr-documentos.txt)
```

Isso cria, para cada documento, a pasta `.cache/ocr/<nome-do-arquivo>/` com:

- `p001.png`, `p002.png`… — a imagem de cada página, a 300 dpi;
- `p001.tesseract.txt`… — a leitura do Tesseract, só como apoio;
- `manifest.json` — o PDF de origem e o arquivo de transcrição que será montado.

## 2. Transcrever cada página

Para cada `pNNN.png` de cada pasta, olhe a imagem e escreva `pNNN.codex.txt` na mesma pasta, em UTF-8. Regras:

- **Transcreva literalmente**, na ordem de leitura (em página de duas colunas, a coluna da esquerda inteira, depois a da
  direita). Preserve a grafia original, inclusive acentuação antiga ("Assembléia"), abreviações e erros de digitação.
  Não corrija, não resuma, não reescreva.
- **Mantenha a estrutura**: uma linha em branco entre parágrafos; títulos e itens numerados ("1) ABERTURA:",
  "Art. 5º", "SÉTIMO ITEM:") cada um começando numa linha própria; listas de presença, um nome por linha.
- **Tabelas**: uma linha por linha da tabela, com as células separadas por ` | `.
- **Deixe de fora**: carimbos e selos de cartório, etiquetas de registro, rubricas, assinaturas manuscritas (mantenha o
  nome impresso abaixo delas, se houver), números de página, marcas d'água, timbres repetidos no topo ou no rodapé de
  cada página (endereço, telefone, CNPJ do escritório).
- **Texto sob carimbo**: transcreva o que dá para ler com segurança; o que não der, `[ilegível]` no lugar da palavra ou
  do trecho. Nunca complete com o que "provavelmente" estava escrito.
- **Manuscrito no corpo do texto** (correção, acréscimo à margem): transcreva entre colchetes, `[manuscrito: …]`.
- **Página em branco ou só com carimbo**: arquivo vazio.
- Use o `pNNN.tesseract.txt` só para conferir uma leitura duvidosa; quando ele discordar da imagem, vale a imagem.
- Só texto: nada de comentário, explicação, markdown ou cabeçalho seu no arquivo.

## 3. Montar e conferir

```bash
XDG_DATA_HOME=$HOME/.local/share uv run python scripts/ocr.py assemble
XDG_DATA_HOME=$HOME/.local/share uv run python scripts/ocr.py check --limiar 0.6
```

`assemble` grava `corpus/ocr/<caminho do original>.txt` (uma página por bloco, separadas por U+000C) para cada
documento com todas as páginas transcritas. `check` compara cada página com o Tesseract e lista as que concordam pouco
(abaixo de 0,6). Abra a imagem de cada página listada e confira a sua transcrição; corrija o `pNNN.codex.txt` se
preciso e rode `assemble` de novo. Concordância baixa não é erro por si: o Tesseract costuma ser o errado.

## 4. O que entregar

- Altere **só** `.cache/ocr/` e os arquivos `corpus/ocr/**/*.txt`. Não mexa no catálogo (`corpus/corpus.yaml`,
  `corpus/fontes.yaml`), no código nem nos PDFs.
- No fim, responda com: quantos documentos e páginas foram transcritos; a lista das páginas que ficaram abaixo de 0,6
  depois da sua conferência, com o motivo (carimbo sobre o texto, manuscrito, página quase ilegível); e os documentos
  que você não conseguiu terminar.
