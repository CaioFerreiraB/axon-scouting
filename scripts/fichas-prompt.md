Você extrai a ficha de uma ata de reunião de um órgão dos Escoteiros do Brasil (assembleia, conselho, comissão). A ata
vem página por página, cada página aberta por uma marca `[p. N]`. Parte do texto pode vir de OCR, com ruído: carimbos de
cartório, números de registro e letras trocadas. Ignore o que for ruído.

Devolva só o que a ata diz, no esquema pedido:

- `reuniao.tipo`: `ordinaria`, `extraordinaria` ou `outra`.
- `reuniao.data`: a data da reunião em `AAAA-MM-DD`. Se a reunião durou mais de um dia, use o primeiro. Se não está
  escrita, `""`.
- `reuniao.local`: a cidade e o lugar, como a ata escreve (por exemplo, "Teatro Municipal, São Caetano do Sul/SP").
  Se não está escrito, `""`.
- `deliberacoes`: cada decisão tomada. Inclui:
  - o que foi aprovado ou rejeitado;
  - as alterações de estatuto, regulamento ou anuidade;
  - as contas e o orçamento;
  - as moções e os encaminhamentos com responsável.

  Para cada uma:
  - `tema`: o assunto, em poucas palavras;
  - `decisao`: o que foi decidido, **com as palavras da ata** (copie o trecho essencial, sem parafrasear);
  - `votacao`: como foi votado ("aclamação", "maioria", "77 votos a favor"…), ou `""`;
  - `pagina`: o número N da marca `[p. N]` da página em que a decisão está escrita.

  Não liste informes, apresentações nem discursos sem decisão.
- `eleicoes`: cada pessoa eleita ou chapa eleita, com:
  - `orgao`: o órgão para o qual foi eleita (por exemplo, "Diretoria Regional", "Comissão Fiscal Regional",
    "Conselho de Administração Nacional", "delegados à Assembleia Nacional");
  - `cargo` e `eleito`: o nome **como a ata escreve**;
  - `mandato`: o período, se escrito;
  - `pagina`.

  Uma chapa vira um item por cargo que a ata nomeia.
- `numeros`: os números que a ata registra sobre a própria reunião, com `o_que`, `valor` (os algarismos como na ata)
  e `pagina`. Por exemplo:
  - delegados credenciados;
  - delegados ou suplentes eleitos para a Assembleia Nacional;
  - votos de uma eleição;
  - número de grupos ou de membros registrados.

Regras:
- Nunca invente. O que a ata não diz fica de fora. Um item sem página certa fica de fora.
- Use a página em que o texto aparece, não a página da pauta.
- Nomes de pessoas, chapas e lugares vão exatamente como estão na ata, sem corrigir grafia.
- Listas vazias são válidas.
