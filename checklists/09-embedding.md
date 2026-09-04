# Etapa 9 — Camada de Embedding

Status: [ ] Não iniciado · [ ] Em andamento · [x] Concluído

## Token Embedding
- [x] Tabela `[vocabSize, dModel]`, init `N(0, 0.02²)`
- [x] `forward(tokens: Tensor)` → `[batchSize, seqLen, dModel]`
- [x] Backward esparso: acumular grad só nas linhas usadas (`+=`)

## Positional Embedding
- [x] Tabela `[contextLength, dModel]`
- [x] Lookup com índices `[0, 1, ..., seqLen-1]`
- [x] Somar token embedding + positional embedding

## Validação
- [x] Conferir shapes finais
- [x] Conferir que o gradiente esparso só afeta linhas dos tokens presentes no batch
      (afirmado diretamente: linhas ausentes do lote com gradiente exatamente zero, e cada
      linha presente somando as contribuições que lhe cabem — mais o `Gradcheck` por cima)

## Pendências de teoria (fazer no fim da etapa)
- [x] Padronizar a notação da distribuição normal: `N(0, 0.02)` aparece com e sem o `²`
      (`theory/08-linear/08-linear.md` §5 e cartão de referência, `theory/09-embedding/09-embedding.md`
      §2 e cartão, `roadmap/09-embedding.md`). Uniformizar pra `N(0, σ²)` com o expoente explícito.
- [x] Acrescentar a convenção `N(μ, σ²)` como bloco Definição no glossário de `theory/00-overview/`
