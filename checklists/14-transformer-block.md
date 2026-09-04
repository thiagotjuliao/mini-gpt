# Etapa 14 — Bloco Transformer Completo

Status: [ ] Não iniciado · [ ] Em andamento · [x] Concluído

Teoria em `theory/14-transformer-block/`: capítulo (8 seções), guia visual (5 figuras) e
15 exercícios — escritos antes da implementação, como manda a convenção da Etapa 4.
Implementação em [TransformerBlock.scala](../gpt/src/main/scala/gpt/nn/TransformerBlock.scala),
suíte em [TransformerBlockSpec.scala](../gpt/src/test/scala/gpt/nn/TransformerBlockSpec.scala) (16 testes).

- [x] `H = X + MultiHeadAttention(LayerNorm₁(X))`
- [x] `Y = H + MLP(LayerNorm₂(H))`
- [x] Confirmar Pre-LayerNorm (não Post-LN) — com teste que reprova a troca
- [x] Expor `ln1`, `attention`, `ln2`, `mlp` como `val` públicos
- [x] `parameters` com 14 tensores, na ordem `ln1 ++ attention ++ ln2 ++ mlp`
- [x] Validações com `require`: `x.rank == 3` e `x.shape.last == dModel`
- [x] Contar parâmetros — exatos `12·dModel² + 11·dModel`
- [x] Gradient check do bloco completo — em `x`, nos 14 parâmetros, e com entrada não contígua
- [x] Causalidade ponta a ponta: mudar o último token não mexe nas saídas anteriores, e o último muda
- [x] Empilhável: aplicar o bloco à própria saída
- [x] Dropout — **fora do escopo**, como o roadmap permite

## O achado da implementação

**`Linear(dModel, dModel)` no lugar de `LayerNorm(dModel)`.** Os dois campos `ln1`/`ln2` nasceram
como `Linear`. O bloco compilava, rodava e devolvia o formato certo — só não normalizava nada.

Dos 16 testes, **2** reprovaram: o formato dos parâmetros de cada `LayerNorm`
(`Array(8,8)` contra `Array(8)`) e a contagem total (`14d² + 9d` contra `12d² + 11d`).
Passaram, entre outros, a composição contra as subcamadas — porque a referência usa o mesmo
`ln1` e concorda com o erro —, o gradient check nos 14 parâmetros, a causalidade e o teste de
pre-LN. É a §7 do capítulo acontecendo ao vivo: o gradient check confere coerência, nunca intenção.

**Correção de contagem no capítulo:** a §1 dizia 12 tensores de parâmetro, e são **14**
(`2 + 6 + 2 + 4`). O erro se propagou para o guia visual, os exercícios, este checklist e o
`HISTORY`; corrigido nos cinco. A contagem de *números* (`12·dModel² + 11·dModel`) sempre esteve certa.

## Validação por mutação

Sete mutações, cada uma revertida em seguida:

| mutação | testes que falharam |
|---|---|
| remover a primeira soma residual | 3 de 16 |
| remover a segunda soma residual | 3 de 16 |
| trocar para post-LN | 2 de 16 |
| trocar a ordem das subcamadas | 2 de 16 |
| `Linear` no lugar de `LayerNorm` | 2 de 16 |
| mesmo `LayerNorm` nas duas posições | 1 de 16 |
| passar `x` no lugar de `h` ao MLP | 1 de 16 |

As duas últimas dependem de um teste só cada — `toSet.size` e a comparação contra a composição.

## Pendência que vence na Etapa 15

**`1/√(2·nLayers)` nas camadas que escrevem no fluxo residual** (`W_O` e a segunda `Linear` do MLP).
A §5 do capítulo derivou o fator: como as variâncias dos `2L` incrementos somam, a saída da pilha
tem variância `1 + 2L` sem escala — 25 com `L = 12`, ou norma `×5`. Multiplicar cada incremento
por `1/√(2L)` faz a soma dar **exatamente 2**, para qualquer profundidade.
