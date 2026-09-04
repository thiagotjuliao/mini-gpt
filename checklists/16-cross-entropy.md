# Etapa 16 — Função de Perda (Cross-Entropy Loss)

Status: [ ] Não iniciado · [ ] Em andamento · [x] Concluído

Teoria em `theory/16-cross-entropy/`: capítulo (8 seções), guia visual (4 figuras) e
15 exercícios — escritos antes da implementação, como manda a convenção da Etapa 4.
Implementação em [CrossEntropy.scala](../gpt/src/main/scala/gpt/loss/CrossEntropy.scala),
suíte em [CrossEntropySpec.scala](../gpt/src/test/scala/gpt/loss/CrossEntropySpec.scala) (15 testes),
mais `Tensor.oneHot` no núcleo com 5 testes no `TensorSpec`.

- [x] `gpt/loss/CrossEntropy.scala` — `object` com `apply(logits, targets)` e `perplexity(loss)`
- [x] `logSoftmax(2)` — o eixo do vocabulário (reusado da Etapa 6)
- [x] Seleção do alvo por máscara one-hot, com `requiresGradient = false`
- [x] `Tensor.oneHot(indices, numClasses)` no `scalagrad`, com validação de faixa
- [x] Média sobre `N = B·T`, não soma
- [x] Retorno é um `Tensor` escalar
- [x] `perplexity = exp(loss)`, fora do autograd
- [x] Validações com `require`: ranks, dimensões batendo, alvos inteiros e em `[0, V)`
- [x] Valor à mão: `2.36971`; média do lote: `1.33821`; `log(V)` com logits uniformes
- [x] Gradiente conferido contra `(p − y)/N`, posição a posição

## Decisões registradas

- **Máscara one-hot em vez de `gather`.** A máscara tem o tamanho dos logits, que já existem, então
  não muda a ordem de memória — e usa só operações testadas. O `gather` fica para quando houver
  desempenho medido pedindo.
- **`oneHot` no `scalagrad`, não em `gpt`.** É maquinaria genérica de tensor, sem nada de GPT
  dentro; mora ao lado de `Tensor.zeros` e `Tensor.fill`. O que é específico da perda — validar que
  o alvo é inteiro e está em `[0, V)` — ficou no `CrossEntropy`.
- **`sum / N` e não `.mean`.** O `.mean` dividiria por `B·T·V`; só `B·T` valores são não nulos
  depois da máscara.

## Bugs pegos na revisão

Quatro, em três rodadas. Todos silenciosos de alguma forma:

1. **`Masks.causalMask` no lugar da máscara one-hot.** A máscara da atenção (`[T,T]`, com `−inf`)
   não tem parentesco com a de seleção de alvo. Estouraria no `*` para quase toda forma, e onde
   não estourasse, `−inf × 0` daria `NaN`.
2. **`oneHot` comparando a linha em vez da coluna** (`i == indices(i)`). Cada linha saía inteira de
   zeros ou inteira de uns. Com a máscara toda zero, a perda seria sempre `0.0`.
3. **Falta do `reshape`.** `[B·T, V]` contra `[B, T, V]` só faz broadcast quando `B = 1` — funciona
   no primeiro teste de console e estoura no resto.
4. **Falta do sinal negativo.** A perda sairia negativa, e o treino minimizaria o oposto do que deve.

## Validação por mutação

| mutação | testes que falharam |
|---|---|
| `logSoftmax` no eixo 1 em vez do 2 | 7 de 15 |
| esquecer o sinal negativo | 6 de 15 |
| somar em vez de mediar | 2 de 15 |
| deslocar os alvos em uma posição | 2 de 15 |
| dividir por `B` em vez de `B·T` | 2 de 15 |
| `softmax` seguido de `log` | 1 de 15 — só o teste de subfluxo |
| `oneHot` com `i == indices(i)` | 2 no `TensorSpec` e 7 no `CrossEntropySpec` |

A linha do `softmax`+`log` é a que ensina: com logits normais os dois caminhos dão o **mesmo
número**, e só o caso com diferença de 800 reprova.
