# Etapa 12 — Multi-Head Attention

Status: [ ] Não iniciado · [ ] Em andamento · [x] Concluído

Teoria em `theory/12-multi-head-attention/`: capítulo (10 seções), guia visual
(4 figuras) e 14 exercícios — escritos antes da implementação.
Implementação em [MultiHeadAttention.scala](../gpt/src/main/scala/gpt/nn/MultiHeadAttention.scala),
suíte em [MultiHeadAttentionSpec.scala](../gpt/src/test/scala/gpt/nn/MultiHeadAttentionSpec.scala) (25 testes).

- [x] `W_Q, W_K, W_V, W_O` todos `[dModel, dModel]` — `W_K` e `W_V` com `useBias = false`
- [x] Validar `dModel % nHeads == 0` — `require` no construtor, mensagem citando os dois números
- [x] Split em heads: reshape `[B,T,nHeads,dHead]` + transpose `[B,nHeads,T,dHead]`
- [x] Attention em batch sobre a dimensão `nHeads` — sem mudança no núcleo: o `matmul`
      unificado já lê dois eixos de lote pelas strides, sem copiar os operandos
- [x] Concat de volta: transpose + reshape para `[B,T,dModel]`
- [x] Projeção final `W_O`
- [x] Contar parâmetros (~`4 * dModel²`) — exatos `4·dModel² + 2·dModel`, independente de `nHeads`
- [x] Gradient check — nos 6 parâmetros, em `x`, e com entrada não contígua

## Pré-requisitos resolvidos antes da camada

- **`useBias` no `Linear`** (item 1 da etapa): fecha a pendência do `b_K` morto da Etapa 11.
- **`matmul` unificado**: um só caminho, duas últimas dimensões são a matriz, o resto é lote.
      Sem teto de rank, e entrada não contígua lida pelas strides — que é exatamente o caso
      de `Q` e `K` recém-transpostos.

## Dois bugs pegos na revisão, ambos do mesmo deslize

O `attentionWeights` recebia `scores` num parâmetro chamado `x`, e `x` significa a entrada
`[B,T,dModel]` em todo o resto da classe. Os dois têm rank e eixos diferentes:

1. **`softmax(x.rank - 1)`** normalizava o eixo das *queries* em vez do das *keys*. Shape idêntico
   (os eixos 2 e 3 têm ambos tamanho `T`), sem `NaN`, causalidade preservada e gradient check
   aprovando. Medido: soma das linhas `0,3507 / 0,8344 / 1,8148` em vez de `1`.
2. **`Masks.causalMask(x.shape(1))`** construía a máscara `[nHeads, nHeads]` em vez de `[T, T]`.
   Falha alta quando `T ≠ nHeads`, e passa por coincidência aritmética quando `T == nHeads`.

Correção: `attentionWeights` passou a receber a entrada de verdade e a derivar cada índice do
tensor a que ele pertence. Virou também a janela de diagnóstico pública que os testes de
propriedade usam.

## Validação por mutação

Cinco mutações, cada uma revertida em seguida:

| mutação | testes que falharam |
|---|---|
| `softmax` no eixo das queries (o bug original) | 7 de 25 |
| split com `transpose` antes do `reshape` | 14 de 25 |
| merge sem o `transpose` de volta | 2 de 25 |
| máscara depois do softmax | 10 de 25 |
| sem a divisão por `√dHead` | 3 de 25 |

**MILESTONE: "o coração do transformer está pronto"** ✅
