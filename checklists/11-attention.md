# Etapa 11 — Scaled Dot-Product Attention

Status: [ ] Não iniciado · [ ] Em andamento · [x] Concluído

Teoria em `theory/11-attention/`: capítulo, guia visual (4 figuras) e 14 exercícios.
Implementação em [Attention.scala](../gpt/src/main/scala/gpt/nn/Attention.scala),
suíte em [AttentionSpec.scala](../gpt/src/test/scala/gpt/nn/AttentionSpec.scala) (18 testes).

- [x] Decidir como projetar `[B, T, dModel]` por `W_* : [dModel, dHead]` — resolvido pela rota 2:
      `matmul3D` virou `matmulBatched` e aceita `(3,2)`, com a matriz compartilhada pelo lote.
      `Linear` funciona com entrada rank 3 sem mudança nenhuma.
- [x] Projeções `Q = X*W_Q`, `K = X*W_K`, `V = X*W_V` — três `Linear(dModel, dHead)`
- [x] `scores = Q @ K.T / sqrt(dHead)`
- [x] Máscara causal triangular inferior, `-inf` nas posições futuras
- [x] `weights = softmax(scores + mask, dim=-1)` — atenção: `softmax` não aceita `-1`,
      o índice tem que ser `rank - 1` explícito
- [x] `output = weights @ V`
- [x] Conferir shapes em cada etapa: `Q,K,V [B,T,dHead]`, `scores/weights [B,T,T]`, `output [B,T,dHead]`
- [x] Gradient check completo do bloco (máscara aplicada ANTES do softmax)

## Testes de propriedade (além do gradient check)

Ver `theory/11-attention/11-attention.md` §8. Nenhum deles é coberto pelo gradient check,
que valida a coerência entre forward e backward — não se o forward é o pretendido.

- [x] **Causalidade:** mudar o último token não altera nenhuma saída anterior
- [x] **Contraprova da causalidade:** mudar o primeiro token *tem* que alterar a última saída
      (uma máscara que bloqueie tudo fora da diagonal passa no teste acima)
- [x] **Primeira posição:** `y₀ == v₀`, sem tolerância
- [x] **Soma das linhas de `P`:** todas valem `1`, incluindo as mascaradas
- [x] **Envelope convexo:** cada saída fica entre o mínimo e o máximo dos values permitidos
- [x] Um teste com entrada não contígua, e outro com `T ≠ dHead`
- [x] Comparação posição a posição contra uma referência do bloco inteiro em Scala puro
- [x] Regressão: as três projeções não compartilham a mesma `Linear`

## Achado: `bk` tem gradiente exatamente zero

Somar uma constante a todas as keys desloca a linha inteira de scores pelo mesmo valor
(`q_i · bk` não depende de `j`), e o softmax é invariante a isso. Então `dbk = 0` sempre.
Medido: `2.2e-16` contra `O(1)` nos outros cinco parâmetros.

Consequência prática: o gradient check **reprova** `bk`, porque a métrica é relativa com piso
`1e-8` no denominador — gradiente verdadeiro zero mais ruído de diferenças finitas (`~1e-11`)
dá erro relativo `~1e-3`. Enquanto o parâmetro existiu, o teste dele foi direto (`grad == 0`),
não por `Gradcheck`.

Com `bq` não acontece: deslocar as queries acrescenta `bq · k_j`, que varia com `j`.

**Resolvido em 2026-08-21, no item 1 da Etapa 12** (ver `HISTORY.md`): o `Linear` ganhou
`useBias: Boolean = true` e a `Attention` passou a construir `key` com `useBias = false`.
`parameters` caiu para cinco — `[Wq, bq, Wk, Wv, bv]` — e `bk` deixou de existir. Com isso o
laço de gradient check perdeu a exceção (roda nos cinco), e o teste de `grad == 0` foi
substituído por um que exige gradiente **não trivial** em `bq`, documentando a assimetria pelo
lado que sobrou. `bq` ficou de propósito: é viés funcional, não morto. `bv` também ficou aqui,
mas sai na Etapa 12 — com `W_O` em cena ele é exatamente redundante com `b_O`.

## Nota: a máscara não é recortável hoje

O roadmap fala em pré-computar a máscara para `contextLength`. Não existe operação de
recorte em `scalagrad` — `indexSelect` seleciona linhas de um tensor rank 2. Construir a
máscara `[T, T]` por `Tensor.make` a cada forward é mais simples, e ela não tem gradiente.
O `slice` deve nascer na Etapa 19, que precisa de `logits[-1]`.
