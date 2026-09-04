# Etapa 13 — Feed-Forward Network (MLP)

Status: [ ] Não iniciado · [ ] Em andamento · [x] Concluído

Teoria em `theory/13-mlp/`: capítulo (9 seções), guia visual (5 figuras) e
15 exercícios — escritos antes da implementação, como manda a convenção da Etapa 4.
Implementação em [MLP.scala](../gpt/src/main/scala/gpt/nn/MLP.scala),
suíte em [MLPSpec.scala](../gpt/src/test/scala/gpt/nn/MLPSpec.scala) (18 testes).

- [x] Estrutura: `Linear(dModel, 4*dModel) → GELU → Linear(4*dModel, dModel)`
- [x] Parâmetros `W1,b1` e `W2,b2` — `Linear` reusada, nada de `matmul` reimplementado
- [x] Validações com `require`: `x.rank >= 2`, `x.shape.last == dModel` e `expansion >= 1`
- [x] Contar parâmetros — exatos `8·dModel² + 5·dModel`, o dobro do bloco de atenção
- [x] Gradient check do bloco completo — em `x`, nos 4 parâmetros, e com entrada não contígua
- [x] Independência entre posições: token repetido dá linha idêntica, e permutar a entrada permuta a saída
- [x] Contraprova de linearidade: `f(2x) − 2·f(x) + f(0) ≠ 0` — o único teste que reprova um MLP sem a GELU
- [x] Comparação de `db1` contra a fórmula — o único teste que reprova o viés somado depois da GELU

## Pré-requisitos resolvidos junto com a camada

- **`require(inputDim >= 1)` e `require(outputDim >= 1)` na `Linear`.** Dimensão não positiva
      criava tensores vazios em silêncio. A checagem mora na classe que cria os tensores, uma vez.
- **`weights`, `bias` e `rng` injetável na `Linear`** (escritos pelo usuário): acesso por nome em
      vez de índice posicional, e inicialização reproduzível. Cobertos por 4 testes novos no `LinearSpec`.

## Dois achados da implementação

**`val dFF` declarado depois dos dois `Linear` que o usam.** O corpo da classe inicializa de cima
para baixo, então `dFF` valia `0` na construção, e as duas matrizes nasciam com zero coluna.
O compilador não emite aviso nenhum. Com o `require` novo na `Linear`, o mesmo código passou a
falhar alto: 16 dos 17 testes acusaram `Output dimension must be at least 1, but got 0`.

**Somar `b1` depois da GELU é invisível para o forward.** O viés nasce zerado, e zero é neutro nas
duas ordens — as duas versões são a *mesma função* numa camada recém-construída. `data` é
`private[scalagrad]`, então a suíte também não tem como preencher o viés por fora. Quem separa é o
backward: `db1 = Σ dA ⊙ gelu'(Z)` contra `Σ dA`. A diferença está na regra da cadeia, não no valor
do viés. O capítulo previa que a referência em Scala puro pegaria; a medição mostrou que não, e a
§8 foi corrigida.

## Validação por mutação

Seis mutações, cada uma revertida em seguida:

| mutação | testes que falharam |
|---|---|
| remover a `gelu` | 3 de 18 |
| `gelu → Linear → Linear` (ativação na entrada) | 3 de 18 |
| `gelu` depois da segunda `Linear` | 3 de 18 |
| `dFF = dModel` em vez de `4·dModel` | 3 de 18 |
| somar `b1` depois da `gelu` | 1 de 18 |
| `W2ᵀ` no lugar de `W2` | 12 de 18 |

Na primeira rodada, a linha do `b1` falhou em **zero** testes. O teste de gradiente foi escrito
por causa dessa medição.

## Pendência aberta

**Inicialização.** O fator 2 do Kaiming compensa um corte já ocorrido, então cabe na segunda
`Linear` (cuja entrada veio da GELU), não na primeira. Medido: `Var(x)=1 → Var(Z)=2 → Var(Y)=1.84`.
Decisão fica para a Etapa 15, junto com o `1/√(2·nLayers)` das camadas residuais — mesma pendência
que a Etapa 12 registrou para as projeções da MHA.
