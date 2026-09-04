# Etapa 15 — Modelo GPT Completo

Status: [ ] Não iniciado · [ ] Em andamento · [x] Concluído

Teoria em `theory/15-gpt-model/`: capítulo (9 seções), guia visual (4 figuras) e
15 exercícios — escritos antes da implementação, como manda a convenção da Etapa 4.
Implementação em [GPT.scala](../gpt/src/main/scala/gpt/model/GPT.scala),
suíte em [GPTSpec.scala](../gpt/src/test/scala/gpt/model/GPTSpec.scala) (16 testes).

- [x] `Embedding(tokens)` — token + posição já vêm somados da Etapa 9
- [x] Empilhar `nLayers` blocos com `foldLeft` — instâncias distintas
- [x] LayerNorm final (`ln_f`)
- [x] Cabeça de linguagem `Linear(dModel, vocabSize, useBias = false)` → logits
- [x] Hiperparâmetros: `vocabSize, dModel, nHeads, nLayers, contextLength, expansion`
- [x] `parameters` achatado e deduplicado (`distinct`)
- [x] Validações com `require`: `nLayers >= 1`, `tokens.rank == 2`, `T <= contextLength`
- [x] Forward: `[B, T]` → `[B, T, vocabSize]`
- [x] (Opcional) Weight tying — **decidido não implementar**; exigiria expor a tabela de token na `Embedding`, e é otimização de tamanho, não de correção
- [x] Escala residual `1/√(2·nLayers)` na inicialização de `W_O` e da segunda `Linear` do MLP — feito em 2026-09-01 (caminho 1 da §6), com a medição por profundidade no capítulo
- [x] `case class GPTConfig` — feito na Etapa 18 (2026-09-01), junto com o `Checkpoint` que a serializa

## O achado da etapa: a demonstração da §2 vale para **uma** camada

A mutação "remover o embedding posicional" passou em **16 de 16** na primeira rodada. O teste que
o capítulo chamava de "o que carrega a etapa" rodava com `nLayers = 2`, e nessa profundidade a
propriedade que ele testa simplesmente não existe.

Medido em Python puro, mesmo prefixo trocado:

| camadas de atenção | diferença em `t=2` |
|---|---|
| 1 | `0.000000` |
| 2 | `5.09e-2` |
| 3 | `6.43e-1` |

A causa é a própria máscara: a posição 0 vê um token, a 1 vê dois, a 2 vê três. Essa assimetria já
aparece na saída da primeira camada, e a segunda a lê. **Um modelo causal profundo consegue
reconstruir a ordem sem tabela posicional nenhuma** — o que a tabela dá é a informação direta,
desde a primeira camada.

Correções feitas: o teste passou a usar `nLayers = 1`, e a §2 do capítulo ganhou a ressalva com os
três números, junto com o cartão de referência, o guia visual e duas questões dos exercícios.

## Um segundo ajuste: o `ε` do gradient check

O check do modelo inteiro falhava por pouco no `ε = 1e-5` padrão (`1.86e-5` contra o teto de
`1e-5`). Não era bug: é a curva em U da Etapa 4 §1, com o mínimo deslocado pela profundidade da
composição. Medido, erro máximo entre 19 parâmetros:

| `ε` | erro máximo |
|---|---|
| `1e-4` | `1.7e-3` |
| `1e-5` | `2.4e-4` |
| `1e-6` | `2e-7` a `9e-7` (quatro modelos independentes) |
| `1e-7` | `4e-6` a `6e-6` |

A suíte usa `ε = 1e-6`, mantendo a tolerância em `1e-5`.

## Validação por mutação

| mutação | testes que falharam |
|---|---|
| cabeça com dimensões trocadas | 8 de 16 |
| `List.fill(nLayers)(umBlocoSó)` | 4 de 16 |
| esquecer o LayerNorm final | 2 de 16 |
| aplicar os blocos na ordem inversa | 2 de 16 |
| remover o embedding posicional | 1 de 16 (só com `nLayers = 1`) |

## A pendência, fechada em 2026-09-01

**Escala residual `1/√(2·nLayers)`** implementada pelo caminho 1 da §6: `Linear` ganhou `initScale`,
`MultiHeadAttention` e `MLP` repassam para a camada de saída, e o `GPT` calcula o fator a partir de
`nLayers`. Medido, desvio do fluxo residual em 8 camadas: `18.39` sem escala contra `4.53` com.

Fica aberta, agora documentada como escolha e não como esquecimento, a **inconsistência entre `0.02`
na `Embedding` e Kaiming na `Linear`** — o caminho 2 da §6. A medição mostrou que é ela que impede o
desvio de ficar constante com a profundidade, como a derivação da Etapa 14 §5 prevê.
