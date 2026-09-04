# Etapa 17 — Otimizador (AdamW)

Status: [ ] Não iniciado · [ ] Em andamento · [x] Concluído

## Teoria (antes do código)
- [x] `theory/17-adamw-optimizer/17-adamw-optimizer.md`
- [x] `theory/17-adamw-optimizer/adamw.html` (guia visual)
- [x] `theory/17-adamw-optimizer/exercises.html` (14 questões)
- [x] Termos novos no glossário do `00-overview`

## API de escrita no parâmetro (pré-requisito, no `scalagrad`)
- [x] `Tensor.updateData(values: Array[Double])` — `data` é `private[scalagrad]`, e sem isso o otimizador não compila de `gpt.optim`
- [x] `require(values.length == size)`
- [x] `require(isContiguous)` — `data` é físico, `gradient` é canônico (bug de 2026-08-21)
- [x] Teste: rejeita array de tamanho errado
- [x] Teste: rejeita view não contígua (`transpose` e `broadcastTo`)

## SGD (ponto de partida)
- [x] `p = p - lr * p.grad`

## Adam
- [x] Momentos `m` (1ª ordem) e `v` (2ª ordem) por parâmetro
- [x] Update com bias correction (`m_hat`, `v_hat`)
- [x] Hiperparâmetros padrão: `β1=0.9, β2=0.95, ε=1e-8, lr=3e-4`

## AdamW
- [x] Weight decay aplicado direto no parâmetro (não somado ao grad)
- [x] `λ = 0.1`

## Estado
- [x] Map `Tensor -> (m_array, v_array)` — chave por identidade de referência
- [x] Arrays dimensionados por `p.size`, nunca por `p.data.length`
- [x] Contador de passos `t`, começando em 1 no primeiro `step()`
- [x] `step()` devolve um `AdamW` novo com `t + 1` (estado imutável, sem `var`)
- [x] `zeroGrad()` via `parameters.foreach(_.gradient.zero())`

## Validação
- [x] Testar update manual de 1 passo com parâmetros conhecidos e conferir na mão
- [x] **Primeiro passo vale exatamente `lr`**, para gradientes de escalas bem diferentes — é o teste que pega a falta da correção de viés
- [x] Invariância de escala: gradientes ×1000 dão o mesmo passo
- [x] Momento absorve troca de sinal sem inverter o passo
- [x] Weight decay: com gradiente zero, `p` cai por `(1 − lr·λ)` a cada passo
- [x] Dois passos seguidos usam o estado do anterior (`t` e `m`/`v` propagados)
