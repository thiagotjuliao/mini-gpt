# Etapa 3 — Operações Elementares com Gradiente

Status: [ ] Não iniciado · [ ] Em andamento · [x] Concluído

## Element-wise
- [x] `add(a, b)` forward + backward
- [x] `sub(a, b)` forward + backward
- [x] `mul(a, b)` forward + backward
- [x] `div(a, b)` forward + backward
- [x] `pow(a, exp)` forward + backward
- [x] `neg(a)` forward + backward

## Funções transcendentais
- [x] `exp(a)` (guardar `c.data` para reuso no backward)
- [x] `log(a)` (clipping ou log-sum-exp para estabilidade)

## Reduções
- [x] `sum(a)` escalar + backward
- [x] `sum(a, dim)` + backward
- [x] `mean(a)` escalar + backward
- [x] `mean(a, dim)` + backward
- [x] `max(a)` + backward (gradiente só no índice do máximo)

## Broadcasting
- [x] `unbroadcast(grad, originalShape)` somando nas dimensões broadcastadas

## Transpose e Reshape
- [x] `transpose(a)` com backward (transpõe o gradiente de volta)
- [x] `reshape(a, newShape)` com backward (reshape do gradiente de volta)

## Matrix Multiplication
- [x] `matmul` 2D forward
- [x] `matmul` 2D backward: `dA = dC @ B.T`, `dB = A.T @ dC`
- [x] `matmul` em batch (3D) forward
- [x] `matmul` em batch backward (por fatia)

## Clamp / Clip
- [x] `clamp(a, min, max)` forward + backward (zero fora do intervalo)

## Validação
- [x] Gradient check em add/mul/matmul — superado pela suíte automática da Etapa 4: o `GradcheckSweepSpec` cobre add, sub, mul, div, pow, neg, exp, log, sum, mean, max, clamp e cinco variantes de matmul (2D, 3D em lote, matriz compartilhada, operando não contíguo e dois eixos de lote)
