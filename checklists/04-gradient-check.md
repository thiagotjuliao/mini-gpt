# Etapa 4 — Verificação Numérica de Gradientes (Gradient Check)

Status: [ ] Não iniciado · [ ] Em andamento · [x] Concluído

## Gradiente numérico
- [x] Implementar diferença central: `(f(p+ε) - f(p-ε)) / (2ε)` com `ε = 1e-5`

## Erro relativo
- [x] `|analítico - numérico| / max(|analítico|, |numérico|, 1e-8)`
- [x] Critério: `<1e-5` ok, `1e-5..1e-3` suspeito, `>1e-3` bug

## Função genérica
- [x] `gradCheck(f: Tensor => Tensor, input: Tensor)`:
  - [x] Forward + backward analítico
  - [x] Loop de perturbação por elemento do input
  - [x] Reporta erro relativo máximo

## Aplicação
- [x] Rodar em todas as ops da Etapa 3 com inputs aleatórios de shapes variadas
- [x] Reservar para rodar também em: `matmul`, `softmax`, `layerNorm`, `attention` quando chegar lá
