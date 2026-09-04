# Etapa 6 — Softmax e Log-Softmax

Status: [ ] Não iniciado · [ ] Em andamento · [x] Concluído

- [x] `softmax(a, dim)` numericamente estável (subtrair max antes de exponenciar)
- [x] Backward do softmax: `dx = s * (dOut - (dOut * s).sum())`
- [x] `log_softmax(a, dim)` via log-sum-exp
- [x] Gradient check com logits de magnitudes bem diferentes entre si

**MILESTONE: "o autograd está completo"** — depois desta etapa, todas as primitivas necessárias para o transformer têm forward+backward testados.
