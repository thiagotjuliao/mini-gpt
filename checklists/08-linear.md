# Etapa 8 — Camada Linear (Fully Connected)

Status: [ ] Não iniciado · [ ] Em andamento · [x] Concluído

- [x] Parâmetros `W [inputDim, outputDim]` e `b [outputDim]` com `requiresGrad = true`
- [x] Inicialização de `W` (Kaiming ou `N(0, 0.02²)`), `b` em zeros
- [x] Forward: `matmul(x, W) + b` (bias broadcastado)
- [x] `parameters(): List[Tensor]` retornando `[W, b]`
- [x] Gradient check
