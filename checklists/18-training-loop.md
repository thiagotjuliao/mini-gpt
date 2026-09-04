# Etapa 18 — Loop de Treinamento

Status: [ ] Não iniciado · [ ] Em andamento · [x] Concluído

## Teoria
- [x] `theory/18-training-loop/18-training-loop.md`
- [x] `theory/18-training-loop/training-loop.html` (guia visual)
- [x] `theory/18-training-loop/exercises.html` (14 questões)
- [x] Termos novos no glossário do `00-overview`

## Loop principal
- [x] Amostrar mini-batch (inputs, targets)
- [x] Forward: `logits = model(inputs)`
- [x] `loss = crossEntropy(logits, targets)`
- [x] Zerar gradientes de todos os parâmetros
- [x] `loss.backward()`
- [x] Gradient clipping (norma global, `maxNorm=1.0`)
- [x] `optimizer.step(parameters)`
- [x] Logar loss a cada N passos

## Learning Rate Schedule
- [x] `LRSchedule.cosine(step, totalSteps, lrMax, lrMin, warmupSteps)` — cosine decay + warmup linear (nome trocado de `getLR` para dizer qual decaimento é)

## Logging
- [x] A cada `logInterval`: passo, lr atual, loss, perplexidade
- [x] Confirmar que a loss desce visivelmente nos primeiros passos

## Checkpoint
- [x] Serializar parâmetros do modelo + estado do otimizador em disco
- [x] Load a partir do checkpoint

## Avaliação
- [x] Split de validação (~10% do corpus) — `BatchSampler.split`, corte em ordem
- [x] Loss de validação a cada `evalInterval`, com `noGrad` ativado

**MILESTONE: "o modelo treina"**

## Ordem crítica a não errar
- [x] Zerar gradientes DEPOIS de logar e ANTES de backward — nunca depois de backward
