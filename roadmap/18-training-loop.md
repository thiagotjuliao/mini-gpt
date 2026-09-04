## Etapa 18 — Loop de Treinamento

### Por que existe

O loop de treinamento orquestra todos os componentes: amostrar dados, fazer forward, calcular perda, fazer backward, atualizar pesos. É onde tudo se conecta e onde bugs arquiteturais se manifestam.

### O que implementar

**Loop principal**

Para cada passo de treinamento:
1. Amostrar um mini-batch `(inputs, targets)` do corpus
2. Forward: `logits = model(inputs)`
3. Calcular loss: `loss = crossEntropy(logits, targets)`
4. Zerar gradientes de todos os parâmetros
5. Backward: `loss.backward()`
6. Gradient clipping: `clipGradNorm(parameters, maxNorm=1.0)`
7. Update: `optimizer.step(parameters)`
8. Logar loss a cada N passos

**Gradient Clipping**

Mesmo com AdamW, gradientes ocasionalmente explodem (especialmente no início do treinamento). Gradient clipping limita a norma do gradiente global: se a norma total de todos os gradientes exceder `maxNorm`, todos os gradientes são reescalados proporcionalmente. A norma global é `sqrt(Σ grad[i]²)` sobre todos os parâmetros. Clipping em `1.0` é padrão para transformers.

**Learning Rate Schedule**

Não usar um learning rate fixo — o padrão para transformers é:
- Cosine decay: o lr começa em `lr_max`, desce suavemente (seguindo uma cosseno) até `lr_min` ao longo do treinamento.
- Opcional: warmup linear nos primeiros `warmupSteps` passos, subindo de 0 até `lr_max`.

Implementar como função `getLR(step, totalSteps, lrMax, lrMin, warmupSteps)`.

**Logging e Métricas**

A cada `logInterval` passos, logar: o passo atual, o learning rate atual, a loss do batch, e a perplexidade. A loss deve descer visivelmente nos primeiros passos — se não descer, há um bug.

**Checkpoint**

Implementar serialização dos parâmetros do modelo e estado do otimizador para disco. Sem isso, se o processo morrer, o treinamento começa do zero. Formato simples: escrever os arrays de `Double` com seus shapes em formato binário ou texto. Também implementar load.

**Avaliação no conjunto de validação**

Separar uma fração do corpus (ex: 10%) como validação. A cada `evalInterval` passos, calcular a loss no conjunto de validação com `noGrad` ativado. Essa é a métrica que importa — a loss de treino pode ser enganosa por overfitting.

### Por que essa etapa importa

Bugs no loop de treinamento são os mais difíceis de diagnosticar porque os sintomas (loss não desce, ou desce mas o modelo gera lixo) são ambíguos. Implementar com muita instrumentação no início. A ordem de operações importa: zerar gradientes *depois* de logar e *antes* de backward — nunca depois de backward.
