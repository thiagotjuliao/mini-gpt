## Etapa 16 — Função de Perda (Cross-Entropy Loss)

### Por que existe

Precisamos de uma métrica escalar que meça o quão bem o modelo está prevendo o próximo token. Essa métrica é o que o autograd vai diferenciar para guiar o treinamento. Cross-entropy é a escolha natural para tarefas de classificação — e prever o próximo token é essencialmente classificação sobre o vocabulário.

### O que implementar

**Cross-Entropy**

Dados os logits `[B, T, vocabSize]` e os targets `[B, T]` (os índices dos tokens corretos), a perda é:

```
loss = mean over all (b, t) of: -log(softmax(logits[b, t])[targets[b, t]])
```

Ou equivalentemente, usando log-softmax:

```
loss = mean(-log_softmax(logits)[target_indices])
```

**Implementação numericamente estável**

Evitar calcular softmax e depois log separadamente — isso pode underflow para valores muito negativos. Usar log-sum-exp:

```
log_softmax(x)[i] = x[i] - log(Σ e^(x[j] - max(x))) - max(x)
```

**Perplexidade**

A perplexidade é `exp(loss)` — uma métrica mais interpretável para modelos de linguagem. Uma perplexidade de `P` significa que o modelo é equivalente a escolher uniformemente entre `P` tokens em cada posição. Perplexidade = vocabSize seria pior caso (modelo aleatório). O objetivo é minimizá-la.

### Por que essa etapa importa

A função de perda é o sinal de treinamento. Um erro aqui — como calcular a perda sobre os tokens de entrada em vez dos targets, ou não fazer a média corretamente — produz treinamento que parece funcionar mas o modelo não aprende nada.
