## Etapa 6 — Softmax e Log-Softmax

### Por que existe

Softmax converte um vetor de valores reais arbitrários (logits) em uma distribuição de probabilidade que soma 1. É usada em dois lugares críticos: dentro do mecanismo de atenção (para transformar scores em pesos que somam 1) e na saída do modelo (para obter probabilidades sobre o vocabulário).

### O que implementar

**Softmax com estabilidade numérica**

A fórmula direta `softmax(x)[i] = e^x[i] / Σ e^x[j]` tem problema numérico: para valores grandes de `x`, `e^x` estoura para infinito. A solução é subtrair o máximo antes de exponenciar:

```
softmax(x)[i] = e^(x[i] - max(x)) / Σ e^(x[j] - max(x))
```

Matematicamente equivalente (o máximo cancela em numerador e denominador), mas numericamente estável. Sempre usar essa forma.

Para tensores ND, a softmax deve operar ao longo de uma dimensão específica (geralmente a última). Implementar `softmax(a, dim)`.

**Backward do Softmax**

O backward do softmax é matematicamente mais rico que parece. A saída `s = softmax(x)` é um vetor onde cada elemento depende de todos os elementos da entrada. Por isso, o Jacobiano `ds/dx` é uma matriz densa.

A fórmula compacta para o backward é: dado o gradiente `dOut`, o gradiente em relação a `x` é:

```
dx = s * (dOut - (dOut * s).sum())
```

onde `(dOut * s).sum()` é o produto interno entre `dOut` e `s`. Essa simplificação vem da estrutura especial do Jacobiano do softmax e economiza toda a computação matricial explícita. Vale derivar essa fórmula do zero para entender.

**Log-Softmax**

`log_softmax(x) = x - log(Σ e^x)` — versão numericamente estável usando o truque do log-sum-exp. Usado em conjunto com NLL Loss para implementar cross-entropy de forma estável.

### Por que essa etapa importa

Softmax aparece duas vezes no transformer: na atenção e na loss. Um backward errado aqui afeta todo o aprendizado. Verificar obrigatoriamente com gradient check, especialmente para casos onde os logits têm valores muito diferentes entre si.
