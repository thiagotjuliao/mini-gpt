## Etapa 17 — Otimizador (AdamW)

### Por que existe

O otimizador define como os parâmetros são atualizados depois que os gradientes são calculados. SGD simples (mover na direção negativa do gradiente) funciona, mas é sensível ao learning rate e converge lentamente para transformers. Adam e sua variante AdamW são o padrão para modelos de linguagem.

### O que implementar

**SGD (ponto de partida)**

`p = p - lr * p.grad` para cada parâmetro `p`. Implementar primeiro para entender e testar o loop de treinamento básico. Funcional, mas necessitará de learning rate muito baixo.

**Adam**

Adam mantém dois momentos por parâmetro:
- `m`: média exponencial dos gradientes (momento de 1ª ordem)
- `v`: média exponencial dos gradientes ao quadrado (momento de 2ª ordem)

A cada passo `t`:
```
m = β1 * m + (1 - β1) * grad
v = β2 * v + (1 - β2) * grad²
m_hat = m / (1 - β1^t)   ← correção de bias
v_hat = v / (1 - β2^t)   ← correção de bias
p = p - lr * m_hat / (sqrt(v_hat) + ε)
```

Hiperparâmetros padrão: `β1=0.9`, `β2=0.95`, `ε=1e-8`, `lr=3e-4`.

A divisão por `sqrt(v_hat)` adapta o learning rate por parâmetro: parâmetros com gradientes grandes recebem atualizações menores, e vice-versa. A correção de bias é necessária nos primeiros passos, quando `m` e `v` ainda estão sendo "aquecidos" a partir de zero.

**AdamW (Weight Decay correto)**

L2 regularization em Adam é implementada incorretamente como `grad += λ * p` (que passa pelo adaptador e é amortecida por `v`). AdamW implementa weight decay diretamente: `p = (1 - lr * λ) * p - lr * m_hat / (sqrt(v_hat) + ε)`. Isso aplica o decaimento de peso antes da atualização, sem interação com o adaptador. É o padrão para transformers. Usar `λ = 0.1`.

**Manutenção de estado**

O otimizador precisa manter `m` e `v` para cada parâmetro ao longo de todo o treinamento. Guardar um mapa de `Tensor -> (m_array, v_array)`. Também manter o contador de passos `t` global.

### Por que essa etapa importa

Com SGD, treinar um transformer requer learning rates muito pequenos e muitos passos. AdamW converge mais rápido e de forma mais estável — permite learning rates maiores sem divergir. A diferença entre Adam e AdamW para regularização parece sutil mas tem impacto real na qualidade do modelo treinado.
