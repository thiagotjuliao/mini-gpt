## Etapa 3 — Operações Elementares com Gradiente

### Por que existe

Com a infraestrutura do autograd pronta, precisamos implementar as operações matemáticas concretas que o transformer usa. Cada operação tem duas partes obrigatórias: o **forward** (como computar o resultado) e o **backward** (como propagar o gradiente). O backward de cada operação é simplesmente a aplicação da regra da cadeia do cálculo.

### O que implementar

**Operações element-wise**

Estas operam elemento por elemento entre dois tensores de mesma shape (ou shapes compatíveis via broadcasting):

- `add(a, b)`: forward é `c[i] = a[i] + b[i]`. Backward: `a.grad[i] += c.grad[i]`, `b.grad[i] += c.grad[i]`. A derivada de uma soma em relação a qualquer operando é 1.

- `sub(a, b)`: similar ao add, mas `b.grad[i] -= c.grad[i]`.

- `mul(a, b)`: forward `c[i] = a[i] * b[i]`. Backward: `a.grad[i] += b.data[i] * c.grad[i]`, `b.grad[i] += a.data[i] * c.grad[i]`. A derivada do produto em relação a um fator é o outro fator.

- `div(a, b)`: forward `c[i] = a[i] / b[i]`. Backward requer cuidado: `a.grad[i] += c.grad[i] / b.data[i]`, `b.grad[i] -= c.grad[i] * a.data[i] / (b.data[i]^2)`.

- `pow(a, exp)`: onde `exp` é um escalar constante. Forward `c[i] = a[i]^exp`. Backward: `a.grad[i] += exp * a.data[i]^(exp-1) * c.grad[i]`.

- `neg(a)`: negação. Backward: `a.grad[i] -= c.grad[i]`.

**Funções transcendentais**

- `exp(a)`: `c[i] = e^a[i]`. Backward: `a.grad[i] += c.data[i] * c.grad[i]`. Note que a derivada de `e^x` é `e^x` — ou seja, o próprio valor forward é reusado no backward. Por isso precisamos guardar `c.data`.

- `log(a)`: `c[i] = ln(a[i])`. Backward: `a.grad[i] += c.grad[i] / a.data[i]`. Atenção: `a.data[i]` deve ser positivo. Em prática, fazemos clipping ou usamos log-sum-exp para estabilidade numérica.

**Operações de redução**

- `sum(a)`: soma todos os elementos, resultado é um escalar. Backward: o gradiente do escalar (um número) se distribui igualmente para todos os elementos: `a.grad[i] += c.grad` para todo `i`. Variante: `sum(a, dim)` que soma ao longo de uma dimensão específica e reduz o tensor em uma dimensão.

- `mean(a)`: média de todos os elementos. Forward é `sum(a) / size`. Backward: `a.grad[i] += c.grad / a.size`. Variante por dimensão também necessária.

- `max(a)`: o valor máximo. Backward: o gradiente flui apenas para o índice onde o máximo foi atingido, zero para os demais. Em caso de empate, convencionalmente o gradiente vai para o primeiro.

**Broadcasting e seus gradientes**

Quando operamos dois tensores de shapes diferentes mas compatíveis (ex: adicionar um vetor `[4]` a uma matriz `[3, 4]`), o tensor menor é "expandido" implicitamente. No backward, como o tensor menor foi usado múltiplas vezes (uma vez por linha da matriz), seus gradientes devem ser somados ao longo das dimensões que foram broadcastadas. Implementar `unbroadcast(grad, originalShape)` que faz esse `sum` nas dimensões corretas.

**Transpose e Reshape**

- `transpose(a)`: para matrizes 2D, troca shape e strides. No backward, o gradiente também é transposto antes de ser acumulado.

- `reshape(a, newShape)`: forward é trivial (reusa os dados). No backward, o gradiente recebido (que tem `newShape`) precisa ser reshaped de volta para `a.shape` antes de acumular.

**Matrix Multiplication (`matmul`)**

Esta é a operação mais importante e a mais usada no transformer. Para matrizes `A [M, K]` e `B [K, N]`:

Forward: `C[i, j] = Σ_k A[i, k] * B[k, j]`

Backward: Dado `dC` (gradiente de C), as regras são:
- `dA = dC @ B.T` — gradiente em relação a A é `dC` multiplicado pela transposta de B
- `dB = A.T @ dC` — gradiente em relação a B é a transposta de A multiplicada por `dC`

Essas fórmulas derivam diretamente da diferenciação matricial e são fundamentais. Vale derivá-las no papel antes de implementar para ter intuição do que significam.

Para batches de matrizes (tensores 3D `[B, M, K]`), o matmul opera em batch: cada "fatia" ao longo da dimensão B é um matmul independente. O backward ainda segue as mesmas regras, mas aplicadas por fatia e com soma ao longo do batch.

**Clamp / Clip**

`clamp(a, min, max)`: limita valores ao intervalo `[min, max]`. O backward passa o gradiente onde `min < a < max`, e zero nos limites (a função não é diferenciável nos limites, mas zero é a convenção padrão).

### Por que essa etapa importa

Todas as camadas do transformer são composições dessas operações primitivas. O grafo de computação de um forward pass de um transformer completo é uma árvore gigante dessas operações encadeadas. Se qualquer uma delas tiver o backward errado, o treinamento vai divergir de formas que são difíceis de debugar sem a próxima etapa.
