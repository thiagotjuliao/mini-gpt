## Etapa 8 — Camada Linear (Fully Connected)

### Por que existe

A camada linear é a operação parametrizada mais básica: aplica uma transformação afim `y = xW + b`. É o bloco fundamental do mecanismo de atenção (as projeções Q, K, V) e do feed-forward network.

### O que implementar

**Parâmetros**

Dois tensores com `requiresGrad = true`: o peso `W` de shape `[inputDim, outputDim]` e o bias `b` de shape `[outputDim]`. São os parâmetros que o otimizador vai atualizar.

**Inicialização dos pesos**

Não inicializar com zeros — todos os neurônios seriam idênticos e aprenderiam a mesma coisa (problema da simetria). Não inicializar com valores muito grandes — gradientes explodem. Não muito pequenos — gradientes somem.

A inicialização de Kaiming (He): `W ~ N(0, 2/inputDim)` — variância `2/inputDim`, ou seja desvio padrão `sqrt(2/inputDim)`. Foi derivada para manter a variância dos ativações estável ao longo das camadas. Para transformer com GELU, uma variante comum é `N(0, 0.02²)` (a usada no GPT-2 original).

O bias pode ser inicializado com zeros.

**Forward**

Para input de shape `[batch, inputDim]`, o forward é `matmul(x, W) + b`. O bias é broadcastado ao longo da dimensão batch automaticamente.

**Backward**

O backward é tratado automaticamente pelo autograd através das operações `matmul` e `add` já implementadas. A camada linear não precisa de backward manual — ela apenas compõe operações primitivas.

**Coleção de parâmetros**

Implementar `parameters(): List[Tensor]` que retorna `[W, b]`. Será usado pelo otimizador para encontrar todos os tensores que precisam de update.

### Por que essa etapa importa

Toda projeção no transformer — W_Q, W_K, W_V, W_O, as duas camadas do MLP — é uma camada linear. Uma implementação correta e eficiente aqui é a base de tudo.
