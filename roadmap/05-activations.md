## Etapa 5 — Funções de Ativação

### Por que existe

Sem funções de ativação, uma rede neural com múltiplas camadas é equivalente a uma única transformação linear. As ativações introduzem não-linearidade, que é o que permite ao modelo aprender padrões complexos. Cada função de ativação tem características diferentes que afetam o fluxo de gradientes e a representação aprendida.

### O que implementar

**ReLU (Rectified Linear Unit)**

Forward: `relu(x) = max(0, x)`. Simples, eficiente. Backward: o gradiente passa onde `x > 0`, e é zero onde `x <= 0`. A região `x <= 0` é chamada de "neurônio morto" — uma vez que um neurônio fica preso lá, seus gradientes são sempre zero e ele nunca mais aprende. Implementar como operação sobre tensores element-wise.

**GELU (Gaussian Error Linear Unit)**

Forward: `gelu(x) = x * Φ(x)`, onde `Φ` é a CDF da distribuição normal padrão. A aproximação prática usada em GPT é: `gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.

GELU é preferida ao ReLU em transformers porque não mata neurônios de forma brusca — ela atenua suavemente valores negativos em vez de zerá-los, o que empiricamente melhora o treinamento de modelos de linguagem.

O backward de GELU é mais complexo — envolve a derivada da tanh e do polinômio interno. Deve ser derivado analiticamente da fórmula de aproximação e verificado com gradient check.

**Sigmoid**

Forward: `σ(x) = 1 / (1 + e^(-x))`. Backward: `σ(x) * (1 - σ(x))`. Útil para entender gates e para referência histórica, mas não usada no transformer principal.

**Tanh**

Forward: `tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`. Backward: `1 - tanh(x)²`. Usada internamente na aproximação de GELU.

### Por que essa etapa importa

GELU é a função de ativação do bloco feed-forward do GPT. Implementá-la corretamente — e verificar seu backward — é necessário antes de montar o MLP.
