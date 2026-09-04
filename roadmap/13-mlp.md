## Etapa 13 — Feed-Forward Network (MLP)

### Por que existe

Após o mecanismo de atenção (que mistura informações entre posições), o MLP processa cada posição **independentemente**, aplicando uma transformação não-linear. É onde o modelo armazena "conhecimento" factual — estudos mostram que o MLP de transformers se comporta como memória associativa.

### O que implementar

**Estrutura**

Duas camadas lineares com uma ativação GELU entre elas:

```
x → Linear(dModel, 4*dModel) → GELU → Linear(4*dModel, dModel) → output
```

O fator de expansão 4x é uma convenção empírica do paper original e mantida no GPT-2/3. Significa que a camada intermediária tem 4 vezes mais neurônios que a dimensão do modelo — criando um "gargalo expandido" onde o modelo pode realizar computações complexas antes de comprimir de volta.

**Parâmetros**

- `W1`: `[dModel, 4*dModel]`, bias `b1`: `[4*dModel]`
- `W2`: `[4*dModel, dModel]`, bias `b2`: `[dModel]`

Total: aproximadamente `8 * dModel²` parâmetros por bloco.

**Por que GELU aqui?**

ReLU zera metade dos neurônios (os negativos) de forma abrupta. GELU faz isso suavemente, com derivada não-zero para valores ligeiramente negativos. Empiricamente, GELU produz modelos de linguagem melhores que ReLU — especialmente em modelos maiores.

### Por que essa etapa importa

O MLP compõe metade do custo computacional de cada bloco transformer (a outra metade é a atenção). Implementado sobre operações já testadas, deve funcionar corretamente se as partes anteriores estiverem corretas.
