## Etapa 14 — Bloco Transformer Completo

### Por que existe

O bloco transformer é a unidade que se repete N vezes no modelo. Combina atenção e MLP com conexões residuais e normalização, numa arquitetura que permite treinar modelos muito profundos de forma estável.

### O que implementar

**Conexões residuais (Residual / Skip Connections)**

A saída de cada sub-camada é somada à entrada antes de passar para a próxima:

```
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

O motivo é o problema do gradiente que desaparece: em redes muito profundas, o gradiente multiplica um Jacobiano em cada camada, e esses Jacobianos tendem a ter autovalores menores que 1, fazendo o gradiente encolher exponencialmente. Com conexões residuais, existe um caminho direto pelo qual o gradiente flui sem passar por nenhum Jacobiano — a "rodovia dos gradientes".

**Pre-LayerNorm vs Post-LayerNorm**

O paper original usava Post-LayerNorm (normaliza depois da soma residual). GPT-2 e posteriores usam Pre-LayerNorm (normaliza antes de entrar em atenção ou MLP, como mostrado acima). Pre-LN é mais estável durante o treinamento e permite não usar warm-up tão agressivo.

**Estrutura completa de um bloco**

```
input x  →
  ├→ LayerNorm → MultiHeadAttention → (+) → soma1
  │                                    ↑
  └────────────────────────────────────┘
soma1 →
  ├→ LayerNorm → MLP → (+) → output
  │                    ↑
  └────────────────────┘
```

**Parâmetros do bloco**

- 2 LayerNorms (cada um com gamma e beta de shape `[dModel]`)
- MultiHeadAttention (4 matrizes de `dModel x dModel`)
- MLP (W1, b1, W2, b2)

**Dropout (opcional para o projeto didático)**

Dropout zerava aleatoriamente uma fração dos ativações durante o treinamento, forçando a rede a aprender representações redundantes. É regularização. Se implementado, precisa de um modo de inferência onde não faz nada. Para manter o escopo didático, pode ser omitido ou implementado como um pós-passo simples.

### Por que essa etapa importa

O bloco é a unidade repetível do modelo. Uma vez implementado e testado, o modelo completo é literalmente uma lista desses blocos. A elegância do transformer está em como essas duas ideias simples (atenção + MLP) combinadas com residuais e normalização produzem modelos de capacidade escalonável.
