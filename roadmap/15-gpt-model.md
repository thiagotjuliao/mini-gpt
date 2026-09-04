## Etapa 15 — Modelo GPT Completo

### Por que existe

Com todos os componentes prontos, montamos o modelo completo: a sequência de transformações que vai de tokens (inteiros) até logits (pontuações sobre o vocabulário).

### O que implementar

**Arquitetura**

```
tokens [B, T]
  → Token Embedding [B, T, dModel]
  + Positional Embedding [B, T, dModel]
  = x [B, T, dModel]
  → Bloco Transformer 1
  → Bloco Transformer 2
  → ...
  → Bloco Transformer N
  → LayerNorm final
  → Linear [dModel, vocabSize]  ← "language model head"
  = logits [B, T, vocabSize]
```

**Hiperparâmetros**

Os que definem a arquitetura:
- `nLayers`: número de blocos transformer (profundidade)
- `nHeads`: número de cabeças de atenção
- `dModel`: dimensão do embedding (deve ser divisível por nHeads)
- `contextLength`: comprimento máximo de sequência
- `vocabSize`: tamanho do vocabulário

Para um modelo pequenininho mas funcional: `nLayers=4`, `nHeads=4`, `dModel=128`, `contextLength=128`. Para algo com qualidade perceptível de texto: `nLayers=6`, `nHeads=6`, `dModel=384`.

**Contagem de parâmetros**

Implementar um método que calcula o número total de parâmetros. Útil para sanity check. Um modelo tiny com as configurações acima terá na ordem de 1-10M parâmetros.

**Weight Tying (opcional)**

Uma otimização comum: compartilhar os pesos da token embedding table com o language model head (transposta). Reduz parâmetros e empiricamente melhora a qualidade. Matematicamente faz sentido: os dois mapeiam entre o mesmo espaço de tokens e o espaço de embedding.

**Coleta de parâmetros**

Implementar `parameters()` que percorre todos os sub-módulos e retorna a lista plana de todos os tensores com `requiresGrad = true`. O otimizador receberá essa lista.

### Por que essa etapa importa

Essa é a etapa de integração. Se todas as partes foram implementadas e testadas corretamente, o modelo simplesmente funciona. Se não funciona aqui, é hora de isolar qual componente está errado — um dos benefícios de ter construído tudo de forma modular.
