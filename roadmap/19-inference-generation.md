## Etapa 19 — Inferência e Geração de Texto

### Por que existe

Um modelo treinado que não gera texto não demonstra nada. A geração é o momento onde vemos o que o modelo aprendeu — e também onde calibramos a qualidade subjetivamente.

### O que implementar

**Modo de inferência**

Ativar `noGrad` globalmente. Não construir o grafo de computação, não acumular gradientes. O modelo rodará mais rápido e com menos memória.

**Loop de geração autoregressiva**

Começar com um prompt (sequência de tokens inicial). Repetir:
1. Se a sequência de contexto exceder `contextLength`, cortar os primeiros tokens (sliding window)
2. Forward: `logits = model(context)`
3. Pegar apenas os logits da última posição: `logits[-1]` — shape `[vocabSize]`
4. Amostrar o próximo token segundo a estratégia escolhida
5. Concatenar o token gerado ao contexto
6. Repetir até gerar o número desejado de tokens

**Greedy Decoding**

O token mais provável a cada passo: `nextToken = argmax(logits)`. Determinístico, mas tende a gerar textos repetitivos.

**Temperature Sampling**

Dividir os logits por uma temperatura `T` antes do softmax: `logits = logits / T`. Com `T < 1`, a distribuição fica mais concentrada (mais determinística). Com `T > 1`, fica mais espalhada (mais aleatória). `T = 1.0` é sem modificação. Amostrar um token da distribuição resultante usando o método da inversa da CDF (ou equivalente).

Para amostrar de uma distribuição de probabilidade `p` sobre `vocabSize` opções: gerar um número uniforme `u ~ U[0, 1]`, calcular a CDF acumulada de `p`, e retornar o primeiro índice onde a CDF excede `u`.

**Top-k Sampling**

Antes de amostrar, zeramos as probabilidades de todos os tokens exceto os `k` mais prováveis, e reamostrar apenas entre eles. Evita gerar tokens muito improváveis (erros), mantendo alguma variedade. `k=40` é um valor comum.

**Decodificação**

Converter os tokens gerados de volta para texto com `decode()`. Imprimir progressivamente conforme gera — mais responsivo e didático.

### Por que essa etapa importa

A geração é o produto final e o teste de fumaça definitivo. Um modelo que gera texto com alguma coerência — mesmo que imperfeita — confirma que toda a cadeia (tokenizador, embeddings, atenção, treinamento) funcionou. A temperatura e top-k mostram concretamente o trade-off entre coerência e criatividade que todo sistema de linguagem enfrenta.
