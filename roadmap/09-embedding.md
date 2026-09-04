## Etapa 9 — Camada de Embedding

### Por que existe

O modelo precisa converter cada token (um inteiro) em um vetor de números reais — a representação densa que as camadas vão processar. Uma abordagem ingênua seria one-hot encoding (vetor com 1 na posição do token e 0 em tudo mais) seguido de uma camada linear. A embedding table é equivalente a isso, mas muito mais eficiente: em vez de multiplicar um vetor esparso por uma matriz, simplesmente fazemos uma lookup — buscamos diretamente a linha correspondente ao token na tabela.

### O que implementar

**Token Embedding**

Uma tabela de shape `[vocabSize, dModel]`. Cada linha é o vetor de embedding de um token. Inicializar com `N(0, 0.02²)` — média zero, desvio padrão `0,02`.

O forward de `forward(tokens: Tensor)` recebe o `[batchSize, seqLen]` produzido pelo `BatchSampler` e retorna um tensor de shape `[batchSize, seqLen, dModel]`, onde cada token foi substituído pela sua linha correspondente na tabela.

O backward é específico: o gradiente que chega de volta à embedding table é **esparso** — só as linhas correspondentes aos tokens presentes no batch recebem gradiente. Acumular o gradiente nas linhas corretas com `+=`.

**Positional Embedding**

Transformers não têm noção intrínseca de ordem — o mecanismo de atenção trata a sequência como um conjunto (bag) de tokens. Para que o modelo possa usar a posição dos tokens, adicionamos embeddings posicionais.

Usaremos **positional embeddings aprendidos** (a abordagem do GPT): uma segunda tabela de shape `[contextLength, dModel]`, onde a linha `i` é o embedding da posição `i`. Para cada sequência, criamos um tensor de índices `[0, 1, 2, ..., seqLen-1]` e fazemos lookup nessa tabela.

O embedding final de cada token é a **soma** do token embedding com o positional embedding correspondente.

**Por que não sinusoidal?**

Positional embeddings sinusoidais (do paper "Attention is All You Need" original) são fixos e derivados por fórmula. Positional embeddings aprendidos (GPT) deixam o modelo descobrir a melhor representação de posição para a tarefa. Para um projeto didático, aprendidos são mais simples de implementar e funcionam bem dentro de `contextLength` fixo.

### Por que essa etapa importa

A embedding layer é a porta de entrada do modelo. Ela é também onde tokens e posições se fundem numa representação unificada que as camadas de atenção vão processar. Um erro aqui — como não somar o positional embedding, ou indexar a tabela errada — produz um modelo que tecnicamente funciona mas ignora a ordem dos tokens.
