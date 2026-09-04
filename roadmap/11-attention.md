## Etapa 11 — Scaled Dot-Product Attention

### Por que existe

Atenção é o mecanismo que permite ao modelo, ao processar um token, "olhar" para outros tokens da sequência e decidir quais são relevantes. Diferente de redes convolucionais (que têm janela fixa) ou RNNs (que têm estado sequencial), a atenção opera sobre todos os pares de tokens simultaneamente — e o que determina a relevância é aprendido durante o treinamento.

### O que implementar

**Queries, Keys e Values**

O mecanismo de atenção começa com três transformações lineares da entrada `X`:
- `Q = X * W_Q`: as queries — "o que estou procurando"
- `K = X * W_K`: as keys — "o que eu ofereço"
- `V = X * W_V`: os values — "o que eu retorno se for relevante"

`W_Q`, `W_K`, `W_V` são pesos da camada linear com shape `[dModel, dHead]`.

**Attention Scores**

`scores = Q @ K.T / sqrt(dHead)` — produto interno entre cada query e cada key, normalizado.

O produto interno mede similaridade: queries e keys alinhadas produzem score alto. A divisão por `sqrt(dHead)` é crítica: sem ela, para dimensões grandes, os produtos internos ficam tão grandes em magnitude que o softmax satura (gradientes próximos de zero), tornando o treinamento lento.

**Máscara Causal**

Em modelos autogressivos (que prevêm o próximo token), cada token só pode "ver" tokens anteriores — nunca o futuro. Implementar uma máscara triangular inferior: preencher com `-inf` todos os scores correspondentes a pares onde a key está numa posição futura em relação à query. Após o softmax, `-inf` vira 0, efetivamente ignorando esses pares.

A máscara tem shape `[seqLen, seqLen]` e pode ser pré-computada uma vez para o `contextLength` máximo.

**Attention Weights e Output**

`weights = softmax(scores + mask, dim=-1)` — distribuição de probabilidade sobre as posições.

`output = weights @ V` — soma ponderada dos values pelos attention weights. Cada token recebe um vetor de saída que é a mistura ponderada de todos os values, onde os pesos são determinados pela relevância.

**Shape tracking ao longo do forward**

Para um batch de `B` exemplos com sequências de `T` tokens e dimensão `dHead`:
- `Q, K, V`: `[B, T, dHead]`
- `scores`: `[B, T, T]`
- `weights`: `[B, T, T]`
- `output`: `[B, T, dHead]`

### Por que essa etapa importa

Atenção é o mecanismo diferenciador dos transformers. Entender completamente o forward — e verificar o backward com gradient check — é o pré-requisito para tudo que vem depois. Muitos bugs sutis aparecem aqui: máscara aplicada depois do softmax em vez de antes, divisão por `sqrt(d)` esquecida, dimensions trocadas no matmul.
