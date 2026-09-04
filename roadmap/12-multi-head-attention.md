## Etapa 12 — Multi-Head Attention

### Por que existe

Uma única cabeça de atenção aprende um tipo de relação entre tokens. Multi-head attention roda várias atenções em paralelo sobre subconjuntos do espaço de embedding — cada cabeça pode aprender relações diferentes (sintáticas, semânticas, posicionais, etc.). Os resultados são concatenados e projetados de volta para a dimensão original.

### O que implementar

**Divisão em cabeças**

Com `nHeads` cabeças e `dModel` dimensão total, cada cabeça opera sobre um subespaço de dimensão `dHead = dModel / nHeads`. Importante: `dModel` deve ser divisível por `nHeads`.

As projeções Q, K, V são feitas para `dModel` inteiro (com `W_Q`, `W_K`, `W_V` de shape `[dModel, dModel]`), e depois as saídas são *reshaped* para `[B, T, nHeads, dHead]` e *transpostas* para `[B, nHeads, T, dHead]`. Cada head é processada independentemente ao longo da dimensão `nHeads`.

**Attention por cabeça**

Com o reshape, o scaled dot-product attention pode ser aplicado em batch sobre todas as cabeças simultaneamente. Os shapes ficam:
- `Q, K, V`: `[B, nHeads, T, dHead]`
- `scores`: `[B, nHeads, T, T]`
- `output`: `[B, nHeads, T, dHead]`

**Concatenação e projeção final**

Após a atenção, transpor de volta para `[B, T, nHeads, dHead]` e reshape para `[B, T, dModel]` — isso concatena os resultados de todas as cabeças. Aplicar uma projeção final `W_O` de shape `[dModel, dModel]` sobre o resultado.

**Lista completa de parâmetros do MHA**

- `W_Q`: `[dModel, dModel]`
- `W_K`: `[dModel, dModel]`
- `W_V`: `[dModel, dModel]`
- `W_O`: `[dModel, dModel]`
- Biases correspondentes (opcionais, GPT-2 os inclui)

Total: aproximadamente `4 * dModel²` parâmetros por camada de atenção.

### Por que essa etapa importa

Multi-head attention é a versão que todos os transformers reais usam. A complexidade está principalmente no reshape/transpose para organizar as cabeças e no tracking correto dos shapes. Um shape errado aqui geralmente produz erro explícito — o que é melhor que um resultado silenciosamente errado.
