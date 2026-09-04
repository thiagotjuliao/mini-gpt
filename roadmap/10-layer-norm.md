## Etapa 10 — Layer Normalization

### Por que existe

Conforme o sinal passa por muitas camadas de um transformer, as ativações podem explodir ou encolher. Layer Normalization normaliza as ativações dentro de cada exemplo, estabilizando o treinamento e permitindo usar learning rates maiores. É uma das razões pelas quais transformers profundos treinam de forma estável.

### O que implementar

**Forward**

Para um tensor de shape `[batch, seqLen, dim]`, a normalização opera sobre a última dimensão (dim) independentemente para cada `(batch, seqLen)`:

1. Calcular a média `μ` ao longo de `dim`
2. Calcular a variância `σ²` ao longo de `dim`
3. Normalizar: `x_norm = (x - μ) / sqrt(σ² + ε)`, onde `ε = 1e-5` evita divisão por zero
4. Escalar e deslocar: `out = γ * x_norm + β`

`γ` (gamma) e `β` (beta) são parâmetros aprendidos de shape `[dim]`. Inicializar `γ = ones` e `β = zeros` (identidade no início).

**Backward**

O backward do LayerNorm é o mais matematicamente envolvido das operações elementares do transformer. Envolve a regra da cadeia através das operações de média, variância e normalização, que são interdependentes. Existem duas abordagens:

Opção A (composição de ops): implementar LayerNorm como composição das operações já implementadas (`sub`, `pow`, `mean`, `add`, `mul`). O autograd cuida do backward automaticamente. Mais simples de implementar, mais lento por envolver muitas operações intermediárias.

Opção B (backward manual): derivar e implementar o backward analiticamente de uma vez. Significativamente mais rápido. A fórmula completa aparece em papers e implementações de referência — vale derivar para entender, depois implementar a forma compacta.

A Opção A é recomendada para começar, e pode ser otimizada depois.

**Por que LayerNorm e não BatchNorm?**

BatchNorm normaliza ao longo do batch — calcula estatísticas usando múltiplos exemplos simultâneos. Isso cria dependência entre exemplos num batch, complica a inferência (que às vezes processa um exemplo por vez), e tem problemas com sequências de comprimento variável. LayerNorm normaliza dentro de cada exemplo individualmente, o que não tem nenhum desses problemas.

### Por que essa etapa importa

Sem LayerNorm, transformers profundos geralmente não convergem. As ativações explodem ou encolhem à medida que passam pelas camadas. É uma das inovações que tornaram o treinamento de modelos grandes estável.
