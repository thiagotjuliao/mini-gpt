# Etapa 10 — Layer Normalization

Status: [ ] Não iniciado · [ ] Em andamento · [x] Concluído

Teoria em `theory/10-layer-norm/`: capítulo, guia visual e 12 exercícios.
Implementação por composição (Opção A), com 16 testes em `LayerNormSpec`.

- [x] Forward: média e variância ao longo da última dimensão
- [x] Normalizar: `(x - μ) / sqrt(σ² + ε)`, `ε = 1e-5`
- [x] Escalar e deslocar: `γ * x_norm + β`
- [x] `γ` init `ones`, `β` init `zeros`
- [x] Backward — começar pela Opção A (composição de ops já existentes)
- [ ] (Opcional depois) Opção B: backward manual otimizado — fórmula já derivada
      e conferida em `theory/10-layer-norm/10-layer-norm.md` §5
- [x] Gradient check (em `x`, `γ` e `β`, com perda ponderada — ver nota abaixo)

## Nota de teste

A perda do gradient check **não pode** ser `forward(x).sum`: com `γ = 1` e `β = 0`
essa soma é `Σ x̂`, identicamente zero, e o check degenera em comparar ruído
numérico contra zero (medido: `2.2e-3` contra `3.1e-10` com pesos distintos).
Mesma degeneração já documentada pro softmax na Etapa 6.
