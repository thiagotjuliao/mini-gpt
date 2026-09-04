# PLAN.md — Construindo uma LLM do Zero em Scala

> Nenhuma biblioteca externa. Nenhuma abstração pronta. Cada linha com propósito.

---

## Filosofia do Projeto

A ideia central aqui não é ter um modelo que compita com GPT-4. É **entender profundamente** o que acontece dentro de qualquer LLM moderna — e a única forma de entender de verdade é construir cada peça com as próprias mãos, errar, debugar, e ver os gradientes fluindo onde deveriam fluir.

O projeto segue uma progressão deliberada: começamos com a estrutura de dados mais primitiva possível (um tensor) e subimos camada por camada até gerar texto. Cada etapa depende das anteriores. Não existe atalho seguro nessa ordem.

A linguagem escolhida é Scala porque: closures de primeira classe tornam o grafo de computação natural de expressar, o sistema de tipos revela bugs de dimensão em tempo de compilação, e o ambiente já é familiar. Nenhuma biblioteca externa significa que cada linha de álgebra linear, cada derivada, cada operação de normalização foi escrita por nós — e isso é exatamente o ponto.

A descrição completa de cada etapa (por que existe, o que implementar, por que importa) está em arquivos separados dentro de `roadmap/`, um por etapa, pra manter este arquivo enxuto e fácil de consultar.

---

## Índice das Etapas

1. [Tensor: A Estrutura de Dados Fundamental](roadmap/01-tensor.md)
2. [Motor de Autodiferenciação (Autograd)](roadmap/02-autograd.md)
3. [Operações Elementares com Gradiente](roadmap/03-elementary-operations.md)
4. [Verificação Numérica de Gradientes (Gradient Check)](roadmap/04-gradient-check.md)
5. [Funções de Ativação](roadmap/05-activations.md)
6. [Softmax e Log-Softmax](roadmap/06-softmax.md)
7. [Tokenização](roadmap/07-tokenization.md)
8. [Camada Linear (Fully Connected)](roadmap/08-linear.md)
9. [Camada de Embedding](roadmap/09-embedding.md)
10. [Layer Normalization](roadmap/10-layer-norm.md)
11. [Scaled Dot-Product Attention](roadmap/11-attention.md)
12. [Multi-Head Attention](roadmap/12-multi-head-attention.md)
13. [Feed-Forward Network (MLP)](roadmap/13-mlp.md)
14. [Bloco Transformer Completo](roadmap/14-transformer-block.md)
15. [Modelo GPT Completo](roadmap/15-gpt-model.md)
16. [Função de Perda (Cross-Entropy Loss)](roadmap/16-cross-entropy.md)
17. [Otimizador (AdamW)](roadmap/17-adamw-optimizer.md)
18. [Loop de Treinamento](roadmap/18-training-loop.md)
19. [Inferência e Geração de Texto](roadmap/19-inference-generation.md)

Cada etapa tem também um checklist de subtarefas correspondente em `checklists/` (ex: `checklists/01-tensor.md`), usado para acompanhar o progresso.

---

## Ordem de Implementação Recomendada e Milestones

```
Etapa 1  → Tensor básico funcionando, testes de indexação
Etapa 2  → Autograd com grafo simples (a + b, c * d), backward correto
Etapa 3  → Todas as ops com gradient check passando
Etapa 4  → Gradient checker genérico pronto e integrado
Etapa 5  → Ativações verificadas
Etapa 6  → Softmax verificada — MILESTONE: "o autograd está completo"
Etapa 7  → Corpus carregado, encode/decode funcionando
Etapa 8  → Linear layer, gradient check OK
Etapa 9  → Embedding, sequência tokenizada vira tensor
Etapa 10 → LayerNorm, gradient check OK
Etapa 11 → Atenção de uma cabeça, gradient check OK
Etapa 12 → Multi-head attention — MILESTONE: "o coração do transformer está pronto"
Etapa 13 → MLP
Etapa 14 → Bloco completo
Etapa 15 → Modelo completo, forward pass produz logits com shape correto
Etapa 16 → Loss calculada corretamente
Etapa 17 → Otimizador, um passo manual com parâmetros diminuindo
Etapa 18 → Loop de treinamento, loss descendo — MILESTONE: "o modelo treina"
Etapa 19 → Geração de texto — MILESTONE: "o modelo gera"
```

A cada milestone, o projeto já é demonstrável. O último milestone é quando você sabe que entendeu tudo.

---

## Estimativa de Cronograma (horas mínimas por etapa)

Estimativa calibrada em 2026-08-04, sem registro real de horas (o `HISTORY.md` loga checkpoints, não tempo gasto) — baseada no número de subtarefas de cada `checklists/*.md`, se a etapa introduz conceito matemático novo, e o quanto de debug as etapas já concluídas exigiram na prática (ex.: Etapa 3 foi a mais longa de longe, esticada por vários dias com bugs reais em quase toda operação nova). Tratar como **piso**, não meta: até aqui, toda etapa revelou pelo menos 1-2 bugs reais na revisão, o que historicamente empurrou o tempo real pra além do estimado ingenuamente. Não inclui tempo de leitura da `theory/` nem os intervalos entre sessões.

| Etapa | Horas mín. | Por quê |
|---|---|---|
| 1. Tensor | 4-6h | fundação, mecânico mas com Box-Muller/strides |
| 2. Autograd | 4-6h | topological sort, backward — conceito crítico |
| 3. Operações elementares | 10-14h | a mais densa (22 itens: broadcast, matmul, transpose/reshape backward) |
| 4. Gradient check | 3-4h | achou até uma race condition real |
| 5. Ativações | 2-3h | reusa infra existente |
| 6. Softmax | 3-4h | Jacobiano + generalização por `dim` |
| 7. Tokenização | 3h | domínio novo (`gpt/data`), mas conceito simples |
| 8. Linear | 2h | composição direta sobre `matmul` já pronto |
| 9. Embedding | 3h | lookup table + gradiente esparso/scatter |
| 10. Layer Normalization | 4h | mean/var/ε, backward não-trivial |
| 11. Scaled Dot-Product Attention | 5h | conceito novo central — máscara causal |
| 12. Multi-Head Attention | 5h | reshape/transpose entre cabeças — **milestone** |
| 13. Feed-Forward (MLP) | 2h | composição simples |
| 14. Bloco Transformer | 3h | conexões residuais, integração |
| 15. Modelo GPT completo | 5h | montagem de tudo |
| 16. Cross-Entropy Loss | 2h | loss + backward via softmax já pronto |
| 17. Otimizador (AdamW) | 4h | estado por parâmetro (momentos) |
| 18. Loop de treinamento | 6h | o mais denso (16 itens: LR schedule, checkpoint, batching) |
| 19. Inferência e geração | 4h | sampling (temperatura, top-k), loop autoregressivo — **milestone final** |

**Subtotal até a Etapa 6 (autograd completo): ~26-37h.** **Subtotal Etapas 7-19: ~48h.** **Total do projeto: ~75-90h de trabalho focado.**
