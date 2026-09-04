# Etapa 19 — Inferência e Geração de Texto

Status: [ ] Não iniciado · [ ] Em andamento · [x] Concluído

## Teoria
- [x] `theory/19-inference-generation/19-inference-generation.md`
- [x] `theory/19-inference-generation/generation.html` (guia visual)
- [x] `theory/19-inference-generation/exercises.html` (14 questões)
- [x] Termos novos no glossário do `00-overview`

## Implementação
- [x] Ativar `noGrad` globalmente — o laço inteiro roda dentro de `Tensor.noGrad`
- [x] Loop autoregressivo:
  - [x] Sliding window se contexto exceder `contextLength`
  - [x] Forward, pegar `logits[-1]`
  - [x] Amostrar próximo token
  - [x] Concatenar ao contexto e repetir
- [x] Greedy decoding (`argmax`)
- [x] Temperature sampling (`logits / T` + amostragem via CDF)
- [x] Top-k sampling — `TopK(k, temperature)`; `k` é obrigatório em vez de ter default 40, para a escolha ficar explícita na chamada (40 documentado como valor usual no cartão de referência)
- [x] `decode()` (sobrecarga `generate(prompt: String, ..., tokenizer)`) e impressão progressiva (callback `onToken`)

**MILESTONE: "o modelo gera"** — teste de fumaça final de toda a cadeia.
