# Etapa 7 — Tokenização

Status: [ ] Não iniciado · [ ] Em andamento · [x] Concluído

## Vocabulário
- [x] Ler corpus de texto
- [x] Coletar caracteres únicos e ordenar
- [x] Construir `charToIdx: Map[Char, Int]` e `idxToChar: Map[Int, Char]`
- [x] `vocabSize`

## Encode/Decode
- [x] `encode(text: String): Array[Int]`
- [x] `decode(tokens: Array[Int]): String`
- [x] Verificar `decode(encode(text)) == text` para amostras do corpus

## Mini-batches
- [x] Amostragem de janelas aleatórias de `contextLength`
- [x] Input = janela, Target = janela deslocada de +1
- [x] Amostrar `B` janelas independentes por batch
