## Etapa 7 — Tokenização

### Por que existe

Modelos de linguagem não processam texto diretamente — processam sequências de inteiros. A tokenização é o processo de mapear texto para esses inteiros. A complexidade da tokenização varia muito: GPT-4 usa BPE com ~100k tokens, nós usaremos tokenização character-level por ser a mais simples e completamente implementável do zero.

### O que implementar

**Construção do vocabulário**

Ler um arquivo de texto (o corpus de treinamento — pode ser uma obra de Machado de Assis, código Scala, ou qualquer texto). Coletar todos os caracteres únicos presentes. Ordená-los. Atribuir um índice inteiro para cada um, de 0 ao tamanho do vocabulário menos 1.

Guardar dois maps: `charToIdx: Map[Char, Int]` e `idxToChar: Map[Int, Char]`. O tamanho do vocabulário `vocabSize` é o número de caracteres únicos — tipicamente entre 60 e 100 para textos em português.

**Encode**

`encode(text: String): Array[Int]` — converte cada caractere do texto no índice correspondente. Simples.

**Decode**

`decode(tokens: Array[Int]): String` — converte de volta. Simples.

**Preparação de mini-batches**

O corpus encodado é um longo array de inteiros. Para treinar, vamos amostrar janelas aleatórias de comprimento `contextLength`. Para cada janela, as entradas são os primeiros `contextLength` tokens e os targets são os mesmos tokens deslocados de uma posição à direita — porque o modelo deve prever o próximo token a cada posição.

Exemplo com `contextLength = 4` e sequência `[5, 3, 7, 2, 9]`:
- Input: `[5, 3, 7, 2]`
- Target: `[3, 7, 2, 9]`

Para um batch de tamanho `B`, amostrar `B` janelas aleatórias independentes do corpus.

### Por que essa etapa importa

O tokenizador define a interface entre o mundo do texto e o modelo. Erros aqui — como encode/decode assimétricos, ou targets incorretamente deslocados — produzem um modelo que tecnicamente treina mas aprende a tarefa errada. Verificar sempre que `decode(encode(text)) == text` para amostras do corpus.
