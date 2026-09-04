# Etapa 7 — Tokenização

## Antes de começar

### O que você vai construir

A ponte entre texto e números. Duas peças:

| Peça | Responsabilidade |
|---|---|
| `Tokenizer` | construir o vocabulário, e converter texto ↔ inteiros |
| `BatchSampler` | recortar o corpus em exemplos de treino, aos lotes |

Esta é a primeira etapa do módulo `gpt/`. Tudo até aqui foi matemática genérica, no módulo `scalagrad` — a partir daqui o código é específico do modelo de linguagem.

### O que você precisa saber antes

Quase nada das etapas anteriores. Tokenização não tem gradiente nenhum: é manipulação de texto e inteiros.

A única dependência é o `Tensor` da Etapa 1, usado para empacotar o resultado final. Você pode ler este capítulo sem lembrar de autograd.

### Onde esta etapa se encaixa

As Etapas 1 a 6 completaram o marco da fundação matemática. Todas as primitivas de que um transformer precisa existem e foram verificadas.

Mas um modelo de linguagem não processa texto. Ele processa **números**. Alguém precisa fazer a tradução, nos dois sentidos — texto entra, números saem, e no final números viram texto de novo.

É o que esta etapa constrói. E ela também define a forma exata dos exemplos de treino, que é o assunto mais sutil aqui.

### Objetivos de aprendizagem

Ao terminar, você deve conseguir:

1. Explicar por que a ordenação do vocabulário importa, e o que quebra sem ela.
2. Justificar por que `decode(encode(t)) == t` é um teste real, e não formalidade.
3. Construir, à mão, o par `(input, target)` de uma janela qualquer.
4. Calcular o intervalo válido de posições de início, e explicar o `-1` extra.

> **Guia visual.** Vocabulário, a janela deslizante e o lote de janelas: [`tokenization.html`](tokenization.html). **Exercícios (10 questões):** [`exercises.html`](exercises.html).

---

## §1. Por que tokenização por caractere

> **Definição — token.** A menor unidade de texto que o modelo enxerga. Cada token vira um número inteiro, e é sempre esse inteiro que trafega pelo modelo.

> **Definição — corpus.** O texto de treino. É dele que sai tanto o vocabulário quanto todos os exemplos.

Como recortar texto em tokens é uma escolha de projeto, e há três caminhos.

**Por palavra.** Cada palavra é um token. Simples, mas quebra em qualquer palavra que não estava no vocabulário — nomes próprios, erros de digitação, palavras novas.

**Por sub-palavra (BPE).** O algoritmo aprende quais sequências de caracteres aparecem juntas com frequência, e agrupa cada uma num token: `"ing"`, `"ção"`, `"pre"`. É o que o GPT-4 usa, com cerca de 100 mil tokens. Eficiente, e bem mais complexo de implementar.

**Por caractere.** Cada caractere é um token. É a escolha deste projeto.

> **Definição — vocabulário.** O conjunto de todos os tokens distintos que o modelo conhece. Seu tamanho, `vocabSize`, é uma consequência do corpus — não um número escolhido antes.

A escolha por caractere tem duas vantagens diretas. O vocabulário é minúsculo — tipicamente 60 a 100 para português, contando letras, acentos, pontuação, espaço e quebra de linha. E **nenhuma palavra fica de fora**: qualquer texto formado pelos caracteres já vistos pode ser codificado.

O custo é o comprimento. Uma palavra de oito letras vira oito tokens, não um ou dois. Como o custo da atenção cresce com o quadrado do comprimento da sequência, isso é caro em modelos grandes.

Para o objetivo aqui — entender o mecanismo, não competir em eficiência — é uma troca que vale.

**Exemplo numérico.** A palavra `"abacate"` sob as três estratégias:

```
por palavra:     ["abacate"]                          1 token
por sub-palavra: ["aba", "cate"]                      2 tokens  (hipotético)
por caractere:   ["a","b","a","c","a","t","e"]        7 tokens
```

Sete tokens para uma palavra. É o preço da simplicidade, e é ele que este projeto escolhe pagar.

---

## §2. O vocabulário

Construir o vocabulário tem três passos: ler o corpus, coletar os caracteres **únicos**, e ordená-los.

Dois mapas guardam a correspondência, em sentidos opostos:

```
charToIdx: Map[Char, Int]      'a' -> 0, 'b' -> 1, ...
idxToChar: Map[Int, Char]      0 -> 'a', 1 -> 'b', ...
```

### Por que ordenar

A ordenação parece detalhe estético. Não é — é sobre **determinismo**.

A ordem de iteração de um `Set` não é garantida em Scala. Sem ordenar, o mesmo corpus poderia gerar mapeamentos diferentes em execuções diferentes.

E aí acontece o desastre. Um modelo treinado com um mapeamento fica **incompatível** com outro mapeamento do mesmo corpus. Os pesos aprenderam que o índice 7 significa `'m'`; se numa execução posterior o 7 virar `'q'`, o modelo gera lixo. Sem erro, sem aviso — só texto sem sentido.

Ordenar alfabeticamente garante que o mesmo corpus sempre produza o mesmo vocabulário.

**Exemplo numérico**, com `corpus = "abacate"`:

```
caracteres na ordem em que aparecem:  a, b, a, c, a, t, e
caracteres únicos:                    {a, b, c, e, t}
ordenados alfabeticamente:            a, b, c, e, t
índices atribuídos:                   0, 1, 2, 3, 4

charToIdx = {a:0, b:1, c:2, e:3, t:4}
idxToChar = {0:a, 1:b, 2:c, 3:e, 4:t}
vocabSize = 5
```

Repare no detalhe que torna este exemplo útil: na ordem de **aparição**, o `t` vem antes do `e`. Na ordem **alfabética**, o `e` vem antes. As duas ordens discordam — então o exemplo distingue de fato uma implementação da outra.

---

## §3. Encode e decode

`encode` troca cada caractere pelo seu índice. `decode` faz o inverso e junta tudo numa `String`.

As duas são funções puras, sem estado além dos dois mapas. E devem ser **exatamente inversas** uma da outra:

```
decode(encode(texto)) == texto
```

### Por que essa verificação importa

Parece formalidade — os dois métodos são triviais, o que poderia dar errado?

O risco é um mapeamento **assimétrico**. Se `charToIdx` e `idxToChar` forem construídos a partir de ordenações diferentes, cada um funciona perfeitamente sozinho. `encode` produz índices válidos. `decode` produz caracteres válidos. Nenhum dos dois lança erro.

Mas eles discordam. E o pipeline de treino inteiro fica corrompido em silêncio.

O teste de ida e volta pega isso na hora, e é por isso que ele está no checklist da etapa.

**Exemplo numérico**, com o vocabulário da §2:

```
encode("abacate") = [0, 1, 0, 2, 0, 4, 3]
                     a  b  a  c  a  t  e

decode([0,1,0,2,0,4,3]) = "a"+"b"+"a"+"c"+"a"+"t"+"e" = "abacate"

decode(encode("abacate")) == "abacate"    ✓
```

### O que fazer com o desconhecido

Um caractere fora do vocabulário faz `encode` **falhar explicitamente**, com uma exceção que informa o caractere e a posição. O mesmo vale para `decode` com um índice inválido.

Poderia-se introduzir um token especial `UNK` para casos desconhecidos, como fazem tokenizadores maiores. Este projeto não faz isso, deliberadamente. Num tokenizador por caractere, um caractere desconhecido significa que o corpus de treino não representa o texto de uso — e isso é melhor descobrir com uma exceção do que mascarar.

> **Armadilha.** Ao ler o corpus de um arquivo, a primeira implementação usou `Source.fromFile(f).getLines().mkString`.
>
> O `getLines()` remove a quebra de linha de cada linha. E o `mkString` sem separador não recoloca nada. Resultado: **a última palavra de cada linha cola na primeira da seguinte**.
>
> ```
> arquivo:  "casa\nazul"
> lido:     "casaazul"
> ```
>
> Por que passou despercebido: o texto lido continua sendo texto válido. O tokenizador funciona, o vocabulário é construído, o treino roda. O modelo apenas aprende com um corpus corrompido — e a quebra de linha, que é um token legítimo e informativo, desaparece completamente do vocabulário.
>
> Havia um segundo problema na mesma linha: o `Source` nunca era fechado, vazando o descritor de arquivo. A correção resolveu os dois de uma vez, com `Using.resource(Source.fromFile(f))(_.mkString)` — que preserva o texto exatamente como está e fecha o arquivo mesmo se a leitura falhar.
>
> **Lição geral:** ao ler texto, prefira a leitura crua a qualquer API que processe linhas. Toda transformação intermediária é uma chance de perder informação silenciosamente.

> **Confira você mesmo.** Por que `vocabSize` precisa ser público, se ele é só um detalhe do tokenizador?
>
> <details><summary>Resposta</summary>
>
> Porque ele **dimensiona o modelo**. A tabela de embedding da Etapa 9 tem uma linha por token do vocabulário, e a camada de saída (Etapas 15 e 16) produz um logit por token. Os dois precisam consultar `vocabSize` para se construir. Não é detalhe interno — é um parâmetro da arquitetura.
> </details>

---

## §4. A janela deslizante, e o deslocamento de uma posição

Depois do `encode`, o corpus inteiro é um único array longo de inteiros. Treinar sobre ele de uma vez não é viável.

> **Definição — janela de contexto (`contextLength`).** Quantos tokens consecutivos o modelo enxerga de uma vez. É o tamanho de cada exemplo de treino, e o limite do que o modelo consegue "lembrar".

Cada exemplo é uma janela de `contextLength` tokens, recortada numa posição aleatória do corpus.

### A parte que confunde

Todo exemplo tem uma entrada e uma resposta esperada. Aqui está o ponto que costuma passar batido na primeira leitura:

**O `target` não é uma janela nova em outro lugar do corpus. É a mesma janela, deslocada uma posição à direita.**

```
input  = corpus[start     : start + contextLength]
target = corpus[start + 1 : start + contextLength + 1]
```

Isso captura exatamente a tarefa do modelo: **prever o próximo token, em toda posição da sequência, dado tudo que veio antes**.

Repare que cada posição da janela é um exemplo de treino por si só. Uma janela de quatro tokens não produz um exemplo — produz quatro, e todos são treinados de uma vez.

### O intervalo válido de `start`

Como o `target` vai uma posição além do `input`, a janela precisa de `contextLength + 1` tokens consecutivos disponíveis, e não apenas `contextLength`.

```
start pode ir de 0 até (corpusLength - contextLength - 1), inclusive
```

Aquele `-1` extra é fácil de esquecer, e produz um estouro de índice ao ler o último elemento do `target`.

**Exemplo numérico**, com `encode("abacate") = [0,1,0,2,0,4,3]` e `contextLength = 4`:

```
corpusLength = 7, contextLength = 4
start válido: 0 até 7-4-1 = 2      ou seja, start ∈ {0, 1, 2}

start = 0:
  input  = corpus[0:4] = [0, 1, 0, 2]      "a", "b", "a", "c"
  target = corpus[1:5] = [1, 0, 2, 0]      "b", "a", "c", "a"
```

Confira posição a posição o que o modelo aprende com essa única janela:

```
posição 0: vendo "a"      → o próximo é "b" (1)   ✓ o corpus é "ab..."
posição 1: vendo "ab"     → o próximo é "a" (0)   ✓ o corpus é "aba..."
posição 2: vendo "aba"    → o próximo é "c" (2)   ✓ o corpus é "abac..."
posição 3: vendo "abac"   → o próximo é "a" (0)   ✓ o corpus é "abaca..."
```

**Verificação independente.** O `target` inteiro é o `input` deslocado. Compare os dois arrays: `input = [0,1,0,2]` e `target = [1,0,2,0]`. Os três últimos elementos do `input` são os três primeiros do `target`. É a mesma sequência, uma casa à frente.

---

## §5. Um lote de janelas independentes

> **Definição — lote (*batch*).** Um conjunto de exemplos processados de uma vez. Treinar com lotes é mais rápido que um exemplo por vez, e produz gradientes menos ruidosos.

Para montar um lote de `B` exemplos, sorteiam-se `B` posições de início **independentes** dentro do intervalo válido. Cada uma vira um par `(input, target)` como na §4.

Empilhando tudo, o resultado são dois tensores de formato `(B, contextLength)`.

As posições não precisam ser vizinhas, nem ordenadas, nem sequer distintas. Janelas de linhas diferentes podem se sobrepor sem problema — elas são exemplos independentes que por acaso compartilham tokens.

**Exemplo numérico**, mesmo corpus, `contextLength = 4`, `B = 2`, com `start = 0` e `start = 2`:

```
start = 0:  input = [0,1,0,2] ("abac")    target = [1,0,2,0] ("baca")
start = 2:  input = [0,2,0,4] ("acat")    target = [2,0,4,3] ("cate")

inputs  (2,4) = [[0,1,0,2],        targets (2,4) = [[1,0,2,0],
                  [0,2,0,4]]                         [2,0,4,3]]
```

É justamente essa independência entre as linhas que permite processar o lote inteiro em paralelo nas etapas seguintes — via a multiplicação de matrizes em lote, já pronta desde a Etapa 3.

> **Armadilha.** Esta é uma armadilha ao contrário, e vale registrar pelo motivo oposto ao das outras.
>
> O `BatchSampler` deste projeto **não teve nenhum bug**. Isso é incomum: quase toda operação nova até aqui teve pelo menos um erro de índice na primeira versão, e a aritmética de janelas é território clássico de erro por um.
>
> A diferença foi o processo. O design — intervalo válido, deslocamento do target, layout do tensor — foi todo discutido e escrito **antes** de qualquer código. Os pontos onde normalmente se erra já estavam decididos quando a implementação começou.
>
> **Lição geral:** aritmética de índices é onde bugs se escondem melhor, e é onde pensar antes de digitar rende mais. Escreva o intervalo válido no papel antes de escrever o laço.

> **Confira você mesmo.** Um corpus tem exatamente `contextLength + 1` tokens. Quantas janelas distintas é possível amostrar?
>
> <details><summary>Resposta</summary>
>
> Apenas uma, com `start = 0`. O intervalo válido é `0` até `corpusLength - contextLength - 1`, que aqui dá `0` até `0`. Todo sorteio devolve a mesma janela. É o caso de borda mínimo, e um bom teste: amostrar dez vezes deve produzir dez janelas idênticas.
> </details>

---

## §6. Para onde isso leva

A saída desta etapa é um array de inteiros, cada um dentro de `[0, vocabSize)`.

E aqui há uma limitação importante de reconhecer: **isso ainda não é algo que uma rede neural consiga processar**.

Um índice é um rótulo arbitrário. O token 5 não é "maior" nem "mais" que o token 3 em nenhum sentido útil — a numeração veio da ordem alfabética, que não tem relação com significado. Somar, multiplicar ou tirar média de índices produz lixo.

Falta converter cada rótulo discreto num **vetor de números reais** que o modelo possa manipular. É a Etapa 9 (Embedding) que faz isso, e é lá que os tokens desta etapa finalmente encontram o mundo dos tensores com gradiente.

Antes disso, a Etapa 8 constrói a camada linear — a operação parametrizada mais básica, e a peça de que todo o resto do modelo é feito.

---

## Cartão de referência

| Conceito | Resumo |
|---|---|
| token | menor unidade de texto que o modelo enxerga |
| vocabulário | conjunto dos tokens distintos; tamanho = `vocabSize` |
| `vocabSize` | consequência do corpus, não escolha prévia |
| por que ordenar | determinismo — o mesmo corpus sempre gera o mesmo mapeamento |
| `encode` / `decode` | inversos exatos; `decode(encode(t)) == t` |
| `contextLength` | quantos tokens o modelo vê de uma vez |
| `input` | `corpus[start : start+contextLength]` |
| `target` | `corpus[start+1 : start+contextLength+1]` |
| intervalo de `start` | `0` até `corpusLength - contextLength - 1` |
| formato do lote | `(B, contextLength)`, para input e target |

### As quatro lições que se repetem

1. **Determinismo não é preciosismo.** Um vocabulário em ordem instável torna um modelo treinado incompatível consigo mesmo.
2. **O `target` é a mesma janela, deslocada.** Uma janela de `n` tokens é composta por `n` exemplos de treino, não um.
3. **O `-1` extra existe porque o par precisa de `contextLength + 1` tokens.** É o erro por um clássico desta etapa.
4. **Leia texto cru.** Toda API que processa linhas é uma chance de perder informação sem avisar.
