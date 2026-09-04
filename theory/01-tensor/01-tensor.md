# Etapa 1 — Tensor

## Antes de começar

### O que você vai construir

Uma classe `Tensor`: a estrutura de dados que carrega todos os números do modelo. Ao final desta etapa ela saberá:

| Capacidade | Métodos |
|---|---|
| Guardar um array N-dimensional | `data`, `shape`, `strides` |
| Navegar por índices lógicos | `get`, `index`, `unravelIndex` |
| Mudar de forma sem copiar dados | `reshape`, `transpose` |
| Forçar uma cópia contígua | `contiguous` |
| Nascer preenchida | `zeros`, `ones`, `fill`, `arange`, `randn` |

Nada aqui aprende nada ainda. Esta etapa constrói o recipiente; o conteúdo vem depois.

### O que você precisa saber antes

Só programação. Arrays, classes, e a noção de que a memória do computador é endereçada por números inteiros.

Nenhum conhecimento de aprendizado de máquina é necessário — nem aqui, nem nas etapas seguintes. Todo termo novo é definido na primeira vez que aparece.

### Onde esta etapa se encaixa

O projeto inteiro constrói um GPT do zero, em Scala, sem bibliotecas de matemática. São 19 etapas, agrupadas em quatro marcos:

| Marco | Etapas | O que passa a existir |
|---|---|---|
| Fundação matemática | 1 a 6 | tensores, derivadas automáticas, todas as operações |
| Peças do modelo | 7 a 15 | tokenizador, camadas, o GPT montado |
| Treinamento | 16 a 18 | função de perda, otimizador, laço de treino |
| Uso | 19 | gerar texto |

Você está no começo do primeiro marco. E ele começa por aqui porque **tudo** no projeto é um tensor: os dados de entrada, os pesos que o modelo aprende, os resultados intermediários, os gradientes. Nenhuma etapa posterior funciona sem esta.

> **Definição — tensor.** Um array de N dimensões. É só isso. O nome vem da física e da matemática, onde significa algo mais específico, mas em aprendizado de máquina virou sinônimo de "array N-dimensional".

### Objetivos de aprendizagem

Ao terminar, você deve conseguir:

1. Calcular a posição física de qualquer elemento a partir do seu índice lógico.
2. Explicar por que `reshape` e `transpose` não copiam nenhum dado.
3. Prever o conteúdo de um tensor a partir do seu `shape` e dos seus `strides`.
4. Dizer quando um tensor deixa de ser contíguo, e por que isso importa.

---

### Antes de mergulhar

Se você ainda não leu a [Visão Geral](../00-overview/00-overview.md), leia antes desta etapa. Ela explica o que é um modelo de linguagem, mostra o mapa das 19 etapas, fixa as convenções de notação e traz o glossário. São vinte minutos que economizam muito mais.

Lá também estão descritos os quatro blocos destacados usados em todos os capítulos: **Definição**, **Armadilha**, **Confira você mesmo** e **Guia visual**.

> **Guia visual.** O desta etapa: [`strides-and-memory.html`](strides-and-memory.html). **Exercícios (10 questões):** [`exercises.html`](exercises.html).

---

## §1. O que é um tensor, de verdade

A ideia é simples e cresce por acréscimo de índices.

| Rank | Nome comum | Exemplo |
|---|---|---|
| 0 | escalar | `5.0` |
| 1 | vetor | `[1.0, 2.0, 3.0]` |
| 2 | matriz | `[[1, 2], [3, 4]]` |
| 3+ | — | um lote de 32 imagens 28×28 tem formato `[32, 28, 28]` |

> **Definição — rank.** O número de dimensões de um tensor. Um vetor tem rank 1; uma matriz tem rank 2.

> **Definição — shape.** A lista com o tamanho de cada dimensão. Um tensor de formato `[2, 3]` tem 2 linhas e 3 colunas.

Não existe mágica matemática nova a partir do rank 3. É sempre "mais um índice".

A dificuldade real desta etapa é outra, e é de engenharia. A memória do computador é **linear**: uma fita de posições numeradas de 0 em diante. Um tensor de rank 3 precisa ser achatado nessa fita. Duas perguntas decorrem disso, e as próximas seções respondem as duas:

1. Como transformar um índice lógico como `(1, 2, 3)` numa posição física da fita?
2. Como mudar a forma de um tensor sem copiar dado nenhum?

**Exemplo numérico.** Um tensor de formato `[2, 3]`:

```
lógico:   [[1, 2, 3],
           [4, 5, 6]]

rank  = 2          (duas dimensões)
shape = [2, 3]     (2 linhas, 3 colunas)
size  = 2 · 3 = 6  (seis números no total)

na memória: [1, 2, 3, 4, 5, 6]     ← uma fita de 6 posições
```

O tensor "parece" bidimensional, mas mora numa fita. Toda a próxima seção é sobre a ponte entre essas duas visões.

---

## §2. Layout row-major e strides

> **Guia visual.** Layout de memória, a fórmula do stride, e por que a transposta deixa de ser contígua: [`strides-and-memory.html`](strides-and-memory.html).

### Duas convenções possíveis

Ao achatar uma matriz numa fita, é preciso escolher a ordem. Existem duas convenções.

> **Definição — row-major.** Percorre a **última** dimensão primeiro. Uma linha inteira fica contígua na memória. É o padrão de C, de Scala e do NumPy.

> **Definição — column-major.** Percorre a **primeira** dimensão primeiro. Uma coluna inteira fica contígua. É o padrão de Fortran, Julia e R.

Este projeto usa row-major. Para um tensor de formato `[3, 4]`:

```
índice lógico (i,j):  (0,0) (0,1) (0,2) (0,3) (1,0) (1,1) ... (2,3)
posição física:         0     1     2     3     4     5   ...   11
```

Repare que `(0,3)` e `(1,0)` são vizinhos na memória, apesar de estarem em linhas diferentes. A fita não conhece linhas.

### O stride

> **Definição — stride.** O stride de uma dimensão é quantas posições você anda na fita ao incrementar aquela dimensão em 1, mantendo as outras fixas.

Para uma matriz `[3, 4]`:

- Andar uma linha (`i += 1`) pula 4 posições. Logo `stride[0] = 4`.
- Andar uma coluna (`j += 1`) pula 1 posição. Logo `stride[1] = 1`.

A regra geral: **o stride de uma dimensão é o produto de todas as dimensões à direita dela.**

```
stride[k]   = d(k+1) · d(k+2) · ... · d(n-1)
stride[n-1] = 1                                  ← a última é sempre 1
```

A última dimensão sempre tem stride 1, porque ela é a mais interna — seus elementos são vizinhos diretos na fita.

Com os strides em mãos, converter índice lógico em posição física é uma multiplicação e uma soma:

```
posição = Σᵢ índice[i] · stride[i]
```

**Exemplo numérico.** Formato `[2, 3, 4]` — pense em 2 páginas, cada uma com 3 linhas de 4 colunas.

Calculando os strides, da direita para a esquerda:

```
stride[2] = 1              (última dimensão, sempre 1)
stride[1] = 4              (uma linha tem 4 elementos)
stride[0] = 3 · 4 = 12     (uma página tem 3 linhas de 4)
```

Agora a posição física do elemento lógico `(1, 2, 3)`:

```
posição = 1·12 + 2·4 + 3·1
        = 12 + 8 + 3
        = 23
```

**Verificação independente.** O tensor tem `2·3·4 = 24` elementos, ou seja, posições de 0 a 23. E `(1,2,3)` é o último elemento da última linha da última página — o canto mais distante. Faz sentido que caia exatamente na última posição, 23.

> **Armadilha.** A primeira versão deste projeto calculou os strides errado para rank 4 ou mais. Formatos de rank 2 e 3 funcionavam, então os testes iniciais passaram.
>
> Por que passou despercebido: com poucas dimensões, várias fórmulas erradas coincidem com a certa. O erro só aparece quando há dimensões suficientes para que a ordem dos produtos importe.
>
> **Lição geral:** ao testar código de indexação, use rank 4. Rank 2 valida muito menos do que parece.

> **Confira você mesmo.** Um tensor tem formato `[5, 2, 6]`. Quais são os seus strides?
>
> <details><summary>Resposta</summary>
>
> `[12, 6, 1]`. Da direita para a esquerda: `stride[2] = 1`; `stride[1] = 6` (o tamanho da dimensão à sua direita); `stride[0] = 2 · 6 = 12` (o produto de tudo à direita).
> </details>

---

## §3. Por que `reshape` e `transpose` são de graça

Esta é a ideia mais importante da etapa.

**`reshape` e `transpose` não copiam nenhum número.** Eles só trocam os metadados — `shape` e `strides`. A fita de dados continua exatamente onde estava. O que muda é apenas *como interpretamos* os índices.

> **Definição — view.** Um tensor que compartilha a fita de dados de outro, com `shape` ou `strides` diferentes. Alterar os dados de uma view altera o outro tensor, porque a memória é a mesma.

### O experimento decisivo

Nada explica isso melhor do que ver o mesmo array produzir dois tensores diferentes. Partimos de:

```
data (na memória) = [1, 2, 3, 4, 5, 6]

A: shape (2,3), strides (3,1)  →  [[1, 2, 3],
                                    [4, 5, 6]]
```

Agora aplicamos as duas operações. Ambas resultam em formato `(3,2)`:

**`reshape` para `(3,2)`** — recalcula os strides canônicos do novo formato:

```
shape (3,2), strides (2,1)

posição(0,0)=0 → 1     posição(0,1)=1 → 2
posição(1,0)=2 → 3     posição(1,1)=3 → 4
posição(2,0)=4 → 5     posição(2,1)=5 → 6

resultado: [[1, 2],
            [3, 4],
            [5, 6]]
```

**`transpose`** — troca `shape` e `strides` de posição:

```
shape (3,2), strides (1,3)      ← os strides foram trocados, não recalculados

posição(0,0)=0·1+0·3=0 → 1     posição(0,1)=0·1+1·3=3 → 4
posição(1,0)=1·1+0·3=1 → 2     posição(1,1)=1·1+1·3=4 → 5
posição(2,0)=2·1+0·3=2 → 3     posição(2,1)=2·1+1·3=5 → 6

resultado: [[1, 4],
            [2, 5],
            [3, 6]]
```

Compare os dois resultados. **Mesma fita de dados. Mesmo formato final `(3,2)`. Conteúdos completamente diferentes.** A única coisa que mudou entre eles foram os strides.

É por isso que essas operações são baratas: elas não tocam nos dados. Todo o trabalho está na aritmética de índices.

### Contiguidade

Repare nos strides da transposta: `(1, 3)`. A última dimensão tem stride 3, não 1.

> **Definição — contíguo.** Um tensor é contíguo quando percorrer sua última dimensão anda de 1 em 1 na memória. Equivale a dizer que seus strides são os canônicos do seu formato.

A transposta **não é contígua**. Ler uma "linha" dela significa pular de 3 em 3 pela fita.

Isso não é um problema de correção — a indexação continua certa. É um problema de desempenho, e às vezes de compatibilidade: certas operações assumem stride 1 na última dimensão. Para esses casos existe `contiguous`, que faz uma cópia real dos dados na nova ordem.

Uma consequência de projeto: como duas views compartilham a mesma fita, elas **não podem** interferir uma na outra. É por isso que o `CLAUDE.md` proíbe mutar `shape` e `strides` depois de criados. Um tensor e sua transposta precisam manter metadados independentes.

> **Armadilha.** Duas versões iniciais deste projeto erraram exatamente aqui.
>
> A primeira fez `transpose` **mutar** o tensor original, em vez de devolver uma view nova. O tensor de origem era destruído no processo.
>
> A segunda usou, dentro de `unravelIndex` e `contiguous`, os strides *reais* do tensor onde deveria usar os strides *canônicos* do formato. Funcionava perfeitamente em tensores contíguos, onde os dois coincidem. Quebrava em qualquer tensor que já tivesse passado por um `transpose`.
>
> **Lição geral:** existem dois conjuntos de strides em jogo — os que o tensor tem, e os que o formato dele teria se fosse contíguo. Saiba sempre qual dos dois a sua conta precisa.

> **Confira você mesmo.** Você transpõe uma matriz e depois chama `reshape` no resultado. Por que `reshape` pode precisar copiar dados nesse caso, se normalmente ele não copia nada?
>
> <details><summary>Resposta</summary>
>
> Porque `reshape` reinterpreta a fita **na ordem em que ela está**. Depois de um `transpose`, a ordem lógica dos elementos não bate mais com a ordem física. Reinterpretar a fita crua produziria o resultado errado. Por isso `reshape` precisa primeiro tornar o tensor contíguo — o que copia os dados na ordem lógica correta.
> </details>

---

## §4. De onde vem `randn` — a transformação de Box-Muller

### O problema

Na Etapa 8 você vai precisar preencher matrizes de pesos com números aleatórios que sigam uma distribuição normal. O motivo completo fica para lá; por ora basta saber que preencher com zeros ou com valores iguais **impede o modelo de aprender**.

> **Definição — distribuição normal, `N(μ, σ²)`.** A conhecida curva em sino. O primeiro parâmetro é a **média** `μ`, o centro da curva. O segundo é a **variância** `σ²`, que mede o espalhamento. Sua raiz `σ` é o **desvio padrão**. Valores perto da média são comuns; valores muito distantes são raros.
>
> A **normal padrão** é o caso `N(0, 1)`: centrada em zero, desvio padrão 1. É a que Box-Muller produz.

Cuidado com uma abreviação comum na literatura de aprendizado de máquina. Textos escrevem `N(0, 0.02)` querendo dizer *desvio padrão* `0,02`, não variância. Ao pé da letra a variância `0,02` daria desvio `√0,02 ≈ 0,141`, sete vezes maior. Este material sempre escreve o expoente — `N(0, 0.02²)` — para que não reste dúvida.

O obstáculo é que geradores de números aleatórios não produzem isso. Eles produzem números **uniformes** em `[0, 1)`, onde todo valor é igualmente provável. É preciso converter uniforme em normal.

### A transformação

Box-Muller consome duas amostras uniformes independentes, `u1` e `u2`, e devolve **duas** amostras normais independentes:

```
r  = √(-2 · ln(u1))
θ  = 2π · u2

z0 = r · cos(θ)
z1 = r · sin(θ)
```

### Por que funciona

A intuição é geométrica, e vale mesmo sem a demonstração formal.

Imagine duas normais independentes como as coordenadas de um ponto no plano. A densidade da gaussiana 2D é proporcional a `e^(-(x²+y²)/2)`. Repare que essa expressão depende **só de `x²+y²`**, ou seja, só da distância até a origem — nunca do ângulo.

Isso tem duas consequências diretas:

- **O ângulo é uniforme.** Nenhuma direção é privilegiada, então `θ` pode ser sorteado uniformemente em `[0, 2π)`. É o papel de `u2`.
- **O raio tem distribuição conhecida.** Como a densidade só depende do raio, `r²` acaba seguindo uma distribuição exponencial. É o papel de `u1`, e daí vem o `√(-2·ln(u1))`.

Box-Muller é literalmente isto: sortear um ângulo e um raio nas distribuições certas, e converter de polar para cartesiano.

**É por isso que as amostras saem em pares.** Um ponto no plano tem duas coordenadas. Você não consegue gerar uma sem gerar a outra. Na implementação, isso vira uma `LazyList` infinita, consumida uma amostra por vez.

**Exemplo numérico**, com `u1 = 0.5` e `u2 = 0.3`:

```
r = √(-2 · ln(0.5)) = √(-2 · (-0.693147)) = √1.386294 = 1.177410
θ = 2π · 0.3 = 1.884956 rad

cos(θ) = -0.309017
sin(θ) =  0.951057

z0 = 1.177410 · (-0.309017) = -0.363840
z1 = 1.177410 ·   0.951057  =  1.119783
```

Duas amostras de um par só de uniformes. Uma negativa, outra positiva — como esperado de uma distribuição centrada em zero.

**Verificação independente.** Não dá para conferir uma amostra isolada: ela é aleatória por definição. O que dá para conferir é o comportamento estatístico. Gerando 100.000 amostras, a média deve ficar perto de 0 e a variância perto de 1. Foi assim que a implementação deste projeto foi validada: média `-0,0037` e variância `0,999`. É o teste certo para código aleatório — não o valor de uma amostra, mas a forma da distribuição.

> **Confira você mesmo.** Por que não dá para escrever um teste que verifique que `randn` devolveu o número correto?
>
> <details><summary>Resposta</summary>
>
> Porque não existe "o número correto" — a saída é aleatória por construção. O que se testa em código aleatório são propriedades estatísticas sobre muitas amostras: média, variância, e o formato do tensor. É um tipo de teste diferente do resto do projeto, onde quase sempre existe um valor esperado exato.
> </details>

---

## §5. Para onde isso leva

O `Tensor` desta etapa é burro, e isso é intencional. Ele guarda números e sabe navegar por eles. Só.

Ele não faz ideia de onde os seus valores vieram, nem do que foi calculado a partir deles. Some um tensor com outro e o resultado não tem memória nenhuma da operação.

A Etapa 2 conserta exatamente isso. Ela ensina cada tensor a **lembrar** de onde veio, e a responder a uma pergunta nova: *se eu mexer um pouquinho neste número aqui, quanto o resultado lá na ponta muda?*

Essa pergunta é a base de qualquer forma de treinamento. Sem ela, não há aprendizado — só aritmética.

---

## Cartão de referência

| Conceito | Resumo |
|---|---|
| rank | número de dimensões |
| shape | tamanho de cada dimensão |
| size | produto do shape — total de elementos |
| stride | quantas posições andar na fita ao incrementar uma dimensão |
| stride da última dimensão | sempre 1 |
| regra do stride | produto de todas as dimensões à direita |
| índice → posição | `Σᵢ índice[i] · stride[i]` |
| row-major | última dimensão percorrida primeiro |
| view | compartilha a fita, muda os metadados |
| contíguo | os strides são os canônicos do formato |
| `reshape` | recalcula os strides do novo formato |
| `transpose` | troca `shape` e `strides` de posição |

### As três lições que se repetem

1. **A fita não muda; os metadados mudam.** `reshape` e `transpose` produzem tensores diferentes a partir da mesma memória. Todo o trabalho está na aritmética de índices.
2. **Existem dois conjuntos de strides.** Os que o tensor tem, e os canônicos do seu formato. Em tensores contíguos eles coincidem — e é justamente por isso que confundi-los passa despercebido nos testes.
3. **Teste com formatos que não colapsam.** Rank 2 e matrizes quadradas escondem erros de indexação que rank 4 e formatos retangulares expõem na hora.
