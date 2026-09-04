# Etapa 9 — Camada de Embedding

## Antes de começar

### O que você vai construir

A porta de entrada do modelo: a camada que transforma números de token em vetores densos.

| Componente | Formato | O que guarda |
|---|---|---|
| `Table` | `[vocabSize, dModel]` | um vetor treinável por token do vocabulário |
| `PosTable` | `[contextLength, dModel]` | um vetor treinável por posição da sequência |
| `forward(tokens)` | → `[B, seqLen, dModel]` | a soma dos dois |

### O que você precisa saber antes

**Da Etapa 7:** o que são tokens, e o formato `(B, contextLength)` que o `BatchSampler` produz.

**Da Etapa 8:** o que é um parâmetro treinável, e a inicialização `N(0, 0.02²)`.

**Da Etapa 3:** o backward do `matmul` — a §3 o reusa para derivar tudo aqui — e o broadcasting.

### Onde esta etapa se encaixa

A Etapa 7 terminou com uma limitação explícita: um índice de token é um rótulo arbitrário, e uma rede neural não consegue fazer nada de útil com ele. O token 5 não é "maior" que o token 3 em nenhum sentido semântico — a numeração veio da ordem alfabética.

Esta etapa resolve isso. É aqui que os tokens finalmente encontram o mundo dos tensores com gradiente.

E é também onde token e **posição** se fundem numa representação única, que as camadas de atenção vão processar.

### Objetivos de aprendizagem

Ao terminar, você deve conseguir:

1. Demonstrar que multiplicar por um vetor one-hot equivale a selecionar uma linha.
2. Derivar o gradiente esparso a partir do backward do `matmul`, sem decorá-lo.
3. Explicar por que a atenção precisa de informação de posição explícita.
4. Distinguir o backward da tabela de tokens do backward da tabela de posições.

> **Guia visual.** A equivalência one-hot, o scatter-add e a soma token mais posição: [`embedding.html`](embedding.html). **Exercícios (12 questões):** [`exercises.html`](exercises.html).

---

## §1. Um índice é uma camada linear degenerada

A Etapa 8 fechou apontando para isto, e vale demonstrar antes de tudo o mais, porque tudo que segue decorre daqui.

> **Definição — vetor one-hot.** Um vetor com `1` numa posição e `0` em todas as outras. `onehot(t)` tem tamanho `vocabSize`, com o `1` na posição `t`.

Seja `Table` a tabela de embeddings, de formato `[vocabSize, dModel]`. Pela definição de `matmul` da Etapa 3 §6:

```
(onehot(t) @ Table)[j] = Σₖ onehot(t)[k] · Table[k,j]
```

Todo termo dessa soma é zero, exceto quando `k = t` — o único lugar onde `onehot(t)` vale 1. Sobra apenas `Table[t,j]`:

```
onehot(t) @ Table  =  Table[t]        a linha t, inteira
```

> **Definição — embedding.** O vetor denso que representa um token. É uma linha da tabela, e ela é treinável: o modelo aprende, durante o treino, o que cada token deve significar.

A camada desta etapa **calcula exatamente esse resultado**, mas sem nunca montar o vetor one-hot. Em vez de uma multiplicação com `vocabSize - 1` termos zerados, é um acesso direto à linha. Matematicamente idêntico; computacionalmente, a diferença entre `O(vocabSize)` e `O(1)` por token.

**Exemplo numérico**, com `vocabSize = 5` e `dModel = 2`:

```
Table = [[ 0.10, -0.20],     ← token 0
         [ 0.30,  0.40],     ← token 1
         [-0.10,  0.50],     ← token 2
         [ 0.05, -0.05],     ← token 3
         [ 0.20,  0.20]]     ← token 4

onehot(1) = [0, 1, 0, 0, 0]

onehot(1) @ Table = 0·[0.10,-0.20] + 1·[0.30,0.40] + 0·[-0.10,0.50]
                  + 0·[0.05,-0.05] + 0·[0.20,0.20]
                  = [0.30, 0.40]
                  = Table[1]        ✓ idêntico a pegar a linha 1 direto
```

---

## §2. A tabela de tokens e o forward

A tabela tem uma linha por token do vocabulário, é treinável, e nasce com `N(0, 0.02²)` — a mesma escala do GPT-2 discutida na Etapa 8 §5.

As linhas começam aleatórias. O que elas *significam* é justamente o que o treino descobre.

`forward` recebe os tokens já codificados pela Etapa 7 e devolve, para cada um, a linha correspondente. Para um lote de formato `[B, seqLen]`, a saída tem formato `[B, seqLen, dModel]` — cada posição da sequência vira um vetor.

**Exemplo numérico**, reusando a `Table` da §1, com uma sequência `tokens = [1, 0, 1]`:

```
tokenEmb[0] = Table[1] = [0.30,  0.40]
tokenEmb[1] = Table[0] = [0.10, -0.20]
tokenEmb[2] = Table[1] = [0.30,  0.40]      ← token 1 de novo, mesma linha
```

Note a posição 2. O token 1 aparece duas vezes na sequência, e as duas vezes leem exatamente a mesma linha. Isso é o esperado no forward — e é o que torna o backward interessante.

---

## §3. Backward: gradiente esparso, derivado

O roadmap descreve o backward como "acumular o gradiente nas linhas usadas". Essa regra não precisa ser aceita de bandeja: ela sai do que já foi provado na Etapa 8.

Empilhe os `onehot` de todos os `n` tokens do lote — achatando `[B, seqLen]` numa lista só — numa matriz `X` de formato `[n, vocabSize]`. Então o forward inteiro é `Y = X @ Table`.

O backward do `matmul` para o segundo operando, da Etapa 8 §3, dá:

```
dTable = Xᵀ @ dY
```

Agora repare no que `Xᵀ` é. Ela tem formato `[vocabSize, n]`, e `Xᵀ[t, i] = 1` exatamente quando o token da posição `i` é `t`. Substituindo:

```
dTable[t, :] = Σᵢ Xᵀ[t,i] · dY[i, :] = Σ_{i : tokenᵢ = t} dY[i, :]
```

Em palavras: **a linha `t` recebe a soma de `dY[i]` para toda posição onde o token era `t`**. E linhas de tokens que não apareceram somam sobre um conjunto vazio, ficando em zero.

> **Definição — gradiente esparso.** Só as linhas efetivamente usadas recebem gradiente. Num vocabulário de 100 tokens e um lote que usou 20 deles, 80 linhas ficam zeradas.

É o mesmo padrão *scatter-add* de `sum(dim)` na Etapa 3: mapear cada posição de entrada a um grupo, e acumular. Nenhuma regra nova — é `matmul` de novo, com um operando que por natureza é esparso.

**Exemplo numérico**, continuando com `tokens = [1, 0, 1]` e um gradiente com valores distintos por posição:

```
dY = [[1, 2], [3, 4], [0.5, -1]]

posição 0: token 1, dY[0] = [1, 2]
posição 1: token 0, dY[1] = [3, 4]
posição 2: token 1, dY[2] = [0.5, -1]      ← token 1 de novo

dTable[0] = dY[1]                          = [3, 4]
dTable[1] = dY[0] + dY[2] = [1+0.5, 2-1]   = [1.5, 1]      ← duas contribuições somadas
dTable[2] = dTable[3] = dTable[4]          = [0, 0]        ← nunca apareceram
```

**Verificação independente.** A linha 1 foi lida duas vezes no forward, nas posições 0 e 2. Pela regra de múltiplos caminhos da Etapa 2 §2, ela deve receber a soma das duas contribuições — e recebe. A linha 0 foi lida uma vez, e recebe uma contribuição. As demais, nenhuma.

### Não há gradiente para o índice

Um ponto que fecha o gancho deixado pela Etapa 7: **não existe gradiente em relação ao array de tokens**.

Índice não é quantidade contínua. Não faz sentido perguntar "a perda aumentaria se este token fosse 2.3 em vez de 2". O gradiente flui para dentro da tabela, nunca para os números que a endereçam.

> **Confira você mesmo.** Um lote usa apenas 12 tokens distintos, num vocabulário de 95. Quantas linhas de `dTable` ficam zeradas — e por que isso não é um bug?
>
> <details><summary>Resposta</summary>
>
> 83 linhas ficam em zero. Não é bug: aqueles tokens simplesmente não apareceram neste lote, então a perda não depende deles nesta passada. O otimizador vai atualizá-los quando eles aparecerem em lotes futuros. É exatamente o que "gradiente esparso" significa, e é por isso que a tabela de embedding é barata de treinar apesar de enorme.
> </details>

---

## §4. Embedding posicional

### Por que é necessário

O mecanismo de atenção (Etapa 11) calcula, para cada posição, uma média ponderada sobre as outras posições. Essa operação, sozinha, **não distingue ordem**.

Embaralhe os tokens de entrada e a atenção produzirá essencialmente o mesmo resultado, a menos da ordem da saída. Ela trata a sequência como um **conjunto**, não como uma sequência.

Isso é fatal para linguagem. "O cachorro mordeu o homem" e "O homem mordeu o cachorro" têm exatamente os mesmos tokens.

A solução é injetar a informação de posição em cada vetor, **antes** que a atenção o veja.

### Como

Uma segunda tabela, `PosTable`, com formato `[contextLength, dModel]` — uma linha por posição possível dentro da janela.

A diferença crucial em relação à tabela de tokens: aqui o índice depende só da **posição**, nunca do conteúdo. Para uma sequência de `seqLen` tokens, os índices são sempre `[0, 1, ..., seqLen-1]`.

Isso significa que o resultado do lookup posicional é **idêntico para toda sequência do lote**. Ele tem formato `[seqLen, dModel]` e é broadcastado para os `B` exemplos — o mesmo mecanismo de padding de rank da Etapa 3 §4.

**Exemplo numérico**, com `contextLength = 4`, `dModel = 2` e `seqLen = 3`:

```
PosTable = [[0.01, 0.02],     ← posição 0
            [0.03, 0.04],     ← posição 1
            [0.05, 0.06],     ← posição 2
            [0.07, 0.08]]     ← posição 3, fora do alcance desta sequência

posEmb = PosTable[0:3] = [[0.01,0.02], [0.03,0.04], [0.05,0.06]]      formato (3,2)
```

A linha 3 existe porque `contextLength` reserva espaço para a sequência mais longa possível. Sequências mais curtas simplesmente não a alcançam.

> **Confira você mesmo.** Duas sequências do mesmo lote começam com tokens diferentes. O embedding posicional da posição 0 é diferente entre elas?
>
> <details><summary>Resposta</summary>
>
> Não — é exatamente o mesmo vetor, `PosTable[0]`, nas duas. O lookup posicional depende só de **onde** o token está, nunca de **qual** token é. O que difere entre as duas sequências vem inteiramente da tabela de tokens; a parcela posicional é idêntica. É por isso que ela é broadcastada sobre o lote em vez de calculada por exemplo.
> </details>

---

## §5. Backward posicional: soma sobre o lote

Como toda sequência do lote usa as **mesmas** linhas de `PosTable`, o backward é o `unbroadcast` de sempre — o mesmo mecanismo do deslocamento `b` na Etapa 8 §3.

A linha `i` recebe a soma de `dY[b, i, :]` sobre **todo** `b` do lote.

Vale contrastar com a §3, porque as duas esparsidades são diferentes:

| | tabela de tokens | tabela de posições |
|---|---|---|
| quem recebe gradiente | só as linhas dos tokens presentes | **toda** linha de `0` a `seqLen-1` |
| de quantas fontes | de cada ocorrência daquele token | de **todo** exemplo do lote, sempre |
| o que fica zerado | tokens ausentes do lote | apenas linhas além de `seqLen` |

**Exemplo numérico**, com `B = 2` e `seqLen = 3`:

```
dY[0] = [[1, 1], [2, 0.5], [0.2, -0.3]]
dY[1] = [[0.5, -1], [1, 2], [-0.4, 0.6]]

dPosTable[0] = dY[0,0] + dY[1,0] = [1+0.5,   1-1  ] = [ 1.5, 0  ]
dPosTable[1] = dY[0,1] + dY[1,1] = [2+1,     0.5+2] = [ 3,   2.5]
dPosTable[2] = dY[0,2] + dY[1,2] = [0.2-0.4, -0.3+0.6] = [-0.2, 0.3]
dPosTable[3] = [0, 0]                                     ← nunca usada
```

---

## §6. Juntando token e posição

O embedding final é a **soma** elemento a elemento:

```
y = tokenEmb + posEmb          com posEmb broadcastado sobre o lote
```

Como a derivada local da soma é `1` dos dois lados (Etapa 3 §1), o gradiente que chega em `y` flui **inteiro** para os dois ramos. Não é metade para cada — é o mesmo `dY` alimentando os backwards das §3 e §5 em paralelo.

**Exemplo numérico completo**, com `B = 2` e tokens escolhidos para que o token 1 apareça em **duas sequências diferentes**:

```
tokens[0] = [1, 0, 1]        tokens[1] = [1, 2, 0]

tokenEmb[0] = [[0.30,0.40], [0.10,-0.20], [0.30, 0.40]]
tokenEmb[1] = [[0.30,0.40], [-0.10,0.50], [0.10,-0.20]]

posEmb (igual para os dois) = [[0.01,0.02], [0.03,0.04], [0.05,0.06]]

y[0] = [[0.31, 0.42], [ 0.13, -0.16], [0.35,  0.46]]
y[1] = [[0.31, 0.42], [-0.07,  0.54], [0.15, -0.14]]
```

Com o mesmo `dY` da §5:

```
dPosTable = [[1.5, 0], [3, 2.5], [-0.2, 0.3], [0, 0]]        idêntico à §5

dTable:
  token 1 aparece em (b0,i0), (b0,i2) e (b1,i0) — três vezes, em duas sequências:
    dTable[1] = dY[0,0] + dY[0,2] + dY[1,0] = [1+0.2+0.5, 1-0.3-1] = [1.7, -0.3]
  token 0 aparece em (b0,i1) e (b1,i2):
    dTable[0] = dY[0,1] + dY[1,2]           = [2-0.4, 0.5+0.6]      = [1.6,  1.1]
  token 2 aparece só em (b1,i1):
    dTable[2] = dY[1,1]                                              = [1.0,  2.0]
  dTable[3] = dTable[4] = [0, 0]
```

**Verificação independente.** Note que `dPosTable` é exatamente o mesmo da §5, apesar de os tokens serem diferentes. Isso confirma a independência entre as duas tabelas: o gradiente posicional não depende de **quais** tokens estão ali, apenas de **onde** eles estão.

E o token 1 acumulou contribuição de três posições espalhadas por duas linhas distintas do lote — o scatter-add soma sobre o lote inteiro achatado, não apenas dentro de uma sequência.

> **Confira você mesmo.** Uma sequência tem o mesmo token repetido nas 3 posições, com `dY = [[1,0],[2,0],[3,0]]`. Qual é o gradiente daquela linha da tabela?
>
> <details><summary>Resposta</summary>
>
> `[6, 0]`. As três contribuições somam: `1 + 2 + 3 = 6` na primeira dimensão, e `0` na segunda. Todas as outras linhas ficam zeradas. É o caso extremo do scatter-add — um único destino recebendo tudo.
> </details>

---

## §7. Por que não sinusoidal

O paper original do Transformer usa embeddings posicionais **fixos**, calculados por uma fórmula de seno e cosseno em frequências diferentes por dimensão. Não há parâmetro a treinar, e em tese eles generalizam para sequências mais longas que qualquer uma vista no treino.

O GPT usa embeddings posicionais **aprendidos** — a `PosTable` desta etapa. O modelo descobre a melhor representação de posição para a tarefa, ao custo de ficar limitado a `contextLength`. Não existe linha para uma posição além disso.

Para um projeto didático com `contextLength` fixo, aprendido é mais simples: mesma mecânica de lookup da tabela de tokens, sem nenhuma fórmula trigonométrica nova. E funciona bem dentro do limite.

É a mesma troca de generalização por simplicidade já feita na Etapa 7 §1, ao escolher tokenização por caractere.

---

## §8. A decisão de implementação: lookup eficiente

A §1 provou que `onehot(t) @ Table` e "pegar a linha `t`" são a mesma coisa. Isso abre dois caminhos de implementação, e a escolha entre eles tem uma consequência arquitetural real.

**Caminho A — `onehot(t) @ Table`, literalmente.** Usa apenas API já pública: `Tensor.make` para montar o one-hot, e `.matmul`, testado desde a Etapa 3. O backward sai de graça, herdado e correto. Nenhuma mudança no módulo `scalagrad`.

O custo é montar um tensor `[n, vocabSize]` majoritariamente zero a cada forward — exatamente a ineficiência que motiva a existência da tabela de embedding.

**Caminho B — lookup direto, `O(1)` por token.** Exige criar um tensor de saída com `previous` e `_backward` próprios. E aí está a barreira: o construtor de `Tensor` é `private[scalagrad]`, e `Tensor.make` — a única fábrica pública — só constrói nós-folha. **Apenas código dentro do módulo `scalagrad` consegue criar um tensor com backward próprio.**

Isso tensiona com uma decisão anterior do projeto, de que `ops/` pararia de crescer após a Etapa 6.

**O que foi decidido:** o caminho B, reabrindo `scalagrad.ops` para um primitivo novo.

O raciocínio: a operação é **genérica**. Selecionar linhas de uma tabela por índice não tem nada de específico de GPT — é mecânica de indexação, da mesma família de `matmul` e `broadcastTo`. Ela pertence ao módulo de tensores tanto quanto qualquer operação da Etapa 3. Deixá-la em `gpt/` por respeito a uma fronteira administrativa seria pior do que mover a fronteira.

O resultado é `Tensor.indexSelect(indices)`, em `scalagrad/ops/IndexOps.scala`. Ele opera sobre uma tabela de rank 2, recebe um array achatado de índices, e devolve `[indices.length, rowDim]`. Quem decide como isso vira `[B, seqLen, dModel]` é o `reshape` do lado de `gpt.nn.Embedding` — a função de `scalagrad` não sabe nada sobre tokens.

> **Armadilha.** A primeira implementação de `indexSelect` colocou a validação de rank **depois** de já ter lido `t.shape(1)`:
>
> ```scala
> val numRows = t.shape(0)
> val rowDim  = t.shape(1)          // <- lê aqui
> require(t.rank == 2, "...")       // <- valida só aqui
> ```
>
> Para uma tabela de rank 1, `t.shape(1)` estoura um `ArrayIndexOutOfBoundsException` cru **antes** de o `require` rodar. A mensagem cuidadosamente escrita — que diz exatamente o que se esperava e o que veio — nunca aparece. No lugar dela, sai um erro genérico de índice, sem contexto nenhum.
>
> **Lição geral:** valide antes de usar. Uma pré-condição que roda depois do acesso que ela deveria proteger não é uma pré-condição — é código morto que dá falsa sensação de segurança.

---

## §9. Para onde isso leva

A saída desta etapa é um tensor `[B, seqLen, dModel]`. Pela primeira vez, uma representação vetorial densa e treinável de cada posição da sequência, com token e posição já fundidos.

O `dModel` fixado aqui atravessa o modelo inteiro. Todas as projeções da atenção e do MLP, nas Etapas 12 e 13, operam nessa mesma dimensão.

A Etapa 10 (LayerNorm) é a primeira camada a processar esse tensor de verdade. Ela normaliza cada vetor de `dModel` independentemente, estabilizando a escala das ativações antes que elas entrem no primeiro bloco de atenção.

E aí, na Etapa 11, o mecanismo central do transformer finalmente aparece.

---

## Cartão de referência

| Conceito | Resumo |
|---|---|
| equivalência fundamental | `onehot(t) @ Table = Table[t]` |
| `Table` | `[vocabSize, dModel]`, treinável, `N(0, 0.02²)` |
| `PosTable` | `[contextLength, dModel]`, treinável |
| índice do token | depende do **conteúdo** |
| índice da posição | depende só de **onde**; igual para todo o lote |
| `dTable[t]` | soma de `dY[i]` para toda posição com token `t` |
| `dPosTable[i]` | soma de `dY[b,i]` sobre **todo** o lote |
| gradiente do índice | não existe — índice não é contínuo |
| combinação | `y = tokenEmb + posEmb`; `dY` flui inteiro para os dois |
| implementação | `indexSelect` em `scalagrad`, lookup `O(1)` |

### As quatro lições que se repetem

1. **Reconhecer um caso degenerado economiza trabalho.** Ver que embedding é `matmul` com one-hot dá o backward de graça, derivado em vez de decorado.
2. **Duas esparsidades diferentes.** A tabela de tokens recebe gradiente só onde o token apareceu; a de posições recebe de todo o lote, sempre.
3. **Fronteiras de módulo servem ao código, não o contrário.** Uma operação genérica pertence ao módulo genérico, mesmo que isso reabra uma fronteira que se julgava fechada.
4. **Valide antes de usar.** Uma pré-condição depois do acesso que ela protegeria nunca chega a rodar.
