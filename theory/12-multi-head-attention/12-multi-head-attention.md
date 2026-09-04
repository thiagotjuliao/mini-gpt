# Etapa 12 — Multi-Head Attention

## Antes de começar

### O que você vai construir

Várias atenções rodando em paralelo, cada uma num pedaço do vetor de cada token.

| Componente | Formato | O que faz |
|---|---|---|
| `W_Q`, `W_K`, `W_V` | `[dModel, dModel]` | projetam a entrada inteira, para todas as cabeças de uma vez |
| divisão em cabeças | `[B, T, dModel]` → `[B, H, T, dHead]` | `reshape` seguido de `transpose` |
| atenção em lote | `[B, H, T, dHead]` | a Etapa 11 inteira, com um eixo a mais |
| concatenação | `[B, H, T, dHead]` → `[B, T, dModel]` | `transpose` seguido de `reshape` |
| `W_O` | `[dModel, dModel]` | projeção final, onde as cabeças se misturam |

### O que você precisa saber antes

**Da Etapa 1:** strides e índice linear. A divisão em cabeças é aritmética de índice, e nada mais.

**Da Etapa 3:** `reshape`, `transpose(dim0, dim1)` e `matmul` em lote. São as quatro operações desta etapa.

**Da Etapa 11:** o caminho `Q·Kᵀ → escala → máscara → softmax → ·V`. Ele não muda aqui. Ganha um eixo.

### Onde esta etapa se encaixa

A Etapa 11 construiu uma cabeça de atenção. Ela funciona, e tem um limite estrutural.

Cada posição da saída é uma média ponderada dos values. Os pesos vêm de uma linha de `P`. E essa linha **é a mesma para todas as coordenadas** daquela posição.

Isso força uma escolha. Se a coordenada 0 precisa vir de um token e a coordenada 1 de outro, uma cabeça só não consegue atender as duas. Ela tem que negociar um meio-termo.

A multi-head attention resolve isso do jeito mais direto possível. Ela roda várias atenções ao mesmo tempo, cada uma com a sua própria matriz de pesos. Cada cabeça olha para onde quiser.

O detalhe que surpreende é o preço. Ele é zero.

### Objetivos de aprendizagem

Ao terminar, você deve conseguir:

1. Explicar por que uma cabeça só é limitada, sem recorrer a "porque é mais expressivo".
2. Mostrar que `H` cabeças custam exatamente o mesmo que uma cabeça de tamanho `dModel`.
3. Descrever o caminho `reshape` → `transpose` e dizer o que quebra se você trocar a ordem.
4. Justificar por que `W_O` existe, e o que o modelo perde sem ele.
5. Dizer quais vieses do bloco são parâmetros mortos, e provar isso.

### Convenções de notação

| Símbolo | Significa |
|---|---|
| `B`, `T` | tamanho do lote e número de tokens da sequência |
| `dModel` | tamanho do vetor de cada token na entrada e na saída |
| `H` | número de cabeças — `nHeads` no código |
| `dHead` | tamanho do vetor dentro de cada cabeça, igual a `dModel / H` |
| `Q`, `K`, `V` | queries, keys e values já divididos — `[B, H, T, dHead]` |
| `P` | os pesos de atenção — `[B, H, T, T]` |
| `C` | a concatenação das cabeças, antes de `W_O` — `[B, T, dModel]` |
| `Y` | a saída do bloco — `[B, T, dModel]` |

A convenção `dX ≡ ∂L/∂X` do overview continua valendo.

> **Guia visual.** A divisão em cabeças, com o índice linear de cada elemento: [`heads.html`](heads.html), FIG 1. **Exercícios (14 questões):** [`exercises.html`](exercises.html).

---

## §1. Uma cabeça precisa escolher

Volte à fórmula da Etapa 11 §6. Para a posição `i`, a saída é:

```
y_i = Σⱼ p_ij · v_j
```

Repare no que `p_ij` **não** tem. Ele não tem índice de coordenada. O mesmo peso `p_ij` multiplica a coordenada 0 do value, a coordenada 1, e todas as outras.

> **Definição — cabeça (*head*).** Uma atenção completa: as suas próprias projeções, a sua própria matriz de pesos `[T, T]`, e a sua própria saída. Todas as coordenadas de uma cabeça compartilham a mesma linha de pesos.

Essa é a restrição real. Não é falta de capacidade em `V`. É que a decisão de "para onde olhar" é tomada uma vez por posição, e vale para o vetor inteiro.

**O exemplo mínimo.** Tome dois tokens, com `dModel = 2`. Os values são:

```
v₀ = [7, 2]        v₁ = [3, 9]
```

Suponha que a resposta certa na posição 1 seja `[7, 9]`. A coordenada 0 tem que vir do token 0. A coordenada 1 tem que vir do token 1.

Com uma cabeça, o peso `p` em `v₀` é único. A saída é `p·v₀ + (1−p)·v₁`, nas duas coordenadas:

```
p = 1.0   →  y = [7.0, 2.0]        erro quadrático 49.000
p = 0.0   →  y = [3.0, 9.0]        erro quadrático 16.000
p = 0.5   →  y = [5.0, 5.5]        erro quadrático 16.250
```

Nenhum acerta. O melhor `p` possível sai de minimizar `16(1−p)² + 49p²`, que dá `p = 32/130 ≈ 0.24615`:

```
melhor y de uma cabeça = [3.98462, 7.27692]      erro 12.06154
```

Confirmei esse mínimo por varredura numérica, com 200 mil valores de `p`. O menor erro encontrado foi `12.06154`, em `p ≈ 0.24616`. As duas rotas concordam.

Agora duas cabeças, cada uma com `dHead = 1`. A cabeça 0 cuida da coordenada 0 e usa `p = [1, 0]`. A cabeça 1 cuida da coordenada 1 e usa `p = [0, 1]`. A concatenação dá `[7, 9]`, com erro **exatamente zero**.

O ganho não veio de mais parâmetros. Veio de duas distribuições em vez de uma.

> **Confira você mesmo.** No exemplo acima, `p = 0.5` deu erro maior que `p = 0`. Isso é estranho?
>
> <details><summary>Resposta</summary>
>
> Não é. O erro na coordenada 1 pesa muito mais que o da coordenada 0. A distância entre `v₀` e `v₁` é `4` na primeira coordenada e `7` na segunda. Como o erro é quadrático, a coordenada 1 domina, e o ótimo puxa `p` para perto de zero. É exatamente o comportamento de "negociar um meio-termo" que motiva esta etapa. Uma cabeça só não empata as duas exigências; ela cede para a mais cara.
> </details>

---

## §2. O orçamento: por que dividir em vez de multiplicar

Existem duas maneiras de ter `H` cabeças. A escolha entre elas decide o custo do bloco.

A primeira é dar a cada cabeça o tamanho cheio. Cada uma teria `W_Q` de formato `[dModel, dModel]`, e o bloco custaria `H` vezes mais.

A segunda é repartir o orçamento. Cada cabeça recebe `dHead = dModel / H` coordenadas. É essa que todo transformer usa.

> **Definição — `dHead`.** O tamanho do subespaço de cada cabeça. Vale `dModel / H`, e por isso `dModel` precisa ser divisível por `H`.

O motivo de repartir fica claro na conta. Juntar `H` projeções `[dModel, dHead]` dá exatamente o mesmo número de pesos que uma projeção `[dModel, dModel]`:

```
dModel = 4, H = 2, dHead = 2
  2 cabeças × (4 × 2)  =  16 pesos
  uma matriz [4, 4]    =  16 pesos

dModel = 768, H = 12, dHead = 64        (GPT-2 pequeno)
  12 cabeças × (768 × 64)  =  589.824 pesos
  uma matriz [768, 768]    =  589.824 pesos
```

Os dois números são iguais porque `H · dHead = dModel`, por construção. Multi-head não é uma versão cara da atenção. É a mesma conta, reorganizada.

Isso tem uma consequência prática forte. As `H` projeções por cabeça nunca são materializadas. Você guarda **uma** matriz `[dModel, dModel]` por papel, e a divisão em cabeças acontece depois, no formato do resultado.

O bloco inteiro tem quatro matrizes `[dModel, dModel]`: `W_Q`, `W_K`, `W_V` e `W_O`. Daí vem a estimativa do roadmap:

```
dModel = 4    →  4 × 16      = 64 pesos
dModel = 768  →  4 × 589.824 = 2.359.296 pesos      ≈ 4 · dModel²
```

Com os vieses que sobrevivem à §8, some `2 · dModel`. Para `dModel = 768` isso dá `2.360.832`, uma diferença de `0,07%`.

> **Confira você mesmo.** Por que `dModel` precisa ser divisível por `H`? O que aconteceria com `dModel = 10` e `H = 4`?
>
> <details><summary>Resposta</summary>
>
> `dHead` seria `2.5`, e não existe fatia com duas coordenadas e meia. Na prática o `reshape` para `[B, T, H, dHead]` falharia: `10 ≠ 4 × 2`. Como o `reshape` da Etapa 3 exige que o número de elementos bata, o erro aparece na hora, com mensagem clara. É por isso que a validação `dModel % H == 0` entra no construtor — melhor recusar cedo, com uma mensagem que nomeia as duas grandezas, do que deixar o `reshape` reclamar de um produto que não fecha.
> </details>

---

## §3. Do `reshape` ao `transpose`: onde as cabeças moram

> **Guia visual.** As duas etapas da divisão, e o resultado do caminho errado: [`heads.html`](heads.html), FIG 1 e FIG 2.

A projeção `X·W_Q` devolve `[B, T, dModel]`. Você precisa de `[B, H, T, dHead]`. São dois passos, e a ordem importa.

**Passo 1: `reshape` para `[B, T, H, dHead]`.** Isso corta a última dimensão em `H` blocos contíguos. A cabeça `h` fica com as coordenadas de `h · dHead` até `(h+1) · dHead − 1`.

**Passo 2: `transpose(1, 2)` para `[B, H, T, dHead]`.** Isso troca o eixo das cabeças com o eixo da sequência. Depois dele, `T` e `dHead` são as duas últimas dimensões, que é o que o `matmul` da atenção precisa.

**O exemplo.** Use um rótulo no valor de cada elemento: o token `t`, coordenada `d`, vale `10t + d`. Com `T = 3`, `dModel = 4` e `H = 2`:

```
X  [T, dModel]
   [ 0,  1,  2,  3]
   [10, 11, 12, 13]
   [20, 21, 22, 23]
```

Depois do `reshape` para `[T, H, dHead]`, cada token vira duas fatias:

```
t=0:  [[0, 1], [2, 3]]
t=1:  [[10, 11], [12, 13]]
t=2:  [[20, 21], [22, 23]]
```

Depois do `transpose`, as fatias se agrupam por cabeça:

```
h=0:  [[0, 1], [10, 11], [20, 21]]      ← coordenadas 0 e 1 dos três tokens
h=1:  [[2, 3], [12, 13], [22, 23]]      ← coordenadas 2 e 3 dos três tokens
```

Leia a linha `h=0`. Ela tem os três tokens, na ordem, cada um com as suas duas primeiras coordenadas. É exatamente uma sequência completa vista por um subespaço. É isso que uma cabeça enxerga.

**Por que não pular o `reshape` intermediário.** A tentação é ir direto de `[T, dModel]` para `[H, T, dHead]` num `reshape` só. Os dois formatos têm 12 elementos, então a operação é aceita. O resultado é lixo:

```
h=0:  [[0, 1], [2, 3], [10, 11]]        ← "token 1" da cabeça 0 são as coordenadas 2 e 3 do token 0
h=1:  [[12, 13], [20, 21], [22, 23]]
```

A cabeça 0 recebeu metade do token 0, metade do token 1, e nada do token 2. O `reshape` percorre a memória em ordem canônica, e essa ordem tem `dModel` variando mais rápido que `T`. Sem o `transpose`, os dois eixos ficam entrelaçados.

**A aritmética por trás.** Em `[T, H, dHead]` o índice linear canônico é:

```
linear(t, h, d) = (t · H + h) · dHead + d
```

Confira em três posições, contra o rótulo `10t + (h · dHead + d)`:

```
[t=0][h=1][d=0]  →  linear 2   →  valor 2     esperado 10·0 + 2 = 2
[t=1][h=0][d=1]  →  linear 5   →  valor 11    esperado 10·1 + 1 = 11
[t=2][h=1][d=1]  →  linear 11  →  valor 23    esperado 10·2 + 3 = 23
```

As três batem. O `reshape` não moveu nenhum número: ele só reinterpretou o mesmo array com um agrupamento novo.

> **Armadilha.** Um `Shape` de rank 0 quebrando o cálculo de strides.
>
> Aconteceu neste projeto, na Etapa 6. O `Shape.canonicalStrides` fazia `values.tail.scanRight(...)`. Com um shape de rank 1, colapsar a única dimensão produz um shape de rank 0, com array vazio. E `.tail` de um array vazio lança exceção em Scala.
>
> Por que passou despercebido: nenhum uso anterior tinha colapsado a única dimensão de um tensor rank 1. Sempre foram tensores de rank 2 ou mais. O caso mais elementar de todos foi o último a ser exercitado.
>
> **Lição geral:** código de shape falha nas pontas, não no meio. Esta etapa acrescenta o rank 4 ao projeto. Teste `H = 1` e teste `T = 1`, porque são as pontas novas que você está criando.

---

## §4. Atenção com dois eixos de lote

Com `Q`, `K` e `V` no formato `[B, H, T, dHead]`, a atenção da Etapa 11 roda sem nenhuma mudança conceitual. O que muda é a contabilidade.

Antes havia um eixo de lote, o `B`. Agora há dois, `B` e `H`. As duas últimas dimensões continuam sendo a matriz de verdade:

| Passo | Operação | Formato |
|---|---|---|
| entrada | — | `[B, T, dModel]` |
| projeção | `x.matmul(W_Q)` | `[B, T, dModel]` |
| divisão | `reshape` + `transpose(1,2)` | `[B, H, T, dHead]` |
| scores | `Q.matmul(K.transpose())` | `[B, H, T, T]` |
| escala e máscara | `/ √dHead`, `+ mask` | `[B, H, T, T]` |
| pesos | `softmax(dim = 3)` | `[B, H, T, T]` |
| saída por cabeça | `P.matmul(V)` | `[B, H, T, dHead]` |
| concatenação | `transpose(1,2)` + `reshape` | `[B, T, dModel]` |
| projeção final | `C.matmul(W_O)` | `[B, T, dModel]` |

Duas observações sobre essa tabela.

**O `transpose()` sem argumentos continua certo.** O default é `(rank−2, rank−1)`, decidido na Etapa 3 exatamente para isto. Num tensor rank 4, ele troca `T` com `dHead`, e entrega `[B, H, dHead, T]`. É o que `Q·Kᵀ` pede.

**A máscara não ganha eixo.** Ela continua `[T, T]`. O broadcasting da Etapa 3 §4 a alinha à direita e a replica sobre `B` e `H`. Todas as cabeças usam a mesma máscara causal, porque causalidade é propriedade da sequência, não da cabeça.

**O exemplo.** Tome duas cabeças com `dHead = 2` e `T = 3`, já projetadas:

```
cabeça 0:  Q = [[1,0], [0,1], [1,1]]    K = [[1,0], [0,1], [1,1]]    V = [[2,0], [0,3], [1,1]]
cabeça 1:  Q = [[0,1], [1,0], [1,-1]]   K = [[0,1], [1,0], [-1,1]]   V = [[5,1], [1,5], [3,3]]
```

A última posição vê as três anteriores. Na cabeça 0, com escala `√2 ≈ 1.41421`:

```
scores / escala = [0.70711, 0.70711, 1.41421]
P               = [0.24826, 0.24826, 0.50349]        soma 1.0
y               = [1.00000, 1.24826]
```

Na cabeça 1, com o **mesmo** token de entrada:

```
scores / escala = [-0.70711, 0.70711, -1.41421]
P               = [0.17837, 0.73368, 0.08795]        soma 1.0
y               = [1.88938, 4.11062]
```

Compare as duas linhas de `P`. A cabeça 0 concentra metade do peso na própria posição. A cabeça 1 concentra 73% na posição 1. É a mesma sequência, lida de dois jeitos, e é o ponto inteiro desta etapa.

> **Armadilha.** Uma suíte verde inteira, com um caminho novo do `matmul` completamente quebrado.
>
> Aconteceu neste projeto, na Etapa 11. O `matmul` ganhou o par de ranks `(3,2)` para atender o `Linear` com entrada de sequência. A suíte tinha 115 testes verdes, e nenhum deles passava por esse caminho.
>
> Por que passou despercebido: os testes existentes cobriam `(2,2)` e `(3,3)`, que continuavam corretos. Um caminho novo sem teste novo é invisível — a contagem de testes sobe, a cobertura do código novo não.
>
> **Lição geral:** ao estender o núcleo, escreva o teste do caminho novo antes de usá-lo na camada. Esta etapa reescreveu o `matmul` para lote de rank qualquer. O caminho novo nasceu sem cobertura, e ganhou nove testes antes de qualquer camada usá-lo.

---

## §5. A volta: `transpose` antes de `reshape`

> **Guia visual.** O caminho de volta, e o que o `reshape` sozinho produz: [`heads.html`](heads.html), FIG 3.

A saída da atenção é `[B, H, T, dHead]`. O bloco precisa devolver `[B, T, dModel]`. A concatenação é o caminho de ida, ao contrário:

```
[B, H, T, dHead]  --transpose(1,2)-->  [B, T, H, dHead]  --reshape-->  [B, T, dModel]
```

A ordem é obrigatória, pelo mesmo motivo da §3. O `reshape` funde as duas últimas dimensões em ordem de memória. Fundir `H` com `dHead` só produz o vetor certo quando as duas já estão adjacentes **dentro** de cada posição `t`.

> **Definição — concatenação de cabeças.** Empilhar as saídas das `H` cabeças lado a lado, formando um vetor de tamanho `dModel` por posição. No código isso é um `reshape`, não uma operação de concatenação — as fatias já estão na memória, na ordem certa.

Use os mesmos rótulos da §3, agora no sentido inverso. Se a cabeça 0 devolve `[[0,1], [10,11], [20,21]]` e a cabeça 1 devolve `[[2,3], [12,13], [22,23]]`, o `transpose` reagrupa por token e o `reshape` costura:

```
t=0:  [0, 1, 2, 3]
t=1:  [10, 11, 12, 13]
t=2:  [20, 21, 22, 23]
```

Que é o `X` da §3, de volta. Ida e volta se cancelam, e isso dá um teste barato: divida e concatene sem atenção no meio, e exija igualdade exata com a entrada.

**Um detalhe de custo que vale saber.** O `reshape` deste projeto chama `contiguous` antes de reinterpretar o array. Na ida isso é grátis, porque a entrada já é contígua. Na volta não é: o tensor acabou de ser transposto, e as strides não são canônicas. Então a concatenação **copia** o tensor inteiro.

Isso é correto e é o que o PyTorch também faz. Vale saber que existe, porque é a única cópia obrigatória do bloco.

> **Confira você mesmo.** Por que o `reshape` da ida não copia, e o da volta copia?
>
> <details><summary>Resposta</summary>
>
> Porque `contiguous` só copia quando as strides não são as canônicas. Na ida, `X` vem de um `matmul`, que produz um tensor recém-criado em ordem canônica — `reshape` só troca a interpretação. Na volta, o `transpose(1,2)` devolve uma view com as strides trocadas, e a ordem física deixa de bater com a lógica. Aí `contiguous` tem que materializar o array na ordem nova. É o mesmo mecanismo da Etapa 1: `transpose` é sempre uma view, e quem precisa de ordem canônica paga a cópia.
> </details>

---

## §6. `W_O`: onde as cabeças conversam

> **Guia visual.** As setas que atravessam a fronteira entre as cabeças: [`heads.html`](heads.html), FIG 4.

Sem a projeção final, as cabeças nunca se falam. A coordenada 0 da saída viria da cabeça 0, e só dela, para sempre. As `H` atenções seriam `H` modelos independentes rodando no mesmo tensor.

`W_O` é o que junta. Ele é `[dModel, dModel]`, e cada coordenada da saída passa a ser uma combinação linear de **todas** as coordenadas de **todas** as cabeças.

> **Definição — projeção de saída (`W_O`).** A matriz aplicada à concatenação. Ela mistura as cabeças entre si e devolve o resultado ao formato `[B, T, dModel]`, para que o bloco possa ser empilhado.

**O exemplo.** Continue com as duas cabeças da §4. Na última posição, a concatenação é:

```
C = [1.00000, 1.24826, 1.88938, 4.11062]
     └── cabeça 0 ──┘  └── cabeça 1 ──┘
```

Tome um `W_O` que mistura de propósito. A linha `k` diz para onde a coordenada `k` da concatenação se espalha:

```
        col0  col1  col2  col3
lin0  [   1,    0,    0,    0 ]     ← cabeça 0
lin1  [   0,    1,    0,    0 ]     ← cabeça 0
lin2  [   2,    0,    1,    0 ]     ← cabeça 1, chega na coluna 0
lin3  [   0,   -1,    0,    1 ]     ← cabeça 1, chega na coluna 1
```

A saída da última posição fica:

```
Y = [4.77876, -2.86237, 1.88938, 4.11062]
```

Confira a primeira coordenada posição a posição:

```
Y[0] = 1 · 1.00000  +  2 · 1.88938  =  4.77876
                          └── da cabeça 1 ──┘
```

Sem `W_O`, `Y[0]` seria `1.00000`, a saída crua da cabeça 0. Com `W_O`, o que a cabeça 1 descobriu chega na coordenada 0. A segunda coordenada mostra o mesmo com sinal negativo: `1.24826 − 4.11062 = −2.86237`.

Esse é o papel de `W_O`, e é por isso que ele é um parâmetro treinado e não uma constante. O modelo aprende **como** combinar o que cada cabeça encontrou.

---

## §7. Backward

O backward do bloco é a composição de peças que já existem. Nenhuma fórmula nova aparece aqui.

`reshape` e `transpose` ganharam backward próprio na Etapa 3. O de `reshape` reordena o gradiente de volta ao formato antigo. O de `transpose` troca os mesmos dois eixos. Nenhum dos dois faz conta: eles movem números.

A atenção interna é idêntica à da Etapa 11 §7, com um eixo de lote a mais. As fórmulas `dV = Pᵀ·dY` e `dP = dY·Vᵀ` valem por cabeça, e o eixo `H` só acompanha.

O que é novo é a projeção final. Ela é um `matmul` comum, então vale a Etapa 3 §6:

```
dC   = dY · W_Oᵀ
dW_O = Cᵀ · dY
```

**O exemplo.** Use a concatenação da §6 e um gradiente vindo de cima com valores bem separados, para que trocas de posição não se escondam:

```
dY  [T, dModel]
   [1,  10,  100,  1000]
   [2,  20,  200,  2000]
   [3,  30,  300,  3000]
```

Com o `W_O` da §6:

```
dC = dY · W_Oᵀ
   [1, 10, 102,  990]
   [2, 20, 204, 1980]
   [3, 30, 306, 2970]
```

Confira a primeira linha à mão. A coordenada 2 de `dC` recolhe a linha 2 de `W_O`, que é `[2, 0, 1, 0]`:

```
dC[0][2] = 1·2 + 10·0 + 100·1 + 1000·0 = 102
dC[0][3] = 1·0 + 10·(−1) + 100·0 + 1000·1 = 990
```

As duas batem com a tabela. E `dW_O` soma sobre as posições, porque `W_O` é compartilhado pelas `T` posições:

```
dW_O[0][0] = 2.00000·1 + 0.66048·2 + 1.00000·3 = 6.32095
```

Depois disso, `dC` volta às cabeças pelo caminho inverso da §5. As colunas 0 e 1 vão para a cabeça 0, as colunas 2 e 3 para a cabeça 1:

```
cabeça 0:  [[1, 10], [2, 20], [3, 30]]
cabeça 1:  [[102, 990], [204, 1980], [306, 2970]]
```

O fatiamento é o `reshape` e o `transpose` fazendo o trabalho deles, em sentido contrário. Nenhuma soma acontece nessa passagem — cada elemento de `dC` tem exatamente um destino.

Há um único ponto onde o gradiente **soma**: a entrada `X`. Ela alimenta `W_Q`, `W_K` e `W_V`, então `dX` recebe três contribuições. É o mesmo acúmulo que a Etapa 2 resolveu com o `Gradient` que só acumula, nunca sobrescreve.

> **Confira você mesmo.** No fatiamento acima, por que nenhuma soma acontece, se na entrada `X` acontece?
>
> <details><summary>Resposta</summary>
>
> Porque a divisão em cabeças é uma bijeção. Cada elemento de `C` veio de exatamente um elemento de uma cabeça, então o gradiente volta um para um. Já `X` é usado três vezes no forward, em três projeções diferentes. A regra da cadeia soma sobre todos os usos de uma variável, e `X` tem três. O número de contribuições no backward é o número de arestas que saem do nó no grafo.
> </details>

---

## §8. Quais vieses sobrevivem

O `Linear` da Etapa 8 vem com viés. Aplicado sem pensar, o bloco teria quatro: `b_Q`, `b_K`, `b_V` e `b_O`. Dois deles não servem para nada, e dá para provar isso.

**`b_K` não pode aprender.** O score é `q_i · k_j`. Somar uma constante a todas as keys acrescenta `q_i · b_K` a cada score da linha `i`. Esse termo não depende de `j`. Ele desloca a linha inteira pelo mesmo valor, e o softmax é invariante a deslocamentos constantes.

Confira. Na cabeça 0 da §4, some `5` a todas as coordenadas de todas as keys:

```
K original:      scores = [0.70711, 0.70711, 1.41421]
                 P      = [0.2482550783, 0.2482550783, 0.5034898435]

K + 5:           scores = [7.77817, 7.77817, 8.48528]
                 P      = [0.2482550783, 0.2482550783, 0.5034898435]
```

Os scores mudaram muito. Os pesos não mudaram em nenhum dígito. Se a saída não depende de `b_K`, então `dbK = 0` sempre.

**`b_V` é redundante com `b_O`.** Como cada linha de `P` soma `1`, um viés somado a todos os values atravessa a média ponderada intacto:

```
P · (V + 1·b_V) = P·V + P·1·b_V = P·V + b_V
```

Verificado com `b_V = [0.5, −2.0]` na cabeça 0:

```
y sem b_V = [1.00000, 1.24826]
y com b_V = [1.50000, −0.75174]
diferença = [0.5, −2.0]        ← exatamente b_V, nas duas coordenadas
```

O viés vira uma constante somada a toda posição. Essa constante passa por `W_O` e aterrissa no mesmo lugar onde `b_O` já está. A saída depende de `b_V` apenas através de `c·W_O + b_O`, com `c` a concatenação dos `b_V` das cabeças. São dois parâmetros para uma coisa só.

Os dois casos têm a mesma raiz. São as duas propriedades estruturais da linha do softmax: ela é invariante a deslocamento, e ela soma `1`.

**`b_Q` sobrevive**, e a assimetria é instrutiva. Deslocar as queries acrescenta `b_Q · k_j` ao score. Esse termo **varia** com `j`, porque cada key é diferente. Ele muda a distribuição de verdade, e por isso `b_Q` aprende.

A decisão deste projeto é desligar o viés onde ele é provadamente inútil, e só aí. Na prática: `useBias = false` em `W_K` e em `W_V`, viés mantido em `W_Q` e `W_O`.

> **Armadilha.** O gradient check reprovando um backward correto.
>
> Aconteceu neste projeto, na Etapa 11. O check de `b_K` falhava em 8 de 10 execuções, com erro relativo `1.1e-3`. O forward estava certo, o backward estava certo, e o parâmetro tinha gradiente verdadeiro zero.
>
> Por que é perigoso: a métrica do `Gradcheck` é relativa, com piso `1e-8` no denominador. Com o gradiente analítico em `1e-16` e o ruído de diferenças finitas em `1e-11`, a divisão dá `~1e-3`. O veredito parece um bug de gradiente, e manda você procurar no lugar errado.
>
> **Lição geral:** antes de caçar um bug com erro na casa de `1e-3`, meça a magnitude absoluta do gradiente. Se ele for zero de verdade, o teste certo é `grad == 0` com tolerância absoluta, não o check numérico.

---

## §9. Implementação: contagem, formatos e testes

**A validação que vem primeiro.** `dModel % H == 0` é pré-condição de argumento do construtor, então é `require`, pela regra da Etapa 7. A mensagem deve citar os dois números, porque quem erra isso erra escolhendo `H`.

**O núcleo já atende.** Esta etapa reescreveu o `matmul`, que deixou de ter caminho separado por rank. A regra passou a ser uma só: as duas últimas dimensões são a matriz, todas as anteriores são lote. O laço percorre as posições da saída e lê os operandos pelas strides.

Três consequências importam aqui:

1. **Rank 4 funciona sem nenhum caso especial.** Rank 6 também — não existe teto, porque nada no código conta dimensões de lote.
2. **Entrada não contígua não é copiada.** `Q` e `K` chegam recém-transpostos, e o `matmul` os lê pelas strides.
3. **O segundo operando pode ser rank 2**, e aí a mesma matriz vale para todo o lote, com o gradiente somando sobre ele. É o caminho das projeções.

**A rota descartada, e por quê.** Achatar `[B, H, T, dHead]` em `[B·H, T, dHead]` antes de multiplicar também funciona, e é o que o PyTorch faz por baixo. A diferença está no preço do `reshape`. Lá ele devolve uma *view* quando as strides permitem; aqui ele passa por `contiguous`.

E a divisão em cabeças é justamente um caso em que a view **não** é possível. Fundir as duas primeiras dimensões exigiria `stride(0) = shape(1) · stride(1)`. Depois do `transpose`, `stride(0)` vale `T·H·dHead` e `shape(1)·stride(1)` vale `H·dHead`. Só batem quando `T = 1`.

Ou seja: achatar copiaria os dois operandos a cada multiplicação, mesmo num tensor que a atenção acabou de produzir. Ler pelas strides evita a cópia inteira. Vale notar que o PyTorch paga esse mesmo preço — o código de MHA do nanoGPT escreve `.contiguous()` explícito na concatenação, e ele está lá porque o `view` falharia sem isso.

**A contagem de parâmetros**, para conferir contra o item do checklist:

| `dModel` | `H` | pesos | vieses | total |
|---|---|---|---|---|
| 4 | 2 | `4 × 16 = 64` | `2 × 4 = 8` | 72 |
| 768 | 12 | `4 × 589.824 = 2.359.296` | `2 × 768 = 1.536` | 2.360.832 |

**Como testar.** Os testes da Etapa 11 §8 continuam valendo, e a divisão em cabeças acrescenta quatro.

**Ida e volta.** Divida em cabeças e concatene de volta, sem atenção no meio. O resultado tem que ser igual à entrada, sem tolerância. Este teste sozinho pega a troca de ordem entre `reshape` e `transpose`.

**Cabeças diferentes.** Com `H = 2`, as duas linhas de `P` de uma mesma posição têm que ser diferentes. Se forem iguais, provavelmente as duas cabeças estão lendo a mesma fatia.

**`H = 1` reproduz a Etapa 11.** Com uma cabeça só, o bloco tem que dar o mesmo resultado da atenção simples, a menos de `W_O`. É a melhor regressão possível, porque compara contra código já testado.

**Formatos que não escondem erro.** Escolha `T ≠ dHead` e `H ≠ dHead`. Com `T = dHead` um `transpose` esquecido ainda produz formatos compatíveis, e o erro vira resultado silenciosamente errado.

> **Armadilha.** A suíte testando a versão anterior do arquivo.
>
> Aconteceu neste projeto, em 2026-08-21. Um parâmetro foi removido de uma camada, a suíte foi rodada, e o teste falhou dizendo que a camada ainda tinha o parâmetro antigo. O fonte no disco estava certo. O `.class` em execução tinha sido compilado antes da mudança, e o cache do sbt o servia de novo a cada execução.
>
> Por que é perigoso: `clean` não resolveu, `touch` não resolveu, e apagar o `target/` não resolveu. No caso normal — mudança feita, suíte verde — nada avisa que o teste rodou contra código velho.
>
> **Lição geral:** quando uma mudança parece não ter efeito, compare o fonte com o bytecode antes de duvidar do seu raciocínio. `javap -p -c` mostra o que está rodando de verdade. Esta etapa mexe muito em formato, e formato errado costuma dar erro explícito — o que torna especialmente confuso um erro que não muda quando você corrige o código.

---

## §10. Para onde isso leva

Este é o milestone do projeto. Com a multi-head attention pronta, o coração do transformer está construído.

Vale ver o que já existe. A atenção mistura posições. O LayerNorm da Etapa 10 mantém a escala estável. Falta a peça que processa cada posição isoladamente, e ela é a Etapa 13: o MLP.

A Etapa 14 junta as três num bloco, com as conexões residuais. Esse bloco é o que se repete `N` vezes num GPT, e a Etapa 15 empilha os blocos com o embedding da Etapa 9 na frente.

Repare que o formato de entrada e o de saída deste bloco são o mesmo, `[B, T, dModel]`. Isso não é coincidência. É a propriedade que permite empilhar, e é para isso que `W_O` devolve o resultado a `dModel`. Uma camada que preserva o formato pode ser aplicada quantas vezes você quiser.

---

## Cartão de referência

| Conceito | Resumo |
|---|---|
| ideia central | `H` atenções em paralelo, cada uma com a sua própria distribuição de pesos |
| a limitação que resolve | numa cabeça, todas as coordenadas compartilham a mesma linha de `P` |
| `dHead` | `dModel / H`; exige divisibilidade |
| custo | **zero** — `H · dHead = dModel`, então `H` cabeças custam o mesmo que uma de tamanho `dModel` |
| projeções | quatro matrizes `[dModel, dModel]`: `W_Q`, `W_K`, `W_V`, `W_O` |
| divisão | `reshape [B,T,H,dHead]` → `transpose(1,2)` → `[B,H,T,dHead]` |
| concatenação | `transpose(1,2)` → `reshape [B,T,dModel]`; esta copia |
| máscara | continua `[T, T]`, broadcast sobre `B` e `H` |
| `W_O` | mistura as cabeças; sem ele, cada coordenada da saída vem de uma cabeça só |
| `dC`, `dW_O` | `dY·W_Oᵀ` e `Cᵀ·dY` |
| vieses vivos | `b_Q` e `b_O`; `b_K` tem gradiente zero e `b_V` é redundante com `b_O` |
| total | `4·dModel² + 2·dModel` |
| núcleo | `matmul` sem caminho por rank: duas últimas dimensões são a matriz, o resto é lote |

### As quatro lições que se repetem

1. **Reorganizar não é o mesmo que gastar.** Multi-head não acrescenta parâmetro nenhum. Ele reparte o mesmo orçamento em subespaços, e ganha `H` distribuições no lugar de uma.
2. **A ordem entre `reshape` e `transpose` não é comutativa.** O `reshape` percorre a memória em ordem canônica. Quem funde eixos que não estão adjacentes embaralha os dados sem erro nenhum.
3. **Parâmetro morto se prova, não se adivinha.** `b_K` e `b_V` saem por argumento — invariância a deslocamento e linhas que somam `1` — e não por analogia com outros modelos.
4. **Formato preservado é o que permite empilhar.** O bloco entra e sai em `[B, T, dModel]`, e é por isso que a Etapa 14 pode repeti-lo.
