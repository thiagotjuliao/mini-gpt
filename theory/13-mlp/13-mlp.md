# Etapa 13 — Feed-Forward Network (MLP)

## Antes de começar

### O que você vai construir

Duas camadas lineares com uma GELU no meio, aplicadas a cada posição isoladamente.

| Componente | Formato | O que faz |
|---|---|---|
| `W1`, `b1` | `[dModel, dFF]` e `[dFF]` | projeta cada token para um espaço maior |
| `gelu` | elemento a elemento | a única não-linearidade da camada |
| `W2`, `b2` | `[dFF, dModel]` e `[dModel]` | comprime de volta ao tamanho de entrada |
| `dFF` | `4 · dModel` | a largura da camada escondida |

Entrada `[B, T, dModel]`, saída `[B, T, dModel]`. O formato não muda.

### O que você precisa saber antes

**Da Etapa 3 §4 e §6:** broadcasting e `matmul` com dimensões de lote. O viés é somado por broadcast, e a mesma matriz de pesos serve para todas as posições.

**Da Etapa 5 §1 e §5:** por que duas camadas lineares em sequência colapsam numa só, e a GELU com o seu backward.

**Da Etapa 8:** a camada `Linear` inteira. Esta etapa não implementa álgebra nova — ela compõe duas `Linear` já testadas.

**Da Etapa 12 §9:** a contagem de parâmetros do bloco de atenção. Vamos comparar as duas metades do bloco.

### Onde esta etapa se encaixa

A atenção resolve um problema, e só um: trazer informação de outras posições. Depois dela, cada token carrega um vetor que já viu o contexto.

Falta processar esse vetor.

Pense numa reunião. A atenção é a hora em que cada pessoa escuta as outras. O MLP é a hora em que cada uma volta para a sua mesa e pensa sozinha sobre o que ouviu. Nenhuma informação nova entra nessa segunda etapa. O que acontece ali é transformação, não comunicação.

Essa divisão de trabalho é a arquitetura inteira do transformer. Uma subcamada mistura posições. A outra pensa dentro de cada posição. O bloco da Etapa 14 é só as duas empilhadas.

### Objetivos de aprendizagem

Ao terminar, você deve conseguir:

1. Explicar por que o MLP é aplicado posição a posição, e provar isso com um exemplo.
2. Dizer o que a expansão para `4·dModel` compra, sem responder "mais capacidade".
3. Ler uma coluna de `W1` como uma chave e uma linha de `W2` como um valor.
4. Derivar o backward completo da camada a partir das peças das Etapas 5 e 8.
5. Escrever o teste que reprova um MLP sem a GELU — o gradient check não reprova.

### Convenções de notação

| Símbolo | Significa |
|---|---|
| `B`, `T` | tamanho do lote e número de tokens da sequência |
| `dModel` | tamanho do vetor de cada token na entrada e na saída |
| `dFF` | largura da camada escondida, igual a `4 · dModel` |
| `X` | a entrada — `[B, T, dModel]` |
| `Z` | a pré-ativação, antes da GELU — `[B, T, dFF]` |
| `A` | a ativação, `gelu(Z)` — `[B, T, dFF]` |
| `Y` | a saída — `[B, T, dModel]` |
| `kᵢ` | a coluna `i` de `W1`, lida como chave |
| `vᵢ` | a linha `i` de `W2`, lida como valor |

A convenção `dX ≡ ∂L/∂X` do overview continua valendo. Aqui `Z` é a pré-ativação, e não o número de cabeças — este capítulo não tem cabeça nenhuma.

> **Guia visual.** O caminho dos formatos, o neurônio como par chave-valor e o fluxo do backward: [`mlp.html`](mlp.html). **Exercícios (15 questões):** [`exercises.html`](exercises.html).

---

## §1. Uma posição de cada vez

O MLP recebe um tensor `[B, T, dModel]`. Mas ele não é uma função de tensores.

Ele é uma função de **um vetor**, com `dModel` entradas e `dModel` saídas. O tensor só diz quantas vezes essa função é aplicada.

> **Definição — camada posição a posição.** Uma camada que aplica a mesma função, com os mesmos pesos, a cada vetor da última dimensão, isoladamente. A saída da posição `t` não depende de nenhuma outra posição.

Isso não é uma decisão de projeto separada. Cai de graça das três operações usadas:

1. O `matmul` trata as dimensões da frente como lote. Cada linha da matriz `[B·T, dModel]` encontra a mesma `W1`.
2. O viés é somado por broadcast. A mesma linha `[dFF]` é replicada sobre `B` e `T`.
3. A GELU é elemento a elemento. Ela nem sabe que existem posições.

Duas consequências valem ser lembradas, porque as duas viram teste na §8.

**Tokens iguais produzem saídas iguais.** Se dois vetores de entrada são idênticos, os dois passam pela mesma função e saem idênticos, não importa onde estejam na sequência.

**Permutar a entrada permuta a saída.** Embaralhe os tokens, aplique o MLP, e o resultado é o mesmo que aplicar o MLP e depois embaralhar. Nada aqui depende de ordem.

**O exemplo numérico.** Use `dModel = 2` e `dFF = 4`. A expansão é 2x, e não 4x, só para os números caberem na página — o código usa 4.

```
W1 = [[ 1.0, -2.0,  0.5,  0.0]        b1 = [0.5, -1.0, 0.25, -0.5]
      [ 0.0,  1.0, -1.0,  2.0]]

W2 = [[ 1.0,  0.0]                    b2 = [0.1, -0.2]
      [-1.0,  2.0]
      [ 0.5,  0.5]
      [ 0.0, -1.0]]
```

Três tokens, com o terceiro igual ao primeiro:

```
X = [[1.0, 2.0], [-1.0, 0.5], [1.0, 2.0]]
```

Passo a passo, `Z = X·W1 + b1`, depois `A = gelu(Z)`, depois `Y = A·W2 + b2`:

| posição | `Z` | `A` | `Y` |
|---|---|---|---|
| 0 | `[1.5, -1.0, -1.25, 3.5]` | `[1.39957, -0.15881, -0.13229, 3.49938]` | `[1.59224, -4.08314]` |
| 1 | `[-0.5, 1.5, -0.75, 0.5]` | `[-0.15429, 1.39957, -0.17004, 0.34571]` | `[-1.53888, 2.16841]` |
| 2 | `[1.5, -1.0, -1.25, 3.5]` | `[1.39957, -0.15881, -0.13229, 3.49938]` | `[1.59224, -4.08314]` |

As linhas 0 e 2 são idênticas nas três etapas, e não por arredondamento. São o mesmo cálculo, feito duas vezes.

**A verificação independente.** Permute a entrada para a ordem `[x₁, x₂, x₀]` e recalcule tudo do zero:

```
Y permutado = [[-1.53888, 2.16841], [1.59224, -4.08314], [1.59224, -4.08314]]
```

É exatamente a tabela acima, com as linhas na nova ordem. As duas rotas concordam nas cinco casas.

> **Confira você mesmo.** Se o MLP não olha para nenhuma outra posição, como ele contribui para o modelo entender contexto?
>
> <details><summary>Resposta</summary>
>
> Porque o vetor que chega até ele **já contém** contexto. A atenção da Etapa 12 escreveu ali um resumo do que as outras posições disseram. O MLP processa esse resumo. Um modelo feito só de MLPs empilhados nunca veria contexto nenhum, e um feito só de atenção só saberia calcular médias ponderadas. As duas subcamadas são indispensáveis, por motivos diferentes.
> </details>

---

## §2. A expansão: por que subir para `4·dModel`

Comece pela pergunta anterior. Por que existe uma camada escondida?

A Etapa 5 §1 já respondeu pela metade. Sem a GELU no meio, `X·W1·W2` é o mesmo que `X·W`, com `W = W1·W2` de formato `[dModel, dModel]`. Duas camadas colapsam numa. A expansão não compraria nada, porque o produto de uma `[d, 4d]` por uma `[4d, d]` continua sendo uma `[d, d]`.

Com a GELU no meio, `dFF` passa a significar alguma coisa. E dá para dizer exatamente o quê.

A saída, agrupada por neurônio, é uma soma de `dFF` termos:

```
Y[t] = Σᵢ A[t,i] · vᵢ  +  b2         com vᵢ a linha i de W2
```

Cada neurônio contribui com um múltiplo do seu próprio vetor `vᵢ`. Nada mais. Ou seja, o que o MLP escreve na saída vive sempre no **espaço gerado pelos `dFF` vetores `vᵢ`**, deslocado por `b2`.

Isso dá um limite duro. Se `dFF < dModel`, a camada só consegue escrever num subespaço de dimensão `dFF`, por mais treinada que esteja. Existem direções da saída que ela nunca alcança.

> **Definição — fator de expansão.** A razão `dFF / dModel`. No transformer original ela é 4, com `dModel = 512` e `dFF = 2048`. O GPT-2 manteve o 4, com `768` e `3072`.

**O exemplo numérico.** Tome `dModel = 3` e `dFF = 2`, o caso estreito:

```
W1 = [[ 1.0, 0.0]      b1 = [0.2, -0.3]      W2 = [[1.0,  2.0, 0.0]     b2 = [0.1, 0.1, 0.1]
      [ 0.0, 1.0]                                  [0.0, -1.0, 3.0]]
      [-1.0, 1.0]]
```

Três entradas bem diferentes entre si:

```
X = [[1.0, 2.0, -0.5], [0.5, -1.0, 2.0], [-2.0, 0.5, 1.5]]
```

As saídas, já com `b2` subtraído:

```
Y - b2 = [[ 1.62411,  2.18651, 3.18511]
          [-0.12607, -0.78271, 1.59171]
          [-0.00131, -1.62673, 4.87232]]
```

Três vetores em `R³`. Se fossem independentes, o determinante seria diferente de zero. Ele vale `-5.6e-16`, que é zero na precisão do `Double`. Os três são coplanares.

**A verificação independente.** Escreva cada linha como combinação de `v₀ = [1, 2, 0]` e `v₁ = [0, -1, 3]`:

```
linha 0:  1.62411·v₀ + 1.06170·v₁ = [ 1.62411,  2.18651, 3.18511]
linha 1: -0.12607·v₀ + 0.53057·v₁ = [-0.12607, -0.78271, 1.59171]
linha 2: -0.00131·v₀ + 1.62411·v₁ = [-0.00131, -1.62673, 4.87232]
```

Bate posição por posição. E os coeficientes não são números novos: são exatamente `A[t,0]` e `A[t,1]`, as ativações dos dois neurônios. O determinante nulo não foi coincidência de valores.

Com `dFF = 4` e o mesmo tipo de conta, o determinante das três saídas dá `-259.029`. O plano some.

**E por que 4, e não 2 ou 8?** Não há derivação. O 4 é convenção empírica, herdada do paper original e mantida por todos os GPT desde então. O que a matemática garante é a direção: `dFF` maior compra mais neurônios, cada um com a sua chave, o seu limiar e o seu valor. O preço é linear em `dFF`, como a §7 mostra.

> **Confira você mesmo.** Com `dFF = dModel`, a camada ainda consegue escrever em qualquer direção da saída?
>
> <details><summary>Resposta</summary>
>
> Em princípio sim, desde que os `dModel` vetores `vᵢ` sejam linearmente independentes. Nada garante isso, mas o treinamento também não tem motivo para colapsá-los. O que se perde com `dFF = dModel` não é alcance, é quantidade de detectores: são `dModel` neurônios em vez de `4·dModel`, e portanto um quarto dos padrões distintos que a camada pode reconhecer.
> </details>

---

## §3. Um neurônio é um par chave-valor

Volte à fórmula da §2 e abra também a primeira metade:

```
Z[t,i] = xₜ · kᵢ + b1[i]              kᵢ = coluna i de W1
A[t,i] = gelu(Z[t,i])
Y[t]   = Σᵢ A[t,i] · vᵢ + b2          vᵢ = linha i de W2
```

Lidas juntas, as três linhas contam uma história bem concreta sobre cada neurônio `i`:

1. Ele **mede** o alinhamento entre o token e a sua chave `kᵢ`, com um produto interno.
2. Ele **compara** essa medida com um limiar, que é `−b1[i]`.
3. Ele **escreve** o seu valor `vᵢ` na saída, com intensidade proporcional ao quanto passou do limiar.

> **Definição — memória associativa.** Uma estrutura que guarda pares chave-valor e devolve o valor cuja chave mais se parece com a consulta. O MLP é uma memória associativa suave: em vez de escolher um par, ele soma todos, ponderados pela ativação.

Repare no contraste com a atenção. Lá, as chaves e os valores vêm dos **outros tokens**, e mudam a cada entrada. Aqui, chaves e valores são **parâmetros**: os mesmos para todas as frases, aprendidos no treino. É por isso que se diz que o MLP é onde o modelo guarda o que sabe.

**O exemplo numérico.** Use os pesos da §1 e o token `x₀ = [1, 2]`:

| neurônio | chave `kᵢ` | `xₜ·kᵢ` | `b1[i]` | `Z` | `A` | valor `vᵢ` | contribuição `A·vᵢ` |
|---|---|---|---|---|---|---|---|
| 0 | `[1, 0]` | `1.0` | `0.5` | `1.5` | `1.39957` | `[1, 0]` | `[1.39957, 0.0]` |
| 1 | `[-2, 1]` | `0.0` | `-1.0` | `-1.0` | `-0.15881` | `[-1, 2]` | `[0.15881, -0.31762]` |
| 2 | `[0.5, -1]` | `-1.5` | `0.25` | `-1.25` | `-0.13229` | `[0.5, 0.5]` | `[-0.06614, -0.06614]` |
| 3 | `[0, 2]` | `4.0` | `-0.5` | `3.5` | `3.49938` | `[0, -1]` | `[0.0, -3.49938]` |

Leia a tabela como uma disputa. O neurônio 3 tem a chave mais alinhada com o token, dispara forte, e domina a saída. O neurônio 2 aponta quase na direção oposta, e a contribuição dele é 50 vezes menor. Os dois vetores `vᵢ` que eles escreveriam são igualmente válidos — quem decide é a chave.

**A verificação independente.** Some as quatro contribuições e acrescente `b2 = [0.1, -0.2]`:

```
[1.39957 + 0.15881 - 0.06614 + 0.0,  0.0 - 0.31762 - 0.06614 - 3.49938] + b2
= [1.49224, -3.88314] + [0.1, -0.2]
= [1.59224, -4.08314]
```

É a linha 0 da tabela de `Y` da §1, obtida lá por dois `matmul`. A leitura chave-valor não é uma metáfora — é a mesma conta, agrupada por neurônio em vez de por coordenada.

> **Confira você mesmo.** Qual o papel de `b1` nessa leitura, e o que muda se ele for zero?
>
> <details><summary>Resposta</summary>
>
> `b1[i]` é o **limiar** do neurônio: ele dispara quando `xₜ·kᵢ > −b1[i]`. Com `b1 = 0`, o limiar de todo neurônio passa a ser exatamente zero, e a ativação depende só do sinal do produto interno. Cada neurônio responderia a metade do espaço, sempre. Repare no neurônio 1 da tabela: o produto interno dele deu `0.0`, e é o `b1 = −1.0` que o mantém desligado. Sem viés, ele estaria em cima da fronteira.
> </details>

---

## §4. Forward, formato a formato

Nenhuma operação nova. Vale só acompanhar os formatos, porque é onde erro de composição aparece.

| passo | operação | formatos | resultado |
|---|---|---|---|
| 1 | `X.matmul(W1)` | `[B,T,dModel] × [dModel,dFF]` | `[B,T,dFF]` |
| 2 | `+ b1` | `[B,T,dFF] + [dFF]` | `[B,T,dFF]` |
| 3 | `.gelu` | elemento a elemento | `[B,T,dFF]` |
| 4 | `.matmul(W2)` | `[B,T,dFF] × [dFF,dModel]` | `[B,T,dModel]` |
| 5 | `+ b2` | `[B,T,dModel] + [dModel]` | `[B,T,dModel]` |

Dois detalhes do núcleo importam aqui, e os dois já estão implementados.

**O segundo operando é rank 2.** O `matmul` aceita isso e usa a mesma matriz para todo o lote. No backward, o gradiente dela **soma** sobre as `B·T` posições. É o caminho descrito na Etapa 3 §6 e reusado pela Etapa 12 §4.

**O viés alinha pela direita.** `[dFF]` contra `[B,T,dFF]` é o caso mais simples do broadcasting da Etapa 3 §4. No backward, o `unbroadcast` soma sobre `B` e `T`, e devolve um gradiente `[dFF]`.

Os passos 1 e 2 juntos são a `Linear` da Etapa 8. Os passos 4 e 5 também. A camada inteira é `Linear → gelu → Linear`, e não deve reimplementar nada disso.

**O exemplo numérico.** Com `B = 1`, `T = 2`, `dModel = 2` e `dFF = 4`, os formatos ficam:

```
X  [1, 2, 2]  →  Z  [1, 2, 4]  →  A  [1, 2, 4]  →  Y  [1, 2, 2]
```

Contando elementos: 4 na entrada, 8 no meio, 4 na saída. Os pesos são `8 + 4 + 8 + 2 = 22` números, quase três vezes o maior tensor de ativação deste exemplo. Em modelos de verdade essa proporção se inverte. A estrutura, porém, é sempre essa: os parâmetros ficam nas matrizes, e o tensor cresce só na camada escondida.

Os valores, com os mesmos pesos da §1:

```
Z = [[ 1.5, -1.0, -1.25, 3.5], [-0.5,  1.5, -0.75, 0.5]]
A = [[ 1.39957, -0.15881, -0.13229, 3.49938], [-0.15429, 1.39957, -0.17004, 0.34571]]
Y = [[ 1.59224, -4.08314], [-1.53888, 2.16841]]
```

Confira uma posição à mão: `Z[0,0] = 1·1 + 2·0 + 0.5 = 1.5`, e `gelu(1.5) = 1.39957`. A GELU aparece só no passo 3, e é a única coisa não-linear no caminho inteiro.

---

## §5. Backward: composição, nada novo

Sete gradientes, todos já derivados em capítulos anteriores. A tabela sai de `Y = A·W2 + b2` e `Z = X·W1 + b1`, com a Etapa 3 §6 para o `matmul` e a Etapa 5 §5 para a GELU.

| gradiente | fórmula | formato | soma sobre |
|---|---|---|---|
| `dA` | `dY · W2ᵀ` | `[B,T,dFF]` | — |
| `dW2` | `Aᵀ · dY` | `[dFF,dModel]` | `B` e `T` |
| `db2` | soma de `dY` | `[dModel]` | `B` e `T` |
| `dZ` | `dA ⊙ gelu'(Z)` | `[B,T,dFF]` | — |
| `dW1` | `Xᵀ · dZ` | `[dModel,dFF]` | `B` e `T` |
| `db1` | soma de `dZ` | `[dFF]` | `B` e `T` |
| `dX` | `dZ · W1ᵀ` | `[B,T,dModel]` | — |

O símbolo `⊙` é o produto elemento a elemento. Ele aparece porque a GELU é elemento a elemento: a derivada de cada posição multiplica só o gradiente daquela posição.

Repare em quem soma e quem não soma. `W1` e `W2` são compartilhados por todas as `B·T` posições, então cada posição contribui para o gradiente deles. `X` e `Z` têm uma cópia por posição, e cada uma recebe o seu.

**A derivada da GELU precisa de `Z`, não de `A`.** Isso não é detalhe de implementação. A fórmula da Etapa 5 §5 tem `x` explícito em dois lugares, e não existe reescrita em função só da saída. O grafo tem que segurar a pré-ativação até o backward rodar.

**O exemplo numérico.** Mesmos pesos, `T = 2`, e um gradiente que chega de cima com magnitudes bem separadas:

```
dY = [[1, 10], [100, 1000]]
```

Passo a passo:

```
dA        = [[1.0, 19.0, 5.5, -10.0], [100.0, 1900.0, 550.0, -1000.0]]
gelu'(Z)  = [[1.12771, -0.08296, -0.12249, 1.00242], [0.13263, 1.12771, 0.00106, 0.86737]]
dZ        = [[1.12771, -1.57632, -0.67371, -10.02423], [13.26301, 2142.65051, 0.58396, -867.36990]]
```

E daí os cinco gradientes finais:

```
dW2 = [[-14.02903, -140.29027]        db2 = [101.0, 1010.0]
       [139.79835, 1397.98350]
       [-17.13623, -171.36230]
       [ 38.07078,  380.70785]]

dW1 = [[-12.13530, -2144.22682, -1.25767,  857.34568]
       [  8.88693,  1068.17262, -1.05544, -453.73340]]

db1 = [14.39072, 2141.07419, -0.08975, -877.39413]

dX  = [[3.94349, -20.95106], [-4271.74602, 407.32674]]
```

**A verificação independente.** Rodei diferenças finitas centrais em cada entrada de `X` e de `W1`, com `ε = 1e-6`, sobre a perda `L = Σ Y ⊙ dY`. Os oito valores de `dW1` e os quatro de `dX` batem com a tabela em todas as cinco casas decimais. É o gradient check da Etapa 4, à mão, no bloco inteiro.

Vale reparar em `db2 = [101, 1010]`. É a soma direta das colunas de `dY`, sem peso nenhum no meio — o viés final recebe inteiro o gradiente que chegou.

> **Armadilha.** O backward das nove operações unárias lia o valor errado da entrada.
>
> Aconteceu neste projeto, em 2026-08-21, e a `gelu` estava entre as nove. O `data` de um tensor é indexado pelas strides reais. O `gradient` é sempre canônico. O código usava o mesmo índice `i` nos dois, o que pareava a derivada local de uma posição com o valor de outra.
>
> Por que é perigoso: num tensor contíguo as duas indexações coincidem, e tudo funciona. O erro só aparece quando a entrada chega transposta — e aí o forward continua perfeito, porque ele lê pelas strides. O gradient check acusou `0.632` de erro na mesma operação que dava `8.5e-11` com entrada contígua.
>
> **Lição geral:** quando duas indexações convivem no mesmo código, escreva o teste que as separa. Aqui isso significa **toda operação nova precisa de um caso com entrada não contígua**, e formato quadrado não protege, porque o `transpose` troca as strides mesmo com dimensões iguais.

> **Confira você mesmo.** Por que `dW1` e `dW2` somam sobre `B` e `T`, mas `dX` não?
>
> <details><summary>Resposta</summary>
>
> Porque o gradiente tem sempre o formato do tensor a que se refere. `W1` é um tensor só, usado `B·T` vezes, então as `B·T` contribuições se acumulam nele. `X` tem `B·T` vetores distintos, cada um usado uma vez, então cada um recebe o seu gradiente próprio. É a mesma regra do operando compartilhado da Etapa 3 §6.
> </details>

---

## §6. Inicialização: onde o fator 2 do Kaiming vale

A Etapa 8 §5 deduziu a inicialização de Kaiming, também chamada He: `Var(w) = 2 / inputDim`. Vale reler de onde vem o 2.

Ele existe para compensar uma perda **já ocorrida**. A ReLU zera metade do sinal que chega, então a entrada daquela camada tem metade da energia que teria. O 2 devolve isso.

A conclusão incomoda: o 2 pertence a uma camada cuja **entrada** veio de uma ativação. Não a uma camada que é seguida por uma.

O MLP tem uma de cada tipo.

**A primeira `Linear` recebe o fluxo residual.** Esse tensor não passou por ativação nenhuma — ele vem de uma soma residual, normalizada pelo LayerNorm. Aplicar o fator 2 ali **dobra** a variância.

**A segunda `Linear` recebe `A`, que acabou de sair da GELU.** Aqui o 2 é justamente o caso para o qual ele foi desenhado.

**O exemplo numérico.** Suponha `Var(x) = 1` na entrada, com `dModel = 768` e `dFF = 3072`, e Kaiming nas duas camadas. A variância de uma soma de `n` produtos independentes é `n · Var(w) · E[entrada²]`:

```
Var(Z) = 768 · (2/768) · 1        = 2.0
E[A²]  = 0.92214                          com Z ~ N(0, 2), por quadratura numérica
Var(Y) = 3072 · (2/3072) · 0.92214 = 1.84429
```

O MLP quase dobra a escala. E dá para ver onde: o primeiro passo levou `1` a `2`, e o segundo levou `2` a `1.84`. A segunda camada praticamente preservou, porque `E[A²] ≈ 0.46 · Var(Z)` — a GELU corta perto de metade, e o 2 repõe.

**A verificação independente.** Refaça a mesma conta com ReLU no lugar da GELU. Para `z ~ N(0, 2)`, `E[relu(z)²]` vale exatamente `1.0`, metade da variância. Aí `Var(Y) = 3072 · (2/3072) · 1.0 = 2.0`, e a segunda camada preserva de forma exata. Foi assim que Kaiming derivou o fator, e a GELU chega perto o suficiente para herdá-lo.

Trocar a primeira camada para Xavier, com `Var(w) = 1/inputDim`, daria `Var(Z) = 1`, `E[A²] = 0.42519` e `Var(Y) = 0.85`. Bem mais perto de preservar.

**O que o projeto faz hoje.** A `Linear` da Etapa 8 usa Kaiming sempre, e o MLP herda isso nas duas camadas. Não é erro grave, por dois motivos. O LayerNorm renormaliza na fronteira de cada bloco, então nada explode com a profundidade. E o mesmo desvio já existe nas quatro projeções da Etapa 12, onde não há ativação nenhuma entre elas.

Fica anotado como pendência, junto com a segunda parte da história: o GPT-2 escala as camadas que escrevem no fluxo residual — `W_O` e a segunda `Linear` do MLP — por `1/√(2·nLayers)`. Esse fator só faz sentido quando existe profundidade para medir, então a decisão é da Etapa 15.

---

## §7. Contagem de parâmetros e o custo do bloco

Somando as quatro peças:

```
W1: dModel · 4·dModel = 4·dModel²        b1: 4·dModel
W2: 4·dModel · dModel = 4·dModel²        b2:   dModel
                                        ------------------
total: 8·dModel² + 5·dModel
```

Comparando com o `4·dModel² + 2·dModel` do bloco de atenção da Etapa 12 §9:

| `dModel` | MLP | atenção | razão |
|---|---|---|---|
| 4 | 148 | 72 | 2.06 |
| 64 | 33.088 | 16.512 | 2.00 |
| 768 | 4.722.432 | 2.360.832 | 2.00 |

**O MLP tem o dobro dos parâmetros da atenção.** Dois terços de cada bloco moram aqui. É uma surpresa comum: a atenção recebe toda a atenção, e o MLP é quem carrega o modelo.

O termo `8·dModel²` domina, e os vieses somem na comparação. Com `dModel = 768` eles são `3.840` de `4.722.432`, ou 0,08%.

**A verificação independente.** Vale conferir a fórmula contra um modelo real. O GPT-2 small tem `dModel = 768`, 12 blocos, vocabulário de 50.257 tokens e contexto de 1.024:

```
MLP por bloco       4.722.432        ← nossa fórmula, exata
atenção por bloco   2.362.368        ← com os 4 vieses que o GPT-2 usa
2 LayerNorm             3.072
                   ------------
bloco               7.087.872   × 12 = 85.054.464

embedding de token  38.597.376        50.257 × 768
embedding posicional   786.432         1.024 × 768
LayerNorm final          1.536
                   ------------
total              124.439.808
```

É o "GPT-2 124M" que aparece nos papers, ao número exato. A nossa contagem de MLP bate com a dele na casa das unidades.

O nosso bloco de atenção fica `1.536` mais leve por bloco, porque a Etapa 12 §8 provou que `b_K` e `b_V` são parâmetros mortos e os removeu. Sobre 12 blocos, são `18.432` parâmetros a menos — e nenhuma capacidade a menos.

---

## §8. Implementação e como testar

**A assinatura.** Uma classe `MLP(dModel: Int, expansion: Int = 4)`, com `dFF = dModel * expansion` calculado no construtor. Dois campos privados do tipo `Linear`, e um `parameters` que concatena os dois na ordem. O `expansion` como parâmetro com default deixa os testes usarem valores pequenos sem tocar no padrão.

**As validações.** Pela regra da Etapa 7, são pré-condições de argumento, então `require`. Três, sendo duas no `forward`: `x.rank >= 2`, porque o `matmul` exige matriz dos dois lados, e `x.shape.last == dModel`. A terceira é `expansion >= 1`, no construtor. Repare que o MLP **não** precisa exigir rank 3, ao contrário da `MultiHeadAttention` — ele não tem eixo de posição para interpretar, e funciona igual sobre `[T, dModel]`.

**Não reimplemente a `Linear`.** A tentação existe, porque `x.matmul(W1) + b1` é uma linha. Mas aí a inicialização, a validação e o `parameters` passam a existir em dois lugares.

Oito testes cobrem a camada. Os cinco primeiros são os de sempre; os três últimos são os que esta etapa acrescenta.

**Formato preservado.** Entrada `[B, T, dModel]`, saída `[B, T, dModel]`, com `B`, `T`, `dModel` e `expansion` distintos entre si. `dModel = 4` e `expansion = 4` daria `dFF = 16`, e vários erros de transposição sobreviveriam a isso.

**Comparação contra uma referência em Scala puro.** Calcule `Y` com laços simples, sem tensores, e compare posição a posição.

**Independência entre posições.** Duas verificações, as duas da §1. Repita o mesmo token em duas posições e exija linhas idênticas. Depois permute a entrada e exija a saída permutada.

**Contagem e formato dos parâmetros.** Quatro tensores, formatos `[dModel, dFF]`, `[dFF]`, `[dFF, dModel]`, `[dModel]`, todos com `requiresGradient`. Some os elementos e compare com `8·dModel² + 5·dModel` para alguns valores de `dModel`.

**Gradient check.** Em `x` e nos quatro parâmetros, mais um caso com entrada não contígua.

**A camada não é linear.** Este é o teste que carrega a etapa. Escolha `x` e confira que `f(2x) ≠ 2·f(x)`, ou que `f(a+b) ≠ f(a) + f(b)`. É o princípio da superposição da Etapa 5 §1, usado como contraprova.

**A GELU está no meio, não no fim.** Compare contra duas referências erradas: uma que aplica a ativação na saída, outra que a aplica na entrada. Os resultados têm que diferir das duas.

**O gradiente de `b1`, contra a fórmula.** Este cobre um ponto cego que nenhum teste de forward alcança. Somar `b1` **depois** da GELU produz, numa camada recém-construída, exatamente os mesmos números — porque `b1` nasce zerado, e zero é neutro nas duas ordens. O backward separa as duas na hora: com o viés no lugar certo, `db1 = Σ dZ = Σ dA ⊙ gelu'(Z)`; com ele depois da ativação, `db1 = Σ dA`. A diferença está na regra da cadeia, não no valor do viés, então ela aparece mesmo com `b1 = 0`. Calcule `db1` pela fórmula da §5, em Scala puro, e compare com o que o `backward()` acumulou.

Por que os três últimos importam tanto: **o gradient check aprova um MLP sem GELU nenhuma.** Ele confere se o backward é a derivada correta do forward que você escreveu, e um forward linear é perfeitamente derivável. O teste de formato também passa. A referência em Scala puro só pega o erro se ela mesma tiver a GELU no lugar certo — e é fácil escrever as duas com o mesmo engano.

> **Armadilha.** Um sinal trocado na derivada da GELU, com forward perfeito.
>
> Aconteceu neste projeto, na Etapa 5. A implementação escreveu `(1 + t²)` onde a regra da cadeia pedia `(1 − t²)`, no termo vindo da derivada da tanh. Em `x = 1`, o gradiente saiu `1.50434` em vez de `1.08296` — quase 40% maior.
>
> Por que é perigoso: o forward continua exato, porque o erro está só no backward. Nenhum teste de valor acusa. O modelo até treina, só que na direção errada, e a diferença aparece como "convergiu pior", não como falha.
>
> **Lição geral:** derivada longa é onde o gradient check paga o próprio custo. A Etapa 4 existe por causa deste bug, e a `gelu` é a função mais complexa que este projeto deriva à mão.

**Mutações que a suíte precisa pegar.** Vale rodar cada uma e contar quantos testes falham. Se alguma passar batido, falta teste. Os números abaixo são os medidos na suíte de 18 testes desta etapa:

| mutação | testes que falharam |
|---|---|
| remover a `gelu` | 3 |
| trocar a ordem para `gelu → Linear → Linear` | 3 |
| aplicar a `gelu` depois da segunda `Linear` | 3 |
| `dFF = dModel` em vez de `4·dModel` | 3 |
| somar `b1` depois da `gelu` | **1** |
| usar `W2ᵀ` no lugar de `W2` | 12 |

A linha do `b1` é a que ensina. Na primeira versão da suíte ela falhou em **zero** testes, e a previsão deste capítulo — de que a referência em Scala puro pegaria — estava errada. A referência não pega, porque com `b1 = 0` as duas ordens são a mesma função. Só o teste de gradiente pega, e ele foi escrito por causa dessa medição.

> **Armadilha.** Uma dimensão que vale zero porque a linha estava no lugar errado.
>
> Aconteceu neste projeto, ao escrever esta camada. O `dFF` foi declarado **depois** dos dois `Linear` que o usam:
>
> ```scala
> private val up = Linear(dModel, dFF)     // dFF ainda vale 0 aqui
> private val down = Linear(dFF, dModel)
> val dFF = dModel * expansion
> ```
>
> O corpo de uma classe inicializa de cima para baixo. Quando `up` é construído, `dFF` ainda tem o valor default do campo `Int`, que é zero. O compilador aceita sem nenhum aviso.
>
> Por que é perigoso: os dois `Linear` nasciam com matrizes de zero coluna, o `forward` rodava, e o resultado era um tensor vazio. Nada estourava. A camada só falharia muito depois, longe da causa.
>
> **Lição geral:** dimensão não positiva não pode ser aceita em silêncio. A correção que expõe o erro é um `require(inputDim >= 1)` e `require(outputDim >= 1)` dentro da `Linear` — uma vez, na classe que cria os tensores, e não em cada camada que a usa. Com ele, o mesmo código passa a falhar na construção, com a mensagem dizendo qual dimensão veio zerada.

**Uma nota sobre contiguidade.** No caminho normal, o tensor que chega à `gelu` sempre é contíguo: ele vem de um `matmul` seguido de uma soma, e as duas operações produzem strides canônicas. Isso não dispensa o teste com entrada transposta. O `x` que o chamador passa pode ter qualquer stride, e o `matmul` o lê direto, sem cópia.

---

## §9. Para onde isso leva

Com o MLP pronto, as duas subcamadas do transformer existem. A Etapa 14 as junta.

O bloco é `LayerNorm → atenção → soma residual → LayerNorm → MLP → soma residual`. A Etapa 10 §8 já discutiu a escolha entre pre-LN e post-LN, e o projeto usa pre-LN. Não há peça nova ali — a Etapa 14 é composição, com as conexões residuais como único conceito inédito.

Repare de novo no formato. Entra `[B, T, dModel]`, sai `[B, T, dModel]`. A atenção da Etapa 12 tem a mesma propriedade, e é por isso que as duas podem ser somadas ao fluxo residual e empilhadas quantas vezes se quiser. A Etapa 15 empilha.

Fica também a pendência da §6, agora com endereço: quando houver `nLayers`, decidir se a segunda `Linear` do MLP e a `W_O` da atenção devem nascer escaladas por `1/√(2·nLayers)`.

---

## Cartão de referência

| Conceito | Resumo |
|---|---|
| estrutura | `Linear(dModel, dFF)` → `gelu` → `Linear(dFF, dModel)` |
| `dFF` | `4 · dModel`; convenção empírica, não derivada |
| posição a posição | a mesma função, com os mesmos pesos, em cada vetor da última dimensão |
| leitura chave-valor | coluna `i` de `W1` é a chave, linha `i` de `W2` é o valor, `b1[i]` é o limiar |
| saída | `Y[t] = Σᵢ gelu(xₜ·kᵢ + b1[i]) · vᵢ + b2` |
| limite de `dFF` | a saída vive no espaço gerado pelos `dFF` vetores `vᵢ` |
| `dA`, `dW2`, `db2` | `dY·W2ᵀ`, `Aᵀ·dY`, soma de `dY` |
| `dZ` | `dA ⊙ gelu'(Z)`; a derivada precisa de `Z`, não de `A` |
| `dW1`, `db1`, `dX` | `Xᵀ·dZ`, soma de `dZ`, `dZ·W1ᵀ` |
| parâmetros | `8·dModel² + 5·dModel` — o dobro do bloco de atenção |
| GPT-2 small | `4.722.432` por MLP, `85M` nos 12 blocos, `124.439.808` no total |
| Kaiming | o fator 2 cabe na segunda `Linear`, cuja entrada veio da GELU |
| o teste que carrega a etapa | a contraprova de linearidade — o gradient check aprova um MLP sem GELU |

### As cinco lições que se repetem

1. **Uma camada sem comunicação também é indispensável.** A atenção move informação entre posições. O MLP transforma o que chegou. Um modelo com só uma das duas não funciona.
2. **A largura é o alcance.** A saída do MLP vive no espaço gerado pelos vetores `vᵢ`, então `dFF` não é um número de conveniência: ele limita o que a camada consegue escrever.
3. **O fator 2 do Kaiming é uma compensação, não um enfeite.** Ele repõe o que uma ativação tirou, e portanto pertence à camada que recebe o sinal já cortado — não à que vem antes de cortá-lo.
4. **O gradient check não confere intenção.** Ele confere se o backward é a derivada do forward escrito. Um MLP sem a não-linearidade passa, e só uma contraprova de linearidade reprova.
5. **Parâmetro que nasce zerado esconde erro de ordem.** Somar `b1` antes ou depois da GELU dá o mesmo número enquanto `b1` for zero, e nenhum teste de forward separa os dois. Quem separa é o gradiente, porque a regra da cadeia muda mesmo quando o valor não muda.
