# Etapa 8 — Camada Linear

## Antes de começar

### O que você vai construir

A primeira **camada** do projeto: uma classe que guarda parâmetros treináveis e aplica uma transformação.

| Componente | O que é |
|---|---|
| `W` | matriz de pesos, formato `[inputDim, outputDim]`, treinável |
| `b` | vetor de deslocamento, formato `[outputDim]`, treinável |
| `forward(x)` | `x.matmul(W) + b` |
| `parameters` | a lista `[W, b]`, para o otimizador encontrar |

### O que você precisa saber antes

**Da Etapa 3:** `matmul` e o broadcasting da soma. Esta camada é literalmente essas duas operações.

**Da Etapa 2:** o que significa `requiresGradient`, e por que um parâmetro precisa dele.

**Da Etapa 4:** `Gradcheck.run`, para verificar a camada ao final.

### Onde esta etapa se encaixa

Até aqui, todo tensor era efêmero. Ele nascia numa operação, era consumido pela seguinte, e desaparecia.

> **Definição — parâmetro treinável.** Um tensor que **persiste** entre passadas, e que o otimizador atualiza a cada passo de treino. Ele não é recalculado a partir de nada — ele *é* o que o modelo aprendeu.

Esta é a primeira vez que parâmetros treináveis e autograd se encontram. E é a peça de que todo o resto do modelo é feito: as projeções `Q`, `K`, `V` e `O` da atenção (Etapa 12) e as duas camadas do MLP (Etapa 13) são todas instâncias desta mesma classe, apenas com formatos diferentes.

### Objetivos de aprendizagem

Ao terminar, você deve conseguir:

1. Explicar por que cada coluna de `W` é um neurônio independente.
2. Mostrar, com números, por que inicializar `W` com zeros impede o aprendizado para sempre.
3. Derivar a variância de Kaiming e explicar de onde vem o fator 2.
4. Justificar por que `b` pode ser zero, mas `W` não pode.

> **Guia visual.** O forward como produto escalar por neurônio, e o ciclo do zero-init: [`linear.html`](linear.html). **Exercícios (10 questões):** [`exercises.html`](exercises.html).

---

## §1. A transformação afim

> **Definição — transformação afim.** Uma transformação linear seguida de um deslocamento: `y = xW + b`. É "afim", e não "linear", porque o `b` desloca a origem.

Uma camada linear aplica a mesma transformação afim a cada exemplo do lote.

A leitura importante é por coluna:

> **Definição — neurônio.** Cada **coluna** de `W` é um neurônio: um vetor de pesos do tamanho de `inputDim`, que faz um produto escalar com a entrada e produz um número de saída.

Ter `outputDim` colunas significa ter `outputDim` neurônios. Cada um aprende a detectar uma combinação linear diferente das características de entrada — e as colunas **nunca se misturam** entre si no forward.

Essa independência entre colunas é o que torna a §4 tão importante. Se duas colunas nascem iguais, nada no mecanismo as separa depois.

**Exemplo numérico**, com `inputDim = 2`, `outputDim = 2` e um único exemplo:

```
x = [1, 2]

W = [[0.1, 0.3],          coluna 0 = neurônio 0 = [0.1, 0.2]
     [0.2, 0.4]]          coluna 1 = neurônio 1 = [0.3, 0.4]

b = [0.01, -0.02]

y[0] = 1·0.1 + 2·0.2 + 0.01 = 0.1 + 0.4 + 0.01 = 0.51
y[1] = 1·0.3 + 2·0.4 - 0.02 = 0.3 + 0.8 - 0.02 = 1.08
```

Repare que o cálculo de `y[0]` usa apenas a coluna 0, e o de `y[1]` apenas a coluna 1. Os dois poderiam ser feitos por processos completamente separados.

---

## §2. Formatos e forward

```
x: [B, inputDim]           B exemplos, cada um com inputDim características
W: [inputDim, outputDim]   outputDim neurônios, cada um com inputDim pesos
b: [outputDim]             um deslocamento por neurônio, compartilhado pelo lote

y = matmul(x, W) + b       formato [B, outputDim]
```

O `+ b` soma um vetor de formato `[outputDim]` a um tensor `[B, outputDim]`. É exatamente o caso de broadcasting da Etapa 3 §4 — `b` é replicado ao longo da dimensão do lote, sem cópia real.

Note que `B` **não** aparece na definição da camada. Ele vem do `x` recebido em cada chamada. A mesma camada atende lotes de qualquer tamanho, o que é essencial para reusá-la no treino (Etapa 18) e na geração (Etapa 19, tipicamente com `B = 1`).

---

## §3. Backward: nada novo, só composição

Este é o ponto central da etapa: **a camada linear não tem backward próprio**.

Ela é `matmul` seguido de `add` — duas operações cujo backward já foi derivado, implementado e verificado na Etapa 3. O autograd encadeia as duas automaticamente. `Linear` é uma classe fina que guarda `W` e `b` e chama duas operações existentes.

Reaproveitando as fórmulas da Etapa 3 §6, com `A = x` e `B = W`:

```
dx = dy @ Wᵀ         (era dA = dC @ Bᵀ)
dW = xᵀ @ dy         (era dB = Aᵀ @ dC)
db = dy.sum(dim=0)   (o unbroadcast de b — soma sobre o lote)
```

A fórmula de `db` merece uma frase. Como `b` foi replicado para todos os `B` exemplos no forward, o backward soma de volta ao longo dessa dimensão. Cada exemplo do lote "vota" no gradiente do deslocamento, e os votos somam. É o mesmo mecanismo do Caso 1 da Etapa 3 §4.

> **Confira você mesmo.** Uma camada tem `inputDim = 4` e `outputDim = 3`, e recebe um lote de 10 exemplos. Quais são os formatos de `dx`, `dW` e `db`?
>
> <details><summary>Resposta</summary>
>
> `dx` é `[10, 4]`, `dW` é `[4, 3]` e `db` é `[3]`. O gradiente sempre tem o formato do tensor a que se refere — regra geral desde a Etapa 2. Repare que `dW` e `db` **não** têm dimensão de lote: eles são parâmetros compartilhados por todos os exemplos, e o lote foi somado para dentro deles.
> </details>

> **Armadilha.** A primeira implementação desta camada tinha um método `backward()` vazio, esperando ser preenchido.
>
> Por que é enganoso: o método existir sugere que falta implementar algo ali. Não falta. Deixá-lo vazio convida alguém a "consertar" mais tarde, escrevendo um backward manual que duplicaria — e provavelmente contradiria — o que o autograd já faz.
>
> **Lição geral:** quando uma camada é pura composição de operações já diferenciáveis, ela não deve ter método de backward nenhum. A ausência do método é a documentação de que ele não é necessário.

---

## §4. Por que não inicializar `W` com zeros

Suponha `W` inteiramente zero. O problema não é a saída inicial ser sem graça. É que **ela nunca deixa de ser**.

### O argumento

Se todas as colunas de `W` são iguais, todos os neurônios calculam a mesma saída, para qualquer entrada. Chame isso de "neurônios gêmeos".

Agora acompanhe o gradiente:

- A coluna `j` de `dW` vale `x · dy[j]`. Ela depende só da entrada e do gradiente que chega naquele neurônio.
- Se dois neurônios produzem a mesma saída, qualquer função razoável da saída — a perda, a camada seguinte — devolve o mesmo gradiente para os dois.
- Gradiente igual significa atualização igual.
- Depois da atualização, as colunas continuam idênticas.

O ciclo se fecha e nunca se rompe. Não importa quantos neurônios a camada declare: com pesos simétricos, ela se comporta como se tivesse **um só**.

> **Definição — problema da simetria.** Neurônios que começam idênticos permanecem idênticos por todo o treino, porque recebem sempre o mesmo gradiente. Inicialização aleatória existe para quebrar essa simetria.

### A prova numérica

`inputDim = 2`, `outputDim = 2`, tudo zerado, `x = [1, 2]`:

```
W = [[0, 0],        b = [0, 0]
     [0, 0]]

y = xW + b = [0, 0]        as duas saídas são idênticas, por construção
```

Suponha uma perda simétrica nas duas saídas, `L = (y₀-1)² + (y₁-1)²`:

```
dy[0] = 2(y₀ - 1) = 2(0-1) = -2
dy[1] = 2(y₁ - 1) = 2(0-1) = -2        idêntico, porque y₀ = y₁ = 0
```

Agora o gradiente dos pesos, `dW = xᵀ @ dy`:

```
dW = [[1],[2]] @ [[-2, -2]] = [[1·(-2), 1·(-2)],   = [[-2, -2],
                                [2·(-2), 2·(-2)]]      [-4, -4]]
```

As duas colunas de `dW` são idênticas: `[-2, -4]` e `[-2, -4]`.

Depois de um passo de gradiente com `lr = 0.1`:

```
W ← W - 0.1·dW = [[0.2, 0.2],
                   [0.4, 0.4]]
```

As colunas continuam iguais. E o argumento se repete no próximo passo, e no seguinte, indefinidamente. Por indução, elas nunca se separam.

Inicializar com valores aleatórios quebra o ciclo logo no primeiro elo: cada coluna começa numa direção diferente, então `y[j]` já difere entre neurônios na primeira passada, e todos os gradientes seguintes diferem também.

> **Armadilha.** Dois erros reais desta implementação combinaram-se para produzir exatamente o cenário acima.
>
> O primeiro foi divisão inteira:
>
> ```scala
> val std = Math.sqrt(2 / inputDim)      // 2 e inputDim são Int
> ```
>
> Em Scala, `/` entre dois `Int` trunca. Para qualquer `inputDim > 2`, isso dá `2 / inputDim == 0`, e portanto `std = 0`.
>
> O segundo foi preencher com uma constante:
>
> ```scala
> val data = Array.fill(inputDim * outputDim)(std)
> ```
>
> Isso põe o **mesmo valor** em todo peso. Não é "desvio padrão `std`" — é o próprio `std` repetido.
>
> Juntos, os dois produzem `W` inteiramente zero: o pior caso possível, e exatamente o que esta seção descreve. A correção precisa das duas partes: `2.0 / inputDim` para forçar ponto flutuante, e `Array.fill(n)(std * Random.nextGaussian())` para sortear cada peso de forma independente.
>
> **Lição geral:** um desvio padrão é um **parâmetro** da distribuição, não um valor a ser copiado. E, em Scala, sempre que uma divisão deveria dar fracionário, escreva o literal com ponto decimal.

> **Confira você mesmo.** Como escrever um teste que pegaria o `Array.fill(n)(std)`, sem depender do valor específico de `std`?
>
> <details><summary>Resposta</summary>
>
> Verificando que os pesos de `W` **não são todos iguais**. Basta coletar os valores num `Set` e conferir que ele tem mais de um elemento. O teste não precisa saber qual deveria ser a distribuição — só que existe alguma variação. É exatamente o teste que a suíte desta etapa passou a ter.
> </details>

---

## §5. Kaiming: preservando a variância

Aleatório não basta. A **escala** dos pesos também importa.

Pesos grandes demais fazem as ativações crescerem camada após camada, até estourar. Pequenos demais, elas encolhem até sumir. Nos dois casos o treino falha, e por motivos difíceis de diagnosticar.

> **Definição — inicialização de Kaiming (ou He).** Escolhe a variância dos pesos de forma que a variância da saída fique aproximadamente igual à da entrada — nem cresce, nem encolhe, ao atravessar a camada.

### A derivação

Seja `z = Σᵢ xᵢ·wᵢ` uma saída antes do deslocamento, soma de `inputDim` termos. Assuma `xᵢ` e `wᵢ` independentes, com `E[wᵢ] = 0`.

Essa média zero não é detalhe: é exatamente por isso que os pesos são centrados em zero, e é o que permite o passo seguinte.

```
Var(z) = Σᵢ Var(xᵢ·wᵢ) = inputDim · Var(x) · Var(w)
```

Para manter `Var(z) = Var(x)`, precisamos de `inputDim · Var(w) = 1`:

```
Var(w) = 1 / inputDim
```

Essa é a inicialização de **Xavier** (ou Glorot), pensada para ativações simétricas como a tanh.

Kaiming ajusta para a ReLU. Como a ReLU zera as entradas negativas, em média **metade** das ativações da camada anterior chega como zero. A variância disponível cai pela metade. Para compensar, dobra-se a variância exigida dos pesos:

```
Var(w) = 2 / inputDim          std(w) = √(2 / inputDim)
```

**Exemplo numérico**, com `inputDim = 8`:

```
Xavier:  std = 1/√8   = 0.3536
Kaiming: std = √(2/8) = √0.25 = 0.5
```

Kaiming pede pesos com o dobro da variância de Xavier — consistente com "metade do sinal some na ReLU, então cada peso restante precisa carregar mais".

### A alternativa do GPT-2

Para a GELU, que este projeto usa, o GPT-2 original adota algo mais simples: `W ~ N(0, 0.02²)`. Um desvio padrão **fixo**, independente de `inputDim`.

Isso funciona por dois motivos. A GELU não corta o sinal tão abruptamente quanto a ReLU (compare as curvas em `activations.html`). E o transformer tem conexões residuais (Etapa 14), que estabilizam a variância entre camadas por outro mecanismo.

Qualquer uma das duas serve aqui: Kaiming pela generalidade, ou `N(0, 0.02²)` para ficar mais perto do GPT-2 real.

> **Confira você mesmo.** Duas camadas usam Kaiming: uma com `inputDim = 8`, outra com `inputDim = 512`. Qual delas recebe pesos maiores, e por quê isso faz sentido?
>
> <details><summary>Resposta</summary>
>
> A de `inputDim = 8`, com `std = 0.5` contra `std ≈ 0.0625`. Faz sentido porque a variância da saída acumula sobre **todas** as entradas somadas. Com 512 termos na soma, cada um precisa contribuir muito menos para o total ficar na mesma escala. Quanto mais entradas, menores os pesos — é exatamente o que `2/inputDim` codifica.
> </details>

---

## §6. Por que o deslocamento pode ser zero

O `b` **não** sofre do problema da §4, mesmo inteiramente zerado.

A quebra de simetria já aconteceu em `W`. Cada neurônio calcula uma combinação linear diferente de `x` desde a primeira passada, com pesos aleatórios distintos. O deslocamento apenas move essa saída já diferenciada — somar a mesma constante não reintroduz simetria nenhuma.

Por isso inicializar `b` com zeros é seguro, e é o que toda a literatura faz.

**Exemplo numérico.** Retomando a §1, com `W` aleatório e `b = [0, 0]`:

```
y[0] = 1·0.1 + 2·0.2 + 0 = 0.5
y[1] = 1·0.3 + 2·0.4 + 0 = 1.1
```

As duas saídas já são diferentes — `0.5` contra `1.1` — e a diferença veio inteiramente de `W`. O `b` não teve papel nenhum nisso, e não precisaria ter.

> **Armadilha.** A primeira implementação criou `b` sem `requiresGradient = true`.
>
> Por que é grave: sem essa marca, `b` nunca entra no grafo de autograd. O `backward()` jamais escreve gradiente nele, e o otimizador da Etapa 17 nunca o atualiza. O deslocamento fica congelado em zero para sempre.
>
> E não há sintoma. O forward funciona, o treino roda, a perda até cai — o modelo apenas aprende com menos parâmetros do que deveria.
>
> **Lição geral:** todo parâmetro treinável precisa de `requiresGradient = true` na criação. Um teste que percorre `parameters` e confirma a marca em todos custa três linhas.

---

## §7. `parameters` — a lista para o otimizador

A camada expõe a lista dos tensores que o otimizador deve atualizar:

```scala
val parameters: List[Tensor] = List(W, b)
```

Sem essa coleção, o laço de treino não teria como distinguir **parâmetros** de **ativações**. Ambos têm `requiresGradient = true` e ambos recebem gradiente no backward — mas só os parâmetros devem ser atualizados. As ativações são recalculadas do zero a cada passada.

Todas as camadas futuras seguem o mesmo contrato, e o modelo completo (Etapa 15) coleta os parâmetros de todas as subcamadas recursivamente.

Note que `parameters` é um `val`, não um método. Como `W` e `b` nunca trocam de referência, não há motivo para reconstruir a lista a cada acesso.

---

## §8. Exemplo numérico completo

Reunindo tudo, com `inputDim = 2`, `outputDim = 2` e `B = 2` — um lote de dois exemplos, para que `db` tenha algo real a somar.

```
x = [[1,  2],       W = [[0.1, 0.3],       b = [0.01, -0.02]
     [3, -1]]            [0.2, 0.4]]
```

**Forward**, uma linha por exemplo:

```
y[0,0] = 1·0.1 + 2·0.2 + 0.01     = 0.51
y[0,1] = 1·0.3 + 2·0.4 - 0.02     = 1.08
y[1,0] = 3·0.1 + (-1)·0.2 + 0.01  = 0.11
y[1,1] = 3·0.3 + (-1)·0.4 - 0.02  = 0.48

y = [[0.51, 1.08],
     [0.11, 0.48]]
```

Suponha que chega da camada seguinte o gradiente `dy = [[1, 2], [0.5, -1]]`, com valores distintos por posição.

**`db = dy.sum(dim=0)`** — soma sobre o lote:

```
db = [1 + 0.5,  2 + (-1)] = [1.5, 1]
```

**`dW = xᵀ @ dy`**, aplicando `dW[i,j] = Σ_lote x[lote,i]·dy[lote,j]`:

```
dW[0,0] = 1·1    + 3·0.5     =  2.5
dW[0,1] = 1·2    + 3·(-1)    = -1
dW[1,0] = 2·1    + (-1)·0.5  =  1.5
dW[1,1] = 2·2    + (-1)·(-1) =  5

dW = [[2.5, -1],
      [1.5,  5]]
```

**`dx = dy @ Wᵀ`**, com `Wᵀ = [[0.1, 0.2], [0.3, 0.4]]`:

```
dx[0,0] = 1·0.1   + 2·0.3     =  0.7
dx[0,1] = 1·0.2   + 2·0.4     =  1.0
dx[1,0] = 0.5·0.1 + (-1)·0.3  = -0.25
dx[1,1] = 0.5·0.2 + (-1)·0.4  = -0.3

dx = [[ 0.7,   1.0],
      [-0.25, -0.3]]
```

**Verificação independente de `dW[0,0]`.** Sem usar a fórmula matricial, direto da regra da cadeia: `dW[0,0]` recebe a contribuição de todo exemplo do lote onde `x[·,0]` multiplica algo que chega em `y[·,0]`.

```
dW[0,0] = x[0,0]·dy[0,0] + x[1,0]·dy[1,0] = 1·1 + 3·0.5 = 2.5
```

Bate com o valor da fórmula. Isso confirma que `dW = xᵀ @ dy` não é fórmula mágica — é a soma da regra da cadeia sobre o lote, escrita em notação de matriz.

---

## §9. Para onde isso leva

`Linear` é a primeira camada com estado treinável, mas ela ainda opera sobre vetores genéricos. Não sabe nada sobre tokens, texto ou vocabulário.

A Etapa 9 (Embedding) é a próxima camada com parâmetros, e tem mecânica diferente. Em vez de multiplicar um vetor de entrada por uma matriz, ela **indexa** uma tabela pelo número do token — a saída da Etapa 7.

O elo entre as duas é mais próximo do que parece: multiplicar por um vetor one-hot é equivalente a selecionar uma linha de uma matriz. A Etapa 9 abre demonstrando exatamente isso.

Depois disso, `Linear` volta a aparecer diretamente e em quantidade. A partir da Etapa 12, as projeções `Q`, `K`, `V` e `O` da atenção são todas instâncias desta mesma classe.

---

## Cartão de referência

| Conceito | Resumo |
|---|---|
| transformação afim | `y = xW + b` |
| coluna de `W` | um neurônio; as colunas nunca se misturam |
| formato de `W` | `[inputDim, outputDim]` |
| formato de `b` | `[outputDim]`, broadcastado sobre o lote |
| `dx` | `dy @ Wᵀ` |
| `dW` | `xᵀ @ dy` |
| `db` | `dy.sum(dim=0)` |
| backward próprio | **não existe** — é composição de `matmul` e `add` |
| Xavier | `Var(w) = 1/inputDim` |
| Kaiming | `Var(w) = 2/inputDim`, o dobro, por causa da ReLU |
| GPT-2 | `N(0, 0.02²)`, fixo |
| `b` inicial | zeros — a simetria já foi quebrada por `W` |
| `parameters` | `List(W, b)`, para o otimizador |

### As quatro lições que se repetem

1. **Composição dispensa backward.** Se a camada é feita só de operações já diferenciáveis, não escreva backward nenhum.
2. **Simetria é permanente.** Neurônios que nascem iguais recebem sempre o mesmo gradiente, e nunca se separam.
3. **Divisão entre inteiros trunca.** `2 / inputDim` é zero para qualquer `inputDim > 2`, e nada avisa.
4. **`requiresGradient` esquecido não dá erro.** O parâmetro simplesmente nunca aprende, e a perda até cai — só menos do que deveria.
