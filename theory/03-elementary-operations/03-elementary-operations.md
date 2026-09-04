# Etapa 3 — Operações Elementares com Gradiente

## Antes de começar

### O que você vai construir

Ao final desta etapa, você terá implementado **doze operações** que sabem calcular tanto o resultado quanto o gradiente:

| Grupo | Operações |
|---|---|
| Aritmética | `add`, `sub`, `mul`, `div`, `pow`, `neg` |
| Transcendentais | `exp`, `log` |
| Reduções | `sum`, `mean`, `max` (globais e por dimensão) |
| Estruturais | `reshape`, `transpose` (backward próprio) |
| Álgebra linear | `matmul` (2D e em lote) |
| Utilidade | `clamp` |

Juntas, elas são a caixa de ferramentas completa de que um transformer precisa. Todas as camadas que você vai construir depois — atenção, normalização, feed-forward — são combinações destas doze.

### O que você precisa saber antes

Este capítulo assume três coisas.

**Da Etapa 1 (Tensor):** o que são `shape` e `strides`, e por que `reshape` e `transpose` não copiam dados. Se "stride" não significa nada para você agora, releia `01-tensor.md` §2 antes de continuar.

**Da Etapa 2 (Autograd):** o grafo computacional, a ordenação topológica, e por que o gradiente é acumulado com `+=` em vez de sobrescrito. Esta etapa preenche o que aquela deixou em aberto.

**De cálculo:** derivada, derivada parcial e regra da cadeia. Você não precisa saber demonstrar teoremas. Precisa entender que a derivada mede *quanto a saída muda quando a entrada muda um pouquinho*, e que a regra da cadeia multiplica essas sensibilidades ao longo de uma composição.

### Onde esta etapa se encaixa

A Etapa 2 construiu o **motor** do autograd. Ele sabe percorrer o grafo na ordem certa e chamar a função `_backward` de cada nó. Mas ele não sabe o que essas funções devem fazer.

Pense num carro. A Etapa 2 montou a transmissão: ela sabe levar força das rodas de volta ao motor, na ordem certa. Esta etapa fabrica as peças que de fato geram essa força — uma por operação.

É por isso que esta é a etapa mais longa do projeto. Cada operação exige uma derivação própria.

### Objetivos de aprendizagem

Ao terminar, você deve conseguir:

1. Derivar a regra local de qualquer operação elemento a elemento, sem consultar tabela.
2. Explicar por que o backward do `exp` reusa a saída e o do `log` usa a entrada.
3. Descrever o que o *broadcasting* faz no forward e por que o backward precisa somar.
4. Derivar `dA = dC @ Bᵀ` a partir da definição de multiplicação de matrizes.
5. Reconhecer, num backward escrito por outra pessoa, os erros que este capítulo cataloga.

---

## Como ler este capítulo

O texto usa quatro tipos de bloco destacado. Vale conhecê-los antes.

> **Definição.** Introduz um termo pela primeira vez. Se um termo aparece sem definição em algum ponto, é um defeito do texto — anote e reclame.

> **Armadilha.** Um erro que **realmente aconteceu** durante a implementação deste projeto. Não são erros hipotéticos: cada um custou tempo real de depuração. São o material mais valioso do capítulo, porque marcam exatamente onde a intuição costuma falhar.

> **Confira você mesmo.** Uma pergunta curta, com resposta logo abaixo. Tente responder antes de olhar. Se errar, releia a seção — não siga em frente.

> **Guia visual.** Aponta para o diagrama correspondente, quando existe um.

Cada seção fecha com um exemplo numérico completo. Refaça-o com papel e caneta pelo menos uma vez. Ler uma derivação dá a sensação de entendimento; reproduzi-la é o que produz entendimento de verdade.

**Guias visuais desta etapa:** [`broadcasting.html`](broadcasting.html) e [`matmul.html`](matmul.html). **Exercícios (22 questões):** [`exercises.html`](exercises.html).

---

## Convenções de notação

Antes de qualquer matemática, é preciso combinar a notação. Esta é a maior fonte de confusão em backpropagation, e vale resolvê-la de uma vez.

> **Definição — a notação `d`.** Escrevemos `dA` como abreviação de `∂L/∂A`. Leia: *"o gradiente da perda `L` em relação a `A`"*.

Repare no que isso **não** significa. `dA` não é "a derivada de `A`". É a derivada de outra coisa — a perda — em relação a `A`.

> **Definição — perda (`L`).** A perda é **um único número**. Ela mede o quão errada está a previsão do modelo. Quanto menor, melhor o modelo. Treinar é ajustar os parâmetros para que `L` diminua.

Que `L` seja um número só, e não um tensor, é o que faz todo o resto funcionar. Se `A` é uma matriz 3×4, então `dA` também é uma matriz 3×4: cada posição guarda quanto aquele elemento de `A` influencia o número `L`. A forma do gradiente sempre acompanha a forma do tensor.

Você vai construir `L` de verdade só na Etapa 16. Até lá, os exemplos usam `L = soma de tudo` ou `L = soma ponderada`, que servem para exercitar o mecanismo.

Resumindo a notação usada daqui em diante:

| Símbolo | Significa |
|---|---|
| `A`, `B`, `C` | tensores no forward |
| `dA` | `∂L/∂A` — gradiente da perda em relação a `A` |
| `dC` | o gradiente que **chega de cima**, vindo da operação seguinte |
| `∂C/∂A` | a derivada **local** da operação, isolada do resto do grafo |

---

## A anatomia de uma operação com gradiente

Todas as doze operações seguem o mesmo esqueleto. Vale internalizá-lo agora, porque cada seção seguinte é uma instância dele.

Uma operação com gradiente tem duas metades.

**No forward**, ela calcula o resultado. `C = f(A, B)`.

**No backward**, ela recebe `dC` — quanto a perda responde a cada posição da saída — e precisa devolver `dA` e `dB`.

A conexão entre as duas metades é a regra da cadeia:

```
dA = dC · ∂C/∂A
```

Em palavras: *o gradiente que chega, multiplicado pela derivada local da operação*.

> **Definição — derivada local.** É a derivada da operação isolada, tratando todos os outros tensores como constantes. Ela ignora completamente o resto do grafo.

Essa separação é o que torna o sistema componível. Ao escrever o backward do `mul`, você não precisa saber nada sobre o que vem antes ou depois dele no grafo. Só precisa da fórmula do próprio `mul`. Compor as operações corretamente é responsabilidade do autograd da Etapa 2, não sua.

Então, ao implementar qualquer operação nova, a pergunta é sempre a mesma: **qual é a derivada local, e de quais valores ela depende?**

Essa segunda parte — *de quais valores ela depende* — é onde a maioria dos bugs mora. Guarde a pergunta. Ela reaparece em quase toda seção deste capítulo.

---

## §1. Operações elemento a elemento

> **Definição — elemento a elemento.** Uma operação é elemento a elemento quando `C[i]` depende apenas de `A[i]` e `B[i]` — a mesma posição, nos dois operandos. Nenhuma posição conversa com outra.

Essa propriedade simplifica muito o backward. Se as posições não se misturam no forward, elas também não se misturam no backward. Cada posição pode ser tratada isoladamente.

### As derivadas locais

| Operação | Forward | `∂C/∂A` | `∂C/∂B` |
|---|---|---|---|
| `add` | `A + B` | `1` | `1` |
| `sub` | `A - B` | `1` | `-1` |
| `mul` | `A · B` | `B` | `A` |
| `div` | `A / B` | `1/B` | `-A/B²` |
| `pow` (expoente `p` fixo) | `Aᵖ` | `p·A^(p-1)` | — |
| `neg` | `-A` | `-1` | — |

Duas linhas dessa tabela merecem atenção.

**A linha do `div`, para `B`.** De onde vem o quadrado no denominador? Reescreva a divisão como multiplicação: `A/B = A · B⁻¹`. Agora derive em relação a `B`, com `A` constante. Pela regra da potência, `d(B⁻¹)/dB = -B⁻²`. Multiplicando por `A`, temos `-A·B⁻²`, que é `-A/B²`. O quadrado não é uma escolha arbitrária: é o que a regra da potência produz.

**As linhas de `mul` e `div`.** Repare que a derivada em relação a `A` depende de `B`, e vice-versa. Isso terá uma consequência prática importante. Para calcular esses backwards, é preciso ter guardado os valores dos dois operandos. As operações `add` e `sub` não têm esse problema: a derivada delas é constante.

### Exemplo numérico

Vamos aplicar as quatro operações binárias aos mesmos números. Isso deixa a comparação direta.

```
A  = [ 6, -2]
B  = [ 3,  4]
dC = [ 1, 10]        ← gradiente vindo de cima
```

Escolhi `dC = [1, 10]` de propósito, com valores diferentes entre si. Um `dC` uniforme (todo `1`) esconderia erros de indexação, porque qualquer posição trocada daria o mesmo resultado.

**`add`:**
```
C  = [6+3, -2+4] = [9, 2]

∂C/∂A = 1  →  dA = dC · 1 = [1, 10]
∂C/∂B = 1  →  dB = dC · 1 = [1, 10]
```
O gradiente atravessa a soma sem alteração nenhuma. Faz sentido: se você aumentar `A[0]` em uma unidade, `C[0]` também sobe exatamente uma unidade.

**`sub`:**
```
C  = [6-3, -2-4] = [3, -6]

dA = dC · 1  = [ 1,  10]
dB = dC · -1 = [-1, -10]
```
O sinal do segundo operando inverte. Aumentar `B` faz `C` **diminuir**, então o gradiente chega com o sinal trocado.

**`mul`:**
```
C  = [6·3, -2·4] = [18, -8]

dA = dC · B = [1·3, 10·4]    = [3, 40]
dB = dC · A = [1·6, 10·(-2)] = [6, -20]
```
Aqui cada gradiente usa o **outro** operando. Note `dB[1] = -20`: é negativo porque `A[1]` é negativo. Aumentar `B[1]` faz `C[1]` ficar mais negativo.

**`div`:**
```
C  = [6/3, -2/4] = [2, -0.5]

dA = dC / B        = [1/3, 10/4]              = [0.3333, 2.5]
dB = dC · (-A/B²)  = [1·(-6/9), 10·(2/16)]    = [-0.6667, 1.25]
```
Confira `dB[1]`: `A[1] = -2`, então `-A/B² = -(-2)/16 = +0.125`. Multiplicado por `dC[1] = 10`, dá `1.25`. O sinal positivo está certo — como `A[1]` é negativo, aumentar `B[1]` aproxima `C[1]` de zero, ou seja, faz `C[1]` **crescer**.

**Verificação independente.** Vamos conferir `dA[1]` do `mul` por outro caminho, sem usar a fórmula. Se `dC = [1, 10]`, a perda correspondente é `L = 1·C[0] + 10·C[1]`. Substituindo `C[1] = A[1]·B[1]`, temos `L = 1·C[0] + 10·A[1]·B[1]`. Derivando diretamente em relação a `A[1]`: `∂L/∂A[1] = 10·B[1] = 10·4 = 40`. Bate com `dA[1] = 40` da fórmula.

> **Confira você mesmo.** Sem olhar a tabela: numa operação `C = A - B`, o gradiente `dB` chega com sinal invertido. Por que `dA` **não** inverte?
>
> <details><summary>Resposta</summary>
>
> Porque `∂C/∂A = +1` e `∂C/∂B = -1`. Aumentar `A` aumenta `C` na mesma proporção; aumentar `B` **diminui** `C`. O sinal do gradiente segue o sinal da derivada local, e só o segundo operando entra subtraindo.
> </details>

### Divisão por zero

Uma decisão de projeto que vale registrar: `div` **não** tem proteção contra divisão por zero.

Se você dividir por zero, o resultado segue o padrão IEEE 754 de ponto flutuante: `x/0.0` dá `Infinity`, e `0.0/0.0` dá `NaN`. É o mesmo comportamento de PyTorch e NumPy na divisão elementar.

Isso é deliberado, não um descuido. Adicionar um `epsilon` de proteção aqui mascararia erros reais em outros lugares. Estabilidade numérica é tratada onde ela de fato importa: no `log` (§2) e no `LayerNorm` (Etapa 10, com `ε = 1e-5` somado à variância).

---

## §2. Funções transcendentais

Duas operações, e uma lição sobre *de quais valores a derivada depende* — a pergunta que ficou pendente na seção sobre a anatomia.

### `exp`

Forward: `C = e^A`.

A função exponencial tem uma propriedade única: **ela é a própria derivada**. `d(eˣ)/dx = eˣ`.

Isso tem uma consequência prática direta. A derivada local `∂C/∂A` é igual a `C` — o valor que o forward **já calculou**. Você não precisa recalcular nenhuma exponencial no backward. Basta reusar a saída.

```
dA = dC · C          ← C é a SAÍDA, não a entrada
```

> **Armadilha.** Este é um erro real, cometido durante a implementação deste projeto. A primeira versão do backward do `exp` escreveu `dA = dC · A`, usando a **entrada** em vez da saída.
>
> Por que é perigoso: o forward continua perfeito. `exp` calcula os valores certos, os testes de forward passam, nada trava. Só os gradientes saem errados — e gradiente errado não avisa. O modelo simplesmente treina mal, sem nenhuma mensagem de erro.
>
> Os números do exemplo abaixo mostram a diferença de magnitude. Não é um desvio sutil.

**Exemplo numérico:**

```
A  = [0, 1, 2]
dC = [1, 10, 100]

C = e^A = [1.0, 2.71828, 7.38906]

CERTO   dA = dC · C = [1·1.0, 10·2.71828, 100·7.38906] = [1.0, 27.183, 738.91]
ERRADO  dA = dC · A = [1·0,   10·1,       100·2      ] = [0,   10,     200   ]
```

A posição 0 é a mais reveladora. Com a entrada `A[0] = 0`, a versão errada zera o gradiente completamente. Mas `e⁰ = 1`, então o gradiente correto é `1`, não `0`. A versão com bug corta o fluxo de gradiente num ponto onde ele deveria passar inteiro.

### `log`

Forward: `C = ln(A)`.

Derivada: `d(ln x)/dx = 1/x`. Então:

```
dA = dC / A          ← A é a ENTRADA
```

Compare com `exp`. Aqui a derivada depende da **entrada**, não da saída. Isso não é uma inconsistência: é só o que as duas funções são. Vale a comparação lado a lado, porque é exatamente a pergunta que a seção de anatomia mandou guardar:

| Operação | A derivada depende de | Reusa o valor do forward? |
|---|---|---|
| `exp` | da saída `C` | sim |
| `log` | da entrada `A` | não |

**Cuidado numérico com `log`.** A função diverge quando `A → 0⁺` e não é definida para `A ≤ 0`. Na prática isso importa muito na Etapa 6 (softmax), onde probabilidades muito pequenas são comuns. A solução não é somar um `epsilon`: é reorganizar a álgebra para nunca precisar do `log` de um número minúsculo. Essa técnica se chama *log-sum-exp*, e você vai construí-la na Etapa 6.

**Exemplo numérico:**

```
A  = [1, 2, 4]
dC = [1, 10, 100]

C  = ln(A) = [0, 0.69315, 1.38629]

dA = dC / A = [1/1, 10/2, 100/4] = [1, 5, 25]
```

Repare como o gradiente **encolhe** conforme `A` cresce. Em `A = 4`, um aumento de uma unidade quase não move o logaritmo — a curva já está bem achatada ali. O gradiente reflete isso.

> **Confira você mesmo.** Você está implementando `sigmoid` (Etapa 5), cujo forward é `σ(x) = 1/(1+e⁻ˣ)`, e cuja derivada é `σ(x)·(1-σ(x))`. Essa derivada depende da entrada ou da saída?
>
> <details><summary>Resposta</summary>
>
> Da **saída**. A fórmula é escrita inteiramente em termos de `σ(x)`, que é o valor que o forward já calculou. Assim como `exp`, o backward pode reusar a saída e não precisa reavaliar nenhuma exponencial. Esse padrão se repete várias vezes na Etapa 5 (`sigmoid` e `tanh` fazem isso; `gelu` não).
> </details>

---

## §3. Reduções

> **Definição — redução.** Uma operação que colapsa vários valores num só. `sum` colapsa em uma soma, `mean` numa média, `max` no maior valor.

Reduções invertem o padrão das seções anteriores. Até aqui, a saída tinha o mesmo formato da entrada. Agora ela é **menor** — às vezes um único número.

Isso muda o backward de forma característica. No forward, muitas entradas viram uma saída. No backward, o caminho é inverso: **um gradiente precisa se espalhar de volta para muitas posições**.

### `sum`

Forward: `C = Σᵢ A[i]`, um escalar.

Derivada local: `∂C/∂A[i] = 1`, para todo `i`. Cada elemento contribui aditivamente e de forma independente.

Backward: `dA[i] = dC` — o mesmo escalar, copiado para **todas** as posições.

```
A  = [1, 4, 2]
C  = 7
dC = 10

dA = [10, 10, 10]
```

> **Armadilha.** Reduções mudam o formato da saída, e é fácil esquecer disso ao construir o tensor de resultado. A primeira versão de `sum`, `mean` e `max` neste projeto construiu a saída com o `shape` da **entrada**, quando o correto é `shape = [1]`.
>
> Como o `data` de saída tem um elemento só, isso violava a invariante interna do `Tensor` e estourava uma exceção para qualquer entrada com mais de um elemento. Só não foi percebido antes porque os primeiros testes manuais usaram tensores de um elemento — onde os dois formatos coincidem por acidente.
>
> **Lição geral:** teste com formatos que não colapsam. Um tensor 1×1 valida muito menos do que parece.

### `mean`

Forward: `C = (Σᵢ A[i]) / n`.

A média é a soma dividida por uma constante. Por linearidade, a derivada é a da soma dividida pela mesma constante:

```
∂C/∂A[i] = 1/n        →        dA[i] = dC / n
```

Intuição: se a média subiu uma unidade, e todos os `n` elementos contribuíram igualmente, cada um "deveria" ter subido `1/n`.

```
A  = [1, 4, 2]
C  = 7/3 ≈ 2.333
dC = 10

dA = [10/3, 10/3, 10/3] ≈ [3.333, 3.333, 3.333]
```

### `max`

Forward: `C = maior valor de A`.

Aqui a derivada não é suave. `max` **seleciona** um elemento e ignora todos os outros.

```
∂C/∂A[i] = 1  se A[i] é o máximo
           0  caso contrário
```

O gradiente flui apenas pelo caminho que "venceu". Todos os outros recebem zero.

```
A  = [3, 9, 1]
C  = 9
dC = 7

dA = [0, 7, 0]
```

Esta é a primeira operação do projeto cuja derivada depende dos **valores** da entrada, e não apenas das posições. O backward precisa saber qual índice venceu no forward.

**E quando há empate?** Dois elementos podem ser iguais ao máximo. Matematicamente, a função não é diferenciável ali. É preciso uma convenção.

> **Definição — convenção de desempate do `max`.** Em caso de empate, o gradiente vai inteiro para a **primeira** ocorrência do máximo. As demais recebem zero. É a mesma semântica do PyTorch.

```
A  = [3, 5, 5, 1]        ← empate entre as posições 1 e 2
C  = 5
dC = 7

dA = [0, 7, 0, 0]        ← só a posição 1 recebe; a 2 fica zerada
```

Note que o gradiente **não** é dividido entre os empatados. Ele vai inteiro para o primeiro.

### Reduções por dimensão

`sum(A, dim)` e `mean(A, dim)` generalizam as versões acima. Em vez de colapsar o tensor inteiro num escalar, elas colapsam apenas um eixo e preservam os demais.

> **Definição — fatia.** Ao reduzir ao longo de `dim`, os elementos que compartilham todos os outros índices formam uma fatia. Cada fatia produz um valor de saída.

A derivada local não muda: continua `1` para a soma e `1/n_dim` para a média. O que muda é o **alcance** do espalhamento. Antes, um único `dC` ia para o tensor inteiro. Agora, cada valor de saída espalha o seu próprio gradiente — mas só dentro da fatia que o originou.

**Exemplo numérico**, `A` de formato `(2,3)`:

```
A = [[1, 4, 2],
     [5, 0, 3]]
```

Somando ao longo de `dim=1` (as colunas colapsam; sobra uma soma por linha):

```
C = sum(A, dim=1) = [1+4+2, 5+0+3] = [7, 8]        formato (2,)
```

Backward, com valores distintos por linha:

```
dC = [10, 100]

dA = [[ 10,  10,  10],
      [100, 100, 100]]                              formato (2,3)
```

Cada elemento recebeu o gradiente da **sua própria linha**. A linha 0 recebeu `10`; a linha 1 recebeu `100`. Compare com `sum` global, onde um único valor iria para as seis posições.

Para `mean(A, dim=1)`, tudo é igual, dividido pelo tamanho do eixo somado (`n_dim = 3`):

```
C  = [7/3, 8/3]
dA = [[10/3, 10/3, 10/3], [100/3, 100/3, 100/3]]
```

> **Armadilha.** O backward da redução por dimensão precisa de dois índices diferentes: o da posição de **entrada** (onde acumular) e o da posição de **saída** (de onde ler o gradiente). Confundi-los é fácil.
>
> A primeira versão deste projeto acumulou no índice de saída em ambos os casos. O resultado: nenhuma exceção, nenhum aviso, e um gradiente silenciosamente corrompido. Como o tensor de saída é bem menor que o de entrada, várias posições escreviam repetidamente nos mesmos poucos slots iniciais, enquanto o resto do array ficava zerado.
>
> No exemplo acima, a versão com bug produzia `dA = [30, 300, 0, 0, 0, 0]` em vez do resultado correto. Nenhum teste que verificasse só o formato pegaria isso.

> **Confira você mesmo.** No exemplo acima, o formato de `dA` é `(2,3)` — igual ao de `A`, e não ao de `C`. Por que o gradiente tem o formato da entrada, e não o da saída?
>
> <details><summary>Resposta</summary>
>
> Porque `dA` responde à pergunta "quanto a perda muda se eu mexer em cada posição de `A`?". Existe uma resposta para **cada** posição de `A`, então `dA` precisa ter uma entrada para cada uma delas. Vale a regra geral das convenções de notação: **o gradiente sempre tem o formato do tensor a que se refere**, não o da saída da operação.
> </details>

---

## §4. Broadcasting

> **Guia visual.** Alinhamento de formatos, a *view* de stride 0, os dois casos do `unbroadcast` e o pipeline completo do `add`: [`broadcasting.html`](broadcasting.html).

Esta é a seção mais longa e a mais conceitualmente carregada do capítulo. Leia com calma.

### O problema

Você tem uma matriz `M` de formato `[3, 4]` e um vetor `v` de formato `[4]`. Quer somar `v` a cada linha de `M`.

A solução ingênua seria copiar `v` três vezes, formando uma matriz `[3, 4]`, e então somar. Funciona, mas desperdiça memória. Em modelos reais, esse desperdício é significativo.

> **Definição — broadcasting.** É a regra que permite operar tensores de formatos diferentes, tratando dimensões de tamanho `1` (ou ausentes) como se fossem repetidas — sem de fato copiar nada.

### O forward: repetir sem copiar

O truque reusa exatamente a ideia da Etapa 1: descrever o mesmo `data` com `shape` e `strides` diferentes.

Lembre como um índice vira posição de memória: multiplica-se cada coordenada pela stride correspondente e soma-se tudo.

A sacada: **se uma stride vale `0`, aquela dimensão desaparece da conta.** Multiplicar qualquer índice por zero dá zero. Todas as coordenadas naquela dimensão passam a ler a mesma posição física.

**Exemplo concreto.** Um tensor `t` de formato `(1, 3)`, com `data = [10, 20, 30]`. Queremos esticá-lo para `(2, 3)`:

```
shape final:    (2, 3)
strides finais: (0, 1)      ← a dimensão que era 1 ganha stride 0
```

Testando alguns índices:

```
index(0, 2) = 0·0 + 2·1 = 2   →  data[2] = 30
index(1, 2) = 1·0 + 2·1 = 2   →  data[2] = 30      ← mesma posição!
```

A "linha 1" lê exatamente os mesmos dados da "linha 0". O array continua com três elementos, mas o tensor se comporta como se tivesse seis.

**Quando os rangos são diferentes**, há um passo a mais. Compare `(3,)` com `(2,3)`: o primeiro tem rank 1, o segundo rank 2. Alinha-se preenchendo `1`s **à esquerda**, e a stride dessa dimensão nova nasce `0`. Então `(3,)` vira `(1,3)` com strides `(0,1)`, e caímos no caso anterior.

### Uma invariante que precisa mudar

Até aqui valia uma regra simples: `data.length == shape.product`. Todo tensor tinha exatamente tantos elementos quanto o formato prometia.

Broadcasting quebra isso de propósito. Agora `data` pode ser deliberadamente **menor** que `shape.product` — no exemplo acima, três elementos para um formato que promete seis.

A invariante correta não é sobre tamanho total, e sim sobre o maior endereço alcançável:

```
Σᵢ (shape[i] - 1) · strides[i]  <  data.length
```

Em palavras: *nenhuma combinação de índices pode ler fora dos limites do array*. Isso cobre `reshape`, `transpose` e broadcasting com uma checagem só.

### O backward: por que é preciso somar

Aqui está o ponto central da seção.

No forward, `v` foi **usado três vezes** — uma por linha de `M`. Isso é exatamente o caso de "múltiplos caminhos" da Etapa 2 §2. E a regra lá é clara: quando uma variável influencia a saída por vários caminhos, as contribuições **somam**.

Então:

```
dv[j] = Σᵢ dC[i, j]         para cada coluna j
```

> **Definição — `unbroadcast`.** A operação que desfaz o broadcasting no backward. Ela soma o gradiente ao longo de toda dimensão que foi esticada no forward, até o formato voltar ao original.

A assinatura real no projeto é:

```
unbroadcast(grad: Gradient, gradShape: Shape, targetShape: Shape): Gradient
```

Ela precisa dos dois formatos porque compara um com o outro para descobrir quais eixos foram esticados.

### Os dois casos

O `unbroadcast` termina de formas ligeiramente diferentes, dependendo da origem da dimensão esticada. Vale separar.

**Caso 1 — a dimensão foi criada pelo alinhamento.** `M` é `(3,4)`, `v` é `(4,)`, e `C = M + v`.

```
dC = [[1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12]]
```

Alinha-se `(4,)` para `(1,4)`. A dimensão 0 vale `1` no formato alinhado, mas `3` em `dC` — foi esticada. Somamos ao longo dela:

```
dv = [1+5+9, 2+6+10, 3+7+11, 4+8+12] = [15, 18, 21, 24]        formato (4,)
```

A dimensão 0 desaparece no fim, porque ela nunca existiu em `v` — foi inventada pelo alinhamento.

**Caso 2 — a dimensão já existia, com tamanho `1`.** Agora um viés por linha: `b` de formato `(3,1)`, somado a `M` de formato `(3,4)`.

Os rangos já batem, então não há alinhamento. A dimensão 1 vale `1` em `b` mas `4` em `dC` — foi esticada. Somamos ao longo dela, com o mesmo `dC` de cima:

```
db = [[1+2+3+4], [5+6+7+8], [9+10+11+12]] = [[10], [26], [42]]     formato (3,1)
```

Aqui a dimensão **permanece**. Ela já existia em `b`, com tamanho `1` — é assim que `b` foi declarado, como um vetor-coluna. O formato final precisa bater com o original.

A diferença entre os dois casos, em uma frase: *"essa dimensão nunca existiu"* significa somar e remover; *"essa dimensão existia, só era 1"* significa somar e manter.

> **Nota de implementação.** A descrição acima é conceitual — pensa em termos de `sum` e `squeeze`. A implementação real não compõe essas operações. Ela percorre cada posição do gradiente, calcula para onde aquela posição deve voltar, e acumula direto. O resultado é idêntico, sem alocar tensores intermediários.

> **Armadilha.** Uma versão inicial de `broadcastTo` neste projeto fazia o alinhamento de rank corretamente, mas esquecia de **esticar** dimensões que já eram `1`. O resultado: funcionava quando os rangos eram diferentes, e virava uma operação nula quando eram iguais. `(1,3) → (2,3)` simplesmente não mudava nada.
>
> Depois, no `add`, houve um segundo erro relacionado: o código calculava o formato broadcastado corretamente e então **ignorava** esse formato no forward, operando sobre os arrays crus.
>
> **Lição geral:** calcular o formato certo e usar o formato certo são duas coisas diferentes. Verifique as duas.

> **Confira você mesmo.** Um tensor de formato `(5, 1)` é somado a outro de formato `(5, 8)`. No backward, o gradiente do primeiro tem qual formato — e ao longo de qual eixo a soma acontece?
>
> <details><summary>Resposta</summary>
>
> Formato `(5, 1)`, com a soma ao longo do eixo 1. É o Caso 2: a dimensão de tamanho `1` já existia no tensor original, então ela é somada mas **mantida**. O gradiente sempre volta ao formato do tensor a que se refere.
> </details>

---

## §5. Transpose e Reshape no backward

> **Guia visual.** O mecanismo por trás das duas operações (layout row-major, strides, por que a transposta deixa de ser contígua) está em [`strides-and-memory.html`](../01-tensor/strides-and-memory.html), da Etapa 1. Aqui tratamos só do backward.

A Etapa 1 mostrou que `transpose` e `reshape` só reinterpretam metadados no forward, sem copiar dados. Falta o backward.

O gradiente que chega tem o formato da **saída**. Ele precisa voltar ao formato da **entrada** antes de ser acumulado.

Há uma diferença importante em relação ao broadcasting. Lá, várias posições de saída vinham da mesma posição de entrada, então o backward precisava somar. Aqui não: cada posição de saída corresponde a **exatamente uma** posição de entrada. Desfazer é pura reindexação, nunca redução.

### `reshape`

`reshape` não reordena elementos. Ele só reinterpreta o mesmo array linear com um agrupamento diferente. Então desfazê-lo é aplicar outro `reshape`, de volta ao formato original.

```
A = [[1, 4, 2],
     [5, 0, 3]]                            array linear: [1, 4, 2, 5, 0, 3]

C = reshape(A, (3,2)) = [[1, 4],
                          [2, 5],
                          [0, 3]]          mesmo array, lido em blocos de 2
```

Backward, com valores distintos por posição:

```
dC = [[10, 20],
      [30, 40],
      [50, 60]]                            array linear: [10, 20, 30, 40, 50, 60]

dA = reshape(dC, (2,3)) = [[10, 20, 30],
                            [40, 50, 60]]
```

Repare que a posição **linear** de cada número não mudou. O `20` era o segundo elemento antes e continua sendo depois. Só o agrupamento em linhas mudou.

### `transpose`

`transpose` troca eixos. Desfazê-lo é aplicar a mesma troca ao gradiente.

```
A  = [[1, 4, 2],
      [5, 0, 3]]

Aᵀ = [[1, 5],
      [4, 0],
      [2, 3]]                              formato (3,2)
```

Backward:

```
dC = [[10, 20],
      [30, 40],
      [50, 60]]

dA = transpose(dC) = [[10, 30, 50],
                       [20, 40, 60]]
```

**Verificação posição a posição.** Como `Aᵀ[j,i] = A[i,j]`, o gradiente de `A[i,j]` tem de ser exatamente `dC[j,i]`. Testando duas posições: `dA[0,1]` deve valer `dC[1,0] = 30` ✓; `dA[1,2]` deve valer `dC[2,1] = 60` ✓. A matriz acima confere.

> **Armadilha.** Ao implementar o backward do `reshape`, este projeto cometeu um erro instrutivo. Como `reshape` pode receber um tensor não contíguo (vindo de um `transpose` anterior), o código criava uma cópia contígua auxiliar — e então acumulava o gradiente **nessa cópia**, em vez de no tensor original.
>
> A cópia auxiliar é órfã: ela não está no grafo, e a ordenação topológica nunca a alcança. Todo gradiente que chegava num `reshape` era descartado silenciosamente, sem nunca chegar ao tensor de verdade.
>
> **Lição geral:** cuidado com tensores intermediários criados dentro de uma operação. O gradiente precisa voltar para o tensor que está **no grafo**, não para um auxiliar local.

---

## §6. Multiplicação de matrizes

> **Guia visual.** Produto escalar linha·coluna, as fórmulas `dA = dC @ Bᵀ` e `dB = Aᵀ @ dC`, e a versão em lote: [`matmul.html`](matmul.html).

Esta é a operação mais importante do projeto. Todo peso treinável do transformer é usado através de um `matmul`.

### O forward

Para `A` de formato `[M, K]` e `B` de formato `[K, N]`:

```
C[i,j] = Σₖ A[i,k] · B[k,j]
```

Cada elemento de `C` é o produto escalar entre uma linha de `A` e uma coluna de `B`.

### Derivando o backward

As fórmulas do backward do `matmul` costumam ser apresentadas como algo que se decora. Não é o caso: elas saem em quatro linhas da definição acima. Vale acompanhar a derivação, porque ela é o modelo para toda derivação parecida no resto do projeto.

Queremos `∂L/∂A[i,k]`, sabendo `dC[i,j]` para todo `i,j`.

**Primeiro passo: quais saídas dependem de `A[i,k]`?** Olhando a fórmula do forward, `A[i,k]` aparece no cálculo de `C[i,j]` para **todo** `j`, com `i` fixo. Ele participa de uma linha inteira de `C`.

**Segundo passo: aplicar a regra da cadeia, somando sobre todos esses caminhos.**

```
dA[i,k] = Σⱼ dC[i,j] · ∂C[i,j]/∂A[i,k]
```

**Terceiro passo: calcular a derivada local.** Na soma `C[i,j] = Σₖ A[i,k]·B[k,j]`, o coeficiente que multiplica `A[i,k]` é `B[k,j]`. Então `∂C[i,j]/∂A[i,k] = B[k,j]`.

```
dA[i,k] = Σⱼ dC[i,j] · B[k,j]
```

**Quarto passo: reconhecer o padrão.** Essa soma é exatamente a definição de um produto de matrizes entre `dC` e `B` transposta:

```
dA = dC @ Bᵀ
```

Pelo mesmo raciocínio, com os papéis trocados:

```
dB = Aᵀ @ dC
```

Repare que nada aqui foi decorado. As duas fórmulas são consequência da definição do forward mais a regra da cadeia.

### Exemplo numérico

```
A = [[1, 2],        B = [[5, 6],
     [3, 4]]             [7, 8]]

C = A @ B = [[1·5+2·7, 1·6+2·8],  = [[19, 22],
             [3·5+4·7, 3·6+4·8]]     [43, 50]]
```

Com `dC` chegando de cima como `[[1,1],[1,1]]` (o gradiente de `L = sum(C)`):

```
dA = dC @ Bᵀ = [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11, 15],
                                                 [11, 15]]

dB = Aᵀ @ dC = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4, 4],
                                                 [6, 6]]
```

**Verificação independente.** Vamos confirmar `dA` sem usar a fórmula matricial. Como `dC` é todo `1`, temos `L = Σ C[i,j] = Σᵢⱼₖ A[i,k]·B[k,j]`. Derivando diretamente em relação a `A[i,k]`, sobra `Σⱼ B[k,j]` — a soma da linha `k` de `B`. A linha 0 de `B` soma `5+6 = 11`; a linha 1 soma `7+8 = 15`. Isso dá `dA[i,0] = 11` e `dA[i,1] = 15` para qualquer `i`, batendo exatamente com a matriz acima.

> **Armadilha.** O formato da saída do `matmul` é `[M, N]` — o número de linhas do primeiro operando pelo número de **colunas do segundo**. A primeira versão deste projeto usou `[M, K]`, esquecendo de olhar para `B`.
>
> O detalhe cruel: esse erro fica **invisível** sempre que `K == N`. Com matrizes quadradas, os dois formatos coincidem e tudo parece funcionar.
>
> **Lição geral:** teste `matmul` com formatos retangulares onde todas as três dimensões são diferentes. Um teste `(2,2) @ (2,2)` valida muito menos do que parece.

### Multiplicação em lote

Para tensores 3D — `[B, M, K]` multiplicado por `[B, K, N]` — o cálculo é literalmente o mesmo, repetido de forma independente para cada fatia do lote. É como executar `B` multiplicações 2D separadas, cada uma com seu próprio `A[b]`, `B[b]` e `dC[b]`.

As fatias nunca conversam entre si. Essa independência é o que permite, mais adiante, processar um batch inteiro de sequências de uma vez.

**Exemplo numérico.** A fatia 0 é exatamente o exemplo 2D de cima. A fatia 1 leva números completamente diferentes:

```
A[0] = [[1, 2],   B[0] = [[5, 6],   C[0] = [[19, 22],
        [3, 4]]           [7, 8]]           [43, 50]]

A[1] = [[0,  1],  B[1] = [[1, 3],   C[1] = [[ 4, 2],
        [2, -1]]          [4, 2]]           [-2, 4]]
```

`C[0]` saiu idêntico ao resultado 2D, com a fatia 1 no meio do caminho. Esse é o teste do parágrafo acima: se um índice vazasse de uma fatia para a outra, `C[0]` mudaria.

### Um operando compartilhado pelo lote

Existe um terceiro caso, e ele é o mais comum de todos numa rede: `[B, M, K]` multiplicado por `[K, N]`, sem dimensão de lote no segundo operando.

Ele aparece toda vez que uma camada aplica os **mesmos pesos** a todos os exemplos. É o caso do `Linear` da Etapa 8 recebendo um lote de sequências, e das projeções da atenção na Etapa 11.

O forward não tem nada de novo. É a fórmula do lote com o índice `b` removido do segundo operando:

```
C[b,m,n] = Σₖ A[b,m,k] · W[k,n]
```

O backward tem. `dA` continua fatia a fatia, porque cada `A[b]` só influenciou o seu próprio `C[b]`:

```
dA[b,m,k] = Σₙ dC[b,m,n] · W[k,n]
```

Mas `W` foi usada por **todas** as fatias. Pela regra da cadeia, o gradiente dela recolhe todos esses caminhos, e isso acrescenta uma soma sobre `b`:

```
dW[k,n] = Σ_b Σₘ dC[b,m,n] · A[b,m,k]
```

Esse `Σ_b` não é uma regra nova. É o mesmo desbroadcast do §4: quem foi compartilhado no forward soma no backward. A diferença é só que aqui o compartilhamento veio do formato do operando, e não de uma dimensão de tamanho `1`.

**Exemplo numérico**, com `B = 2`, `M = 2`, `K = 2` e `N = 3`:

```
A[0] = [[1, 2],    W = [[1, 0, 2],     C[0] = [[1, 2,  8],
        [3, 4]]         [0, 1, 3]]             [3, 4, 18]]

A[1] = [[5, 6],                        C[1] = [[5, 6, 28],
        [7, 8]]                                [7, 8, 38]]
```

Com `dC[0] = [[1,2,3],[4,5,6]]` e `dC[1] = [[7,8,9],[10,11,12]]`:

```
dA[0] = [[ 7, 11],      dW da fatia 0 = [[ 13,  17,  21],
         [16, 23]]                       [ 18,  24,  30]]

dA[1] = [[25, 35],      dW da fatia 1 = [[105, 117, 129],
         [34, 47]]                       [122, 136, 150]]

dW total = [[118, 134, 150],
            [140, 160, 180]]
```

**Verificação independente.** As duas contribuições parciais somam exatamente o total: `13 + 105 = 118`, `17 + 117 = 134`, e assim por diante nas seis posições. Perturbando cada entrada de `A` e de `W` por `±1e-6` e medindo pela diferença central da Etapa 4 §1, todos os valores acima se confirmam.

Repare no tamanho dos números de `dW`: eles são bem maiores que os de `dA`. Faz sentido — `dW[k,n]` é uma soma de `B · M` parcelas, enquanto cada `dA[b,m,k]` soma apenas `N`.

> **Confira você mesmo.** Você multiplica `A` de formato `(4, 7)` por `B` de formato `(7, 3)`. Quais são os formatos de `C`, `dA` e `dB`?
>
> <details><summary>Resposta</summary>
>
> `C` tem formato `(4, 3)` — linhas de `A` por colunas de `B`. `dA` tem formato `(4, 7)` e `dB` tem `(7, 3)`, porque o gradiente sempre acompanha o formato do tensor a que se refere. Conferindo pelas fórmulas: `dC @ Bᵀ` é `(4,3) @ (3,7) = (4,7)` ✓ e `Aᵀ @ dC` é `(7,4) @ (4,3) = (7,3)` ✓.
> </details>

---

## §7. Clamp

Forward: `clamp(A, min, max)` satura os valores fora do intervalo. Valores abaixo de `min` viram `min`; acima de `max`, viram `max`; no meio, passam intactos.

O backward segue diretamente do que a função faz em cada região.

Na região onde o valor **não** foi alterado (`min < A < max`), a saída acompanha a entrada exatamente. A derivada é `1`.

Na região saturada, a saída é constante — mexer na entrada não muda mais nada. A derivada é `0`.

Nos pontos exatos `A = min` e `A = max`, a função tem um "bico" e não é diferenciável. Convenciona-se gradiente zero ali também, por simplicidade. É a mesma convenção que a ReLU vai usar na Etapa 5.

**Exemplo numérico**, com `clamp(A, 0, 5)` e valores escolhidos para cobrir os três regimes:

```
A  = [-3,  2,  7,  5,  0,  4.5]
C  = [ 0,  2,  5,  5,  0,  4.5]
dC = [ 1,  1,  1,  1,  1,  1  ]

A[0] = -3     saturado abaixo do min    → dA[0] = 0
A[1] =  2     dentro do intervalo       → dA[1] = 1
A[2] =  7     saturado acima do max     → dA[2] = 0
A[3] =  5     exatamente no max (bico)  → dA[3] = 0    (convenção)
A[4] =  0     exatamente no min (bico)  → dA[4] = 0    (convenção)
A[5] =  4.5   dentro do intervalo       → dA[5] = 1

dA = [0, 1, 0, 0, 0, 1]
```

Apenas as posições **estritamente** dentro do intervalo deixam o gradiente passar. Saturadas e bordas exatas cortam o fluxo.

---

## §8. Para onde isso leva

Com estas doze operações, você tem toda a matemática de que o transformer precisa por baixo dos panos. Nenhuma etapa futura vai introduzir uma primitiva matemática realmente nova — só combinações destas.

Mas isso levanta um problema. Você derivou cada backward à mão e conferiu com exemplos numéricos, também à mão. Isso funcionou para doze operações. Não vai funcionar para `softmax`, `layerNorm`, `attention` e cada camada nova que vier.

Pior: erros de gradiente são **silenciosos**. Você viu isso três vezes neste capítulo. O `exp` usando a entrada, a redução acumulando no índice errado, o `reshape` escrevendo num tensor órfão — nenhum deles trava, nenhum emite aviso. O forward continua perfeito. O modelo só treina mal.

A Etapa 4 resolve isso. Ela constrói um verificador que compara o gradiente que você escreveu contra uma aproximação numérica calculada de forma completamente independente — perturbando a entrada por um `ε` minúsculo e medindo a variação da saída. Se os dois baterem, o backward está certo.

É uma etapa curta, e é a que dá confiança para todo o resto do projeto.

---

## Cartão de referência

Todas as derivadas locais deste capítulo, num lugar só.

| Operação | Forward | Backward |
|---|---|---|
| `add` | `A + B` | `dA = dC`, `dB = dC` |
| `sub` | `A - B` | `dA = dC`, `dB = -dC` |
| `mul` | `A · B` | `dA = dC·B`, `dB = dC·A` |
| `div` | `A / B` | `dA = dC/B`, `dB = -dC·A/B²` |
| `neg` | `-A` | `dA = -dC` |
| `pow` | `Aᵖ` | `dA = dC·p·A^(p-1)` |
| `exp` | `e^A` | `dA = dC·C` — usa a **saída** |
| `log` | `ln(A)` | `dA = dC/A` — usa a **entrada** |
| `sum` | `Σ A[i]` | `dA[i] = dC` (copiado para todos) |
| `mean` | `Σ A[i] / n` | `dA[i] = dC/n` |
| `max` | maior valor | `dA[i] = dC` só no primeiro máximo, `0` no resto |
| `sum(dim)` | soma por fatia | `dA[i] = dC` da fatia de `i` |
| `mean(dim)` | média por fatia | `dA[i] = dC/n_dim` da fatia de `i` |
| broadcasting | estica com stride 0 | soma ao longo dos eixos esticados |
| `reshape` | reagrupa | `reshape` de volta |
| `transpose` | troca eixos | mesma troca aplicada a `dC` |
| `matmul` | `Σₖ A[i,k]·B[k,j]` | `dA = dC @ Bᵀ`, `dB = Aᵀ @ dC` |
| `clamp` | satura fora de `[min,max]` | `dA = dC` estritamente dentro, `0` fora e nas bordas |

### As cinco lições que se repetem

1. **Pergunte de quais valores a derivada depende.** `exp` usa a saída, `log` usa a entrada, `mul` usa o outro operando. Errar isso não trava nada — só corrompe o gradiente.
2. **O gradiente tem o formato do tensor a que se refere**, nunca o da saída da operação.
3. **Muitos-para-um no forward vira soma no backward.** Broadcasting e reduções seguem essa regra. `reshape` e `transpose` não, porque são um-para-um.
4. **Teste com formatos que não colapsam.** Matrizes quadradas e tensores de um elemento escondem erros de formato.
5. **Use valores distintos nos exemplos.** Um `dC` todo `1` esconde erros de indexação, porque qualquer posição trocada dá o mesmo resultado.
