# Etapa 6 — Softmax e Log-Softmax

## Antes de começar

### O que você vai construir

Duas operações que transformam números arbitrários numa distribuição de probabilidade:

| Operação | O que faz |
|---|---|
| `softmax(dim)` | converte um vetor de números reais em probabilidades que somam 1 |
| `logSoftmax(dim)` | o logaritmo do anterior, calculado de forma numericamente estável |

Ao final desta etapa, o marco **"o autograd está completo"** é atingido: todas as primitivas matemáticas que um transformer usa terão forward e backward verificados.

### O que você precisa saber antes

**Da Etapa 3:** as reduções por dimensão (`sum(dim)`), e o broadcasting. O softmax reusa exatamente a mesma mecânica de agrupamento por fatia.

**Da Etapa 4:** `Gradcheck.run` — esta etapa tem uma sutileza de teste que só ele resolve.

**Da Etapa 5:** a ideia de que uma ativação transforma valores. O softmax é diferente de todas elas, e a §3 explica por quê.

### Onde esta etapa se encaixa

Todas as operações até aqui têm uma propriedade em comum, e ela passou despercebida justamente por ser universal: **cada posição da saída dependia de uma única posição da entrada**. Elemento a elemento, ativações, até as reduções — em todas, `∂saída[i]/∂entrada[j]` era zero sempre que `i ≠ j`.

O softmax quebra isso. Cada saída depende de **todas** as entradas. A derivada deixa de ser um vetor e vira uma matriz.

É essa estrutura que torna o backward desta etapa interessante — e é a mesma estrutura que aparece no coração da atenção, na Etapa 11.

### Objetivos de aprendizagem

Ao terminar, você deve conseguir:

1. Explicar por que subtrair o máximo não altera o resultado do softmax.
2. Derivar o Jacobiano `∂sᵢ/∂xⱼ` nos dois casos, `i = j` e `i ≠ j`.
3. Simplificar o backward de `O(N²)` para `O(N)` algebricamente.
4. Explicar por que testar softmax com `.sum` puro **não funciona**.

> **Guia visual.** A dependência total entre entradas e saídas, e a matriz do Jacobiano: [`softmax.html`](softmax.html). **Exercícios (12 questões):** [`exercises.html`](exercises.html).

---

## §1. Forward, e o truque da estabilidade

### A definição

> **Definição — logit.** Um número real, sem faixa fixa, que representa o "quanto o modelo acha" de cada opção. Pode ser negativo, pode ser enorme. Logits **não** são probabilidades — eles viram probabilidades depois de passar pelo softmax. É a saída bruta da última camada do modelo.

> **Definição — softmax.** `softmax(x)ᵢ = eˣⁱ / Σⱼ eˣʲ`. Converte um vetor de logits numa distribuição de probabilidade.

Duas propriedades saem direto da fórmula. Todo valor de saída é positivo, porque a exponencial é sempre positiva. E a soma de todos eles é exatamente 1, porque o denominador é a soma dos numeradores.

O papel da exponencial é amplificar diferenças. Uma vantagem pequena nos logits vira uma vantagem grande nas probabilidades.

### O problema numérico

A exponencial cresce muito rápido. Já em `x = 1000`, o `Double` estoura e devolve `Infinity`. E `Infinity / Infinity` é `NaN`.

Não é um cenário raro. Logits de modelos de linguagem reais alcançam magnitudes assim.

### A solução

Subtraia o maior valor de todos, antes de exponenciar:

```
softmax(x)ᵢ = e^(xᵢ - m) / Σⱼ e^(xʲ - m)          onde m = max(x)
```

**Isso não muda o resultado.** A demonstração é curta. Como `e^(xᵢ-m) = eˣⁱ · e⁻ᵐ`, o fator `e⁻ᵐ` aparece em todo termo do numerador e do denominador:

```
(eˣⁱ · e⁻ᵐ) / Σⱼ(eˣʲ · e⁻ᵐ) = (e⁻ᵐ · eˣⁱ) / (e⁻ᵐ · Σⱼeˣʲ) = eˣⁱ / Σⱼeˣʲ
```

O `e⁻ᵐ` cancela. Resultado idêntico ao original.

O ganho é puramente numérico. Depois da subtração, o maior expoente é sempre exatamente `0`, e `e⁰ = 1`. Nada estoura. O pior caso passa a ser `e` elevado a um número muito negativo, que apenas se aproxima de zero — nunca explode.

> **Definição — invariância por deslocamento.** `softmax(x + c) = softmax(x)` para qualquer constante `c` somada a **todos** os elementos. Subtrair o máximo é um caso particular, com `c = -m`.

Guarde essa propriedade. Ela reaparece na §3 como verificação do Jacobiano.

**Exemplo numérico**, `x = [1.0, 2.0, 3.0]`:

```
m = 3.0
x - m = [-2.0, -1.0, 0.0]

e⁻² ≈ 0.13534    e⁻¹ ≈ 0.36788    e⁰ = 1.0
soma ≈ 1.50321

s ≈ [0.09003, 0.24473, 0.66524]        soma ≈ 1.0 ✓
```

**Verificação da invariância.** Tome `x = [1000, 1001, 1002]` — as mesmas diferenças relativas, mas em magnitude que estouraria a exponencial direta. Subtraindo o máximo, `x - m = [-2, -1, 0]`, **idêntico** ao caso acima. O resultado é exatamente o mesmo, sem nenhum overflow.

> **Confira você mesmo.** Por que subtrair o **máximo**, especificamente? Subtrair a média também cancelaria na divisão.
>
> <details><summary>Resposta</summary>
>
> Qualquer constante cancela — a média funcionaria matematicamente. O máximo é escolhido porque garante que o maior expoente seja exatamente `0`, e portanto que nenhuma exponencial passe de `1`. Com a média, um valor bem acima dela ainda poderia estourar. O máximo é a única escolha que dá garantia, não apenas probabilidade.
> </details>

---

## §2. Operando ao longo de uma dimensão

Na prática os tensores têm mais de uma dimensão. Um lote de sequências produz uma matriz de logits, e cada linha precisa ser normalizada **separadamente**.

`softmax(a, dim)` aplica a normalização de forma independente em cada fatia ao longo de `dim`, preservando as demais dimensões. É o mesmo espírito de `sum(dim)` e `mean(dim)` da Etapa 3, e reusa a mesma mecânica de agrupamento por fatia.

**Exemplo numérico**, `A` de formato `(2,3)`, normalizando ao longo de `dim=1`:

```
A = [[1, 2, 3],
     [1, 1, 1]]

linha 0: mesma conta da §1              → [0.09003, 0.24473, 0.66524]
linha 1: x-m = [0,0,0] → e⁰=1 nos três  → soma 3 → [0.33333, 0.33333, 0.33333]

softmax(A, dim=1) = [[0.09003, 0.24473, 0.66524],
                      [0.33333, 0.33333, 0.33333]]
```

A linha 1 é instrutiva: quando todos os logits são iguais, o modelo não tem preferência, e o softmax devolve a distribuição uniforme. Cada linha soma 1 de forma independente.

> **Armadilha.** O cálculo exige **duas passadas** por fatia: a primeira encontra o máximo, a segunda calcula as exponenciais. A primeira implementação deste projeto fundiu as duas num laço só, usando "o maior valor visto até agora" no lugar do máximo definitivo.
>
> Por que passou despercebido: **a soma continua dando 1**. Qualquer verificação do tipo "as probabilidades somam 1?" aprova a versão com bug. Só a distribuição relativa sai errada.
>
> O contra-exemplo que expôs o problema, com `x = [1, 5, 3]`:
>
> ```
> CORRETO          [0.01588, 0.86681, 0.11731]      soma = 1.0
> BUG (máx corrente)[0.46831, 0.46831, 0.06338]     soma = 1.0
> ```
>
> As duas primeiras posições saem empatadas na versão com bug, quando na verdade uma delas deveria ter 55 vezes a probabilidade da outra.
>
> **Lição geral:** desconfie de invariantes fáceis. "Soma 1" é necessário, mas longe de suficiente. Escolha um teste que distinga a resposta certa da errada.

> **Armadilha.** Duas outras falhas apareceram na mesma implementação, e ambas são de leitura de tensor.
>
> O código lia `t.data(i)` diretamente, em vez de `t.get(multiIdx*)`. Isso só funciona em tensores **contíguos** — e quebraria com qualquer tensor vindo de um `transpose`, cenário garantido na atenção da Etapa 11. E a saída foi construída com `t.strides` em vez dos strides canônicos do formato.
>
> **Lição geral:** é o mesmo par de strides da Etapa 1, §3. Ler pelo array cru assume contiguidade; ler pelo `get` respeita os strides reais. Em código genérico, use sempre o segundo.

---

## §3. O Jacobiano: por que o backward não é elemento a elemento

Aqui está a diferença estrutural em relação a tudo que veio antes.

Em toda operação anterior, `∂saída[i]/∂entrada[j]` era zero para `i ≠ j`. No softmax isso não vale: `xⱼ` entra no **denominador** de todas as saídas. Mexer numa única entrada mexe em todas as saídas.

> **Definição — Jacobiano.** A matriz de todas as derivadas parciais `∂sᵢ/∂xⱼ`. Para uma entrada de `N` elementos, é uma matriz `N × N`.

### Derivando

Seja `S = Σₖ eˣᵏ`, de modo que `sᵢ = eˣⁱ / S`.

**Caso `i = j`.** Aqui `xᵢ` aparece nos dois lugares: no numerador e no denominador. Pela regra do quociente, com `∂S/∂xᵢ = eˣⁱ`:

```
∂sᵢ/∂xᵢ = (eˣⁱ·S - eˣⁱ·eˣⁱ) / S² = sᵢ - sᵢ² = sᵢ(1 - sᵢ)
```

**Caso `i ≠ j`.** Agora `xⱼ` aparece **apenas** no denominador. O numerador é constante em relação a ele:

```
∂sᵢ/∂xⱼ = -eˣⁱ·eˣʲ / S² = -sᵢ·sⱼ
```

Os dois casos se unificam com o delta de Kronecker, que vale 1 quando `i = j` e 0 caso contrário:

```
∂sᵢ/∂xⱼ = sᵢ·(δᵢⱼ - sⱼ)
```

Em notação matricial, `J = diag(s) - s·sᵀ`. Note que essa matriz é **simétrica**, já que `sᵢsⱼ = sⱼsᵢ`. Isso vai simplificar a conta da próxima seção.

**Exemplo numérico**, com `s ≈ [0.09003, 0.24473, 0.66524]` da §1:

```
J₀₀ = s₀(1-s₀) ≈  0.08193     J₀₁ = -s₀s₁ ≈ -0.02203     J₀₂ = -s₀s₂ ≈ -0.05990
J₁₀ = -s₁s₀    ≈ -0.02203     J₁₁ = s₁(1-s₁) ≈ 0.18483   J₁₂ = -s₁s₂ ≈ -0.16280
J₂₀ = -s₂s₀    ≈ -0.05990     J₂₁ = -s₂s₁ ≈ -0.16280     J₂₂ = s₂(1-s₂) ≈ 0.22271
```

**Verificação independente.** Some cada linha:

```
linha 0:  0.08193 - 0.02203 - 0.05990 ≈ 0
```

Toda linha soma zero. Isso **não** é coincidência: é consequência direta da invariância por deslocamento da §1. Se `Σⱼ ∂sᵢ/∂xⱼ` fosse diferente de zero, somar uma constante a todo `x` mudaria `sᵢ` — contradizendo `softmax(x+c) = softmax(x)`.

Uma propriedade provada na primeira seção, confirmando um cálculo feito na terceira. É o tipo de checagem cruzada que dá confiança na derivação.

---

## §4. Simplificando o backward de `O(N²)` para `O(N)`

Montar a matriz `N × N` e multiplicá-la por `dOut` funcionaria, mas seria caro — e desperdiçaria a estrutura especial de `J`.

Pela regra da cadeia, `dxⱼ = Σᵢ dOutᵢ · ∂sᵢ/∂xⱼ`. Substituindo a fórmula da §3:

```
dxⱼ = Σᵢ dOutᵢ·sᵢ·(δᵢⱼ - sⱼ)
    = Σᵢ dOutᵢ·sᵢ·δᵢⱼ  -  Σᵢ dOutᵢ·sᵢ·sⱼ
```

Trate os dois termos separadamente.

No **primeiro**, o delta zera tudo exceto `i = j`. Sobra apenas `dOutⱼ·sⱼ`.

No **segundo**, `sⱼ` não depende de `i`, então sai da soma. Sobra `sⱼ · Σᵢ dOutᵢ·sᵢ`.

```
dxⱼ = sⱼ·dOutⱼ - sⱼ·(Σᵢ dOutᵢ·sᵢ) = sⱼ · [dOutⱼ - (dOut · s).sum()]
```

Em notação vetorial:

```
dx = s · (dOut - (dOut · s).sum())
```

A matriz `N × N` desapareceu. O que sobrou foi um produto interno — um único número — subtraído de cada posição, e depois uma multiplicação elemento a elemento. Custo linear.

**Exemplo numérico**, com o mesmo `s` e `dOut = [0.1, 0.2, 0.3]`:

```
(dOut · s).sum() = 0.1·0.09003 + 0.2·0.24473 + 0.3·0.66524 ≈ 0.25752

dx₀ = 0.09003 · (0.1 - 0.25752) ≈ 0.09003 · (-0.15752) ≈ -0.01418
dx₁ = 0.24473 · (0.2 - 0.25752) ≈ 0.24473 · (-0.05752) ≈ -0.01408
dx₂ = 0.66524 · (0.3 - 0.25752) ≈ 0.66524 ·   0.04248  ≈  0.02826
```

**Verificação independente**, agora pelo Jacobiano completo da §3. Conferindo apenas `dx₀`, com `dx₀ = Σᵢ dOutᵢ·Jᵢ₀`:

```
dx₀ = 0.1·0.08193 + 0.2·(-0.02203) + 0.3·(-0.05990)
    ≈ 0.00819 - 0.00441 - 0.01797
    ≈ -0.01418
```

Idêntico ao valor obtido pela fórmula simplificada. A simplificação algébrica é exatamente equivalente ao produto matricial completo — só muito mais barata.

Note também que `dx₀ + dx₁ + dx₂ ≈ 0`, a mesma propriedade de soma zero vista nas linhas do Jacobiano.

> **Confira você mesmo.** Para um vetor de 1000 logits, quantas multiplicações a versão com Jacobiano explícito faria, contra a versão simplificada?
>
> <details><summary>Resposta</summary>
>
> O Jacobiano exige montar uma matriz `1000 × 1000` e multiplicá-la por `dOut`: cerca de **um milhão** de multiplicações, mais a memória da matriz. A versão simplificada faz um produto interno (1000), uma subtração por posição (1000) e uma multiplicação por posição (1000) — cerca de **três mil**, sem alocar matriz nenhuma. É a diferença entre `O(N²)` e `O(N)`, e ela vem de quatro linhas de álgebra.
> </details>

> **Armadilha.** O backward tem duas passadas, e a primeira acumula `Σᵢ dOutᵢ·sᵢ` por grupo. A implementação inicial indexou essa soma pelo índice do **grupo** em vez do índice do **elemento** — escreveu `data(idx)·grad(idx)` onde deveria ser `data(i)·grad(i)`.
>
> Num tensor 1D com três elementos há um único grupo, então `idx` valia sempre `0`. A soma resultava em `3·data(0)·grad(0)`, ignorando completamente as posições 1 e 2.
>
> **Lição geral:** o mesmo par de índices da Etapa 3 — entrada contra saída, elemento contra grupo. Sempre que houver dois espaços de indexação, escreva qual é qual antes de escrever a conta.

> **Armadilha.** Um bug apareceu num lugar inesperado: `softmax` sobre um tensor de **rank 1** — o caso mais elementar, um vetor simples — lançava exceção.
>
> A causa estava no código de formato da Etapa 1. Agrupar por fatia num tensor rank 1 colapsa a única dimensão, produzindo um formato de rank 0. E o cálculo de strides canônicos fazia `.tail` num array já vazio, o que lança exceção em Scala.
>
> Nenhum uso anterior tinha colapsado a única dimensão de um vetor — todos os testes de redução usaram rank 2 ou mais.
>
> **Lição geral:** o caso mais simples nem sempre é o mais testado. Rank 0 e rank 1 são fronteiras legítimas, e costumam ser os últimos a receber teste.

---

## §5. Log-Softmax

> **Definição — log-softmax.** `logSoftmax(x)ᵢ = xᵢ - log(Σⱼ eˣʲ)`. O logaritmo do softmax, calculado sem nunca formar o softmax explicitamente.

Por que não simplesmente aplicar `log` no resultado do `softmax`? Por precisão.

Uma probabilidade `sᵢ` pode ser minúscula — algo como `1e-30`. Calcular `log` de um número desses, **depois** de já ter perdido precisão na divisão, acumula erro. A fórmula acima evita isso: usa apenas uma subtração e um `log` de uma soma, nunca uma divisão por um número muito pequeno.

A versão estável reusa o mesmo truque da §1, e tem nome próprio:

> **Definição — log-sum-exp.** `log(Σⱼ eˣʲ) = m + log(Σⱼ e^(xʲ-m))`, com `m = max(x)`. Permite calcular o logaritmo de uma soma de exponenciais sem que nenhuma delas estoure.

**Backward.** Derivando `yᵢ = xᵢ - log(S)` diretamente, com `∂S/∂xⱼ = eˣʲ`:

```
∂yᵢ/∂xⱼ = δᵢⱼ - eˣʲ/S = δᵢⱼ - sⱼ
```

Esta matriz **não** é simétrica, ao contrário da do softmax. Mas a regra da cadeia simplifica de forma parecida:

```
dxⱼ = Σᵢ dOutᵢ·(δᵢⱼ - sⱼ) = dOutⱼ - sⱼ·Σᵢ dOutᵢ
```

```
dx = dOut - s · dOut.sum()
```

É ainda mais simples que o backward do softmax. Não precisa do produto interno `dOut·s` — apenas da soma de `dOut`.

**Exemplo numérico**, mesmo `s` e mesmo `dOut`, com `Σdᵢ = 0.6`:

```
dx₀ = 0.1 - 0.09003·0.6 ≈ 0.1 - 0.05402 ≈  0.04598
dx₁ = 0.2 - 0.24473·0.6 ≈ 0.2 - 0.14684 ≈  0.05316
dx₂ = 0.3 - 0.66524·0.6 ≈ 0.3 - 0.39914 ≈ -0.09914
```

De novo a soma dá aproximadamente zero — a mesma invariância por deslocamento, agora do lado do log-softmax.

### Uma sutileza de teste que só o Gradcheck resolve

Ao escrever os testes desta etapa, apareceu um problema que vale registrar, porque ele não é óbvio.

A tentação é testar com `a.softmax(dim).sum`. **Isso não funciona.**

O motivo: a soma das probabilidades dentro de um grupo é sempre exatamente `1`, independente de `x`. Ou seja, é uma função **constante**. A derivada de uma constante é zero, e o teste perde o que medir.

Vale ver o que acontece de fato, porque não é o que a intuição sugere. Medindo neste projeto, com `x = [1, 2, 3, 0.5]`:

```
gradiente analítico de sum(softmax(x))   = [0, 0, 0, 0]    exatamente zero
gradiente numérico                       ≈ ruído de arredondamento, ~1e-11

Gradcheck.run com softmax(x).sum          = 1.7e-3      ← acima do limite de 1e-5
Gradcheck.run com (softmax(x)*pesos).sum  = 4.6e-10     ← um teste de verdade
```

O teste degenerado **não** aprova em silêncio: ele compara ruído contra zero, e o erro relativo do `Gradcheck` divide esse ruído pelo piso de `1e-8`. O resultado não diz nada sobre o backward — nesta implementação ele até reprova o código correto. Seja qual for o veredito, ele é ruído, e é por isso que a perda não pode ser constante.

A correção é multiplicar por pesos fixos antes de somar:

```
(a.softmax(dim) * pesos).sum
```

Isso quebra a degenerescência: agora a saída realmente varia com `x`, e o gradiente tem algo para medir.

> **Confira você mesmo.** É a mesma técnica de "valores distintos" que o projeto usa em toda parte. Por que ela resolve dois problemas diferentes aqui?
>
> <details><summary>Resposta</summary>
>
> Primeiro, quebra a constância: sem os pesos, a saída é sempre 1, o gradiente verdadeiro é zero, e o teste passa a comparar ruído numérico contra zero. Segundo, com pesos **distintos** entre si, um erro de indexação no backward produz um resultado diferente do correto — enquanto pesos uniformes deixariam qualquer troca de posição passar despercebida.
> </details>

---

## §6. Para onde isso leva

Com softmax e log-softmax verificados, fecha-se o **marco "o autograd está completo"**. Todas as primitivas matemáticas que um transformer usa — element-wise, matmul, ativações e agora normalização — têm forward e backward testados e confirmados numericamente.

Daqui em diante, nenhuma matemática realmente nova aparece. As Etapas 7 a 19 combinam essas primitivas em camadas, montam o modelo, e o treinam.

O softmax reaparece em dois lugares específicos, e vale saber quais desde já:

- Na **atenção** (Etapa 11), transformando scores de atenção em pesos que somam 1. É lá que a estrutura "cada saída depende de todas as entradas" desta etapa deixa de ser uma curiosidade matemática e vira o mecanismo central do modelo.
- Na **cross-entropy** (Etapa 16), via `logSoftmax`, para treinar o modelo a prever o próximo token.

A próxima etapa muda de assunto. A Etapa 7 sai da matemática e entra em texto: como transformar caracteres em números que o modelo possa consumir.

---

## Cartão de referência

| Conceito | Fórmula |
|---|---|
| softmax | `sᵢ = eˣⁱ / Σⱼ eˣʲ` |
| versão estável | `sᵢ = e^(xᵢ-m) / Σⱼ e^(xʲ-m)`, com `m = max(x)` |
| invariância | `softmax(x + c) = softmax(x)` |
| Jacobiano | `∂sᵢ/∂xⱼ = sᵢ(δᵢⱼ - sⱼ)` |
| forma matricial | `J = diag(s) - s·sᵀ` (simétrica) |
| backward simplificado | `dx = s · (dOut - (dOut·s).sum())` |
| log-softmax | `yᵢ = xᵢ - log(Σⱼ eˣʲ)` |
| log-sum-exp | `log(Σⱼeˣʲ) = m + log(Σⱼe^(xʲ-m))` |
| backward do log-softmax | `dx = dOut - s · dOut.sum()` |
| propriedade dos dois | as componentes de `dx` somam zero |

### As quatro lições que se repetem

1. **Cancelamento algébrico é uma ferramenta numérica.** Subtrair o máximo não muda a matemática e resolve o overflow por completo.
2. **Invariantes fáceis enganam.** "Soma 1" aprova uma distribuição completamente errada. Escolha testes que distingam certo de errado.
3. **Estrutura especial vale simplificação.** O Jacobiano do softmax é `N × N`, mas a regra da cadeia colapsa para `O(N)` com quatro linhas de álgebra.
4. **Cuidado com funções constantes em testes de gradiente.** Se a saída não varia com a entrada, o gradiente verdadeiro é zero e o teste passa a medir ruído — o veredito dele não significa nada.
