# Etapa 5 — Funções de Ativação

## Antes de começar

### O que você vai construir

Quatro funções não-lineares, cada uma com forward e backward:

| Função | Saída em | Papel no projeto |
|---|---|---|
| `relu` | `[0, ∞)` | a mais simples; base de comparação |
| `sigmoid` | `(0, 1)` | comprime para uma faixa tipo probabilidade |
| `tanh` | `(-1, 1)` | como sigmoid, mas centrada em zero |
| `gelu` | `(-0.17, ∞)` | a usada de fato no GPT |

Todas cabem na mesma fábrica de operações unárias já construída na Etapa 3. Nenhuma infraestrutura nova é necessária — só matemática.

### O que você precisa saber antes

**Da Etapa 3:** a fábrica de operações unárias, e a ideia de que o backward pode reusar o valor da saída.

**Da Etapa 4:** `Gradcheck.run`. Você vai precisar dele nesta etapa mais do que em qualquer outra até agora.

### Onde esta etapa se encaixa

Todas as operações até aqui são lineares ou quase. Somar, multiplicar, multiplicar matrizes — nada disso dobra o espaço de forma interessante.

Esta etapa introduz a primeira coisa genuinamente **não-linear** do projeto. E, como a §1 mostra com números, é essa não-linearidade que separa uma rede neural de verdade de uma única transformação linear disfarçada de várias camadas.

Sem ativação, empilhar cem camadas produz exatamente o mesmo poder de uma só.

### Objetivos de aprendizagem

Ao terminar, você deve conseguir:

1. Demonstrar, com números, que compor camadas lineares não aumenta o poder do modelo.
2. Explicar o problema do "neurônio morto" e por que a GELU o evita.
3. Derivar a derivada da sigmoid pela regra do quociente.
4. Derivar a derivada da GELU, combinando regra do produto e regra da cadeia.

> **Guia visual.** As quatro curvas e suas derivadas, lado a lado: [`activations.html`](activations.html). **Exercícios (12 questões):** [`exercises.html`](exercises.html).

---

## §1. Por que não-linearidade importa

### O colapso

> **Definição — camada linear.** Uma transformação da forma `f(x) = Wx + b`, onde `W` é uma matriz de pesos e `b` um vetor de deslocamento. Você vai construir isso na Etapa 8.

Empilhe duas camadas lineares, sem nada entre elas:

```
f₂(f₁(x)) = W₂(W₁x + b₁) + b₂
          = (W₂W₁)x + (W₂b₁ + b₂)
          = W'x + b'
```

O resultado é `W'x + b'` — **uma única camada linear**, apenas com pesos diferentes.

Isso vale para qualquer profundidade. Cem camadas lineares empilhadas colapsam numa só. Profundidade sem ativação não acrescenta nada.

### A prova numérica

> **Definição — princípio da superposição.** Uma função `f` é linear se `f(a) + f(b) = f(a+b)` para quaisquer `a` e `b`. É o teste formal de linearidade.

Vamos testar duas redes de uma dimensão, com pesos `w₁ = 2` e `w₂ = 3`.

**Sem ativação**, `f(x) = w₂ · (w₁ · x) = 6x`:

```
f(-1) = -6        f(1) = 6

f(-1) + f(1) = -6 + 6 = 0
f(-1 + 1)    = f(0)   = 0

0 = 0    ✓ superposição vale — é linear
```

**Com uma ReLU no meio**, `g(x) = w₂ · relu(w₁ · x)`:

```
g(-1) = 3 · relu(-2) = 3 · 0 = 0
g(1)  = 3 · relu(2)  = 3 · 2 = 6

g(-1) + g(1) = 0 + 6 = 6
g(-1 + 1)    = g(0)  = 3 · relu(0) = 0

6 ≠ 0    ✗ superposição falha — não é linear
```

A diferença é `6`, e ela é toda mérito da ReLU. A função `g` é construída só com multiplicações — que são lineares — e uma ReLU no meio. Essa única não-linearidade basta para quebrar a superposição.

É exatamente essa quebra que dá à rede o poder de aproximar funções que uma reta jamais alcançaria.

---

## §2. ReLU

> **Definição — ReLU (*Rectified Linear Unit*).** `relu(x) = max(0, x)`. Passa valores positivos intactos e zera os negativos.

É a ativação mais simples que existe, e por muito tempo foi a mais usada em redes profundas.

**Backward:**

```
∂relu/∂x = 1   se x > 0
           0   se x ≤ 0
```

No ponto exato `x = 0` a função tem um "bico" e não é diferenciável. Convenciona-se gradiente zero ali — a mesma convenção já usada no `clamp` da Etapa 3.

**Exemplo numérico**, com valores distintos por posição:

```
A  = [-3, -0.5,  0,  2,  5]
dC = [ 1,    2,  3,  4,  5]

C = relu(A) = [0, 0, 0, 2, 5]

A[0] = -3     x ≤ 0            → dA[0] = 0
A[1] = -0.5   x ≤ 0            → dA[1] = 0
A[2] = 0      x ≤ 0 (convenção)→ dA[2] = 0
A[3] = 2      x > 0            → dA[3] = 1 · 4 = 4
A[4] = 5      x > 0            → dA[4] = 1 · 5 = 5

dA = [0, 0, 0, 4, 5]
```

### O neurônio morto

> **Definição — neurônio.** Uma unidade que combina várias entradas num único número, multiplicando cada entrada por um peso e somando tudo. Na Etapa 8 você verá que cada **coluna** da matriz de pesos é um neurônio.

A ReLU tem um defeito sério, e ele decorre direto do backward acima.

Suponha que os pesos de um neurônio evoluam de forma a deixá-lo sempre com entrada negativa. Então a ReLU sempre zera a saída dele. E o gradiente ali é sempre **zero**.

Gradiente zero significa atualização zero. Os pesos daquele neurônio nunca mais mudam. Ele fica permanentemente inútil.

> **Definição — neurônio morto.** Um neurônio cuja entrada ficou permanentemente na região negativa da ReLU. Como o gradiente ali é zero, ele nunca mais é atualizado, e permanece inútil pelo resto do treino.

Note que o problema é auto-reforçado: sem gradiente, não há como sair da região que causou o problema. É uma armadilha da qual o neurônio não escapa sozinho.

É a principal razão prática para o transformer preferir GELU, como a §5 mostra.

> **Armadilha.** A primeira implementação de `relu` neste projeto usou `x >= 0` no backward, dando gradiente `1` no zero exato.
>
> Por que importa: diverge da convenção do próprio projeto, já fixada no `clamp` da Etapa 3, e diverge do PyTorch, que usa comparação estrita. Um único ponto do domínio — mas convenções inconsistentes dentro da mesma biblioteca causam confusão real depois.
>
> **Lição geral:** ao encontrar um ponto não-diferenciável, verifique qual convenção o resto do código já adotou. Consistência vale mais que a escolha em si.

---

## §3. Sigmoid

> **Definição — sigmoid.** `σ(x) = 1 / (1 + e⁻ˣ)`. Comprime qualquer número real para o intervalo aberto `(0, 1)`.

Entradas muito negativas se aproximam de 0; muito positivas, de 1. O valor em zero é exatamente `0.5`.

**Derivando o backward** pela regra do quociente. Com `g(x) = 1 + e⁻ˣ`, temos `σ = 1/g` e `g'(x) = -e⁻ˣ`:

```
σ'(x) = -g'(x) / g(x)² = e⁻ˣ / (1 + e⁻ˣ)²
```

Isso já está correto, mas dá para reescrever de forma muito mais útil. Separe a fração em dois fatores:

```
σ'(x) = [1/(1+e⁻ˣ)] · [e⁻ˣ/(1+e⁻ˣ)]
```

O primeiro fator é `σ(x)`. E o segundo pode ser simplificado somando e subtraindo 1 no numerador:

```
e⁻ˣ/(1+e⁻ˣ) = (1 + e⁻ˣ - 1)/(1+e⁻ˣ) = 1 - σ(x)
```

Portanto:

```
σ'(x) = σ(x) · [1 - σ(x)]
```

Repare no que ganhamos. A derivada é escrita **inteiramente em termos da saída**. Assim como no `exp` da Etapa 3, o backward pode reusar o valor já calculado no forward, sem reavaliar nenhuma exponencial.

**Exemplo numérico**, `x = [0, 1]` e `dC = [1, 2]`:

```
σ(0) = 1/(1+1)     = 0.5
σ(1) = 1/(1+e⁻¹)   ≈ 0.73106

dx[0] = σ(0)·(1-σ(0))·dC[0] = 0.5 · 0.5 · 1      = 0.25
dx[1] = σ(1)·(1-σ(1))·dC[1] = 0.73106 · 0.26894 · 2 ≈ 0.39322
```

Note que `0.25` é o valor **máximo** que `σ'` pode assumir. Isso acontece em `x = 0` e decai rapidamente para os lados — em `x = 5`, a derivada já é menor que `0.007`. Uma sigmoid saturada quase não deixa gradiente passar, o que é um problema em redes profundas.

---

## §4. Tanh

> **Definição — tanh (tangente hiperbólica).** `tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)`. Comprime para o intervalo `(-1, 1)`, simetricamente em torno de zero.

A diferença prática em relação à sigmoid é o **centro**. A sigmoid é centrada em `0.5`; a tanh, em `0`. Saídas centradas em zero costumam facilitar o treino, porque não introduzem um viés sistemático na camada seguinte.

**Backward:**

```
d(tanh)/dx = 1 - tanh(x)²
```

É uma identidade padrão, análoga à de `sec²` em trigonometria. Assim como a sigmoid, ela reusa o valor da saída.

**Exemplo numérico**, `x = [0, 1]` e `dC = [1, 2]`:

```
tanh(0) = 0
tanh(1) ≈ 0.76159

dx[0] = (1 - 0²) · dC[0]        = 1 · 1              = 1
dx[1] = (1 - 0.76159²) · dC[1]  = (1 - 0.58002) · 2 ≈ 0.83996
```

Vale registrar esta derivada com cuidado. Ela reaparece **dentro** da GELU, na próxima seção, e será reusada via regra da cadeia — não rederivada do zero.

> **Confira você mesmo.** Três das quatro ativações deste capítulo reusam o valor da saída no backward. Qual não reusa, e por quê?
>
> <details><summary>Resposta</summary>
>
> A **ReLU**. A derivada dela depende de saber se a entrada era positiva, e a saída sozinha não responde isso: uma saída `0` pode vir de qualquer entrada negativa. A GELU também precisa da entrada, porque a fórmula da derivada contém `x` explicitamente. Sigmoid e tanh são as que se escrevem inteiramente em termos da saída.
> </details>

---

## §5. GELU — a ativação do GPT

> **Definição — GELU (*Gaussian Error Linear Unit*).** A ativação usada no GPT. A forma exata envolve a função de distribuição acumulada da normal; na prática usa-se a aproximação abaixo, a mesma do GPT-2.

```
gelu(x) ≈ 0.5 · x · (1 + tanh(c · (x + 0.044715·x³)))       com c = √(2/π) ≈ 0.797885
```

### Por que ela, e não ReLU

A ReLU corta abruptamente: tudo abaixo de zero vira exatamente zero, e o gradiente morre junto.

A GELU **atenua suavemente**. Para `x` negativo, a saída não é zero — é um número pequeno e negativo. E, crucialmente, a derivada ali **não é zero**.

Isso resolve o problema do neurônio morto da §2. Um neurônio que caiu na região negativa ainda recebe algum gradiente, e portanto ainda pode se recuperar. Empiricamente, modelos de linguagem treinam melhor com GELU.

### Derivando o backward

Esta é, de longe, a derivada mais complexa do projeto. Vale ir devagar.

O forward é um **produto** de dois fatores, e o segundo contém uma **composição**. Isso exige regra do produto e regra da cadeia juntas.

Abreviando `t = tanh(u(x))`, com `u(x) = c·(x + 0.044715x³)`, o forward é:

```
gelu(x) = 0.5 · x · (1 + t)
```

**Passo 1 — regra do produto**, tratando `x` e `(1+t)` como os dois fatores:

```
d(gelu)/dx = 0.5 · (1 + t)  +  0.5 · x · dt/dx
```

**Passo 2 — regra da cadeia** para o `dt/dx`, reusando a derivada da tanh da §4:

```
dt/dx = (1 - t²) · u'(x)
```

**Passo 3 — a derivada interna.** Derivando `u(x) = c·(x + 0.044715x³)`:

```
u'(x) = c · (1 + 3·0.044715·x²) = c · (1 + 0.134145·x²)
```

Repare de onde vem o `0.134145`: é `3 × 0.044715`, resultado da regra da potência aplicada ao termo cúbico. Não é uma constante nova, é a antiga multiplicada por 3.

**Juntando tudo:**

```
d(gelu)/dx = 0.5·(1 + t) + 0.5·x·(1 - t²)·c·(1 + 0.134145·x²)
```

**Exemplo numérico**, em `x = 1`:

```
u(1) = 0.797885 · (1 + 0.044715) = 0.797885 · 1.044715 ≈ 0.833562
t    = tanh(0.833562)                                  ≈ 0.68238

gelu(1) = 0.5 · 1 · (1 + 0.68238) ≈ 0.84119
```

O valor de referência do GPT-2 para `gelu(1)` é `0.8413`. Bate.

Agora a derivada:

```
u'(1) = 0.797885 · (1 + 0.134145) = 0.797885 · 1.134145 ≈ 0.90489

d(gelu)/dx |ₓ₌₁ = 0.5·(1 + 0.68238) + 0.5·1·(1 - 0.68238²)·0.90489
                = 0.84119 + 0.5·(1 - 0.46564)·0.90489
                = 0.84119 + 0.5·0.53436·0.90489
                ≈ 0.84119 + 0.24177
                ≈ 1.08296
```

> **Armadilha.** A primeira implementação deste projeto escreveu `(1 + t²)` onde deveria estar `(1 - t²)`, no termo vindo da derivada da tanh.
>
> Um único sinal. O forward continua perfeito, o código roda sem erro, e o gradiente sai **1.50434** em vez de **1.08296** — quase 40% maior.
>
> Como foi encontrado: o `Gradcheck` da Etapa 4. Não havia teste manual cobrindo esse ponto, e conferir a álgebra à mão de novo dificilmente teria pego o sinal.
>
> **Lição geral:** esta é exatamente a situação para a qual a Etapa 4 foi construída. Derivadas longas têm muitos pontos onde um sinal, um coeficiente ou uma composição podem escapar — e nenhum deles produz sintoma visível.

Três lugares específicos onde essa derivação pode dar errado, e vale conferir com o verificador depois de escrever:

- o sinal do `(1 - t²)`, como no bug acima;
- confundir `0.044715` com `0.134145` — lembre que o segundo é o triplo do primeiro;
- esquecer a regra do produto, e derivar apenas o `tanh` sem o termo `0.5·(1+t)`.

> **Confira você mesmo.** Por que a GELU precisa do valor de `x` no backward, enquanto sigmoid e tanh não precisam?
>
> <details><summary>Resposta</summary>
>
> Porque `x` aparece **explicitamente** na fórmula da derivada, em dois lugares: no fator `0.5·x` da regra do produto, e dentro de `u'(x) = c·(1 + 0.134145x²)`. Não há como reescrever isso apenas em termos de `gelu(x)`. Sigmoid e tanh têm identidades que permitem essa reescrita; a GELU não.
> </details>

---

## §6. Para onde isso leva

O projeto tem agora a não-linearidade que faltava. Com ela, empilhar camadas passa a valer a pena.

Todas as quatro ativações desta etapa compartilham uma característica: elas agem **posição a posição**. Cada elemento do tensor é transformado isoladamente, sem olhar para os vizinhos.

A Etapa 6 quebra esse padrão pela primeira vez. O `softmax` normaliza um vetor **inteiro** numa distribuição de probabilidade — cada saída depende de **todas** as entradas. Isso torna o backward qualitativamente diferente de tudo que veio antes, e é o mecanismo central por trás da atenção do transformer.

---

## Cartão de referência

| Função | Forward | Backward | Reusa a saída? |
|---|---|---|---|
| `relu` | `max(0, x)` | `1` se `x > 0`, senão `0` | não |
| `sigmoid` | `1/(1+e⁻ˣ)` | `σ·(1-σ)` | sim |
| `tanh` | `(eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)` | `1 - tanh²` | sim |
| `gelu` | `0.5x(1 + tanh(u))` | `0.5(1+t) + 0.5x(1-t²)u'` | parcialmente |

Com `u(x) = c(x + 0.044715x³)`, `u'(x) = c(1 + 0.134145x²)` e `c = √(2/π) ≈ 0.797885`.

### As quatro lições que se repetem

1. **Sem não-linearidade, profundidade não existe.** Cem camadas lineares empilhadas são matematicamente idênticas a uma só.
2. **Gradiente zero é permanente.** A ReLU pode desligar um neurônio para sempre, porque sem gradiente não há como sair da região que causou o problema.
3. **Derivadas longas escondem erros de sinal.** O `(1+t²)` da GELU produz um gradiente 40% errado sem nenhum sintoma no forward.
4. **Reusar a saída é a regra, não a exceção.** Sigmoid, tanh e `exp` fazem isso. Sempre pergunte se a derivada pode ser escrita em termos do que já foi calculado.
