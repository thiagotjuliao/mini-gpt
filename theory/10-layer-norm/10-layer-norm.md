# Etapa 10 — Layer Normalization

## Antes de começar

### O que você vai construir

A camada que mantém as ativações numa escala estável, vetor a vetor.

| Componente | Formato | O que faz |
|---|---|---|
| `γ` (gama) | `[H]` | ganho treinável, um por posição do vetor; nasce em `1` |
| `β` (beta) | `[H]` | deslocamento treinável; nasce em `0` |
| `forward(x)` | `[B, T, H]` → `[B, T, H]` | normaliza cada vetor de tamanho `H`, depois aplica `γ` e `β` |

### O que você precisa saber antes

**Da Etapa 3:** as reduções por eixo (`mean(dim, keepDim)`, `sum(dim, keepDim)`), o `pow`, e o broadcasting. A camada inteira é feita disso.

**Da Etapa 8:** o que é um parâmetro treinável, e por que um parâmetro precisa de `requiresGradient`.

**Da Etapa 9:** o formato `[B, T, dModel]` que sai da camada de embedding. É a entrada desta aqui.

### Onde esta etapa se encaixa

As Etapas 8 e 9 construíram camadas que não precisaram de backward próprio. `Linear` é `matmul` mais soma; `Embedding` é seleção de linha mais soma. Em ambas, o autograd da Etapa 2 fez o trabalho sozinho.

Esta etapa é diferente. A normalização acopla todas as posições de um vetor: mudar `x[0]` muda a média, que muda **todas** as saídas. É a primeira vez que a derivada de uma saída depende de todas as entradas do grupo.

E é a peça que torna possível empilhar blocos. Sem ela, transformers profundos costumam não convergir.

### Objetivos de aprendizagem

Ao terminar, você deve conseguir:

1. Explicar o que a normalização apaga da entrada, e por que `γ` e `β` existem para devolver parte disso.
2. Justificar `ε` com números, não como "evita divisão por zero".
3. Derivar `dx` percorrendo os três caminhos que ligam `x` à saída.
4. Explicar por que a normalização é feita dentro do exemplo, e não ao longo do lote.

### Convenções de notação

| Símbolo | Significa |
|---|---|
| `H` | o tamanho do vetor normalizado — a última dimensão |
| `μ` | a média das `H` posições de um vetor |
| `σ²` | a variância dessas mesmas `H` posições |
| `s` | `√(σ² + ε)`, o denominador da normalização |
| `x̂` | o vetor normalizado, antes de `γ` e `β` |

A convenção `dX ≡ ∂L/∂X` do overview continua valendo.

> **Guia visual.** A normalização como duas operações geométricas, os três caminhos do backward e o contraste com BatchNorm: [`layer-norm.html`](layer-norm.html). **Exercícios (12 questões):** [`exercises.html`](exercises.html).

---

## §1. O problema: a escala das ativações

> **Definição — ativação.** O valor de saída de uma camada, para uma entrada específica. Diferente de um parâmetro: o parâmetro é aprendido e fica; a ativação é calculada e some no fim da passada.

Um transformer empilha dezenas de camadas. Cada uma multiplica, soma e mistura. Se cada camada aumenta a escala dos números só um pouco, doze camadas depois esse "pouco" virou um fator enorme. Se cada uma encolhe, o sinal desaparece.

Isso não é hipótese: é multiplicação repetida. Um fator de 1,5 por camada, doze vezes, dá 130. Um fator de 0,7 dá 0,014.

Números grandes demais saturam as ativações e estouram os gradientes. Pequenos demais fazem o gradiente sumir. Nos dois casos o treino trava.

A solução desta etapa é direta: **em pontos fixos da rede, reescreva cada vetor numa escala padronizada**. Não importa em que escala ele chegou.

> **Definição — normalizar.** Subtrair a média e dividir pelo desvio padrão, de modo que o resultado tenha média `0` e desvio `1`.

O efeito é fácil de ver com números. Tome `x = [2, 3, 6, 9]`, e depois o mesmo vetor multiplicado por 100 e deslocado em 1000:

```
x        = [   2,    3,    6,    9]   →  x̂ = [-1.0954, -0.7303, 0.3651, 1.4606]
x · 100  = [ 200,  300,  600,  900]   →  x̂ = [-1.0954, -0.7303, 0.3651, 1.4606]
x + 1000 = [1002, 1003, 1006, 1009]   →  x̂ = [-1.0954, -0.7303, 0.3651, 1.4606]
```

Os três produzem exatamente o mesmo `x̂`. A normalização **descarta** a escala e o deslocamento, e preserva só o formato relativo do vetor — quem é maior que quem, e por quanto.

Guarde essa frase. A §4 existe porque descartar isso o tempo todo seria perda de informação.

---

## §2. A fórmula, passo a passo

> **Definição — Layer Normalization.** Normalizar cada vetor da última dimensão de forma independente, usando apenas as suas próprias `H` posições, e em seguida aplicar um ganho e um deslocamento treináveis.

Para um tensor `[B, T, H]`, isso acontece `B · T` vezes, uma para cada vetor de tamanho `H`. Cada uma dessas contas ignora completamente as outras.

Quatro passos:

```
1.  μ  = (1/H) · Σᵢ xᵢ                    média das H posições
2.  σ² = (1/H) · Σᵢ (xᵢ - μ)²             variância das H posições
3.  x̂ᵢ = (xᵢ - μ) / √(σ² + ε)             normalização
4.  yᵢ = γᵢ · x̂ᵢ + βᵢ                     ganho e deslocamento
```

Repare no divisor do passo 2: é `H`, não `H - 1`. Estatística amostral usa `H - 1` para estimar a variância de uma população maior. Aqui não se estima nada — as `H` posições **são** tudo que existe. O divisor é `H`.

**Exemplo numérico** com `H = 4`, `ε = 1e-5`, `γ = [1, 0.5, 2, 1.5]` e `β = [0, 1, -1, 0.5]`:

```
x        = [2, 3, 6, 9]

μ        = (2 + 3 + 6 + 9) / 4 = 5
desvios  = [-3, -2, 1, 4]
σ²       = (9 + 4 + 1 + 16) / 4 = 30 / 4 = 7.5
s        = √(7.5 + 0.00001) = 2.738615

x̂        = [-3, -2, 1, 4] / 2.738615
         = [-1.0954, -0.7303, 0.3651, 1.4606]

y        = [1·(-1.0954) + 0,  0.5·(-0.7303) + 1,  2·0.3651 - 1,  1.5·1.4606 + 0.5]
         = [-1.0954, 0.6349, -0.2697, 2.6909]
```

Confira o `x̂`: a soma das quatro posições dá `0.0000000000`, e a variância delas dá `0.99999867`. Média zero e variância um, como prometido. O `0.99999867` em vez de `1` exato é a assinatura do `ε` — a §3 volta nisso.

> **Confira você mesmo.** No exemplo acima, `y` tem média zero e variância um?
>
> <details><summary>Resposta</summary>
>
> Não. Quem tem média zero e variância um é `x̂`. Depois disso, `γ` multiplica cada posição por um valor diferente e `β` desloca cada uma por outro. A média de `y` é `0.4901` e sua variância é `1.9889`. A normalização é um passo intermediário, não a saída da camada.
> </details>

---

## §3. Por que `ε` existe

A explicação usual é "evita divisão por zero". Está certa, mas é a metade menos interessante.

**A metade óbvia.** Se as `H` posições forem todas iguais, a variância é exatamente zero:

```
x = [7, 7, 7, 7]    μ = 7    σ² = 0

sem ε:  x̂ᵢ = (7 - 7) / √0 = 0 / 0 = NaN
com ε:  x̂ᵢ = (7 - 7) / √(0.00001) = 0 / 0.003162 = 0
```

Um `NaN` contamina tudo que toca. Ele atravessa o forward, atravessa o backward e transforma todos os parâmetros do modelo em `NaN` numa única atualização. E não trava o programa: o treino continua rodando, produzindo nada.

**A metade que importa.** O problema não começa no zero exato — começa perto dele. Tome um vetor quase constante:

```
x = [7, 7, 7, 7.001]

σ²           = 0.0000001875
√σ²          = 0.00043301        ← denominador minúsculo
√(σ² + ε)    = 0.00319179        ← ε domina

sem ε:  x̂ = [-0.5774, -0.5774, -0.5774,  1.7321]
com ε:  x̂ = [-0.0783, -0.0783, -0.0783,  0.2350]
```

Sem `ε`, uma diferença de um milésimo entre as posições é amplificada até virar um desvio de 1,73. O ruído numérico do vetor sai da camada com a mesma força que um sinal de verdade teria.

Com `ε`, o denominador tem um piso. Vetores com variação real passam praticamente intactos, porque para eles `σ² ≫ ε`. Vetores quase constantes são amortecidos.

É por isso que `ε` fica **dentro** da raiz, somado à variância. Fora dela, `√σ² + ε`, ele seria só uma constante somada a um denominador que ainda pode ser zero — não resolveria nem o `NaN`.

> **Confira você mesmo.** No exemplo principal da §2, `σ² = 7.5`. Quanto o `ε = 1e-5` mudou o resultado?
>
> <details><summary>Resposta</summary>
>
> Quase nada, e é esse o objetivo. `√7.5 = 2.738613` e `√7.50001 = 2.738615`. A diferença aparece na sexta casa decimal. `ε` é um piso de segurança, e um piso só é notado por quem chega perto dele.
> </details>

---

## §4. `γ` e `β`: devolver ao modelo o que a normalização tirou

A §1 mostrou que normalizar **apaga** escala e deslocamento. Isso estabiliza o treino, mas é uma imposição forte: nenhuma camada consegue mais entregar um vetor com escala própria.

`γ` e `β` desfazem essa imposição de forma controlada. São dois vetores de tamanho `H`, treináveis, um valor por posição:

```
yᵢ = γᵢ · x̂ᵢ + βᵢ
```

> **Definição — ganho e deslocamento (`γ`, `β`).** Parâmetros treináveis que reescalam e reposicionam cada posição do vetor normalizado, de forma independente das demais.

Eles nascem em `γ = 1` e `β = 0`. Com esses valores, `y = x̂` — a camada começa sendo exatamente a normalização pura. A partir daí, o treino decide posição por posição se convém amplificar, encolher ou deslocar. No limite, o modelo pode aprender a desfazer boa parte da normalização, se isso ajudar.

Aqui vale contrastar com a Etapa 8. Lá, inicializar a matriz `W` com um valor constante era um **bug**: duas colunas idênticas recebem gradiente idêntico e permanecem gêmeas para sempre. Por que `γ = 1` em toda posição não tem o mesmo problema?

Porque `γ` não mistura posições. Cada `γᵢ` multiplica **apenas** a posição `i`, então cada um recebe um gradiente diferente já na primeira passada, vindo do `x̂ᵢ` correspondente. Em `W`, a coluna `j` via exatamente a mesma entrada que a coluna `k`. A simetria que trava o aprendizado é a simetria entre parâmetros que enxergam a mesma coisa, não a igualdade de valores iniciais.

**Backward.** Os dois são diretos, porque a operação é elemento a elemento. Pela regra do produto e da soma da Etapa 3 §1:

```
dγᵢ = Σ_linhas  dyᵢ · x̂ᵢ
dβᵢ = Σ_linhas  dyᵢ
```

O somatório percorre todas as `B · T` linhas do lote. É o mesmo mecanismo do bias da Etapa 8 §3, pela mesma razão. `γ` e `β` têm formato `[H]`, e foram broadcastados sobre as outras dimensões. O gradiente precisa então ser desbroadcastado de volta, somando o que se espalhou.

**Exemplo numérico**, com duas linhas no lote. A primeira é a da §2; a segunda é `x = [4, 4, 6, 6]`, cuja média é `5` e variância `1`, logo `x̂ = [-1, -1, 1, 1]`.

```
linha 1:  x̂ = [-1.0954, -0.7303, 0.3651, 1.4606]     dy = [0.5, -1.0, 2.0, 3.0]
linha 2:  x̂ = [-1.0000, -1.0000, 1.0000, 1.0000]     dy = [1.0,  2.0, -1.0, 0.5]

dγ linha 1 = [-0.5477,  0.7303,  0.7303,  4.3818]
dγ linha 2 = [-1.0000, -2.0000, -1.0000,  0.5000]
dγ total   = [-1.5477, -1.2697, -0.2697,  4.8818]     ← soma posição a posição

dβ total   = [0.5+1.0, -1.0+2.0, 2.0-1.0, 3.0+0.5]
           = [ 1.5000,  1.0000,  1.0000,  3.5000]     ← só a soma dos dy
```

---

## §5. Backward de `x` — a derivação

Este é o backward mais envolvido do projeto até aqui, e o motivo é estrutural. Olhe o caminho de `x₀` até a saída:

```
x₀ ──────────────────────────────► x̂₀        (direto, no numerador)
x₀ ──► μ ───────────────────────► x̂₀ ... x̂₃  (μ entra em TODAS as saídas)
x₀ ──► σ² ──────────────────────► x̂₀ ... x̂₃  (σ² também)
```

Mexer em `x₀` muda a média, e a média entra em todas as saídas. Por isso `dx₀` não depende só de `dx̂₀`: depende do vetor inteiro.

Vamos pelos três caminhos. Comece pelo mais fácil.

**Passo 1 — de `y` para `x̂`.** A operação é `yᵢ = γᵢ x̂ᵢ + βᵢ`, elemento a elemento:

```
dx̂ᵢ = dyᵢ · γᵢ
```

**Passo 2 — o caminho direto.** Tratando `μ` e `s` como constantes por um momento, `x̂ᵢ = (xᵢ - μ)/s` dá contribuição `dx̂ᵢ / s`.

**Passo 3 — o caminho pela média.** `μ` depende de todo `xⱼ` com peso `1/H`. Uma mudança em `μ` desloca todas as saídas por `-1/s`:

```
∂L/∂μ = Σᵢ dx̂ᵢ · (-1/s)        e      ∂μ/∂xⱼ = 1/H
```

Contribuição para cada `xⱼ`: `-(1/H) · Σᵢ dx̂ᵢ / s`. Ou seja, **a média dos `dx̂` é subtraída de todo mundo**.

**Passo 4 — o caminho pela variância.** `σ²` depende de `xⱼ` através de `(xⱼ - μ)²`. Derivando `x̂ᵢ = (xᵢ - μ)(σ² + ε)^(-1/2)` em relação a `σ²`, e depois `σ²` em relação a `xⱼ`, os termos se organizam em algo notavelmente limpo:

```
contribuição = -(1/H) · x̂ⱼ · Σᵢ (dx̂ᵢ · x̂ᵢ) / s
```

**Juntando os três.** Escrevendo `m₁` para a média dos `dx̂` e `m₂` para a média dos produtos `dx̂ᵢ · x̂ᵢ`:

```
m₁ = (1/H) Σᵢ dx̂ᵢ
m₂ = (1/H) Σᵢ dx̂ᵢ · x̂ᵢ

dxᵢ = ( dx̂ᵢ - m₁ - x̂ᵢ · m₂ ) / s
```

Três termos, e cada um tem leitura própria. O primeiro é o gradiente que chegou naquela posição. O segundo remove o componente que apenas deslocaria o vetor inteiro. O terceiro remove o componente que apenas o reescalaria.

Faz sentido que sejam exatamente esses dois. A §1 mostrou que a normalização **ignora** deslocamento e escala da entrada. Um gradiente que empurrasse `x` nessas duas direções não mudaria a saída em nada — e o backward, corretamente, o descarta.

**Exemplo numérico**, continuando a linha 1 da §4:

```
x̂   = [-1.0954, -0.7303, 0.3651, 1.4606]     s  = 2.738615
dy  = [0.5, -1.0, 2.0, 3.0]                  γ  = [1, 0.5, 2, 1.5]

dx̂  = [0.5·1, -1.0·0.5, 2.0·2, 3.0·1.5] = [0.5, -0.5, 4.0, 4.5]

m₁  = (0.5 - 0.5 + 4.0 + 4.5) / 4 = 2.125000
m₂  = (0.5·(-1.0954) + (-0.5)·(-0.7303) + 4·0.3651 + 4.5·1.4606) / 4 = 1.962671

dx₀ = (0.5 - 2.125 - (-1.0954 · 1.962671)) / 2.738615 =  0.191702
dx₁ = (-0.5 - 2.125 - (-0.7303 · 1.962671)) / 2.738615 = -0.435136
dx₂ = (4.0 - 2.125 - (0.3651 · 1.962671)) / 2.738615 =    0.422964
dx₃ = (4.5 - 2.125 - (1.4606 · 1.962671)) / 2.738615 =   -0.179530
```

**Verificação independente**, por dois caminhos que não usam a fórmula acima.

O primeiro é numérico. Perturbando cada `xᵢ` por `±1e-6` e medindo a variação da perda pela diferença central da Etapa 4, sai `[0.191702, -0.435136, 0.422964, -0.179530]`. O erro máximo contra a fórmula é `1.3e-9`.

O segundo é geométrico, e confirma o parágrafo sobre os dois componentes removidos:

```
Σᵢ dxᵢ        = -1.4e-16      ← soma zero: nenhum empurrão de deslocamento
Σᵢ dxᵢ · x̂ᵢ   =  3.8e-06      ← projeção nula sobre x̂: nenhum empurrão de escala
```

Os dois deveriam ser exatamente zero, e o segundo não é por causa do `ε` — o mesmo `ε` que já tinha aparecido como `0.99999867` na §2. Refazendo a conta com `ε = 0`, o resíduo cai para `5.3e-16`, que é zero até o limite do `Double`.

> **Confira você mesmo.** Se `dy` for igual em todas as posições e `γ = 1`, quanto vale `dx`?
>
> <details><summary>Resposta</summary>
>
> Zero em todas as posições. Com `dx̂` constante e igual a `c`, temos `m₁ = c` e `m₂ = c · média(x̂) = 0`. Sobra `(c - c - 0)/s = 0`. Faz sentido: um `dy` uniforme pede para empurrar o vetor inteiro na mesma direção, e a normalização descarta exatamente isso.
> </details>

---

## §6. Duas implementações

O roadmap oferece dois caminhos, e a §5 acabou de justificar por que vale conhecer os dois.

**Opção A — composição.** Escrever o forward com as operações que já existem e deixar o autograd derivar tudo:

```
xc   = x - x.mean(dim = rank-1, keepDim = true)
var  = xc.pow(2).mean(dim = rank-1, keepDim = true)
x̂    = xc / (var + ε).pow(0.5)
y    = x̂ * γ + β
```

Nenhum backward manual. Cada operação já tem o seu, testado desde a Etapa 3, e a regra da cadeia da Etapa 2 costura tudo. Custa tensores intermediários e um grafo com mais nós.

O `keepDim = true` não é detalhe. Sem ele, a média de um `[B, T, H]` sai como `[B, T]`, e a subtração seguinte alinha as dimensões pela direita — `T` contra `H`. Ou o formato é incompatível e o erro aparece, ou por azar `T == H` e ele **não** aparece: a conta roda subtraindo o valor errado de cada posição. Com `keepDim = true` a média sai `[B, T, 1]`, e o `1` broadcasta sobre `H` exatamente como se quer.

**Opção B — fórmula fechada.** Implementar o backward da §5 diretamente, num nó só. Menos alocação, um grafo curto, e a fórmula já está derivada e verificada acima.

**Comece pela A.** Ela é mais curta, mais difícil de errar, e o gradient check da Etapa 4 vale como referência para a B depois. Quando a B existir, o teste natural é comparar as duas: mesmo `x`, mesmo `dy`, e os gradientes têm que bater em toda posição.

**Uma armadilha de teste, herdada da Etapa 6.** A perda óbvia para o gradient check seria `forward(x).sum`. Ela não serve, pelo mesmo motivo do softmax: com `γ = 1` e `β = 0`, essa soma é `Σ x̂`, que vale zero para qualquer `x`. É uma função constante, e o gradiente verdadeiro dela é zero.

Medido neste projeto, com as três linhas de exemplo:

```
gradiente analítico de sum(LN(x))    = [0, 0, ...]   exatamente zero
gradiente numérico                    ≈ ruído, ~2e-11

Gradcheck com forward(x).sum          = 2.2e-3     ← acima do limite de 1e-5
Gradcheck com (forward(x)*pesos).sum  = 3.1e-10    ← um teste de verdade
```

O teste degenerado compara ruído contra zero. Ele não aprova em silêncio — nesta implementação chega a reprovar código correto —, mas o veredito dele, qualquer que seja, não fala sobre o backward. Use pesos distintos.

> **Armadilha.** Escrever `1 / H` para a média, com `H` inteiro.
>
> Em Scala, `1 / 4` é divisão inteira e vale `0`. A média inteira vira zero, `x̂` vira zero, e a camada devolve exatamente `β` para qualquer entrada. Este bug já aconteceu neste projeto: na Etapa 8, `Math.sqrt(2 / inputDim)` truncava para `0` sempre que `inputDim > 2`, e a matriz de pesos nascia inteiramente zerada.
>
> Por que passou despercebido: o forward roda, o formato da saída está correto, e nenhuma exceção é lançada. Só os valores estão errados.
>
> **Lição geral:** em conta com divisão, force um dos operandos a `Double` (`1.0 / H`). O compilador não vai avisar — os dois tipos são válidos.

> **Armadilha.** Criar `γ` e `β` sem `requiresGradient = true`.
>
> Também já aconteceu aqui: na Etapa 8, o bias `b` do `Linear` nasceu fora do grafo. O modelo treina, a perda até cai — mas dois parâmetros nunca são atualizados, e o otimizador não tem como reclamar de algo que não está na lista.
>
> Por que passou despercebido: com `γ = 1` e `β = 0`, a camada normaliza corretamente. Ela só nunca aprende a ajustar a escala, e a diferença aparece como um modelo um pouco pior, não como um erro.
>
> **Lição geral:** todo tensor que o otimizador deve atualizar precisa de `requiresGradient = true` **e** de estar em `parameters`. Um teste que verifica as duas coisas custa três linhas.

---

## §7. Por que dentro do exemplo, e não ao longo do lote

> **Definição — BatchNorm.** A alternativa histórica: normalizar cada posição usando a média e a variância **daquela posição ao longo de todos os exemplos do lote**.

A diferença é a direção em que as estatísticas são calculadas. LayerNorm olha para dentro de um exemplo. BatchNorm olha para o lado, entre exemplos.

Isso soa como detalhe até se olhar um caso concreto. Tome um lote com duas sequências em escalas diferentes:

```
linha A = [  2,   3,   6,   9]
linha B = [200, 300, 600, 900]
```

Com LayerNorm, cada linha usa só as suas próprias estatísticas:

```
A → [-1.0954, -0.7303, 0.3651, 1.4606]
B → [-1.0954, -0.7303, 0.3651, 1.4606]
```

Com BatchNorm, a posição 0 é normalizada usando `[2, 200]`, a posição 1 usando `[3, 300]`, e assim por diante. Média `101` e desvio `99` na posição 0, média `151.5` e desvio `148.5` na posição 1:

```
A → [-1, -1, -1, -1]
B → [ 1,  1,  1,  1]
```

Toda a estrutura interna de cada linha foi destruída. Pior: **a saída de A depende de B**. Trocar a outra sequência do lote muda o que A vira, mesmo sem tocar em A.

Isso traz três problemas concretos para um modelo de linguagem.

O primeiro é a geração, na Etapa 19, que processa uma sequência de cada vez. Com `B = 1`, a variância ao longo do lote é exatamente zero, e não há estatística nenhuma para usar. BatchNorm precisa guardar médias acumuladas do treino e trocar de comportamento na inferência.

O segundo é o comprimento variável. Sequências de tamanhos diferentes num lote significam posições com quantidades diferentes de exemplos reais.

O terceiro é a reprodutibilidade. Com BatchNorm, a saída para uma entrada depende de quem mais estava no lote.

LayerNorm não tem nenhum desses. A conta de uma linha só depende daquela linha — no treino, na geração, com lote de 1 ou de 1000.

---

## §8. Onde ela entra no bloco

Duas posições possíveis, e a escolha tem consequência.

> **Definição — post-LN e pre-LN.** Post-LN aplica a normalização **depois** da subcamada e da soma residual: `x + Sub(x)`, depois normaliza. Pre-LN normaliza **antes** da subcamada e soma o resultado: `x + Sub(LN(x))`.

O paper original do transformer usa post-LN. O GPT-2 mudou para pre-LN, e é o que este projeto vai construir na Etapa 14.

A razão aparece quando se pergunta o que acontece com o caminho direto — aquele `x +` que atravessa o bloco sem passar por nada. Em pre-LN, esse caminho fica intocado: o `x` original chega ao fim do bloco exatamente como entrou, e o gradiente volta por ele sem ser reescalado. Em post-LN, toda soma residual passa por uma normalização, e o caminho direto é reescalado a cada bloco.

Com muitos blocos empilhados, a diferença é entre um treino que começa estável e um que precisa de aquecimento cuidadoso da taxa de aprendizado.

**Exemplo numérico.** Usando o `x = [2, 3, 6, 9]` da §2, com `γ = 1` e `β = 0`, e supondo que a subcamada devolva `[0.1, 0.1, 0.1, 0.1]`:

```
pre-LN:   x + Sub(LN(x))  = [2.1, 3.1, 6.1, 9.1]     ← escala original preservada
post-LN:  LN(x + Sub(x))  = [-1.0954, -0.7303, 0.3651, 1.4606]
```

Em post-LN, a escala de `x` foi descartada na saída do bloco. O bloco seguinte não tem como saber quão grande era o sinal que chegou.

---

## §9. Para onde isso leva

Esta é a primeira camada com backward de verdade, e a técnica usada aqui se repete.

O padrão da §5 tem três passos: listar os caminhos, derivar cada um, e reconhecer que os termos extras removem as direções que a operação ignora. É o mesmo que a Etapa 6 usou no softmax, e o que a Etapa 11 vai usar na atenção. Não por acaso: softmax e LayerNorm são as duas operações do transformer que acoplam todas as posições de um eixo.

A Etapa 11 traz a atenção, o mecanismo central do modelo. A partir daí, LayerNorm aparece duas vezes por bloco, e o bloco se repete.

---

## Cartão de referência

| Conceito | Resumo |
|---|---|
| eixo normalizado | a última dimensão, `H`, dentro de cada exemplo |
| `μ`, `σ²` | média e variância das `H` posições; variância divide por `H`, não `H-1` |
| `s` | `√(σ² + ε)`, com `ε = 1e-5` **dentro** da raiz |
| `x̂` | `(x - μ)/s`, com média `0` e variância `1` |
| `γ`, `β` | `[H]`, treináveis, init `1` e `0` — identidade no começo |
| o que a normalização apaga | escala e deslocamento da entrada |
| `dγ` | `Σ_linhas dy · x̂` |
| `dβ` | `Σ_linhas dy` |
| `dx` | `(dx̂ - m₁ - x̂·m₂)/s`, com `m₁` e `m₂` as médias de `dx̂` e `dx̂·x̂` |
| conferência de `dx` | `Σ dx = 0` e `Σ dx·x̂ = 0` |
| implementação | comece pela composição de ops; fórmula fechada depois |
| posição no bloco | pre-LN, como o GPT-2 |

### As quatro lições que se repetem

1. **Normalizar é descartar informação de propósito.** `γ` e `β` existem para devolver, sob controle do treino, o que foi descartado.
2. **`ε` não é só antizero.** Ele é um piso que impede a amplificação de ruído em vetores quase constantes.
3. **Quando uma operação acopla um eixo inteiro, o backward ganha termos de correção.** Eles removem as direções que o forward ignora — e é isso que os torna previsíveis, não decorados.
4. **Toda propriedade do forward tem eco no backward.** A invariância a deslocamento e escala vira `Σ dx = 0` e `Σ dx·x̂ = 0`, que servem como teste.
