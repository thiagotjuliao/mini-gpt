# Etapa 2 — Autograd

## Antes de começar

### O que você vai construir

O mecanismo que calcula derivadas **automaticamente**. Ao final desta etapa, o `Tensor` da Etapa 1 saberá:

| Capacidade | O que passa a existir |
|---|---|
| Lembrar de onde veio | `previous` — os tensores que originaram este |
| Saber como devolver o gradiente | `_backward` — a regra local da operação |
| Propagar gradiente pelo grafo todo | `backward()` |
| Limpar gradientes entre passos | `zeroGrad()` |
| Desligar o mecanismo quando não é preciso | `noGrad` |

Esta é a etapa que separa "uma biblioteca de arrays" de "uma biblioteca que treina modelos".

### O que você precisa saber antes

**Da Etapa 1:** o que é um `Tensor`, e que ele é comparado por referência — dois tensores com os mesmos valores são nós **diferentes** do grafo.

**De cálculo:** derivada e regra da cadeia. Se `dz/dx = dz/dy · dy/dx` não é familiar, revise antes de continuar. É a única ferramenta matemática do capítulo, mas ela é usada o tempo todo.

### Onde esta etapa se encaixa

A Etapa 1 construiu o recipiente. Ele guarda números e nada mais — não faz ideia de onde os valores vieram.

Esta etapa constrói o **motor**. Ao final, cada tensor sabe responder à pergunta que torna o aprendizado possível: *se eu mexer um pouquinho neste número, quanto o resultado final muda?*

Uma comparação útil: pense num carro. A Etapa 1 fabricou as peças. Esta etapa monta a transmissão — o sistema que leva força de uma ponta à outra, na ordem certa. A Etapa 3 fabrica os pistões que de fato geram essa força, um por operação.

### Objetivos de aprendizagem

Ao terminar, você deve conseguir:

1. Explicar o que é a perda `L` e por que ela precisa ser um único número.
2. Percorrer um grafo pequeno à mão e calcular todos os gradientes.
3. Justificar por que o backward exige ordenação topológica, e não uma ordem qualquer.
4. Mostrar, com números, o que quebra se o gradiente for sobrescrito em vez de acumulado.

> **Guia visual.** O grafo como DAG, a ordenação topológica e a convergência de gradiente: [`computational-graph.html`](computational-graph.html). **Exercícios (10 questões):** [`exercises.html`](exercises.html).

---

## Convenções de notação

Esta notação vale para todos os capítulos daqui em diante.

| Símbolo | Significa |
|---|---|
| `A`, `B`, `C` | tensores no forward |
| `dA` | `∂L/∂A` — gradiente da **perda** em relação a `A` |
| `dC` | o gradiente que chega de cima, vindo da operação seguinte |
| `∂C/∂A` | a derivada **local** da operação, isolada do grafo |
| `L` | a perda |

Uma armadilha de leitura, que vale resolver antes de qualquer conta:

**`dA` não significa "a derivada de `A`".** Significa a derivada de **outra coisa** — a perda — em relação a `A`. Toda vez que você vir um `d` colado num nome, leia "quanto a perda responde a isto".

---

## §1. O problema que estamos resolvendo

### O que é treinar

> **Definição — parâmetro (ou peso).** Um número dentro do modelo que pode ser ajustado. Um GPT pequeno tem milhões deles; o GPT-3 tem 175 bilhões. Eles começam aleatórios e vão sendo corrigidos.

> **Definição — perda (`L`).** **Um único número** que mede o quão errada está a previsão do modelo. Quanto menor, melhor. Se o modelo acerta em cheio, a perda é zero.

Treinar é um problema de otimização, e ele cabe numa frase: **ajustar os parâmetros para que a perda diminua.**

Que `L` seja um número só, e não um tensor, é o que faz todo o mecanismo funcionar. Existe uma única quantidade para minimizar, e portanto uma única pergunta a fazer sobre cada parâmetro: *aumentar este peso faz a perda subir ou descer, e com que intensidade?*

Essa pergunta é exatamente `∂L/∂w`.

### Como o gradiente é usado

Sabendo `∂L/∂w`, a correção é direta:

```
w ← w - lr · ∂L/∂w
```

> **Definição — taxa de aprendizado (`lr`, de *learning rate*).** Um número pequeno que controla o tamanho do passo. Alto demais, o ajuste passa do ponto e oscila. Baixo demais, o treino leva uma eternidade.

Repare no **sinal de menos**. O gradiente aponta na direção em que a perda **cresce**. Como queremos que ela diminua, andamos no sentido contrário. Daí o nome do método: *gradiente descendente*.

### Um passo de treino completo

Nada torna isso concreto como fazer as contas. Vamos treinar o menor modelo possível.

**A tarefa:** aprender a função `y = 2x`. O modelo tem um único parâmetro `w`, e prevê `ŷ = w · x`. Ele começa errado, com `w = 0.5`.

**O exemplo de treino:** `x = 3`, e a resposta certa é `y = 6`.

**A perda:** o erro ao quadrado, `L = (ŷ - y)²`. O quadrado serve para que errar para cima e errar para baixo custem igual.

**A derivada da perda**, pela regra da cadeia — sendo `L = (ŷ-y)²` e `ŷ = w·x`:

```
∂L/∂ŷ = 2(ŷ - y)
∂ŷ/∂w = x
∂L/∂w = 2(ŷ - y) · x
```

Agora três passos, com `lr = 0.05`:

```
passo      w      ŷ = w·x     L = (ŷ-y)²     ∂L/∂w      novo w
   1    0.5000     1.5000      20.250000    -27.0000    1.8500
   2    1.8500     5.5500       0.202500     -2.7000    1.9850
   3    1.9850     5.9550       0.002025     -0.2700    1.9985
```

Confira o primeiro passo à mão. Com `w = 0.5`, a previsão é `1.5`, enquanto a resposta certa é `6`. O erro é `-4.5`, e a perda é `20.25`. A derivada vale `2·(-4.5)·3 = -27`. Ela é **negativa**, o que significa: aumentar `w` faz a perda cair. Então o ajuste soma — `0.5 - 0.05·(-27) = 1.85`.

Olhe a coluna da perda: `20.25 → 0.2025 → 0.002025`. Ela cai por um fator de 100 a cada passo, e `w` caminha para `2` — o valor certo, que o modelo nunca viu.

**Isto é aprendizado.** Todo o resto do projeto é escala: milhões de parâmetros em vez de um, e uma perda mais elaborada. O mecanismo é este.

### Por que automatizar

O exemplo acima tem um parâmetro e duas operações. Deu para derivar à mão.

Um transformer tem milhões de parâmetros, compostos em centenas de operações encadeadas. Derivar cada `∂L/∂w` manualmente é impossível — e refazer tudo a cada mudança de arquitetura, mais ainda.

> **Definição — autograd.** O mecanismo que aplica a regra da cadeia automaticamente, por toda a computação, sem que ninguém escreva as derivadas do modelo à mão.

É isso que esta etapa constrói.

---

## §2. Regra da cadeia

Se `y = f(x)` e `z = g(y)`, então:

```
dz/dx = dz/dy · dy/dx
```

A leitura importante é de **fluxo**: para saber como `z`, lá na saída, responde a `x`, lá na entrada, multiplicamos as sensibilidades de cada etapa do caminho. Isso vale para cadeias de qualquer comprimento — basta multiplicar mais termos.

**Exemplo numérico.** Tome `y = 3x` e `z = y²`, com `x = 2`.

```
forward:   y = 3·2 = 6        z = 6² = 36

derivadas locais:
  dy/dx = 3          (a derivada de 3x)
  dz/dy = 2y = 12    (a derivada de y², avaliada em y=6)

regra da cadeia:
  dz/dx = 12 · 3 = 36
```

**Verificação independente.** Substituindo antes de derivar: `z = (3x)² = 9x²`, logo `dz/dx = 18x = 18·2 = 36`. Os dois caminhos batem.

Repare no que a verificação mostra. Não é preciso compor as funções simbolicamente. Basta cada operação conhecer a **sua própria** derivada, e multiplicar ao longo do caminho. É exatamente por isso que o autograd é possível.

### Quando há mais de um caminho

Aqui está o detalhe que mais causa bug no projeto inteiro.

Se uma variável influencia a saída por **vários caminhos**, a regra muda: as contribuições **somam**.

```
x usado em dois lugares   →   dL/dx = (contribuição do caminho 1) + (contribuição do caminho 2)
```

Isso é a regra da cadeia multivariável. É a razão de existir o `+=` na acumulação de gradiente, e a seção §6 mostra, com números, o que acontece quando alguém esquece disso.

---

## §3. O grafo de computação

Cada operação cria um tensor de saída que **lembra de onde veio**. Dois campos bastam:

- `previous` — os tensores de entrada que geraram este.
- `_backward` — uma função que sabe empurrar o gradiente de volta para essas entradas, usando a derivada local da operação.

> **Definição — DAG (grafo acíclico dirigido).** Os nós são tensores; as arestas apontam de uma operação para suas entradas. É acíclico porque nenhum cálculo pode depender do próprio resultado.

> **Definição — folha e raiz.** As **folhas** são os tensores que não vieram de operação nenhuma: as entradas e os parâmetros. A **raiz** é o tensor final, do qual o backward parte — normalmente a perda.

Este é o grafo que acompanha o capítulo inteiro:

```
a ──┐
    ├─(add)── c ──┐
b ──┘             ├─(mul)── d
a ────────────────┘
```

Repare que `a` aparece **duas vezes**: uma diretamente no `mul`, outra através de `c`. É o caso de "múltiplos caminhos" da seção anterior, e não é um exemplo artificial — conexões residuais, que você vai construir na Etapa 14, têm exatamente esta forma.

**Exemplo numérico**, com `a = 2` e `b = 3`:

```
c = a + b = 2 + 3 = 5
d = c · a = 5 · 2 = 10

previous(c) = {a, b}
previous(d) = {c, a}
folhas: a, b        raiz: d
```

Note que `previous(d)` contém `a` diretamente, e também `c` — que por sua vez contém `a`. É o mesmo tensor `a` alcançável por dois caminhos distintos. Guarde isso: é o que torna a próxima seção necessária.

---

## §4. Por que a ordem importa

> **Guia visual.** O caso do grafo diamante, e por que uma ordem ingênua falha: [`computational-graph.html`](computational-graph.html).

O `backward()` precisa visitar os nós numa ordem específica. A regra é:

**Quando processamos um nó, todo o gradiente que ele vai receber já precisa ter chegado.**

Se um nó for processado cedo demais, ele propaga adiante um gradiente **incompleto** — faltando as contribuições dos caminhos ainda não percorridos. E o erro se espalha para tudo que vem depois dele.

> **Definição — ordenação topológica.** Uma ordem dos nós em que todo nó aparece depois de todos os seus predecessores.

O backward usa essa ordem **invertida**: começa na raiz, e só chega numa folha depois de ter passado por todos os seus consumidores.

**Por que funciona.** Se `X` é predecessor de `Y`, a busca em profundidade garante que `X` entra na lista antes de `Y`. Invertendo a lista, `Y` vem antes de `X` — que é exatamente o que o backward precisa: consumidores antes de dependências.

**Exemplo numérico**, com o grafo do §3:

```
ordem topológica:            a, b, c, d
ordem do backward (reversa): d, c, b, a
```

Acompanhe por que essa ordem é a única que funciona. O tensor `a` recebe gradiente de duas fontes: de `d` (pelo `mul`) e de `c` (pelo `add`). Na ordem reversa, tanto `d` quanto `c` são processados **antes** de `a`. Quando chega a vez de `a`, as duas contribuições já chegaram.

Agora imagine processar `a` logo depois de `d`, antes de `c`. O gradiente de `a` sairia valendo `5`, faltando a contribuição `2` que viria por `c`. O valor correto é `7`.

> **Armadilha.** A ordenação topológica deste projeto teve dois bugs reais, e ambos são instrutivos.
>
> O primeiro era um laço infinito: a recursão empilhava os predecessores mas esquecia de remover o nó já processado do topo da pilha. Travava em qualquer grafo — ou seja, em todos.
>
> O segundo é mais sutil, e passou por vários testes. O algoritmo funcionava para folhas compartilhadas, mas duplicava indefinidamente quando o nó compartilhado por dois consumidores era ele mesmo um nó **intermediário**. O ramo que emitia um nó "pronto" esquecia de marcá-lo como visitado.
>
> Por que passou despercebido: o grafo diamante clássico, com uma folha compartilhada, passava. Só o caso do intermediário compartilhado quebrava — que é precisamente a forma de uma conexão residual.
>
> **Lição geral:** ao testar um algoritmo de grafo, o caso interessante não é o nó compartilhado. É o nó compartilhado que **não é folha**.

> **Confira você mesmo.** Por que o backward não pode simplesmente visitar os nós na ordem em que foram criados, invertida?
>
> <details><summary>Resposta</summary>
>
> Porque a ordem de criação não garante nada sobre as dependências do grafo. Dois ramos independentes podem ser criados intercalados, e a ordem invertida colocaria um consumidor depois de sua dependência. A ordenação topológica olha para as arestas reais, não para o relógio.
> </details>

---

## §5. Um backward completo, à mão

Vamos percorrer o grafo inteiro. `a = 2`, `b = 3`.

**Forward:**
```
c = a + b = 5
d = c · a = 10
```

**Semeando a raiz.** O backward começa com `d.grad = 1`. Isso não é uma convenção arbitrária: é `∂d/∂d`, a derivada de `d` em relação a si mesmo, que vale 1 por definição.

**Passo 1 — o `mul` (`d = c · a`).** A derivada de um produto em relação a um fator é o outro fator:

```
∂d/∂c = a = 2          ∂d/∂a = c = 5

c.grad += d.grad · a = 1 · 2 = 2
a.grad += d.grad · c = 1 · 5 = 5      ← primeira contribuição a `a`
```

**Passo 2 — o `add` (`c = a + b`).** A derivada da soma é 1 dos dois lados:

```
∂c/∂a = 1              ∂c/∂b = 1

a.grad += c.grad · 1 = 2    →  a.grad total = 5 + 2 = 7
b.grad += c.grad · 1 = 2
```

**Resultado:** `a.grad = 7`, `b.grad = 2`, `c.grad = 2`.

**Verificação independente.** Vamos conferir por um caminho que não usa o grafo. Substituindo `c = a+b` na expressão de `d`:

```
d = (a + b) · a = a² + ab

∂d/∂a = 2a + b = 2·2 + 3 = 7    ✓
∂d/∂b = a = 2                    ✓
```

A derivação simbólica direta bate com o que o algoritmo do grafo produziu. Isso confirma que "somar as contribuições de todos os caminhos" não é uma regra inventada — é o que o cálculo exige.

---

## §6. Por que acumular (`+=`) e nunca sobrescrever (`=`)

Esta é a seção mais curta do capítulo e a que mais previne bugs.

Quando um tensor é usado em várias operações, cada uma contribui uma parcela do gradiente total. Se a segunda contribuição **sobrescrevesse** a primeira em vez de somar, a primeira seria perdida.

**Exemplo numérico.** Mesmo grafo, rodado dos dois jeitos:

```
                    a.grad    b.grad    c.grad
acumulando (+=)         7         2         2      ← correto
sobrescrevendo (=)      2         2         2      ← errado
```

Veja onde o `7` se perde. Na ordem reversa, o `mul` roda primeiro e escreve `5` em `a.grad`. Depois o `add` roda e escreve `2`. Com `+=`, o total é `7`. Com `=`, o `5` é simplesmente apagado, e sobra `2`.

Compare com a derivação simbólica do §5: `∂d/∂a = 2a + b = 7`. A versão que sobrescreve devolve `2` — ela perdeu inteiramente o termo `2a`, que é justamente o caminho direto pelo `mul`.

E o mais perigoso: **nada quebra.** Nenhuma exceção, nenhum aviso. O código roda, devolve um número plausível, e o modelo só treina mal.

### A consequência: `zeroGrad()`

Se o gradiente acumula, ele acumula **entre passos de treino também**. Sem limpar, o gradiente do passo anterior soma ao do passo atual, e o modelo caminha numa direção que mistura informação velha com nova.

Daí a existência de `zeroGrad()`, que precisa ser chamado antes de cada novo `backward()`.

> **Armadilha.** A primeira versão de `backward()` neste projeto **não semeava** a raiz com `1.0`. Todo `_backward` rodava sobre um array de gradiente zerado.
>
> Por que é traiçoeiro: multiplicar por zero propaga zero. O algoritmo percorria o grafo inteiro, na ordem certa, executando tudo corretamente — e escrevia zero em toda parte. Nenhum erro, nenhuma exceção. Simplesmente não fazia nada.
>
> **Lição geral:** a regra da cadeia é uma cadeia de multiplicações. Ela precisa começar em 1, não em 0.

> **Confira você mesmo.** No grafo do §5, suponha que `a` fosse usado em **três** lugares em vez de dois, e que a terceira contribuição valesse `4`. Qual seria `a.grad`?
>
> <details><summary>Resposta</summary>
>
> `11`. As contribuições somam: `5` (pelo `mul`) mais `2` (pelo `add`) mais `4` (pelo novo caminho). O número de caminhos não muda a regra — cada um acrescenta a sua parcela.
> </details>

---

## §7. `noGrad` — desligando o mecanismo

Construir o grafo custa. Cada operação guarda referências aos seus predecessores e cria uma closure. Isso consome memória e tempo.

Às vezes esse custo é puro desperdício. Ao **gerar texto** com um modelo já treinado (Etapa 19), nunca vamos chamar `backward()`. O grafo é construído e imediatamente jogado fora.

> **Definição — `noGrad`.** Um bloco dentro do qual as operações pulam a construção do grafo. O forward continua funcionando normalmente; o que se perde é a capacidade de fazer o backward depois.

```
fora de noGrad:     a + b  →  resultado com previous = {a, b}
dentro de noGrad:   a + b  →  resultado com previous = {}, requiresGradient = false
```

O valor numérico do resultado é idêntico nos dois casos. Só o rastro desaparece.

> **Armadilha.** Este ponto gerou dois bugs, e o segundo reapareceu três vezes ao longo do projeto.
>
> O primeiro: a flag global nasceu com valor padrão `false`. Como o único jeito de mexer nela era o `noGrad`, que apenas desliga, o autograd inteiro ficava morto por padrão. Uma soma comum saía sem grafo nenhum.
>
> O segundo é de precedência de operadores:
>
> ```scala
> val reqGrad = t1.requiresGradient || t2.requiresGradient && Tensor.gradEnabled
> ```
>
> Em Scala, `&&` liga mais forte que `||`. Isso lê como `t1 || (t2 && gradEnabled)`. Se `t1` já exigisse gradiente, o resultado saía `true` mesmo dentro de um `noGrad`. A correção são parênteses explícitos: `(t1 || t2) && gradEnabled`.
>
> **Lição geral:** esse mesmo erro de precedência reapareceu em `sum(dim)` na Etapa 3 e em `reshape`/`transpose` depois. Ao combinar uma condição de negócio com uma flag global, ponha os parênteses mesmo quando parecerem desnecessários.

---

## §8. Para onde isso leva

O autograd está pronto — mas ele é uma casca.

Ele sabe percorrer o grafo na ordem certa. Sabe acumular contribuições de múltiplos caminhos. Sabe quando não construir grafo nenhum. O que ele **não** sabe é a derivada de nada.

Cada `_backward` precisa ser escrito, operação por operação. Qual é a derivada de uma multiplicação? De uma exponencial? De uma multiplicação de matrizes?

A Etapa 3 responde a essas perguntas doze vezes — uma por operação. É a etapa mais longa do projeto, e ao final dela você terá toda a caixa de ferramentas matemática que um transformer usa.

---

## Cartão de referência

| Conceito | Resumo |
|---|---|
| perda `L` | um único número; quanto menor, melhor o modelo |
| `∂L/∂w` | quanto a perda responde a um parâmetro |
| gradiente descendente | `w ← w - lr · ∂L/∂w` |
| por que o menos | o gradiente aponta para onde a perda **cresce** |
| `previous` | os tensores que originaram este |
| `_backward` | a regra local, que empurra o gradiente para trás |
| folha | tensor sem predecessores: entrada ou parâmetro |
| raiz | tensor final, de onde o backward parte |
| semente do backward | `1.0`, porque `∂L/∂L = 1` |
| ordem do backward | topológica reversa: consumidores antes de dependências |
| múltiplos caminhos | as contribuições **somam** |
| `zeroGrad()` | limpa entre passos, porque o gradiente acumula |
| `noGrad` | pula a construção do grafo; o forward não muda |

### As quatro lições que se repetem

1. **A perda é um número só.** É isso que permite fazer uma única pergunta sobre cada parâmetro, e é a base de todo o mecanismo.
2. **Somar, nunca sobrescrever.** Um tensor usado em `n` caminhos recebe `n` contribuições, e todas contam.
3. **A ordem não é detalhe.** Processar um nó cedo demais propaga gradiente incompleto para tudo que vem depois.
4. **Erros de gradiente são silenciosos.** Semente zerada, contribuição sobrescrita, precedência trocada — nenhum deles lança exceção. O forward continua perfeito e o modelo só treina mal. É por isso que a Etapa 4 existe.
