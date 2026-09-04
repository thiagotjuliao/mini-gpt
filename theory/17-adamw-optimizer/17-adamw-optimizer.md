# Etapa 17 — AdamW: o otimizador

## Antes de começar

### O que você vai construir

A primeira peça do projeto cujo trabalho é **mudar** alguma coisa.

| Componente | Onde mora | O que faz |
|---|---|---|
| `Tensor.updateData` | `scalagrad.core` | escreve valores novos no `data` de um parâmetro |
| `AdamW` | `gpt.optim` | lê `p.gradient`, calcula o passo, escreve em `p` |
| estado `m`, `v` | dentro do `AdamW` | dois arrays por parâmetro, que sobrevivem entre passos |
| contador `t` | dentro do `AdamW` | quantos passos já foram dados |
| `step()` | `AdamW` | atualiza os parâmetros e devolve **um `AdamW` novo**, com `t + 1` |

Zero tensores novos no grafo. O otimizador não participa do `backward()` — ele roda depois que o `backward()` terminou.

### O que você precisa saber antes

**Da Etapa 1 §3:** strides, e o que significa um tensor ser contíguo. A §8 depende disso.

**Da Etapa 2:** que `p.gradient` acumula, e que `backward()` preenche o gradiente de todo parâmetro alcançável a partir da perda.

**Da Etapa 8 §7:** a lista `parameters`, que cada camada expõe. O `GPT` da Etapa 15 concatena todas elas.

**Da Etapa 16:** a perda. É dela que o `backward()` parte.

Cálculo: você precisa saber que a derivada aponta para onde a função **cresce**. Todo o resto sai daí.

### Onde esta etapa se encaixa

Até aqui o projeto sabe medir o erro e sabe distribuir a culpa. Não sabe consertar nada.

O `backward()` da Etapa 2 responde uma pergunta local. Se este parâmetro subir um pouquinho, a perda sobe ou desce, e com que inclinação? A resposta é o gradiente. Ela não diz **quanto** mexer, e é essa a decisão que falta.

Parece uma decisão pequena. É a diferença entre um modelo que converge em mil passos e um que oscila para sempre.

### Objetivos de aprendizagem

Ao terminar, você deve conseguir:

1. Explicar por que um único learning rate não serve para parâmetros com gradientes de escalas diferentes.
2. Calcular `m` e `v` à mão para três passos, e dizer o que cada um mede.
3. Mostrar que o primeiro passo do Adam vale exatamente `lr`, e por que a correção de viés garante isso.
4. Explicar por que L2 dentro do Adam decai errado, e o que o AdamW muda.
5. Dizer por que o estado do otimizador é indexado por identidade de referência, e não por valor.

### Convenções de notação

| Símbolo | Significa |
|---|---|
| `p` | um parâmetro — um `Tensor` com `requiresGradient = true` |
| `g` | `dp`, o gradiente da perda em relação a `p`, já calculado pelo `backward()` |
| `m` | primeiro momento: média móvel de `g` |
| `v` | segundo momento: média móvel de `g²` |
| `m̂`, `v̂` | os mesmos, depois da correção de viés |
| `t` | o número do passo, contando a partir de **1** |
| `lr` | taxa de aprendizado — `3e-4` neste projeto |
| `β1`, `β2` | as taxas de esquecimento de `m` e `v` — `0.9` e `0.95` |
| `ε` | `1e-8`, só para não dividir por zero |
| `λ` | a força do weight decay — `0.1` |

A convenção `dX ≡ ∂L/∂X` continua valendo. Aqui ela aparece com o nome curto `g`, porque a fórmula do Adam a repete muitas vezes.

> **Guia visual.** As duas médias móveis, a correção de viés e a diferença entre L2 e AdamW: [`adamw.html`](adamw.html). **Exercícios (14 questões):** [`exercises.html`](exercises.html).

**O exemplo que atravessa o capítulo.** Um parâmetro de três elementos, e três passos de gradiente:

```
p₀ = [ 0.60, -0.20,  0.05]

g₁ = [ 0.30,  0.03, -0.60]
g₂ = [ 0.10,  0.05, -0.20]
g₃ = [-0.20,  0.04, -0.10]
```

As três posições contam histórias diferentes. A posição 0 tem gradiente que **troca de sinal** no passo 3. A posição 1 tem gradiente dez vezes menor que as outras. A posição 2 tem o maior gradiente, e sinal constante.

---

## §1. O passo mais simples: SGD

> **Definição — otimizador.** A regra que decide como transformar os gradientes em mudanças nos parâmetros. Ele roda depois do `backward()`, uma vez por lote.

O gradiente aponta para onde a perda **cresce**. Queremos que ela diminua. Então andamos no sentido oposto:

```
p ← p − lr · g
```

> **Definição — taxa de aprendizado (`lr`).** O fator que multiplica o gradiente para virar um passo. É o único controle de "quanto mexer" que o SGD tem.

Isso é o **SGD**, de *stochastic gradient descent*. É estocástico porque `g` vem de um lote sorteado, e não do corpus inteiro. Cada passo anda numa estimativa ruidosa da direção certa.

Duas propriedades do SGD importam para o resto do capítulo. O passo é **proporcional ao gradiente**. E o passo **não tem memória**: o que aconteceu antes não influencia este.

**O exemplo numérico.** Com `lr = 3e-4` e `g₁`:

```
passo = lr · g₁ = [9.0e-5, 9.0e-6, -1.8e-4]
p₁    = [0.599910, -0.200009, 0.050180]
```

A posição 1 andou `9.0e-6`. A posição 2 andou `1.8e-4`, vinte vezes mais. A razão entre os passos é exatamente a razão entre os gradientes, e é isso que a §2 questiona.

---

## §2. Por que um único learning rate não serve

Um GPT pequeno tem dezenas de milhares de parâmetros. Eles não vivem na mesma escala.

O gradiente de um peso da atenção e o de um `gamma` do LayerNorm podem diferir por três ordens de grandeza. A causa é estrutural. Cada parâmetro está a uma profundidade diferente do grafo, multiplicado por um conjunto diferente de fatores na regra da cadeia.

Com um `lr` só, você escolhe entre dois males.

**`lr` grande.** Bom para os parâmetros de gradiente pequeno, que finalmente saem do lugar. Os de gradiente grande dão passos enormes, passam do ponto, e a perda oscila ou explode.

**`lr` pequeno.** Seguro para os de gradiente grande. Os de gradiente pequeno quase não se movem, e nunca aprendem nada.

Não existe valor que sirva para os dois. O `lr` é um número, e o problema tem milhares de escalas.

A saída do Adam é abandonar a proporcionalidade. Em vez de "ande proporcional ao gradiente", a regra vira **"ande um passo de tamanho `lr`, no sentido que o gradiente indica"**. O tamanho passa a ser decidido pelo `lr`, e o gradiente decide só a direção.

**O exemplo numérico.** Depois dos três passos de SGD com os gradientes do capítulo:

| posição | soma dos três `g` | deslocamento com SGD |
|---|---|---|
| 0 | `+0.20` | `6.00e-5` |
| 1 | `+0.12` | `3.60e-5` |
| 2 | `−0.90` | `2.70e-4` |

A posição 1 andou 7,5 vezes menos que a posição 2. Não porque ela precise andar menos, e sim porque o gradiente dela é menor. A §4 refaz esta tabela com Adam, e a ordem se inverte.

> **Confira você mesmo.** Por que não basta dar um `lr` diferente para cada camada, e resolver o problema à mão?
>
> <details><summary>Resposta</summary>
>
> Porque a escala não é fixa. O gradiente de um mesmo parâmetro muda de ordem de grandeza ao longo do treino — grande no começo, minúsculo perto da convergência. Um `lr` por camada seria um ajuste estático para um problema que se move. O Adam mede a escala **a cada passo**, e por isso acompanha.
> </details>

---

## §3. Momento: a média móvel exponencial

O gradiente de um lote é ruidoso. Ele mistura duas coisas: a direção real de descida, e o acaso de quais exemplos caíram no lote.

A ideia do momento é simples. Em vez de usar o gradiente deste passo, use uma média dos gradientes recentes. O ruído tende a se cancelar, e a direção consistente sobrevive.

```
m ← β1 · m + (1 − β1) · g
```

> **Definição — média móvel exponencial.** Uma média onde cada valor novo entra com peso `1 − β1`, e tudo que já estava lá encolhe por `β1`. O passado nunca é descartado, só desbotado: o gradiente de `k` passos atrás ainda pesa `β1` elevado a `k`.

Com `β1 = 0.9`, cada gradiente novo entra com 10% e o histórico fica com 90%. A meia-vida é de cerca de sete passos, porque `0.9⁷ ≈ 0.478`.

Por que isso ajuda, em uma frase. Se dois passos seguidos apontam para lados opostos, eles se cancelam em `m`, e o otimizador não sai correndo atrás do ruído.

**O exemplo numérico.** Os três passos, posição por posição:

```
m₁ = 0.1·g₁                = [ 0.0300,  0.00300, -0.0600]
m₂ = 0.9·m₁ + 0.1·g₂       = [ 0.0370,  0.00770, -0.0740]
m₃ = 0.9·m₂ + 0.1·g₃       = [ 0.0133,  0.01093, -0.0766]
```

Olhe a **posição 0**. O gradiente foi `+0.30`, `+0.10`, depois `−0.20`. Ele trocou de sinal. Mas `m` continua positivo, e só caiu de `0.0370` para `0.0133`. O otimizador vai continuar andando no mesmo sentido, com um passo bem menor. É o comportamento desejado: uma discordância isolada reduz a confiança, mas não inverte a decisão.

Confira à mão a posição 0 do passo 3. `0.9 · 0.0370 + 0.1 · (−0.20) = 0.0333 − 0.0200 = 0.0133`. ✓

---

## §4. O segundo momento: o passo que não depende da escala

O momento resolveu o ruído. Não resolveu a escala, porque `m` continua tendo o tamanho do gradiente.

Para normalizar, precisamos de uma medida do **tamanho típico** do gradiente daquele parâmetro. É o segundo momento:

```
v ← β2 · v + (1 − β2) · g²
```

O `g²` é elemento a elemento. Como todo termo é positivo, `v` não sofre cancelamento. Ele mede magnitude e ignora sinal. A raiz `√v` é a escala típica do gradiente, na mesma unidade dele.

E aí vem o passo do Adam:

```
p ← p − lr · m / (√v + ε)
```

> **Definição — passo adaptativo.** A divisão por `√v` cancela a unidade do gradiente. `m` tem a escala de `g`, `√v` também, e a razão entre os dois é **adimensional** — um número perto de 1, para qualquer parâmetro.

Essa é a propriedade central da etapa. Multiplique todos os gradientes por 1000. O `m` fica 1000 vezes maior, o `√v` também, e a razão **não muda**. O passo continua o mesmo.

O `ε = 1e-8` existe só para o caso de `v` ser zero. Ele é pequeno demais para afetar qualquer conta real.

**O exemplo numérico — a invariância, medida.** Rodando o passo 1 com `g₁`, e depois com `1000 · g₁`:

```
com g₁         passo = [ 2.99999990e-4,  2.99999900e-4, -2.99999995e-4]
com 1000·g₁    passo = [ 3.00000000e-4,  2.99999999e-4, -3.00000000e-4]
diferença máxima: 1.0e-10
```

Mil vezes mais gradiente, e o passo é o mesmo até a décima casa. A diferença de `1e-10` é o `ε`, e nada mais.

**O exemplo numérico — os três passos.** Com `v` acumulado junto:

```
v₁ = 0.05·g₁²              = [0.00450000, 0.00004500, 0.01800000]
v₂ = 0.95·v₁ + 0.05·g₂²    = [0.00477500, 0.00016775, 0.01910000]
v₃ = 0.95·v₂ + 0.05·g₃²    = [0.00653625, 0.00023936, 0.01864500]
```

E agora a tabela da §2, refeita com Adam:

| posição | deslocamento com SGD | deslocamento com Adam |
|---|---|---|
| 0 | `6.00e-5` | `6.33e-4` |
| 1 | `3.60e-5` | `8.88e-4` |
| 2 | `2.70e-4` | `7.99e-4` |

A ordem se inverteu. A posição 1, que o SGD deixou quase parada, foi a que **mais** andou. Ela percorreu quase 25 vezes mais do que percorreria com SGD. Os três deslocamentos ficaram dentro de um fator de 1,4 entre si, apesar de gradientes que diferem por vinte vezes. É o `lr` decidindo o tamanho, como a §2 prometeu.

> **Confira você mesmo.** Se `v` mede magnitude e ignora sinal, por que não usar o módulo de `g` direto, em vez da raiz de `g²`?
>
> <details><summary>Resposta</summary>
>
> Para um único passo dá no mesmo. A diferença aparece na média móvel. Média de quadrados pesa mais os gradientes grandes, e é isso que se quer: um pico isolado deve segurar o passo por vários passos seguintes. Média de módulos reagiria menos, e o parâmetro passaria do ponto logo depois de um pico.
> </details>

---

## §5. Correção de viés

Tem um problema de partida. `m` e `v` começam em zero, e zero não é uma estimativa. É ausência de informação.

No primeiro passo, `m₁ = 0.1 · g₁`. Isso é dez vezes menor que `g₁`. O `m` não está medindo a média dos gradientes. Ele está medindo a média entre os gradientes vistos e um monte de zeros que nunca existiram.

O viés é calculável, e por isso corrigível. Depois de `t` passos com gradiente constante `g`, o `m` acumulado vale `(1 − β1ᵗ) · g`. Basta dividir por esse fator:

```
m̂ = m / (1 − β1ᵗ)
v̂ = v / (1 − β2ᵗ)
```

> **Definição — correção de viés.** A divisão que compensa o arranque em zero. O fator `1 − β1ᵗ` vale `0.1` no primeiro passo e tende a 1 conforme `t` cresce. A correção é forte no começo e some sozinha.

Note que `t` conta a partir de 1, e não de 0. Com `t = 0` o denominador seria zero.

**O exemplo numérico — o passo 1 vale exatamente `lr`.** Com `t = 1`:

```
m̂₁ = m₁ / (1 − 0.9)  = m₁ / 0.10  = [ 0.30,  0.03, -0.60]   ← é g₁ de volta
v̂₁ = v₁ / (1 − 0.95) = v₁ / 0.05  = [ 0.09, 0.0009,  0.36]   ← é g₁² de volta
```

A correção desfez exatamente a diluição. E aí o passo:

```
passo₁ = lr · m̂₁ / (√v̂₁ + ε) = lr · g / |g| = [3e-4, 3e-4, -3e-4]
p₁     = [0.5997, -0.2003, 0.0503]
```

Os três valores têm o mesmo tamanho, `3e-4`, que é o `lr`. Os gradientes eram `0.30`, `0.03` e `−0.60`, com vinte vezes de diferença entre eles. Só o **sinal** sobreviveu.

**Sem a correção, o mesmo passo sairia menor.** O fator seria `0.1 / √0.05 = 0.4472`. O primeiro passo teria 45% do tamanho pretendido, e a defasagem levaria dezenas de passos para sumir:

| `t` | fator sem correção |
|---|---|
| 1 | `0.4472` |
| 3 | `0.7176` |
| 5 | `0.8610` |
| 10 | `1.0282` |
| 20 | `1.0967` |
| 100 | `1.0029` |

O fator não sobe monotonicamente até 1. Ele passa de 1 por volta do passo 9, tem pico de `1.0967` no passo 20, e só então converge. A causa é `β2 > β1`. Os dois vieses somem em ritmos diferentes, e o de `v` demora mais.

> **Armadilha.** Um erro que não trava nada: **esquecer a correção de viés**. O treino roda, a perda cai, o modelo aprende. Ele só aprende um pouco pior nos primeiros passos, o que é invisível num gráfico de perda.
>
> Por que é perigoso: não há sintoma. Compare com o que aconteceu na Etapa 16, com `softmax` seguido de `log` no lugar do `logSoftmax`. Com logits de tamanho normal os dois caminhos davam o **mesmo número**, e catorze dos quinze testes passavam. Só o caso extremo, com diferença de 800 entre os logits, reprovava.
>
> **Lição geral:** para código que só erra "um pouco", o teste tem que atacar o regime onde o erro é grande. Aqui esse regime é `t = 1`. Um teste que verifica que o primeiro passo tem tamanho `lr` pega a falta da correção na hora. Um teste que roda cem passos e olha a perda não pega nunca.

---

## §6. Weight decay, e por que o "W" existe

> **Definição — weight decay.** Uma pressão constante que puxa todo parâmetro na direção do zero, independente do gradiente. Serve para o modelo não decorar o corpus: pesos grandes só sobrevivem se o gradiente insistir neles.

A forma clássica de fazer isso chama-se **regularização L2**, e consiste em somar `λ · p` ao gradiente:

```
g' = g + λ · p          ← L2, o jeito clássico
```

Com SGD isso funciona perfeitamente. O passo vira `p ← p − lr·g − lr·λ·p`, e o segundo termo encolhe `p` de forma limpa.

Com Adam, não. O `g'` inteiro atravessa a normalização da §4, e **o decaimento é dividido por `√v̂` junto com o resto**. O que era uma pressão constante vira uma pressão dividida pelo tamanho do gradiente daquele parâmetro.

O AdamW conserta separando os dois. O decaimento não passa pelo gradiente, e é aplicado direto no parâmetro:

```
p ← p − lr·λ·p − lr · m̂ / (√v̂ + ε)      ← AdamW
```

O "W" é de *decoupled weight decay*, ou decaimento desacoplado. O primeiro termo não sabe nada sobre `m` e `v`.

**O exemplo numérico.** Um parâmetro `p = 0.60`, com `λ = 0.1` e `lr = 3e-4`. A coluna do meio mostra quanto o decaimento move `p` quando entra pelo gradiente, para cinco escalas de gradiente:

| escala de `g` | decaimento via L2 | decaimento no AdamW |
|---|---|---|
| `1.0` | `1.80e-5` | `1.80e-5` |
| `0.3` | `6.00e-5` | `1.80e-5` |
| `0.1` | `1.80e-4` | `1.80e-5` |
| `0.03` | `6.00e-4` | `1.80e-5` |
| `0.003` | `6.00e-3` | `1.80e-5` |

A coluna do AdamW é constante, porque `lr · λ · p = 3e-4 · 0.1 · 0.60 = 1.8e-5` não depende de `g`. A coluna do L2 varia por um fator de **333** entre a primeira linha e a última.

E varia no sentido errado. Quem tem gradiente minúsculo é o parâmetro que o modelo já resolveu, e que o gradiente não está mais empurrando. É justamente ele que recebe o decaimento mais forte. Quem tem gradiente enorme quase não é decaído. A regularização acaba mais agressiva onde havia menos motivo para intervir.

**A verificação independente.** No AdamW o decaimento é puramente multiplicativo: `p ← (1 − lr·λ) · p = 0.99997 · p`. Ele não depende do gradiente, então dá para prever o efeito de 200 passos sem simular nada:

```
previsto:   0.60 · 0.99997²⁰⁰ = 0.596411   → o decaimento tira 0.003589
simulado:   200 passos com o gradiente junto → tirou 0.003411
```

Os dois batem na terceira casa. A diferença sobra porque `p` também está descendo por causa do gradiente, e um `p` menor sofre um decaimento absoluto menor.

> **Confira você mesmo.** Com `lr = 3e-4` e `λ = 0.1`, quantos passos um parâmetro leva para cair à metade, se o gradiente dele for zero o tempo todo?
>
> <details><summary>Resposta</summary>
>
> São `log(0.5) / log(1 − 3e-5) ≈ 23.105` passos. O decaimento é bem mais lento do que `λ = 0.1` sugere. Quem decide o ritmo é o produto `lr · λ`, e não o `λ` sozinho. Mudar o `lr` muda a força da regularização junto, e é por isso que os dois costumam ser ajustados em conjunto.
> </details>

---

## §7. O estado do otimizador, e a instância nova a cada passo

O AdamW precisa lembrar de três coisas entre um passo e o outro: `m`, `v` e `t`. Isso é novidade no projeto. Todas as peças anteriores eram funções puras de tensores.

**Onde `m` e `v` moram.** Um array de cada, por parâmetro, com uma posição por elemento. O modelo tem uma lista de parâmetros, então o estado é um mapa:

```scala
Map[Tensor, (Array[Double], Array[Double])]
```

Usar `Tensor` como chave funciona pelo mesmo motivo que faz o `Set[Tensor]` do grafo funcionar, na Etapa 2 §3. `Tensor` é uma `class` comum, e não uma `case class`, então a igualdade é **por referência**. Dois parâmetros diferentes com exatamente os mesmos valores são chaves diferentes, que é o que se quer. Com `case class` a igualdade seria estrutural, e dois pesos inicializados por acaso com os mesmos números colidiriam no mapa. Eles compartilhariam o momento um do outro.

> **Definição — hiperparâmetro.** Um número que você escolhe antes do treino e que não é aprendido: `lr`, `β1`, `β2`, `ε` e `λ`. Parâmetros o gradiente ajusta; hiperparâmetros, não.

**Como o estado avança.** O `step()` não muda campo nenhum do otimizador. Ele calcula o `m` e o `v` novos, escreve nos parâmetros, e devolve um `AdamW` novo com `t + 1`:

```scala
final class AdamW(
    parameters: List[Tensor],
    lr: Double = 3e-4,
    beta1: Double = 0.9,
    beta2: Double = 0.95,
    eps: Double = 1e-8,
    weightDecay: Double = 0.1,
    t: Int = 0,
    state: Map[Tensor, (Array[Double], Array[Double])] = Map.empty
) {
  def step(): AdamW = ???
  def zeroGrad(): Unit = ???
}
```

O loop de treino da Etapa 18 vira um `foldLeft` sobre os lotes, carregando o otimizador como acumulador. Sem `var`, como o resto do projeto.

**A honestidade sobre o custo.** Estado imutável significa dois arrays novos por parâmetro, a cada passo. Some o array do próprio `updateData` e são três alocações do tamanho do parâmetro, por passo. Não é grátis.

Vale medir antes de concluir qualquer coisa. O precedente do projeto é o `matmul` de 2026-08-21, onde a formulação mais funcional foi também a mais rápida. Foram `288 ms` com `updated`, `126 ms` com `while`, e `58 ms` com aritmética de deslocamento e `foldLeft`. Se o custo aqui aparecer numa medição, a saída provavelmente é reformular, e não trocar por `var`.

> **Armadilha.** O tamanho dos arrays `m` e `v` deve vir de `p.size`, e nunca de `p.data.length`.
>
> Os dois coincidem para todo parâmetro criado por `Tensor.make` ou por `randn`. Não coincidem em geral: `data` é o buffer físico, e uma view pode anunciar um `shape` menor que ele. Passar `data.length` no lugar de `shape.size` já causou inconsistência real neste projeto, e foi o motivo de `Gradient.zeros` ganhar a sobrecarga que recebe `Shape`.
>
> **Lição geral:** o invariante é uma posição por elemento do tensor. Quem define isso é o `shape`, e nunca o comprimento do buffer.

**O exemplo numérico.** O estado depois dos três passos, para o parâmetro de três elementos:

```
t = 3
m = [ 0.01330000,  0.01093000, -0.07660000]
v = [ 0.00653625,  0.00023936,  0.01864500]
p = [ 0.59936724, -0.20088846,  0.05079852]
```

Nove números de estado para três de parâmetro. O Adam custa **três vezes** a memória do modelo, e é por isso que o tamanho do otimizador aparece nas contas de memória de treino.

---

## §8. Escrever no parâmetro: `updateData`

Falta o mecanismo. O `AdamW` mora em `gpt.optim`, e o campo `data` do `Tensor` é `private[scalagrad]`. De fora do módulo, ele nem compila.

Isso não é um descuido. O único lugar que escreve em `data` hoje é o `Gradcheck`, e ele só consegue porque mora dentro do `scalagrad`. Escrever num tensor é perigoso o bastante para ter uma porta só, estreita e vigiada. É a mesma filosofia do `Gradient`, que expõe `accumulate` e se recusa a expor um `update` genérico.

A porta desta etapa:

```scala
def updateData(values: Array[Double]): Unit
```

Duas pré-condições, ambas `require`. O chamador é externo, e a Etapa 7 fixou essa regra.

**`values.length == size`.** Um array curto escreveria metade do parâmetro e só então lançaria, deixando o modelo num estado meio atualizado.

**`isContiguous`.** Esta é a que importa, e a próxima armadilha explica.

> **Armadilha.** `data` é indexado **fisicamente**, pelas strides reais. `gradient` é indexado **canonicamente**. Usar o mesmo `i` nos dois é o bug de 2026-08-21.
>
> Ele atingiu nove operações unárias de uma vez: `neg`, `pow`, `exp`, `log`, `clamp`, `relu`, `sigmoid`, `tanh` e `gelu`. Num tensor contíguo as duas indexações coincidem, então tudo passa. O gradient check só reprovou quando a entrada era um tensor transposto, com erro de `0.632` contra `8.5e-11` da mesma operação sobre entrada contígua.
>
> Por que passou despercebido: o forward continua perfeito. Só o gradiente sai errado, e gradiente errado não avisa.
>
> **Lição geral:** onde `data` e `gradient` se encontram, ou você garante contiguidade ou traduz o índice. O `updateData` escolhe garantir, com `require(isContiguous)`. Todo parâmetro nasce contíguo, então a checagem nunca dispara por uso legítimo. Ela existe para o dia em que alguém passar uma view.

**O `zeroGrad`, e por que ele não precisa de API nova.** O `Gradient` só acumula. Se você não zerar entre os passos, o gradiente do lote 2 soma no do lote 1, e o modelo anda numa direção que é a soma de tudo que já viu.

Isso não exige método novo. O campo `gradient` é público, e `Gradient.zero()` também. O otimizador já tem a lista de parâmetros:

```scala
parameters.foreach(_.gradient.zero())
```

Existe também `Tensor.zeroGrad()`, que percorre o grafo a partir de um tensor e zera todos os nós que requerem gradiente. Chamado na perda, ele zera o modelo inteiro. Chamado num parâmetro, zera só ele, porque um parâmetro é folha e não tem nada antes no grafo.

**O exemplo numérico.** O passo 1, do gradiente ao array escrito:

```
p.gradient          = [ 0.30,  0.03, -0.60]     ← o backward() deixou isto aqui
passo calculado     = [ 3.0e-4, 3.0e-4, -3.0e-4]
decaimento (λ=0.1)  = [ 1.8e-5, -6.0e-6, 1.5e-6]

values = p − decaimento − passo
       = [0.59968200, -0.20029400, 0.05029850]

p.updateData(values)
p.gradient.zero()
```

Repare no sinal do decaimento na posição 1. O parâmetro vale `−0.20`, então `lr·λ·p` é **negativo**, e subtraí-lo empurra o valor para cima. Para cima é a direção do zero, que é o que o weight decay promete. O decaimento puxa para zero, e não para baixo.

> **Confira você mesmo.** O `updateData` exige contiguidade. O `broadcastTo` devolve uma view com stride 0, e o `transpose` devolve uma view com strides trocadas. Qual dos dois faria o `require` disparar?
>
> <details><summary>Resposta</summary>
>
> Os dois, porque nenhuma das duas views tem strides canônicas. E é bom que disparem, por motivos diferentes. A view transposta tem o `data` em outra ordem, e escrever nela por índice canônico embaralharia os valores. A view de broadcast é pior: várias posições do shape anunciado apontam para o mesmo elemento físico, e nem existe um array de `size` valores para escrever nela.
> </details>

---

## §9. Para onde isso leva

Todas as peças existem. O modelo produz logits, a perda vira um número, o `backward()` distribui a culpa, e agora o otimizador transforma culpa em correção.

A Etapa 18 amarra os quatro num laço:

```
para cada lote:
    logits = modelo.forward(inputs)
    loss   = CrossEntropy(logits, targets)
    otimizador.zeroGrad()
    loss.backward()
    otimizador = otimizador.step()
```

Três coisas que a Etapa 18 acrescenta, e que dependem desta.

**Learning rate schedule.** O `lr` não fica em `3e-4` o treino inteiro. Ele sobe do zero nos primeiros passos, no aquecimento, e depois desce suavemente. O `t` que o `AdamW` já carrega é exatamente o contador que o schedule consulta.

**Checkpoint.** Salvar o modelo é salvar os valores dos parâmetros. Carregar é escrevê-los de volta num modelo já construído, e o `updateData` desta etapa é o caminho para isso. Salvar `m` e `v` junto é o que permite retomar um treino sem perder o momento acumulado.

**A primeira medida real.** Com o loop fechado, a perda vira uma série temporal, e a Etapa 16 já disse o que esperar. Ela começa perto de `log(V)` e cai. Se não cair, os suspeitos estão aqui: `lr` grande demais, correção de viés faltando, ou gradiente não zerado entre os passos.

---

## Cartão de referência

| Fórmula | O que faz |
|---|---|
| `p ← p − lr·g` | SGD: passo proporcional ao gradiente |
| `m ← β1·m + (1−β1)·g` | média móvel do gradiente: filtra ruído |
| `v ← β2·v + (1−β2)·g²` | média móvel do quadrado: mede a escala |
| `m̂ = m/(1−β1ᵗ)` e `v̂ = v/(1−β2ᵗ)` | desfaz o viés do arranque em zero |
| `p ← p − lr·λ·p − lr·m̂/(√v̂+ε)` | o passo do AdamW, completo |

| Hiperparâmetro | Valor | Papel |
|---|---|---|
| `lr` | `3e-4` | tamanho do passo |
| `β1` | `0.9` | memória do gradiente, com meia-vida de ~7 passos |
| `β2` | `0.95` | memória da escala, mais lenta que `β1` |
| `ε` | `1e-8` | só evita divisão por zero |
| `λ` | `0.1` | força do decaimento; o que age é `lr·λ` |

**As lições que se repetem:**

**Normalizar é o que torna um hiperparâmetro utilizável.** O `lr` só funciona como "tamanho do passo" porque a divisão por `√v̂` tirou a unidade do gradiente. É o mesmo movimento do LayerNorm na Etapa 10 e da divisão por `√d_k` na Etapa 11. Três lugares diferentes, sempre pelo mesmo motivo.

**Corrigir o arranque importa mais do que parece.** O `m` e o `v` começam em zero por falta de alternativa. O viés que isso cria é grande justo no começo, quando o modelo está mais sensível.

**Bug que só erra um pouco precisa de teste no regime extremo.** A falta da correção de viés e o `softmax` seguido de `log` da Etapa 16 são a mesma classe de problema. São certos no caso comum, errados no caso raro, e silenciosos nos dois.

**Onde `data` e `gradient` se encontram, pare e pense.** Um é físico, o outro é canônico. Foi o bug de 2026-08-21, e o `require(isContiguous)` do `updateData` existe para ele não voltar.

**Estado imutável cabe aqui.** O `step()` devolve um `AdamW` novo, e o loop de treino é um `foldLeft`. O único mutável do projeto continua sendo o que precisa ser: o `Gradient`, e agora o `data` dos parâmetros.
