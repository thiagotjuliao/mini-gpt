# Etapa 18 — O loop de treinamento

## Antes de começar

### O que você vai construir

O laço que faz as dezessete etapas anteriores virarem um modelo treinado.

| Componente | Onde mora | O que faz |
|---|---|---|
| `Gradient.scale` | `scalagrad.core` | reescala o gradiente acumulado, sem apagá-lo |
| `LRSchedule.cosine` | `gpt.train` | a taxa de aprendizado de cada passo |
| `GradientClipping` | `gpt.train` | segura a norma global dos gradientes |
| `Checkpoint` | `gpt.train` | salva e restaura modelo e otimizador |
| `Trainer.evaluate` | `gpt.train` | perda de validação, sem construir grafo |
| `Trainer.train` | `gpt.train` | o laço, e o histórico do que aconteceu |

Nenhum parâmetro novo. Esta etapa não acrescenta capacidade ao modelo — ela organiza o uso do que já existe.

### O que você precisa saber antes

**Da Etapa 2:** que `backward()` preenche o gradiente e que `Gradient` **só acumula**. A §1 depende disso.

**Da Etapa 7 §5:** o `BatchSampler`, que entrega `inputs` e `targets` já deslocados.

**Da Etapa 16:** a perda, e que ela começa em `log(V)` ou acima.

**Da Etapa 17:** o `AdamW`, o `step(stepLr)` e o `updateData`.

### Onde esta etapa se encaixa

Todas as peças estão prontas e nenhuma delas foi usada de verdade.

Até aqui cada etapa foi testada isoladamente: o gradient check confirma uma operação, o `GPTSpec` confirma um forward. Nada disso mostra se as peças **juntas** aprendem. É o laço que responde.

E é aqui que bugs arquiteturais aparecem. Eles são os piores de diagnosticar, porque o sintoma é sempre o mesmo: a perda não desce. A causa pode estar em qualquer uma das dezessete etapas.

### Objetivos de aprendizagem

Ao terminar, você deve conseguir:

1. Escrever as seis operações de um passo na ordem certa, e dizer o que quebra em cada troca.
2. Explicar por que a norma do clipping é global, e não por parâmetro.
3. Calcular a taxa de aprendizado de um passo qualquer, no aquecimento e no cosseno.
4. Dizer por que a avaliação roda dentro de `noGrad`, e o que aconteceria sem ele.
5. Ler uma curva de perda e separar "está aprendendo devagar" de "tem bug".

### Convenções de notação

| Símbolo | Significa |
|---|---|
| `step` | o número do passo de treino, contando de **1** |
| `totalSteps` | quantos passos o treino inteiro tem |
| `‖g‖` | a norma global: uma raiz sobre os quadrados de **todos** os gradientes |
| `maxNorm` | o teto do clipping — `1.0` neste projeto |
| `lrMax`, `lrMin` | os extremos do schedule — `3e-4` e `3e-5` |

> **Guia visual.** A ordem das seis operações, o cosseno com aquecimento e a diferença entre norma global e por parâmetro: [`training-loop.html`](training-loop.html). **Exercícios (14 questões):** [`exercises.html`](exercises.html).

---

## §1. As seis operações, e por que a ordem é contrato

Um passo de treino é isto:

```
1. amostrar          (inputs, targets) = sampler.sample(batchSize)
2. forward           logits = model.forward(inputs)
                     loss   = CrossEntropy(logits, targets)
3. zerar             optimizer.zeroGrad()
4. backward          loss.backward()
5. clipar            GradientClipping.clipByGlobalNorm(parameters, maxNorm)
6. atualizar         optimizer.step(lr)
```

> **Definição — passo de treino.** Uma passada completa: um lote entra, a perda sai, e todo parâmetro se move uma vez. É a unidade que o `t` do otimizador conta.

A ordem não é estilo. Três trocas quebram o treino, e duas delas em silêncio.

**Zerar depois do `backward` apaga o trabalho.** O passo 3 tem que vir antes do 4. Invertidos, os gradientes são calculados e imediatamente zerados, o otimizador recebe zeros, e nenhum parâmetro se move. A perda fica parada num valor plausível para sempre.

**Não zerar nunca soma tudo que já passou.** O `Gradient` só acumula, por decisão da Etapa 2 — múltiplos caminhos do grafo precisam somar suas contribuições numa mesma passada. Fora de uma passada, essa mesma soma vira lixo: no passo 10, o gradiente seria a soma dos dez lotes vistos. O modelo anda numa direção que mistura dados antigos com atuais, e o treino piora conforme avança.

**Clipar depois de atualizar não faz nada.** O passo 5 tem que estar entre o 4 e o 6. Depois do `step`, os parâmetros já se moveram — reescalar o gradiente ali só afeta o próximo passo, com o valor errado.

Note que o `zeroGrad` do otimizador zera **os parâmetros**, e não o grafo inteiro. É suficiente, e por um motivo específico: cada `forward` constrói um grafo novo, cujos nós intermediários nascem com gradiente zero. Só os parâmetros atravessam os passos.

**O exemplo numérico.** Um parâmetro com gradiente `0.4` em três lotes seguidos, com e sem o `zeroGrad`:

| passo | gradiente do lote | com `zeroGrad` | sem `zeroGrad` |
|---|---|---|---|
| 1 | `0.4` | `0.4` | `0.4` |
| 2 | `0.4` | `0.4` | `0.8` |
| 3 | `0.4` | `0.4` | `1.2` |

Na terceira linha o otimizador vê um gradiente três vezes maior do que o lote reporta. Com Adam, isso nem muda muito o passo — a normalização por `√v̂` absorve boa parte da escala. É o que torna o bug tão discreto: ele distorce a direção, não o tamanho.

---

## §2. Gradient clipping: uma norma para todos

Gradientes explodem. Acontece principalmente no começo, quando os pesos ainda são aleatórios e uma sequência azarada produz um erro enorme. Um passo gigante joga o modelo para uma região ruim, e às vezes ele não volta.

> **Definição — norma global do gradiente.** A raiz da soma dos quadrados de **todos** os gradientes de **todos** os parâmetros, como se fossem um vetor só. É o comprimento do passo que o modelo está prestes a dar.

A regra do clipping tem uma linha:

```
se ‖g‖ > maxNorm:   todo gradiente é multiplicado por maxNorm / ‖g‖
```

Multiplicar tudo pelo mesmo fator encolhe o vetor sem girá-lo. A direção do passo é preservada exatamente; só o tamanho muda.

**Por que global, e não uma norma por parâmetro.** Essa é a parte que se erra. Clipar cada parâmetro separadamente parece equivalente e não é: cada um receberia um fator diferente, e a direção conjunta mudaria.

**O exemplo numérico.** Dois parâmetros, `a` com gradiente `[3, −4]` e `b` com gradiente `[0.1]`, e `maxNorm = 1.0`.

```
norma global = √(9 + 16 + 0.01) = 5.001000
fator        = 1 / 5.001000     = 0.199960

a → [0.599880, -0.799840]
b → [0.019996]
```

Agora o mesmo corte, feito por parâmetro. A norma de `a` é `5.0` e a de `b` é `0.1`:

```
a → [0.600000, -0.800000]     (cortado, fator 0.2)
b → [0.100000]                (intacto: 0.1 já está abaixo de 1.0)
```

Os dois resultados parecem quase idênticos. Confira a razão entre a primeira componente de `a` e a de `b`:

| | `a₀ / b₀` |
|---|---|
| antes do corte | `30.0` |
| com norma global | `30.0` |
| com norma por parâmetro | `6.0` |

A norma global preserva a razão exatamente. A norma por parâmetro a reduz por cinco: ela deu a `b` um peso relativo cinco vezes maior do que o gradiente pedia. O modelo passa a andar numa direção que ninguém calculou.

> **Confira você mesmo.** Por que o `clipByGlobalNorm` devolve a norma de **antes** do corte, em vez da de depois?
>
> <details><summary>Resposta</summary>
>
> Porque a de depois não informa nada: quando houve corte ela vale sempre `maxNorm`, e quando não houve é igual à de antes. A de antes é diagnóstico. Se ela vive em `0.3`, o clipping nunca age e o teto está frouxo. Se vive em `50`, alguma coisa está errada antes do clipping — provavelmente a inicialização ou a taxa de aprendizado.
> </details>

---

## §3. O learning rate schedule

A Etapa 17 tratou o `lr` como constante. Nenhum treino sério faz isso.

> **Definição — schedule.** Uma função que devolve a taxa de aprendizado de cada passo. Duas fases: o aquecimento, que sobe, e o decaimento, que desce.

**O aquecimento (*warmup*).** Nos primeiros `warmupSteps` passos, a taxa sobe linearmente de quase zero até `lrMax`. O motivo está na Etapa 17: no primeiro passo o Adam dá um passo de tamanho `lr` **para todo parâmetro**, guiado por um gradiente que veio de um único lote com pesos aleatórios. É a pior estimativa que o treino inteiro vai ver, e é justamente ali que o passo tem tamanho cheio. O aquecimento reduz o estrago.

**O decaimento em cosseno.** Depois do aquecimento, a taxa desce de `lrMax` a `lrMin` seguindo meia volta de cosseno:

```
progresso = (step − warmupSteps) / (totalSteps − warmupSteps)
lr        = lrMin + ½ · (lrMax − lrMin) · (1 + cos(π · progresso))
```

O cosseno desce devagar no começo, rápido no meio e devagar de novo no fim. As duas pontas planas são o ponto: no começo você ainda quer passos grandes, e no fim quer ajuste fino sem sobressalto.

**O exemplo numérico.** Com `lrMax = 3e-4`, `lrMin = 3e-5`, `warmupSteps = 100` e `totalSteps = 1000`:

| `step` | `lr` | fase |
|---|---|---|
| 1 | `3.000000e-6` | aquecimento, 1% de `lrMax` |
| 50 | `1.500000e-4` | aquecimento, metade |
| 100 | `3.000000e-4` | o pico |
| 200 | `2.918585e-4` | cosseno, ainda quase no topo |
| 550 | `1.650000e-4` | o meio exato |
| 775 | `6.954058e-5` | descendo rápido |
| 1000 | `3.000000e-5` | `lrMin` |
| 1200 | `3.000000e-5` | além do fim: fica parado |

**A verificação independente.** No meio do cosseno, `progresso = 0.5` e `cos(π/2) = 0`. A fórmula vira `lrMin + ½(lrMax − lrMin)`, que é a média aritmética dos extremos: `(3e-4 + 3e-5)/2 = 1.65e-4`. Bate com a linha do passo 550.

A última linha é uma decisão de projeto. Sem travar o progresso em 1, um treino que passasse de `totalSteps` veria o cosseno **subir de volta** — a taxa cresceria de novo no fim, e o modelo desandaria depois de já ter convergido.

> **Confira você mesmo.** Por que o aquecimento nunca começa em zero exato?
>
> <details><summary>Resposta</summary>
>
> Porque `lr = 0` faz o primeiro passo não mover parâmetro nenhum. O passo é gasto: o forward roda, o backward roda, e nada acontece. A rampa começa em `lrMax / warmupSteps`, que é pequeno mas move.
> </details>

---

## §4. O laço sem `var`

O laço parece o lugar onde a imutabilidade finalmente cede. Tem um contador, um otimizador que muda de estado e um histórico que cresce.

Não cede. O `step()` da Etapa 17 devolve um `AdamW` novo, e isso é exatamente o que um `foldLeft` precisa:

```scala
(1 to config.steps).foldLeft(TrainingResult(optimizer, List.empty)) { (acc, step) =>
  ...
  TrainingResult(nextOptimizer, metrics :: acc.history)
}
```

O acumulador carrega o otimizador e o histórico. O `step` vem do intervalo, e não de um contador que alguém incrementa.

Uma sutileza de ordem: o histórico é construído com `::`, que insere na frente e custa tempo constante. Ele sai ao contrário, e o `train` inverte uma vez no fim. Construir com `:+` daria a ordem certa direto, ao custo de percorrer a lista inteira a cada passo.

**O exemplo numérico.** Cinco passos, e o que o acumulador carrega:

| depois do passo | `optimizer.t` | `history` (topo primeiro) |
|---|---|---|
| 1 | 1 | `[1]` |
| 2 | 2 | `[2, 1]` |
| 3 | 3 | `[3, 2, 1]` |
| 5 | 5 | `[5, 4, 3, 2, 1]` |

E o `reverse` final devolve `[1, 2, 3, 4, 5]`. O teste `keep the history in chronological order` fixa isso, porque é o tipo de coisa que ninguém percebe até ler um gráfico de perda de trás para frente.

---

## §5. Treino e validação

A perda de treino mede o lote que o modelo acabou de ver. Ela pode cair por dois motivos bem diferentes: o modelo aprendeu o padrão, ou o modelo decorou os exemplos.

> **Definição — conjunto de validação.** Uma fatia do corpus separada antes do treino, e nunca usada para atualizar parâmetros. A perda nela é a que mede aprendizado de verdade.

> **Definição — overfitting.** Quando a perda de treino continua caindo e a de validação começa a subir. O modelo está memorizando o corpus em vez de aprender a estrutura dele.

A avaliação roda dentro de `Tensor.noGrad`. Não é só economia de memória:

**Sem `noGrad`, a avaliação suja o treino.** O `forward` construiria um grafo, e o `backward` não seria chamado — mas o `CrossEntropy` marca os tensores com `requiresGradient`, e nada garante que o gradiente acumulado ali fique inerte. Pior: o grafo inteiro fica pendurado na memória a cada lote de avaliação, sem ninguém para consumi-lo.

O teste `evaluate should leave the gradients untouched` fixa a propriedade: depois de três lotes de avaliação, todo gradiente de todo parâmetro continua exatamente zero.

**O exemplo numérico.** Um modelo recém-inicializado com `vocabSize = 3`, avaliado em três lotes:

```
log(3)        = 1.0986      ← o piso
perda medida  = 1.10 a 1.84 ← conforme a inicialização
```

O piso é o do modelo que chuta uniformemente, e a §7 mostra por que a medida fica acima dele. Serve de teste: uma avaliação inicial **abaixo** de `log(V)` indica bug antes mesmo de o treino começar.

---

## §6. Checkpoint

Treinar leva horas. Um processo que morre sem checkpoint recomeça do zero.

O formato deste projeto é binário e direto: um número mágico, a versão, a `GPTConfig`, o `t` do otimizador, a quantidade de parâmetros e um marcador de "tem momentos". Depois, por parâmetro na ordem de `model.parameters`: o tamanho, os valores, e os arrays `m` e `v`.

Três decisões dentro disso.

**`m` e `v` vão junto.** É a que mais importa. Restaurar só os pesos parece suficiente e não é: o otimizador voltaria com `t = 0` e momentos zerados, e a correção de viés trataria o próximo passo como o **primeiro**. Depois de mil passos de momento acumulado, o modelo daria um passo de tamanho `lr` cheio, guiado por um único lote. O treino continua, e a curva ganha um degrau que ninguém sabe explicar.

**A configuração vai junto, e o tamanho de cada parâmetro também.** Carregar um checkpoint num modelo de forma diferente é erro do chamador, não corrupção silenciosa. Sem a conferência, um `dModel` diferente leria bytes do parâmetro seguinte e produziria um modelo com pesos embaralhados que ainda roda.

Guardar a `GPTConfig` faz duas coisas. A mensagem de erro passa a nomear as duas configurações, em vez de reclamar do tamanho do parâmetro 7. E o `Checkpoint.loadModel` consegue **reconstruir** o modelo sozinho, o que é o que permite gerar texto de um treino antigo sem lembrar com que dimensões ele rodou.

**A restauração é in-place, via `updateData`.** Os parâmetros são `val` dentro de `Linear` e `LayerNorm`, e outros nós do grafo guardam a referência deles. Trocar a instância quebraria essas referências — é o mesmo argumento que fez o `Gradient` ser mutável na Etapa 2.

**O exemplo numérico.** Um modelo com `vocabSize = 6`, `dModel = 8`, `nHeads = 2`, `nLayers = 1`, treinado por dois passos e salvo. Um segundo modelo, com inicialização aleatória diferente, carrega o arquivo:

```
antes do load:   logits(A) ≠ logits(B)
depois do load:  logits(A) = logits(B), posição a posição
```

E o teste que fecha a questão dos momentos: dar **mais um passo** dos dois lados produz exatamente os mesmos parâmetros. Um terceiro modelo, que carregou os pesos mas com otimizador zerado, diverge no mesmo passo.

---

## §7. Ler a curva

O log de cada passo traz cinco números: `step`, `lr`, `‖g‖`, `loss` e a perplexidade. Aprender a lê-los economiza horas.

**A perda inicial.** Ela parte de `log(V)` **ou de um pouco acima**, e nunca de baixo. Com `V = 65`, o piso é `4.174`.

O piso não é aproximação, é garantia. Um modelo recém-inicializado não chuta uniformemente: os logits dele são aleatórios, com alguma dispersão. Como o alvo não tem relação nenhuma com esses logits, a perda média é a média de `−log(pᵢ)` sobre as posições, e a desigualdade de Jensen dá `média(−log pᵢ) ≥ −log(média pᵢ) = log(V)`. A igualdade só vale se `p` for exatamente uniforme.

Medido neste projeto, com `V = 3` e `log(3) = 1.0986`, a perda do primeiro passo caiu entre `1.10` e `1.84` conforme a inicialização. Um pouco acima do piso é o normal.

**Muito** acima é bug — alvos desalinhados, ou inicialização com desvio grande demais, que espalha os logits e afasta `p` da uniforme. Abaixo do piso é pior: o modelo está vendo a resposta, e o suspeito é a máscara causal.

**A perda não desce.** Nesta ordem: o `lr` chegou a subir do zero? O `zeroGrad` está antes do `backward`? A norma do gradiente é maior que zero? Um `‖g‖` exatamente zero significa que o gradiente não está chegando aos parâmetros.

**A perda desce e volta a subir.** Taxa alta demais, ou clipping frouxo. A norma no log diz qual: se ela dispara antes da subida, é clipping.

**A perda cai e a validação sobe.** Overfitting. É o único caso desta lista que não é bug.

**O exemplo numérico.** O teste de fumaça deste projeto usa um corpus perfeitamente cíclico — `0, 1, 2, 0, 1, 2, ...` — onde o token seguinte é sempre determinado pelo atual. Um modelo com `dModel = 16` e uma camada, em 60 passos:

```
perda do passo 1:              ≈ 1.0986 = log(3)
média dos 10 últimos passos:   < 0.3
```

Cair abaixo de `0.3` num corpus determinístico é o mínimo aceitável. Se a cadeia inteira — tokenização, embedding, atenção, perda, otimizador — não conseguir aprender **essa** regra, o bug não está nos hiperparâmetros.

> **Confira você mesmo.** A perda de treino está em `0.5` e a de validação em `2.3`. O que fazer primeiro?
>
> <details><summary>Resposta</summary>
>
> Nada de código: é overfitting, não bug. O modelo tem capacidade demais para o corpus, ou o corpus é pequeno demais. As saídas são aumentar o `weightDecay`, reduzir o modelo, ou arranjar mais texto. Mexer no laço não resolve — ele está funcionando exatamente como deveria, e a validação é a prova.
> </details>

---

## §8. Para onde isso leva

O modelo treina. Falta ver o que ele aprendeu.

A Etapa 19 usa duas coisas construídas aqui. O `noGrad` da avaliação é o mesmo modo em que a geração roda — nenhum grafo, nenhum gradiente. E o checkpoint é o que permite gerar texto de um modelo treinado em outra sessão, sem retreinar.

A diferença é que a geração é **autoregressiva**: a saída de um passo vira a entrada do seguinte. O modelo passa a se alimentar do próprio texto, e é isso que transforma um classificador de próximo token num gerador.

---

## Cartão de referência

| Peça | Assinatura | O que devolve |
|---|---|---|
| `LRSchedule.cosine` | `(step, totalSteps, lrMax, lrMin, warmupSteps)` | a taxa daquele passo |
| `GradientClipping.globalNorm` | `(parameters)` | `‖g‖` |
| `GradientClipping.clipByGlobalNorm` | `(parameters, maxNorm)` | `‖g‖` **antes** do corte |
| `Trainer.evaluate` | `(model, sampler, batchSize, batches)` | perda média, sem grafo |
| `Trainer.train` | `(model, optimizer, sampler, config, ...)` | `TrainingResult` |
| `Checkpoint.save` / `load` | `(file, model, optimizer)` | — / o otimizador restaurado |

| Hiperparâmetro | Valor | Papel |
|---|---|---|
| `maxGradNorm` | `1.0` | teto da norma global |
| `warmupSteps` | ~1% a 10% de `totalSteps` | protege os primeiros passos |
| `lrMin` | `lrMax / 10` | o piso do cosseno |
| `evalInterval` | dezenas de passos | a curva que realmente importa |

**As lições que se repetem:**

**Ordem é contrato quando há estado.** Nas dezessete etapas anteriores tudo era função pura, e a ordem era só a das dependências. Aqui trocar dois passos de lugar produz um treino que roda e não aprende.

**Reescalar preserva direção; reescalar em grupos, não.** A norma global existe por isso, e o exemplo da razão `30 → 6` é a demonstração.

**O que arranca do zero precisa de cuidado especial.** O aquecimento e a correção de viés da Etapa 17 resolvem o mesmo problema em lugares diferentes: os primeiros passos são os menos confiáveis e os mais influentes.

**Instrumentar é mais barato que depurar.** Cinco números por passo custam nada e transformam "a perda não desce" numa pergunta respondível.

**Salve o estado inteiro, não só o resultado.** Pesos sem `m` e `v` retomam com um degrau invisível na curva.
