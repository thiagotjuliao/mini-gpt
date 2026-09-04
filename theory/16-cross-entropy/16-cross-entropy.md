# Etapa 16 — Cross-Entropy: a perda

## Antes de começar

### O que você vai construir

Um número. Um só, para o lote inteiro.

| Componente | Formato | O que faz |
|---|---|---|
| entrada: `logits` | `[B, T, vocabSize]` | o que o modelo da Etapa 15 produziu |
| entrada: `targets` | `[B, T]` | o token que realmente veio depois |
| `logSoftmax` | `[B, T, vocabSize]` | log das probabilidades, de forma estável |
| seleção do alvo | `[B, T]` | pega, em cada posição, o log da probabilidade do token certo |
| média e sinal | **escalar** | a perda |

Zero parâmetros. A perda não aprende nada — ela mede.

### O que você precisa saber antes

**Da Etapa 2:** o que é `L`, e por que `backward()` começa nela. Esta etapa constrói o primeiro `L` de verdade do projeto.

**Da Etapa 6, em especial §1 e §5:** o softmax, o truque log-sum-exp e o `logSoftmax` já implementado.

**Da Etapa 7 §5:** o `BatchSampler`, que já entrega `inputs` e `targets` deslocados de uma posição.

**Da Etapa 15:** os logits, e o que a posição `t` está prevendo.

### Onde esta etapa se encaixa

O modelo está pronto e não sabe nada. Falta dizer a ele o que é errar.

Toda a maquinaria das quinze etapas anteriores existe para uma coisa: derivar um número em relação a milhares de parâmetros. Esse número não existia ainda. A notação `dX ≡ ∂L/∂X` aparece desde a Etapa 2, e até agora o `L` era uma promessa — nos testes, um `.sum` qualquer fazia o papel dele.

Aqui a promessa é cumprida. E o número tem que ser escolhido com cuidado, porque **é ele que define o que o modelo vai aprender**.

### Objetivos de aprendizagem

Ao terminar, você deve conseguir:

1. Escrever a cross-entropy a partir da definição e explicar por que o `−log` está ali.
2. Mostrar dois casos em que softmax seguido de log quebra, e o log-softmax não.
3. Derivar `∂L/∂logits = (p − y)/N` e dizer por que a fórmula é tão simples.
4. Prever a perda de um modelo recém-inicializado, e conferir contra a medida.
5. Converter perda em perplexidade e interpretar o número.

### Convenções de notação

| Símbolo | Significa |
|---|---|
| `V` | `vocabSize` |
| `N` | número de posições no lote, `B · T` |
| `z` | os logits de uma posição — um vetor de `V` números |
| `p` | `softmax(z)`, as probabilidades da posição |
| `y` | o vetor one-hot do token correto daquela posição |
| `L` | a perda: **um único número** |

A convenção `dX ≡ ∂L/∂X` do overview continua valendo, e agora com um `L` de verdade.

> **Guia visual.** O caminho dos formatos, os dois modos de falha numérica e a curva do `−log`: [`cross-entropy.html`](cross-entropy.html). **Exercícios (15 questões):** [`exercises.html`](exercises.html).

---

## §1. O que é uma perda

> **Definição — função de perda.** Uma função que recebe a saída do modelo e a resposta correta, e devolve **um único número**: quanto menor, melhor o modelo se saiu. É o que o `backward()` diferencia.

O "único número" não é detalhe de implementação. O gradiente responde à pergunta "se eu mexer neste parâmetro, quanto muda **a** perda?", e essa pergunta só faz sentido com um alvo escalar. Uma saída vetorial teria um gradiente por componente, e nada diria qual seguir.

Prever o próximo token é um problema de **classificação**: dadas `V` opções, qual é a certa? A perda natural para classificação é a cross-entropy, e a §2 mostra por quê.

**O exemplo numérico.** Um modelo com `V = 4` produz, numa posição, estas probabilidades:

```
p = [0.62518, 0.22999, 0.09351, 0.05132]        soma = 1.0
```

Se o token correto era o de índice 2, o modelo deu a ele `9,4%`. A perda precisa transformar esse `0.09351` num número que sirva de sinal — e é isso que a próxima seção faz.

---

## §2. Cross-entropy: a fórmula, e de onde ela vem

A forma geral compara duas distribuições sobre as mesmas `V` opções:

```
L = − Σᵢ yᵢ · log(pᵢ)
```

No nosso caso `y` é **one-hot**: vale 1 no token correto e 0 no resto. Toda a soma colapsa num termo só:

```
L = − log(p_alvo)
```

> **Definição — cross-entropy.** O logaritmo negativo da probabilidade que o modelo atribuiu à resposta certa. Ela ignora completamente como o modelo distribuiu o resto da massa.

Por que `−log`, e não `1 − p` ou `(1 − p)²`? Três propriedades que só o logaritmo tem juntas.

**Acerto perfeito custa zero.** `−log(1) = 0`. Não há prêmio por acertar; há ausência de punição.

**Errar com confiança custa muito, e sem limite.** Quando `p_alvo → 0`, a perda vai a infinito. Um modelo que dá `0.1%` ao token certo paga `6.9`; um que dá `50%` paga `0.69`. A punição por confiança errada é ilimitada, e é isso que impede o modelo de apostar tudo numa opção.

**Somar perdas é multiplicar probabilidades.** `−log(a) − log(b) = −log(a·b)`. A perda total de uma sequência é o log da probabilidade de o modelo ter gerado a sequência inteira. Minimizar a soma é maximizar essa probabilidade.

**O exemplo numérico.** Com o `p` da §1 e alvo 2:

```
L = −log(0.09351) = 2.36971
```

E a tabela que mostra a forma da punição:

| `p` do token certo | perda |
|---|---|
| `0.99` | `0.01005` |
| `0.50` | `0.69315` |
| `0.10` | `2.30259` |
| `0.01` | `4.60517` |
| `0.001` | `6.90776` |

Cada fator de 10 a menos na probabilidade custa `2.30259` a mais — que é `log(10)`. A escala é logarítmica por construção.

> **Confira você mesmo.** Por que a perda ignora como o modelo distribuiu a probabilidade entre os tokens **errados**?
>
> <details><summary>Resposta</summary>
>
> Porque `y` é one-hot, e os termos com `yᵢ = 0` somem da soma. Mas a probabilidade dos errados não é irrelevante — ela entra por outro caminho. As probabilidades somam 1, então dar mais massa a um token errado tira massa do certo, e a perda sobe. O que a fórmula ignora é **qual** dos errados recebeu; dar 30% ao token 5 ou ao token 40 custa o mesmo.
> </details>

---

## §3. Log-softmax: por que não softmax seguido de log

A fórmula pede `log(softmax(z))`. Escrever isso em dois passos é o caminho errado, e há dois modos de falha independentes.

**Estouro para cima.** O softmax exponencia. Com `z = [1000, 1001, 1002]`, `e^1000` não cabe num `Double`:

```
exp direto:        OverflowError
com max subtraído: [0.09003, 0.24473, 0.66524]
```

O truque da Etapa 6 §5 resolve: subtrair o máximo do grupo antes de exponenciar não muda o resultado, porque o fator comum cancela entre numerador e denominador.

**Estouro para baixo.** Este o max sozinho não resolve. Com `z = [0, −800]`:

```
softmax(z) = [1.0, 0.0]              ← a segunda posição virou zero exato
log(0.0)   = −infinito
```

A probabilidade verdadeira é `e^−800`, que é pequena mas não é zero. Ela some ao ser representada como `Double`, e o `log` seguinte devolve infinito. A perda daquela posição contamina o lote inteiro.

O `logSoftmax` calcula o log **sem nunca materializar a probabilidade**:

```
logSoftmax(z)ᵢ = (zᵢ − max) − log(Σⱼ e^(zⱼ − max))
```

Com `z = [0, −800]`, ele devolve `[0.0, −800.0]` — exato, sem passar perto de zero.

> **Definição — log-sum-exp.** A identidade `log Σ e^zⱼ = max + log Σ e^(zⱼ − max)`, que permite calcular o log de uma soma de exponenciais sem estourar. É o que o `logSoftmax` da Etapa 6 já implementa.

**O exemplo numérico.** Com `z = [1000, 1001, 1002]`, os três caminhos:

```
softmax e depois log:   erro de estouro antes de chegar ao log
logSoftmax:             [-2.40761, -1.40761, -0.40761]
conferência:            exp desses três → [0.09003, 0.24473, 0.66524], soma 1.0
```

> **Armadilha.** Normalizar o eixo errado.
>
> Aconteceu neste projeto, na Etapa 12. O softmax rodou sobre o eixo das *queries* em vez do das *keys*, e o formato da saída era idêntico nos dois casos.
>
> Por que é perigoso: aqui o risco é o mesmo e o alvo é o eixo do vocabulário. Sobre `[B, T, V]`, `logSoftmax(2)` normaliza cada posição sobre o vocabulário, que é o certo. `logSoftmax(1)` normalizaria cada token do vocabulário sobre as posições — números plausíveis, formato idêntico, e uma perda que mede outra coisa.
>
> **Lição geral:** derive o eixo do tensor a que ele pertence, e teste com `T ≠ V`. Com os dois iguais, o eixo trocado sobrevive a qualquer teste de formato.

---

## §4. O gradiente é `p − y`

Aqui está o resultado mais elegante do projeto. A derivada da cross-entropy em relação aos **logits** é:

```
∂L/∂z = p − y
```

A probabilidade que o modelo deu, menos o que ele deveria ter dado. Nada de logaritmo, nada de exponencial.

A derivação vem em dois passos. Primeiro, `L = −log p_a`, com `a` o índice do alvo, e `p = softmax(z)`. Do Jacobiano do softmax da Etapa 6 §3:

```
∂p_a/∂zᵢ = p_a · (δᵢₐ − pᵢ)          com δᵢₐ = 1 se i = a, senão 0
```

Segundo, a regra da cadeia com `∂L/∂p_a = −1/p_a`:

```
∂L/∂zᵢ = −(1/p_a) · p_a · (δᵢₐ − pᵢ) = pᵢ − δᵢₐ
```

O `p_a` cancela. É por isso que a fórmula não tem divisão — e é por isso que cross-entropy e softmax são sempre implementados juntos: separados, cada um tem um Jacobiano feio; fundidos, a derivada é uma subtração.

Repare no que a fórmula diz. O gradiente é positivo em todo token que recebeu probabilidade sem merecer, e negativo no token certo. O passo do otimizador vai empurrar os logits errados para baixo e o certo para cima, na proporção exata do erro.

E a soma dos gradientes de uma posição é zero: `Σpᵢ − 1 = 0`. Faz sentido — somar uma constante a todos os logits não muda nada, então mover todos juntos não pode mudar a perda.

**O exemplo numérico.** Com `z = [2.0, 1.0, 0.1, −0.5]` e alvo 2:

```
p        = [0.62518, 0.22999, 0.09351, 0.05132]
y        = [0, 0, 1, 0]
∂L/∂z    = [0.62518, 0.22999, -0.90649, 0.05132]        soma = 0.0
```

**A verificação independente.** Diferenças finitas centrais em cada logit, com `ε = 1e-6`, sobre `L = −logSoftmax(z)[2]`:

```
numérico = [0.62518, 0.22999, -0.90649, 0.05132]
```

Batem nas cinco casas. O token 0, que recebeu `62,5%` sem ser o certo, é o que mais será empurrado para baixo.

> **Confira você mesmo.** Se o modelo acertasse com certeza absoluta — `p = [0, 0, 1, 0]` com alvo 2 — qual seria o gradiente?
>
> <details><summary>Resposta</summary>
>
> Exatamente zero, em todas as coordenadas: `p − y = 0`. Nenhum parâmetro se move, o que é coerente com a perda também ser zero. Na prática o softmax nunca chega a `1` exato, então o gradiente fica muito pequeno mas não nulo — e é o que faz o treino desacelerar naturalmente nas posições que o modelo já domina.
> </details>

---

## §5. A média, e o alvo deslocado

Até aqui, uma posição. O lote tem `N = B · T` delas, e a perda do lote é a **média**:

```
L = (1/N) · Σ_{b,t} −log p[b, t, alvo(b,t)]
```

E o gradiente de cada logit ganha o mesmo divisor: `(p − y)/N`.

**Por que média, e não soma.** Com a soma, dobrar o tamanho do lote dobraria a perda e dobraria os gradientes — e a taxa de aprendizado precisaria ser reajustada a cada mudança de `batchSize`. A média torna a escala do gradiente independente do tamanho do lote.

**O alvo é o token seguinte.** A posição `t` prevê o token `t+1`. Isso já está resolvido neste projeto: o `BatchSampler` da Etapa 7 devolve `inputs = corpus[i .. i+T)` e `targets = corpus[i+1 .. i+T]`, então `logits[b,t]` e `targets[b,t]` já estão alinhados. A perda **não** desloca nada — se ela deslocasse de novo, estaria comparando com o token errado.

**O exemplo numérico.** Um lote com `B = 2`, `T = 2`, `V = 4`. As quatro perdas por posição, e a média:

```
perdas = [2.36971, 1.38629, 0.13993, 1.45693]
L      = 1.33821
```

E o gradiente da primeira posição, agora dividido por `N = 4`:

```
∂L/∂z[0,0] = [0.15630, 0.05750, -0.22662, 0.01283]
```

São os mesmos números da §4 divididos por 4. A verificação é direta: `0.62518/4 = 0.15630` ✓.

> **Armadilha.** Índices guardados como `Double`.
>
> Aconteceu neste projeto, na Etapa 9. O `Tensor` só guarda `Double`, e os `targets` que chegam aqui são `1.0`, `4.0`, `2.0`. Convertê-los para `Int` com truncamento silencioso aceita `2.9999999` como `2`.
>
> Por que é perigoso: um alvo errado não estoura nada. A perda continua sendo um número plausível, o gradiente continua fluindo, e o modelo aprende a associação errada. O sintoma aparece semanas depois como "não converge".
>
> **Lição geral:** valide na fronteira. A `Embedding` já exige que cada índice seja inteiro; a perda precisa da mesma checagem, e também de que cada alvo esteja em `[0, V)`.

---

## §6. Sanidade: `log(V)` e perplexidade

Esta seção é a que mais economiza tempo de depuração no resto do projeto.

Um modelo recém-inicializado não sabe nada. As `V` opções são aproximadamente equiprováveis, então `p_alvo ≈ 1/V`, e a perda é:

```
L₀ ≈ −log(1/V) = log(V)
```

| `V` | perda esperada no início |
|---|---|
| 4 | `1.38629` |
| 65 | `4.17439` |
| 50.257 | `10.82491` |

Este é um **teste**, não uma curiosidade. Se o primeiro `loss` do treino sai muito longe de `log(V)`, algo está errado antes de o treino começar — inicialização com escala absurda, alvo desalinhado, softmax no eixo errado.

> **Definição — perplexidade.** `exp(L)`. É a perda trazida de volta para a escala de contagem: uma perplexidade de `P` significa que o modelo está tão incerto quanto alguém escolhendo uniformemente entre `P` opções.

A perplexidade de um modelo aleatório é exatamente `V` — o que dá uma leitura imediata do progresso. Com vocabulário de 65 caracteres, sair de perplexidade 65 para 10 significa que o modelo passou a hesitar entre umas 10 opções, e não 65.

| perda | perplexidade | leitura |
|---|---|---|
| `4.17439` | `65.0` | não aprendeu nada (com `V = 65`) |
| `2.30259` | `10.0` | hesita entre ~10 caracteres |
| `0.69315` | `2.0` | quase decidido, entre dois |
| `0.01005` | `1.01` | praticamente certo |

**O exemplo numérico.** Com `V = 65` e todos os logits iguais — o caso exato, não aproximado:

```
−logSoftmax([0, 0, ..., 0])[7] = 4.17439 = log(65)
```

Qualquer índice de alvo dá o mesmo, porque todos os logits são iguais.

> **Confira você mesmo.** No primeiro passo do treino a perda medida dá `9.2`, com `V = 65`. O que isso sugere?
>
> <details><summary>Resposta</summary>
>
> Que algo está errado. `9.2` é bem acima de `log(65) = 4.17`, e um modelo aleatório não consegue ser *pior* que aleatório por acaso — para isso ele precisa estar ativamente atribuindo probabilidade baixa aos tokens certos. As suspeitas, em ordem: alvos desalinhados com os logits, inicialização com desvio padrão grande demais (logits enormes, softmax saturado), ou o eixo errado no `logSoftmax`.
> </details>

---

## §7. Implementação e como testar

**Onde mora.** `gpt/loss/CrossEntropy.scala`, o primeiro arquivo do pacote `loss`. Um `object` com dois métodos, `apply(logits, targets)` e `perplexity(loss)` — não há estado nem parâmetro a guardar.

**A assinatura.** `apply(logits: Tensor, targets: Tensor): Tensor`, com `logits` em `[B, T, V]`, `targets` em `[B, T]`, e a saída um escalar. Devolver um `Tensor` de um elemento, e não um `Double`, é obrigatório: é nele que o `backward()` vai ser chamado.

**Como selecionar o alvo, sem operação nova.** O `scalagrad` não tem um `gather`. Duas rotas:

1. **Máscara one-hot.** Construa um tensor `[B, T, V]` com `1.0` na posição do alvo e `0.0` no resto, multiplique elemento a elemento pelo `logSoftmax`, e some. É literalmente a fórmula geral da §2, e usa só operações já testadas.
2. **Um `gather` novo no núcleo**, como o PyTorch faz.

A rota 1 é a recomendada aqui. Ela **não** piora a ordem de memória: a máscara tem exatamente o tamanho dos logits, que já existem. O que ela custa é um passo a mais no grafo, e é um preço didático justo — o `gather` fica para quando houver motivo de desempenho medido.

Um cuidado: a máscara é constante. Ela nasce com `requiresGradient = false`, senão o `parameters` do treino ganharia um passageiro.

**As validações**, todas `require`: `logits.rank == 3`, `targets.rank == 2`, as duas primeiras dimensões batendo, cada alvo inteiro, e cada alvo em `[0, V)`.

Seis frentes de teste.

**Valor conhecido, calculado à mão.** Uma posição, `V = 4`, logits `[2.0, 1.0, 0.1, −0.5]`, alvo 2, perda `2.36971`. É o teste que amarra a fórmula.

**A média sobre o lote.** Quatro posições com perdas conhecidas, média `1.33821`. Pega o erro de somar em vez de mediar, e o de dividir pelo número errado.

**`log(V)` com logits uniformes.** Logits todos iguais dão exatamente `log(V)`, para qualquer alvo. Este é o teste que replica a checagem de sanidade da §6.

**Estabilidade.** Logits deslocados por uma constante grande — `z` e `z + 1000` — têm que dar a **mesma** perda. E um caso com diferença de 800 entre logits não pode produzir `NaN` nem infinito.

**O gradiente é `(p − y)/N`.** Compare o gradiente acumulado nos logits contra a fórmula fechada, posição a posição. Este é o teste que carrega a etapa: ele confere a **intenção**, coisa que o gradient check não faz.

**Gradient check.** Nos logits, com `ε` pequeno. Vale lembrar da Etapa 15: a composição profunda deslocou o mínimo da curva em U, e aqui a função é rasa — o `1e-5` padrão deve bastar.

**Mutações que a suíte precisa pegar:**

| mutação | testes que falharam |
|---|---|
| `logSoftmax` no eixo 1 em vez do 2 | 7 de 15 |
| esquecer o sinal negativo | 6 de 15 |
| somar em vez de mediar | 2 de 15 |
| deslocar os alvos em uma posição | 2 de 15 |
| dividir por `B` em vez de `B·T` | 2 de 15 |
| usar `softmax` e depois `log` | **1** de 15 |

A última linha é a que ensina. Com logits de tamanho normal, `softmax` seguido de `log` dá o **mesmo número** do `logSoftmax` — o valor conhecido passa, a média passa, o `log(V)` passa, o gradiente passa. Só o caso com diferença de 800 entre logits reprova. Uma suíte sem teste de estabilidade daria essa troca por boa, e o erro apareceria meses depois como um `NaN` no meio do treino.

---

## §8. Para onde isso leva

O ciclo de treino tem quatro peças, e agora três existem: o modelo (Etapa 15), a perda (esta), e o `backward()` que o autograd fornece desde a Etapa 2.

Falta a quarta: **o otimizador**. Ter o gradiente de cada parâmetro não adianta se ninguém mexe nos parâmetros. A Etapa 17 constrói o AdamW, que percorre a lista de `parameters` e aplica o passo.

Vale notar o que muda de natureza a partir daqui. Até a Etapa 15, cada etapa era uma função pura: entra tensor, sai tensor. O otimizador **muda estado** — é a primeira peça do projeto cujo trabalho é modificar coisas no lugar. O `Gradient` mutável, decidido lá na Etapa 2, existe justamente para esse momento.

---

## Cartão de referência

| Conceito | Resumo |
|---|---|
| perda | um único número; menor é melhor; é o que o `backward()` diferencia |
| cross-entropy | `L = −log(p_alvo)`; a soma geral colapsa porque `y` é one-hot |
| por que `−log` | acerto custa 0, erro confiante custa ilimitado, somar perdas multiplica probabilidades |
| log-softmax | evita estouro para cima (`e^1000`) e para baixo (`log(0) = −∞`) |
| gradiente | `∂L/∂z = p − y`, dividido por `N`; a soma dele numa posição é zero |
| por que é simples | o `p_a` da regra da cadeia cancela com o `1/p_a` do `−log` |
| média | sobre `N = B·T`, para que a escala não dependa do lote |
| alvo | o token seguinte; o `BatchSampler` já entrega deslocado |
| sanidade | `L₀ ≈ log(V)` — `4.17439` para `V = 65` |
| perplexidade | `exp(L)`; vale `V` num modelo aleatório |
| o teste que carrega a etapa | comparar o gradiente contra `(p − y)/N` |

### As quatro lições que se repetem

1. **A perda define o que o modelo aprende.** Ela não é uma métrica de acompanhamento: é o único sinal que o treino tem. Um erro aqui produz um treino que roda perfeitamente e não ensina nada.
2. **Funções fundidas têm derivadas mais simples que as partes.** Softmax e `−log` separados têm Jacobianos incômodos; juntos, a derivada é `p − y`. O mesmo motivo pelo qual `logSoftmax` existe como operação própria.
3. **Estabilidade numérica se resolve na formulação, não na tolerância.** O log-sum-exp não é um remendo: é a mesma matemática escrita de um jeito que o `Double` aguenta.
4. **Todo número tem um valor esperado antes de você medir.** `log(V)` no primeiro passo é uma previsão que a teoria faz e o código tem que confirmar. Quando a previsão falha, o bug está antes do treino.
