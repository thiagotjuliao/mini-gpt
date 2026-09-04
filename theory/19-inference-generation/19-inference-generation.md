# Etapa 19 — Inferência e geração de texto

## Antes de começar

### O que você vai construir

O produto final. É aqui que o projeto para de ser um exercício e vira um gerador de texto.

| Componente | Onde mora | O que faz |
|---|---|---|
| `Greedy` | `gpt.generate` | escolhe sempre o token mais provável |
| `Temperature(t)` | `gpt.generate` | amostra da distribuição inteira, mais ou menos concentrada |
| `TopK(k, t)` | `gpt.generate` | amostra só entre os `k` mais prováveis |
| `Sampler.next` | `gpt.generate` | logits entram, um índice de token sai |
| `Generator.generate` | `gpt.generate` | o laço autoregressivo, com janela deslizante |

Nenhum parâmetro novo, e nenhum gradiente. O modelo só é lido.

### O que você precisa saber antes

**Do overview §1:** que o modelo devolve uma distribuição, e que **escolher** um token a partir dela é uma decisão separada. Esta etapa é essa decisão.

**Da Etapa 6:** o softmax e a subtração do máximo.

**Da Etapa 7:** o `Tokenizer`, com `encode` e `decode`.

**Da Etapa 15:** que os logits saem com formato `[B, T, V]`, e que a posição `t` prevê o token `t + 1`.

**Da Etapa 18 §5:** o `noGrad`.

### Onde esta etapa se encaixa

O modelo treinado sabe uma coisa: dada uma sequência, qual token vem depois. Isso ainda não é gerar texto.

A ponte é a repetição, e ela é mais simples do que parece. Preveja um token, acrescente ao contexto, preveja de novo. O modelo passa a se alimentar do próprio texto.

> **Definição — geração autoregressiva.** Gerar uma sequência repetindo a previsão do próximo elemento, realimentando cada previsão como entrada da seguinte.

É essa realimentação que transforma um classificador de próximo token num gerador. Nada mais é acrescentado ao modelo.

### Objetivos de aprendizagem

Ao terminar, você deve conseguir:

1. Escrever o laço autoregressivo, e dizer por que só a última posição dos logits importa.
2. Explicar o que a janela deslizante corta, e por que corta pela esquerda.
3. Prever o efeito de uma temperatura de `0.5` e de `2.0` sobre uma distribuição dada.
4. Amostrar um token à mão pela inversa da CDF.
5. Dizer por que `top-k` existe, e o que ele resolve que a temperatura não resolve.

### Convenções de notação

| Símbolo | Significa |
|---|---|
| `V` | `vocabSize` |
| `z` | os logits da última posição — um vetor de `V` números |
| `p` | a distribuição depois do softmax |
| `T` | a temperatura |
| `k` | quantos candidatos o `top-k` mantém |

> **Guia visual.** O laço autoregressivo, a janela deslizante, o efeito da temperatura e a amostragem por CDF: [`generation.html`](generation.html). **Exercícios (14 questões):** [`exercises.html`](exercises.html).

**O exemplo que atravessa o capítulo.** Um vocabulário de quatro tokens, e os logits que o modelo produziu numa posição:

```
z = [2.0, 1.0, 0.0, -1.0]
```

---

## §1. O laço autoregressivo

O laço inteiro cabe em cinco linhas:

```
tokens = prompt
repita maxNewTokens vezes:
    contexto = tokens.takeRight(contextLength)
    z        = logits do modelo na ÚLTIMA posição do contexto
    próximo  = escolher um token a partir de z
    tokens   = tokens :+ próximo
```

**Só a última posição importa.** O modelo devolve logits `[1, T, V]` — uma distribuição por posição. As `T − 1` primeiras preveem tokens que você já tem; só a última prevê algo novo. As outras são descartadas.

Isso é desperdício de cálculo, e é deliberado. Reaproveitar as posições anteriores exigiria guardar as chaves e valores da atenção entre passos, que é o *KV cache* dos sistemas reais. Aqui o modelo é pequeno e a clareza vale mais.

> **Definição — prompt.** A sequência inicial de tokens que condiciona a geração. O modelo precisa de pelo menos um, porque prever o próximo token exige um token atual.

**O exemplo numérico.** Um modelo treinado no corpus cíclico `0, 1, 2, 0, 1, 2, ...`. O prompt é `[0, 1, 2]`, com doze tokens novos, em modo `Greedy`:

```
prompt:   [0, 1, 2]
gerado:   [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
completo: [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
```

O ciclo continua exatamente. Este é o teste de fumaça definitivo do projeto: se ele passa, a cadeia inteira funciona — tokenização, embedding, atenção causal, perda, autograd, otimizador e amostragem.

---

## §2. A janela deslizante

A sequência cresce a cada passo. O modelo não aceita sequência maior que `contextLength`, porque a tabela de posições da Etapa 9 só cobre esse tanto.

A saída é cortar. `takeRight(contextLength)` mantém os tokens mais recentes e descarta os mais antigos.

**Por que pela esquerda.** Porque a previsão do próximo token depende muito mais do que veio logo antes. Cortar pela direita descartaria justamente o contexto imediato, que é o mais informativo.

O custo é real: o modelo **esquece**. Um nome mencionado além da janela deixa de existir para ele. Não há memória fora do contexto, e é por isso que o tamanho da janela é uma das características mais citadas de um modelo de linguagem.

**O exemplo numérico.** Com `contextLength = 4` e o prompt `[0, 1, 2]`:

| passo | sequência | contexto usado | o que foi esquecido |
|---|---|---|---|
| 1 | `[0,1,2]` | `[0,1,2]` | — |
| 2 | `[0,1,2,0]` | `[0,1,2,0]` | — |
| 3 | `[0,1,2,0,1]` | `[1,2,0,1]` | o `0` inicial |
| 4 | `[0,1,2,0,1,2]` | `[2,0,1,2]` | `0, 1` |

A partir do passo 3 a janela desliza, e a saída continua tendo o comprimento total pedido. O teste `let generation run past the context length` existe para isso: sem a janela, o `require` do modelo dispararia no passo 3.

> **Confira você mesmo.** Por que a última posição do contexto é `context.length - 1`, e não `contextLength - 1`?
>
> <details><summary>Resposta</summary>
>
> Porque nos primeiros passos a sequência ainda é **menor** que a janela. Com o prompt `[0,1,2]` e `contextLength = 4`, o contexto tem três tokens, e a última posição é a 2. Usar `contextLength - 1` leria a posição 3, que não existe naquele tensor. O erro só apareceria em prompts curtos, o que é o caso comum.
> </details>

---

## §3. Greedy: o mais provável, sempre

A estratégia mais simples: pegue o índice do maior logit.

```
próximo = argmax(z)
```

Nem precisa de softmax. A exponencial é monótona, então o maior logit é sempre o de maior probabilidade — a normalização não muda quem ganha.

**A vantagem é ser determinístico.** Mesmo prompt, mesma saída, sempre. Isso torna o `Greedy` a escolha certa para teste: o teste do §1, que confere o ciclo aprendido, só é possível porque não há aleatoriedade nele.

**A desvantagem é repetir.** Escolher sempre o mais provável leva o modelo a ciclos: uma vez numa sequência que se realimenta, ele nunca sai. Num corpus cíclico isso é exatamente o que se quer. Em texto natural, produz frases que se repetem sem fim.

**O exemplo numérico.** Com `z = [2.0, 1.0, 0.0, -1.0]`, o `Greedy` devolve `0`, em qualquer execução e com qualquer semente. E se houver empate no topo? Este projeto mantém o **primeiro** índice. É uma regra arbitrária, mas fixa, e o resultado continua reproduzível.

---

## §4. Temperatura

Se o `Greedy` é rígido demais, o extremo oposto é amostrar da distribuição inteira. A temperatura controla o quanto entre os dois.

> **Definição — temperatura.** Um divisor aplicado aos logits antes do softmax. `T < 1` concentra a distribuição, `T > 1` a espalha, e `T = 1` não muda nada.

```
p = softmax(z / T)
```

Por que dividir funciona: dividir por `T < 1` **aumenta** as diferenças entre os logits, e o softmax amplifica diferenças exponencialmente. Dividir por `T > 1` encolhe as diferenças, e todos se aproximam.

**O exemplo numérico.** Com `z = [2.0, 1.0, 0.0, -1.0]`:

| `T` | `p₀` | `p₁` | `p₂` | `p₃` | efeito |
|---|---|---|---|---|---|
| `0.1` | `0.999955` | `0.000045` | `0.000000` | `0.000000` | quase `Greedy` |
| `0.5` | `0.864955` | `0.117059` | `0.015842` | `0.002144` | concentrada |
| `1.0` | `0.643914` | `0.236883` | `0.087144` | `0.032059` | o softmax puro |
| `2.0` | `0.455054` | `0.276004` | `0.167405` | `0.101536` | espalhada |
| `10.0` | `0.288651` | `0.261183` | `0.236328` | `0.213838` | quase uniforme |

Os dois limites são instrutivos. Com `T → 0` a distribuição vira um pico, e a amostragem coincide com o `Greedy`. Com `T → ∞` ela vira uniforme, e o modelo passa a chutar. A escolha do `T` é literalmente o botão entre coerência e criatividade.

> **Confira você mesmo.** Com `T = 10`, o token 3 — o **menos** provável de todos — recebe `21%`. Isso é um problema?
>
> <details><summary>Resposta</summary>
>
> É o problema que o `top-k` da §5 resolve. Um token que o modelo considera ruim ainda recebe uma fatia grande, e a cada passo há uma chance real de ele sair. Um único token improvável desvia o texto, e os passos seguintes são condicionados por esse desvio. É por isso que temperatura alta sozinha degenera rápido.
> </details>

---

## §5. Top-k

A temperatura mexe na distribuição inteira, e não sabe distinguir "o quarto candidato razoável" de "um token absurdo". O `top-k` corta pelo ranking.

> **Definição — top-k.** Manter só os `k` tokens de maior logit e renormalizar entre eles. Todo o resto recebe probabilidade exatamente zero.

A implementação põe `-inf` nos logits descartados, em vez de zerar as probabilidades depois. O motivo é que `exp(-inf) = 0` exatamente, então eles somem no próprio softmax, sem precisar de uma segunda normalização.

**O exemplo numérico.** Com `z = [2.0, 1.0, 0.0, -1.0]` e `k = 2`:

```
depois do corte:  z = [2.0, 1.0, -inf, -inf]
depois do softmax: p = [0.731059, 0.268941, 0.0, 0.0]
```

Compare com a linha `T = 1.0` da §4. O token 0 subiu de `0.643914` para `0.731059`, e os dois descartados foram a zero. A massa que eles tinham foi redistribuída entre os sobreviventes, na proporção deles.

**Empates na fronteira mantêm mais de `k`.** Com logits `[3, 1, 1, 0]` e `k = 2`, os dois empatados em `1` ficam os dois. A alternativa seria desempatar por índice, o que escolheria pela ordem alfabética do vocabulário — e essa não é uma razão.

O `top-k` compõe com a temperatura: primeiro corta, depois divide. Cortar em `k = 40` e usar `T = 0.8` é a combinação comum.

---

## §6. Amostrar pela inversa da CDF

Falta um detalhe: como sortear um índice segundo `p`.

> **Definição — CDF.** A soma acumulada das probabilidades. A posição `i` da CDF vale `p₀ + p₁ + ... + pᵢ`, e a última vale 1.

O método tem três passos. Some as probabilidades acumulando, sorteie `u` uniforme em `[0, 1)`, e devolva o primeiro índice cuja acumulada passa de `u`.

Funciona porque a fatia que o índice `i` ocupa no intervalo `[0, 1)` tem largura exatamente `pᵢ`. Sortear um ponto uniforme e ver em qual fatia ele caiu é, por construção, sortear segundo `p`.

**O exemplo numérico.** Com o `p` de `T = 1.0`:

```
p   = [0.643914, 0.236883, 0.087144, 0.032059]
cdf = [0.643914, 0.880797, 0.967941, 1.000000]
```

| `u` sorteado | primeira acumulada acima de `u` | token |
|---|---|---|
| `0.30` | `0.643914` | 0 |
| `0.65` | `0.880797` | 1 |
| `0.90` | `0.967941` | 2 |
| `0.98` | `1.000000` | 3 |

**A verificação independente.** As larguras das fatias são `0.643914`, `0.880797 − 0.643914 = 0.236883`, `0.967941 − 0.880797 = 0.087144` e `1 − 0.967941 = 0.032059`. São exatamente as probabilidades de volta, o que confirma que a construção não distorceu nada.

> **Armadilha.** A soma acumulada pode parar um epsilon **abaixo** de 1, por arredondamento de ponto flutuante. Se `u` cair nessa fresta, nenhum índice satisfaz a condição, e a busca não encontra nada.
>
> Por que é perigoso: acontece raramente e depende do sorteio, então o teste passa quase sempre. Quando acontece, o retorno é um índice inválido — e o token inválido só estoura mais tarde, no `decode`.
>
> **Lição geral:** toda busca que depende de uma soma de floats atingir um valor exato precisa de um caso de saída. Aqui, a última posição é a resposta certa quando a busca falha, e o teste força isso com um gerador que devolve `0.9999999999999999`.

---

## §7. O modo de inferência

A geração inteira roda dentro de `Tensor.noGrad`.

Não é otimização opcional. Sem ele, cada token gerado construiria um grafo de autograd completo, que ninguém consumiria — cem tokens gerados são cem grafos pendurados na memória. E os parâmetros ficariam marcados com gradiente que o próximo treino somaria ao seu.

É o mesmo mecanismo da avaliação da Etapa 18 §5, usado pelo mesmo motivo: aqui o modelo só é lido.

**O exemplo numérico.** O teste `generation should not accumulate gradients on the parameters` faz três coisas. Ele zera os gradientes, gera dez tokens com amostragem por temperatura, e confere que todo gradiente continua exatamente `0.0`.

---

## §8. Para onde isso leva

Não há Etapa 20. O projeto termina aqui, e produziu um GPT decoder-only completo. Escrito do zero, sem nenhuma biblioteca de matemática ou de aprendizado de máquina.

Vale medir a distância percorrida. A Etapa 1 construiu um array com um shape. A Etapa 19 gera texto. Entre as duas não há nenhum salto. Cada etapa usa só o que as anteriores construíram, e toda derivada foi obtida à mão e conferida por diferença finita.

O que ficou de fora, e que seria o caminho natural para continuar:

**KV cache.** A §1 recalcula o contexto inteiro a cada token. Guardar chaves e valores entre passos torna a geração linear em vez de quadrática.

**Tokenização por sub-palavra.** A Etapa 7 tokeniza por caractere, o que é simples e desperdiça contexto. O BPE é o padrão real.

**Backend de GPU.** O `scalagrad` foi separado do `gpt` exatamente para permitir isso sem tocar no modelo.

**Nucleus sampling (`top-p`).** Uma variante do `top-k` que corta por massa acumulada em vez de por contagem, adaptando quantos candidatos manter a cada passo.

---

## Cartão de referência

| Estratégia | Assinatura | Quando usar |
|---|---|---|
| `Greedy` | — | testes, e quando reprodutibilidade importa |
| `Temperature(t)` | `t > 0` | texto natural; `0.7` a `1.0` é a faixa usual |
| `TopK(k, t)` | `k ≥ 1` | quando a temperatura sozinha degenera; `k = 40` é comum |

| Peça | Assinatura |
|---|---|
| `Sampler.next` | `(logits, strategy, rng) → Int` |
| `Sampler.softmax` | `(logits) → Array[Double]`, estável |
| `Sampler.keepTopK` | `(logits, k) → Array[Double]` com `-inf` fora do topo |
| `Generator.generate` | `(prompt, maxNewTokens, strategy, onToken) → Array[Int]` |

**As lições que se repetem:**

**Repetição é o que transforma previsão em geração.** O modelo não ganhou nenhuma capacidade nesta etapa. O que mudou foi realimentar a saída na entrada.

**Escolher é separado de estimar.** O modelo devolve a distribuição inteira, e três estratégias diferentes leem a mesma distribuição de três jeitos. Essa separação estava anunciada desde o overview §1.

**Subtrair o máximo continua sendo necessário.** O softmax da geração é o mesmo da Etapa 6, e `softmax([1000, 999])` só devolve `[0.731059, 0.268941]` por causa dele.

**Busca sobre soma de floats precisa de saída.** A acumulada pode parar abaixo de 1. O caso raro tem que estar previsto no código, e não descoberto em produção.

**O `noGrad` é o modo, não uma otimização.** Ler o modelo e treinar o modelo são operações diferentes, e o código diz qual das duas está acontecendo.
