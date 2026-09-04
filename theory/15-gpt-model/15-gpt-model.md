# Etapa 15 — Modelo GPT Completo

## Antes de começar

### O que você vai construir

O modelo inteiro: de inteiros que representam texto até uma pontuação por token do vocabulário.

| Componente | Formato | O que faz |
|---|---|---|
| `Embedding` | `[vocabSize, dModel]` e `[contextLength, dModel]` | traduz cada token e cada posição em vetor |
| `L` × `TransformerBlock` | 14 tensores cada | a pilha — o modelo propriamente dito |
| `LayerNorm` final | `[dModel]` × 2 | devolve escala previsível ao fluxo residual |
| cabeça de linguagem | `[dModel, vocabSize]` | projeta cada posição sobre o vocabulário |

Entrada `[B, T]` de inteiros, saída `[B, T, vocabSize]` de logits. É a primeira vez no projeto que a entrada e a saída **não** têm o mesmo formato.

### O que você precisa saber antes

**Da Etapa 7:** o tokenizador. Ele produz os inteiros que este modelo recebe.

**Da Etapa 9:** a `Embedding`, que neste projeto já soma token e posição num só passo.

**Da Etapa 14 inteira:** o bloco, e em especial a §5 sobre o fluxo residual. A variância que cresce com a profundidade é o que motiva o LayerNorm final desta etapa.

**Da Etapa 6 §1:** o que é um logit. A cabeça de linguagem produz logits, e o softmax que os transforma em probabilidade só aparece na Etapa 16.

### Onde esta etapa se encaixa

Esta é a etapa de integração. Nenhuma matemática nova, nenhuma operação nova — só montagem.

Se as catorze etapas anteriores estiverem corretas, o modelo funciona na primeira tentativa. Se não funcionar, o problema está numa peça específica. E o modo de descobrir qual é justamente o que o projeto vem construindo: cada peça tem a sua própria suíte.

Ao fim desta etapa você tem um GPT completo, capaz de receber texto e produzir uma distribuição sobre o próximo token. Ele ainda não sabe nada — os pesos são aleatórios, e não existe perda nem otimizador. Mas o forward está inteiro.

### Objetivos de aprendizagem

Ao terminar, você deve conseguir:

1. Traçar o caminho `[B, T]` → `[B, T, vocabSize]` nomeando cada passo e cada formato.
2. Provar que sem embedding posicional **uma** camada de atenção causal não distingue a ordem do prefixo — e dizer o que muda com duas.
3. Explicar por que existe um LayerNorm depois do último bloco.
4. Dizer o que os pesos amarrados economizam, e por que a operação faz sentido.
5. Justificar a escala `1/√(2·nLayers)` e decidir onde ela entra no código.

### Convenções de notação

| Símbolo | Significa |
|---|---|
| `tokens` | a entrada, `[B, T]` de inteiros em `[0, vocabSize)` |
| `vocabSize` | quantos tokens distintos existem |
| `contextLength` | quantas posições a tabela posicional cobre |
| `L` | número de blocos — `nLayers` no código |
| `logits` | a saída, `[B, T, vocabSize]`; pontuação bruta, antes do softmax |
| `E` | a tabela de embedding de token, `[vocabSize, dModel]` |

A convenção `dX ≡ ∂L/∂X` do overview continua valendo.

> **Guia visual.** O caminho completo, a prova da permutação e a divisão dos parâmetros: [`gpt.html`](gpt.html). **Exercícios (15 questões):** [`exercises.html`](exercises.html).

---

## §1. De inteiros a logits

Seis passos, e um deles muda o formato:

```
tokens  [B, T]          inteiros
  → Embedding           [B, T, dModel]     token + posição, somados
  → Bloco 1             [B, T, dModel]
  → ...                 [B, T, dModel]
  → Bloco L             [B, T, dModel]
  → LayerNorm final     [B, T, dModel]
  → cabeça de linguagem [B, T, vocabSize]  ← o único passo que muda o formato
```

A coluna do meio é a mesma da Etapa 14 §1, repetida `L` vezes. Foi para isso que o bloco preservou o formato.

> **Definição — cabeça de linguagem (*language model head*).** A camada linear final, que projeta cada vetor de `dModel` números sobre o vocabulário inteiro. A saída dela na posição `t` é a pontuação de cada token do vocabulário como candidato a **próximo** token.

Repare no que a saída significa. Para uma sequência de 5 tokens, o modelo não produz uma resposta: produz **cinco**, uma por posição. A posição 0 prevê o token 1, a posição 1 prevê o token 2, e assim por diante. É a máscara causal da Etapa 11 §5 que torna isso legítimo — nenhuma dessas previsões enxergou a resposta.

**O exemplo numérico.** Com `vocabSize = 65`, `dModel = 128`, `L = 4` e uma entrada de `B = 2`, `T = 8`:

| passo | formato | elementos |
|---|---|---|
| `tokens` | `[2, 8]` | 16 |
| depois da `Embedding` | `[2, 8, 128]` | 2.048 |
| depois de cada bloco | `[2, 8, 128]` | 2.048 |
| depois do LayerNorm final | `[2, 8, 128]` | 2.048 |
| `logits` | `[2, 8, 65]` | 1.040 |

Dezesseis inteiros entram, 1.040 números saem. O modelo devolve 65 pontuações para cada uma das 16 posições.

> **Armadilha.** Índices de token guardados como `Double`.
>
> Aconteceu neste projeto, na Etapa 9. O `Tensor` só guarda `Double`, então os índices que entram no modelo são `1.0`, `2.0`, `3.0`. Um índice fracionário — vindo de uma conta, de um `mean`, de um erro de digitação — seria truncado em silêncio, e o modelo leria a linha errada da tabela.
>
> Por que é perigoso: não há erro, não há aviso. O treino simplesmente aprende a associação errada, e o sintoma é "o modelo não converge".
>
> **Lição geral:** quando um tipo é mais permissivo que o domínio, a validação tem que ser explícita. A `Embedding` deste projeto tem um `require` de que cada índice seja inteiro, exatamente por isso.

---

## §2. Por que a posição precisa entrar

A `Embedding` da Etapa 9 soma duas tabelas: uma indexada pelo token, outra pela posição. A pergunta que ficou em aberto lá é por que a segunda existe.

A resposta é uma propriedade estrutural, e ela se demonstra.

Considere a atenção causal **sem** informação de posição. Na posição `t`, a máscara deixa visíveis as posições `0` até `t`. Os pesos de atenção saem de produtos internos entre a query de `t` e as keys dessas posições. A saída é uma soma ponderada dos values.

Nada nessa conta usa **onde** cada token está. O resultado depende apenas de *quais* tokens estão no prefixo — do conjunto, não da ordem.

> **Definição — equivariância a permutação.** Uma função é equivariante a permutação quando reordenar a entrada apenas reordena a saída, sem mudar valores. Uma camada de atenção sem posição é pior que isso na presença da máscara causal: reordenar o prefixo não muda **nada** na saída das posições seguintes.

**O exemplo numérico.** Uma cabeça de atenção causal, `dModel = 4`, três tokens distintos:

```
a = [1.0, -2.0, 0.5, 3.0]     b = [-1.5, 0.5, 2.0, -0.5]     c = [0.25, 1.0, -1.0, 2.0]
```

Rodando com `[a, b, c]` e depois com `[b, a, c]` — o prefixo trocado, o último token igual:

```
saída em t=2, com [a, b, c]:  [-1.873728, 0.438537, -1.909045, 1.416224]
saída em t=2, com [b, a, c]:  [-1.873728, 0.438537, -1.909045, 1.416224]

diferença máxima em t=2:  0.0        ← exatamente zero, não arredondamento
diferença máxima em t=0:  4.69444
```

A posição 0 muda, e muito, porque ali o token é outro. A posição 2 não muda **em nada**. Para ela, `[a, b]` e `[b, a]` são o mesmo prefixo.

Um modelo assim não consegue distinguir "o gato comeu o rato" de "o rato comeu o gato".

**A verificação independente.** Some um vetor por posição — `P₀`, `P₁`, `P₂`, distintos entre si — e repita o experimento:

```
diferença máxima em t=2, agora:  0.417607
```

A degenerescência sumiu. O token `a` na posição 0 deixou de ser o mesmo objeto que o token `a` na posição 1. Cada um chegou somado a um vetor diferente.

**O limite da demonstração: uma camada.** A invariância acima vale para **uma** camada de atenção. Empilhando duas, ela desaparece:

| camadas de atenção | diferença em `t=2`, prefixo trocado |
|---|---|
| 1 | `0.000000` |
| 2 | `5.09e-2` |
| 3 | `6.43e-1` |

O motivo é a própria máscara. A posição 0 enxerga um token, a posição 1 enxerga dois, e a 2 enxerga três — então as saídas da primeira camada já carregam uma assimetria que depende de onde cada posição está. A segunda camada lê isso.

Ou seja: um modelo causal profundo **consegue** reconstruir a ordem sem tabela posicional nenhuma. Isso é conhecido, e modelos sem codificação de posição de fato treinam. O que a tabela oferece é a informação de forma **direta**, em vez de exigir que o modelo a deduza de um efeito colateral da máscara.

Para o teste da §8 isso é decisivo: só com `nLayers = 1` a remoção da tabela posicional produz saída idêntica. Com dois blocos, o teste passa de graça, sem provar nada.

**Por que somar, e não concatenar.** É o mesmo argumento da Etapa 14 §2. Concatenar dobraria `dModel` e custaria parâmetros. Somar mantém o formato, e o modelo aprende a separar as duas informações dentro do mesmo vetor. As tabelas são treináveis, e nada obriga as duas a usarem as mesmas direções.

**O `contextLength` é um teto real.** A tabela posicional tem uma linha por posição, e posições além dela simplesmente não existem. Um modelo com `contextLength = 128` não tem o que somar na posição 128. Por isso a `Embedding` valida `T ≤ contextLength` em vez de deixar o índice estourar.

> **Confira você mesmo.** Se a máscara causal já diz a cada posição quantos tokens vêm antes dela, isso não seria informação de posição suficiente?
>
> <details><summary>Resposta</summary>
>
> Numa camada, não: a máscara diz à posição `t` **quantos** tokens ela pode ver, e os dois casos comparados têm exatamente o mesmo tamanho de prefixo. Com duas camadas ou mais, a resposta muda — a assimetria da máscara vira sinal, e o modelo consegue deduzir posição a partir dela, como a tabela de três linhas acima mostra. O que a tabela posicional garante é que a informação esteja disponível de forma direta, desde a primeira camada, em vez de precisar ser reconstruída.
> </details>

---

## §3. A pilha: `L` blocos e nada de novo

```
x = embedding(tokens)
x = bloco₁(x)
x = bloco₂(x)
...
x = blocoₗ(x)
```

Em Scala isso é um `foldLeft` sobre a lista de blocos, com o embedding como valor inicial. Nenhuma operação nova.

Duas coisas que **não** são compartilhadas entre os blocos, e vale afirmar:

**Cada bloco tem os seus próprios pesos.** `L` blocos são `L` conjuntos independentes de 14 tensores. Compartilhar pesos entre blocos é uma arquitetura diferente, e não é a do GPT.

**Cada bloco tem os seus próprios LayerNorm.** É a mutação que a Etapa 14 mediu. Usar a mesma instância em dois lugares não muda formato nem trava nada, e só aparece na contagem de tensores distintos.

> **Definição — profundidade.** O número de blocos, `nLayers`. É o hiperparâmetro que mais mexe na contagem de parâmetros: cada bloco custa `12·dModel² + 11·dModel`, e a pilha custa isso vezes `L`.

**O exemplo numérico.** Com `dModel = 128`, cada bloco tem `12·16.384 + 11·128 = 198.016` parâmetros. Quatro blocos são `792.064` — e a §7 vai mostrar que isso é 96% do modelo tiny inteiro.

---

## §4. O LayerNorm final e a cabeça de linguagem

A Etapa 14 §5 deixou este gancho aberto. Em pre-LN, cada `LayerNorm` normaliza só o que **entra** numa subcamada. O fluxo residual em si atravessa a pilha inteira sem nunca ser renormalizado, acumulando `2L` incrementos.

O resultado é previsível pela conta de lá: com incrementos de variância 1, a saída da pilha tem variância `1 + 2L`.

| `L` | variância na saída da pilha | norma cresce |
|---|---|---|
| 4 | 9 | `×3.00` |
| 6 | 13 | `×3.61` |
| 12 | 25 | `×5.00` |

A cabeça de linguagem receberia um vetor cuja escala depende de quantos blocos existem. O `LayerNorm` final resolve isso: ele entrega à projeção um vetor de média 0 e desvio 1, com `γ` e `β` treináveis por cima.

> **Definição — `ln_f`.** O nome que o GPT-2 dá ao LayerNorm final. Todo modelo pre-LN tem um; modelos post-LN não precisam, porque cada bloco já normaliza na saída.

**A cabeça, e o viés que ela não tem.** A projeção final é uma `Linear(dModel, vocabSize)` **sem viés**, e o GPT-2 faz o mesmo. Vale descartar primeiro o argumento do `b_K` da Etapa 12 §8. O próximo passo é um softmax, que é invariante a somar uma constante a todos os logits. Um viés por token do vocabulário **não** é constante — ele muda cada logit de forma diferente, então não seria morto. O que pesa contra ele é outro: `vocabSize` parâmetros a mais para representar uma frequência base que a tabela de embedding já consegue representar.

**O exemplo numérico.** Com `dModel = 128` e `vocabSize = 65`, a cabeça é uma matriz `[128, 65]`, ou `8.320` pesos. Cada logit é um produto interno entre o vetor da posição e uma coluna dessa matriz:

```
logit[b, t, v] = Σ_d  x[b, t, d] · W[d, v]
```

Ou seja: **cada token do vocabulário tem um vetor de `dModel` números**, e o logit dele é o alinhamento com o vetor da posição. É a mesma leitura chave-valor da Etapa 13 §3, e é o que torna a §5 possível.

> **Confira você mesmo.** Por que o LayerNorm final não torna o LayerNorm de dentro dos blocos desnecessário?
>
> <details><summary>Resposta</summary>
>
> Porque eles protegem coisas diferentes. Os de dentro garantem que **cada subcamada** receba entrada de escala controlada, em todos os `L` níveis da pilha. O final garante que **a cabeça de linguagem** receba a mesma coisa. Sem os de dentro, a atenção do bloco 12 receberia um vetor com variância 25, e os scores dela sairiam numa escala completamente diferente dos do bloco 1.
> </details>

---

## §5. Pesos amarrados

A tabela de embedding é `[vocabSize, dModel]`: uma linha por token. A cabeça de linguagem é `[dModel, vocabSize]`: uma coluna por token.

As duas guardam a mesma coisa — um vetor por token do vocabulário. Uma é usada para **entrar** no espaço do modelo, a outra para **sair** dele.

> **Definição — pesos amarrados (*weight tying*).** Usar a mesma matriz nas duas pontas, transposta na saída: `logits = x · Eᵀ`, com `E` a tabela de embedding. O parâmetro é um só, e o gradiente dele recebe contribuição dos dois usos.

O argumento a favor é geométrico. Se o vetor do token "gato" é `e`, então prever "gato" deveria significar que o estado da posição aponta na direção de `e`. Manter duas representações independentes do mesmo token permite que elas divirjam, sem que nada no treino as obrigue a concordar.

**O que isso economiza.** Exatamente `vocabSize · dModel` parâmetros. E não é pouco:

| modelo | sem amarrar | com pesos amarrados | economia |
|---|---|---|---|
| tiny (`vocab 65`, `d 128`, `L 4`) | 825.344 | 817.024 | 1,0% |
| GPT-2 small (`vocab 50257`, `d 768`, `L 12`) | 163.018.752 | 124.421.376 | **23,7%** |

A diferença entre as duas linhas é o tamanho do vocabulário. Com 65 tokens a tabela é irrelevante; com 50 mil, ela é quase um quarto do modelo.

**A verificação independente.** O número famoso do GPT-2 small é `124.439.808` parâmetros. A nossa contagem amarrada dá `124.421.376`. A diferença de `18.432` é a de sempre: os vieses `b_K` e `b_V` que a Etapa 12 §8 removeu, `1.536` por bloco. **O GPT-2 é 124M porque amarra os pesos** — sem amarrar seriam 163M.

**No nosso código, a transposta é de graça.** O `matmul` lê os operandos pelas strides, sem copiar, como a Etapa 12 §9 estabeleceu. Então `x.matmul(E.transpose())` não aloca nada — a transposta é uma view.

O custo é de acoplamento. A `Embedding` precisa expor a tabela de token, e o modelo passa a ter um parâmetro que aparece em dois lugares do grafo. O `parameters` deve listá-lo **uma vez** — listar duas faria o otimizador da Etapa 17 aplicar o passo duas vezes ao mesmo tensor.

> **Confira você mesmo.** Com pesos amarrados, o gradiente da tabela `E` recebe contribuição de dois caminhos. Isso é um problema?
>
> <details><summary>Resposta</summary>
>
> Não — é o comportamento correto, e o `scalagrad` já faz. Um tensor usado duas vezes no grafo acumula os dois gradientes, exatamente como o operando compartilhado da Etapa 3 §6. O problema apareceria no otimizador, e por outro motivo: se `parameters` listasse o mesmo tensor duas vezes, ele seria atualizado duas vezes por passo. Por isso a lista precisa ser deduplicada.
> </details>

---

## §6. A escala residual: fechando a pendência

Esta pendência está aberta desde a Etapa 12, e a Etapa 14 §5 derivou a fórmula. Aqui ela se decide, porque é aqui que `nLayers` existe.

Recapitulando: a saída da pilha tem variância `1 + 2L`. Multiplicar a saída de cada subcamada por `1/√(2L)` faz a soma dar exatamente `2`, para qualquer profundidade.

**Onde o GPT-2 aplica isso.** Não no forward, e sim na **inicialização**. As duas matrizes que escrevem no fluxo residual nascem com desvio padrão `0.02 / √(2L)`, em vez de `0.02`. São a `W_O` da atenção e a segunda `Linear` do MLP. Depois disso o treino faz o que quiser com elas.

Isso muda o que é preciso implementar. Não é um fator no `forward`; é um argumento a mais na inicialização de duas camadas.

| `L` | `1/√(2L)` |
|---|---|
| 4 | `0.35355` |
| 6 | `0.28868` |
| 12 | `0.20412` |

**A inconsistência que fica visível aqui.** Este projeto usa duas inicializações diferentes: a `Embedding` nasce com `N(0, 0.02²)`, à moda do GPT-2, e a `Linear` usa Kaiming, `σ = √(2/inputDim)`. Com `dModel = 128`, Kaiming dá `σ = 0.125` — mais de seis vezes o `0.02`.

Nenhuma das duas está errada, mas a mistura não tem justificativa. Há dois caminhos coerentes:

1. **Manter Kaiming e escalar só as duas projeções residuais** por `1/√(2L)`. Mudança mínima: a `Linear` ganha um parâmetro de escala opcional, e `MultiHeadAttention` e `MLP` o repassam para a camada de saída.
2. **Adotar `0.02` em tudo**, como o GPT-2, com `0.02/√(2L)` nas residuais. Mais fiel ao modelo de referência, e mexe na inicialização de todas as camadas — inclusive nas que a Etapa 8 §5 justificou com Kaiming.

A Etapa 13 §6 já mostrou que o fator 2 do Kaiming só se justifica na segunda `Linear` do MLP, cuja entrada veio da GELU. Nas outras ele dobra a variância sem motivo. Isso pesa a favor do caminho 2. Mas o caminho 1 preserva a derivação da Etapa 8, que é material didático já escrito.

**Decidido: caminho 1**, implementado em 2026-09-01. A `Linear` ganhou um `initScale` opcional, `MultiHeadAttention` e `MLP` o repassam para a camada de saída, e o `GPT` calcula `1/√(2·nLayers)` e passa a cada bloco. O caminho 2 fica registrado como possibilidade.

**A medição, e o que ela mostrou.** Desvio padrão do fluxo residual na saída de uma pilha de blocos, com entrada `N(0,1)` e `dModel = 32`:

| `L` | sem escala | com `1/√(2L)` |
|---|---|---|
| 1 | `2.14` | `1.67` |
| 2 | `3.64` | `2.05` |
| 4 | `8.11` | `2.95` |
| 8 | `18.39` | `4.53` |
| 16 | `40.00` | `6.94` |

A escala funciona: sem ela o desvio cresce quase linearmente com a profundidade, e com ela cresce bem mais devagar. Em 16 camadas a diferença é de quase seis vezes.

**Mas ela não fica constante, como a derivação previa.** A conta da Etapa 14 §5 diz que a soma telescopa para `2`, ou seja, desvio `1.41` em qualquer profundidade. O medido vai de `1.67` a `6.94`.

A causa é a ressalva de dois parágrafos acima. A derivação supõe que cada subcamada devolve variância 1, e neste projeto ela devolve mais: o fator 2 do Kaiming está em **todas** as `Linear`, não só na que vem depois da GELU. Cada bloco amplifica um pouco antes de a escala residual dividir, e o excedente se acumula.

Ou seja, a medição confirma empiricamente o argumento que já pesava a favor do caminho 2. Corrigir isso de vez significa uniformizar a inicialização — e essa é uma mudança maior, que troca a derivação da Etapa 8 §5 por outra. Por ora, o mecanismo está implementado, testado e medido, que era o objetivo desta seção.

---

## §7. Contagem de parâmetros do modelo inteiro

Somando tudo, com `V = vocabSize` e `C = contextLength`:

```
embedding de token:      V · dModel
embedding posicional:    C · dModel
pilha:                   L · (12·dModel² + 11·dModel)
LayerNorm final:         2 · dModel
cabeça de linguagem:     dModel · V        (zero, se amarrada)
```

Nas três configurações que interessam:

| | tiny | médio | GPT-2 small |
|---|---|---|---|
| `vocabSize` | 65 | 65 | 50.257 |
| `dModel` | 128 | 384 | 768 |
| `L` | 4 | 6 | 12 |
| `contextLength` | 128 | 256 | 1.024 |
| embeddings | 24.704 | 123.264 | 39.383.808 |
| pilha | 792.064 | 10.642.176 | 85.036.032 |
| cabeça (amarrada) | 0 | 0 | 0 |
| **total** | **817.024** | **10.766.208** | **124.421.376** |

Duas leituras valem guardar.

**No tiny, a pilha é 96% do modelo.** Com vocabulário de 65 caracteres, embeddings e cabeça quase não pesam. Toda a capacidade está nos blocos.

**No GPT-2, os embeddings são 32%.** Com 50 mil tokens, a tabela sozinha tem 38,6 milhões de parâmetros — quase metade do que a pilha inteira custa.

O roadmap estimava "1 a 10M" para a configuração tiny. A conta real dá `817 mil`, porque a estimativa presumia um vocabulário maior. Com a tokenização por caractere da Etapa 7, o vocabulário é pequeno por construção.

---

## §8. Implementação e como testar

**A assinatura.** `GPT(vocabSize: Int, dModel: Int, nHeads: Int, nLayers: Int, contextLength: Int, expansion: Int = 4)`. São cinco hiperparâmetros obrigatórios, e vale considerar agrupá-los num `case class GPTConfig` — a Etapa 18 vai querer serializar isso num checkpoint.

**Os campos:** `embedding`, `blocks: List[TransformerBlock]`, `lnFinal`, `head`. Todos públicos, pelo mesmo motivo da Etapa 14: sem acesso às peças, o teste de composição não é escrevível.

**O forward é um `foldLeft`:**

```
head.forward(lnFinal.forward(blocks.foldLeft(embedding.forward(tokens))((x, b) => b.forward(x))))
```

**As validações.** `tokens.rank == 2` e `T ≤ contextLength` já vivem na `Embedding`. Duplicá-las aqui é defensável pela mesma razão da Etapa 14 — a mensagem nomeia a camada que o chamador invocou. `nLayers >= 1` é `require` do construtor.

**`parameters` precisa deduplicar.** Com pesos amarrados, a tabela de token aparece em dois lugares. `distinct` sobre a lista resolve, e o teste `toSet.size == parameters.size` protege.

Oito frentes de teste.

**Formato de ponta a ponta.** `[B, T]` entra, `[B, T, vocabSize]` sai. Com `B`, `T`, `dModel`, `vocabSize` e `nLayers` distintos entre si.

**Composição contra as peças.** Reproduza o forward chamando `embedding`, os blocos em ordem, `lnFinal` e `head`, e compare posição a posição.

**A ordem da pilha importa.** Compare contra uma referência que aplique os blocos na ordem inversa. Com `L ≥ 2` os resultados têm que diferir — se não diferirem, provavelmente todos os blocos são a mesma instância.

**Blocos independentes.** `blocks.map(_.parameters).toSet.size == nLayers` — cada bloco tem os seus próprios tensores. O erro que isso pega é `List.fill(nLayers)(bloco)` com um único bloco construído fora, que é uma linha muito parecida com a certa.

**A posição importa.** Este é o teste que carrega a etapa, e sai direto da §2. Com `T ≥ 3`, permutar dois tokens do prefixo tem que mudar a saída da última posição. **Use `nLayers = 1` aqui**: com dois blocos, a assimetria da máscara já distingue a ordem sozinha, e o teste passaria mesmo sem tabela posicional nenhuma.

**Causalidade ponta a ponta.** Mudar o último token não pode alterar os logits das posições anteriores. Vale a contraprova: o último **tem** que mudar.

**`parameters` e contagem.** Sem repetição, todos com `requiresGradient`, e soma igual à fórmula da §7 — nas versões amarrada e não amarrada.

**Gradient check.** Nos parâmetros, com um modelo minúsculo. Aqui vale um aviso de custo: o check perturba cada elemento duas vezes, e cada perturbação é um forward completo. Com `vocabSize = 7`, `dModel = 4`, `L = 1` e `T = 3`, isso é viável; com o tiny de 817 mil parâmetros, não é. Escolha a menor configuração que ainda exercite todos os caminhos.

> **Armadilha.** Um `Linear` no lugar de um `LayerNorm`.
>
> Aconteceu neste projeto, na Etapa 14, ontem. Os dois campos `ln1` e `ln2` do bloco nasceram como `Linear(dModel, dModel)`. Compila, roda, devolve o formato certo — e não normaliza nada.
>
> Por que é perigoso: dos 16 testes do bloco, só 2 reprovaram. Passou até a comparação contra a composição das subcamadas, porque a referência usa `block.ln1` e portanto **concorda com o erro**. Uma referência montada a partir dos campos do objeto sob teste herda os erros que estão nos campos.
>
> **Lição geral:** o teste de composição precisa de companhia estrutural. O que pegou foi o formato dos parâmetros e a contagem total — as duas asserções que olham o objeto de fora. Nesta etapa isso vale em dobro, porque o modelo é composição de composições.

**Mutações que a suíte precisa pegar:**

| mutação | quem deveria pegar |
|---|---|
| esquecer o LayerNorm final | a composição contra as peças |
| aplicar os blocos na ordem inversa | a referência de ordem |
| `List.fill(nLayers)(umBlocoSó)` | a independência entre blocos |
| remover o embedding posicional | o teste de permutação, e **só** com `nLayers = 1` |
| listar o parâmetro amarrado duas vezes | `toSet.size == parameters.size` |
| trocar a cabeça por `[vocabSize, dModel]` | o formato da saída |

---

## §9. Para onde isso leva

O modelo está completo. Ele recebe texto tokenizado e devolve, para cada posição, uma pontuação por token do vocabulário.

O que falta é fazê-lo aprender, e são três etapas:

1. **Etapa 16 — a perda.** Cross-entropy compara os logits com o token que realmente veio depois, e produz **um único número**. É a primeira vez no projeto que existe um `L` no sentido do `dX ≡ ∂L/∂X` que a notação usa desde a Etapa 2.
2. **Etapa 17 — o otimizador.** AdamW percorre a lista de `parameters` e ajusta cada tensor na direção oposta ao gradiente.
3. **Etapa 18 — o laço.** Amostrar um lote, calcular a perda, propagar, dar o passo, repetir.

Vale notar que o `parameters` desta etapa é o que a Etapa 17 vai consumir. A lista achatada, deduplicada e completa é o contrato entre as duas.

---

## Cartão de referência

| Conceito | Resumo |
|---|---|
| caminho | `tokens [B,T]` → embedding → `L` blocos → `ln_f` → cabeça → `logits [B,T,V]` |
| saída | uma distribuição por posição; a posição `t` prevê o token `t+1` |
| embedding posicional | sem ele, **uma** camada causal não distingue a ordem do prefixo — diferença exata `0.0`; com duas, `5.1e-2` |
| `contextLength` | teto real: a tabela posicional tem uma linha por posição |
| `ln_f` | existe porque em pre-LN o fluxo residual nunca é renormalizado; variância `1 + 2L` |
| cabeça de linguagem | `[dModel, vocabSize]`, sem viés; cada coluna é o vetor de um token |
| pesos amarrados | `logits = x·Eᵀ`; economiza `V·dModel` — 23,7% do GPT-2 small |
| escala residual | `1/√(2L)` na **inicialização** de `W_O` e da segunda `Linear` do MLP |
| total | `V·d + C·d + L·(12d² + 11d) + 2d` (+ `d·V` se não amarrar) |
| tiny | `817.024` parâmetros, sendo 96% na pilha |
| GPT-2 small | `124.421.376` amarrado, `163.018.752` sem amarrar |

### As quatro lições que se repetem

1. **Somar é o jeito barato de misturar informação.** Posição e token entram no mesmo vetor pela mesma soma que a Etapa 14 usa no residual — sem custo de formato e sem parâmetro extra.
2. **O que a arquitetura não distingue, o modelo não aprende.** A ordem do prefixo é invisível para uma camada de atenção causal, e nenhum treino conserta isso. Com profundidade a máscara vaza posição, mas a tabela é o que torna a informação direta — e é a diferença entre uma propriedade que se prova e uma que se espera emergir.
3. **Duas representações do mesmo objeto podem ser uma só.** Os pesos amarrados são a mesma matriz nas duas pontas, e é isso que faz o GPT-2 caber em 124M em vez de 163M.
4. **Composição de composições precisa de asserção estrutural.** Comparar contra uma referência montada com os próprios campos do objeto herda os erros que estão nos campos — formato e contagem são o que olha de fora.
