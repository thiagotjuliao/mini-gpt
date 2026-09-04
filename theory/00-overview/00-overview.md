# Visão Geral — o que estamos construindo

Este é o ponto de entrada do material. Ele não implementa nada. Ele responde às perguntas que os outros capítulos assumem respondidas: o que é um modelo de linguagem, o que exatamente estamos construindo, e como as 19 etapas se encaixam numa coisa só.

Se você está começando, **leia este documento inteiro antes da Etapa 1**. Ele leva uns vinte minutos e economiza muito mais.

---

## Comece por aqui

### Para quem é este material

Para alguém que sabe programar e nunca estudou aprendizado de máquina.

Não há pré-requisito de ML, IA ou estatística avançada. Todo termo do assunto é definido na primeira vez que aparece, e o §5 reúne todos num glossário. Se em algum ponto um termo surgir sem definição, isso é um defeito do texto — anote e reclame.

### O que você precisa saber antes

**Programação.** Arrays, classes, funções. O código é Scala, mas as ideias não dependem da linguagem.

**Cálculo básico.** Três coisas, e só elas:

| Conceito | O que você precisa saber |
|---|---|
| derivada | que ela mede o quanto a saída muda quando a entrada muda um pouquinho |
| derivada parcial | derivar em relação a uma variável, tratando as outras como constantes |
| regra da cadeia | que `dz/dx = dz/dy · dy/dx` |

**Multiplicação de matrizes.** Se `C = A @ B` não é familiar, a Etapa 3 §6 rederiva tudo do zero — mas ajuda ter visto antes.

### O que você não precisa saber

Não é preciso conhecer redes neurais, transformers, atenção, PyTorch ou qualquer biblioteca. Não é preciso saber álgebra linear além de multiplicar matrizes. Não é preciso saber probabilidade além de "números que somam 1".

O material constrói tudo isso.

### Como ler

Todos os capítulos usam quatro blocos destacados.

> **Definição.** Introduz um termo pela primeira vez.

> **Armadilha.** Um erro que **realmente aconteceu** durante a construção deste projeto. Cada um custou tempo real de depuração. São o material mais valioso do livro, porque marcam exatamente onde a intuição costuma falhar.

> **Confira você mesmo.** Uma pergunta curta, com resposta recolhida. Tente responder antes de abrir. Se errar, releia a seção — não siga em frente.

> **Guia visual.** Aponta para o diagrama da pasta. O deste capítulo: [`gpt-architecture.html`](gpt-architecture.html). **Exercícios (10 questões):** [`exercises.html`](exercises.html).

Cada seção fecha com um exemplo numérico. Refaça pelo menos alguns com papel e caneta — ler uma derivação dá a *sensação* de entendimento; reproduzi-la é o que produz entendimento.

---

## §1. O que é um modelo de linguagem

### A ideia inteira, numa frase

Um modelo de linguagem faz **uma única coisa**: dado um pedaço de texto, ele estima a probabilidade de cada token possível ser o próximo.

É só isso. Não há uma segunda capacidade escondida.

> **Definição — modelo de linguagem.** Uma função que recebe uma sequência de tokens e devolve uma distribuição de probabilidade sobre o próximo token.

Isso costuma decepcionar na primeira leitura. Parece pouco para explicar um sistema que escreve código, redige textos e responde perguntas.

### Por que isso basta

A ponte entre "prever o próximo token" e "escrever um texto" é a repetição.

Preveja um token. Acrescente-o ao texto. Preveja de novo, agora com o texto maior. Repita.

> **Definição — geração autoregressiva.** Gerar texto repetindo a previsão do próximo token, realimentando cada previsão como entrada da seguinte.

É por isso que prever o próximo token é uma tarefa tão exigente. Para acertar o próximo token de "a capital da França é", o modelo precisa ter aprendido geografia. Para acertar o de "def soma(a, b): return", precisa ter aprendido Python. A tarefa é simples de enunciar e arbitrariamente difícil de cumprir bem.

### Exemplo numérico

Vamos usar o vocabulário que atravessa o material inteiro — o do corpus `"abacate"`, com cinco caracteres distintos:

```
vocabulário: {a:0, b:1, c:2, e:3, t:4}      vocabSize = 5
```

Suponha o contexto `"aba"`, e um modelo já treinado nesse corpus. Ele produz um número bruto por token do vocabulário:

> **Definição — logit.** O número bruto que o modelo produz para cada token. Pode ser negativo, pode ser grande, e **não** é uma probabilidade ainda.

```
token 0 ('a'):  logit = 1.0
token 1 ('b'):  logit = 0.5
token 2 ('c'):  logit = 3.0
token 3 ('e'):  logit = 0.2
token 4 ('t'):  logit = 0.8
```

Passando pelo softmax, que você constrói na Etapa 6, os logits viram probabilidades:

```
token 0 ('a'):  0.0974
token 1 ('b'):  0.0591
token 2 ('c'):  0.7199      ← o mais provável
token 3 ('e'):  0.0438
token 4 ('t'):  0.0798
                ------
soma:           1.0000
```

O modelo aposta 72% em `'c'`. E está certo: no corpus `"abacate"`, depois de `"aba"` vem exatamente `'c'`.

**Verificação independente.** Note que as cinco probabilidades somam exatamente 1. Isso não é coincidência — é uma garantia estrutural do softmax, e é o que permite ler esses números como probabilidades. A Etapa 6 demonstra por quê.

Repare também no que o modelo **não** faz: ele não escolhe. Ele devolve a distribuição inteira. Escolher um token a partir dela é uma decisão separada, e a Etapa 19 mostra que há várias estratégias possíveis.

---

## §2. O caminho de um texto pelo modelo

> **Guia visual.** O diagrama completo, com os formatos de tensor em cada estágio: [`gpt-architecture.html`](gpt-architecture.html).

Um texto atravessa seis estágios. Cada um é construído por uma etapa específica deste material.

**1. Texto vira números.** Cada caractere é trocado pelo seu índice no vocabulário. *(Etapa 7)*

**2. Números viram vetores.** Cada índice é trocado por uma linha de uma tabela treinável, e somado a um vetor que codifica a **posição** na sequência. *(Etapa 9)*

**3. Os vetores conversam entre si.** É a atenção: cada posição olha para as anteriores e incorpora informação delas. *(Etapas 11 e 12)*

**4. Cada vetor é processado individualmente.** Uma pequena rede densa refina cada posição. *(Etapa 13)*

Os estágios 3 e 4 formam um **bloco**, e o modelo empilha vários. *(Etapa 14)*

**5. Vetores viram logits.** Uma última camada linear projeta cada vetor de volta ao tamanho do vocabulário. *(Etapa 15)*

**6. Logits viram probabilidades.** Softmax. *(Etapa 6)*

### Exemplo numérico: seguindo os formatos

Nada torna isso concreto como acompanhar os formatos dos tensores. Tome um lote de 2 sequências, de 4 tokens cada, com embeddings de dimensão 8 e vocabulário de 5:

```
                                      formato          construído na
texto: ["abac", "acat"]               —                Etapa 7
  ↓ tokenizar
tokens                                [2, 4]           Etapa 7
  ↓ embedding de token + posição
vetores                               [2, 4, 8]        Etapa 9
  ↓ bloco 1  (atenção + MLP)
vetores                               [2, 4, 8]        Etapas 11-14
  ↓ bloco 2
vetores                               [2, 4, 8]        Etapas 11-14
  ↓ projeção final
logits                                [2, 4, 5]        Etapa 15
  ↓ softmax
probabilidades                        [2, 4, 5]        Etapa 6
```

Três observações que valem guardar.

**Os blocos não mudam o formato.** Entra `[2,4,8]`, sai `[2,4,8]`. É exatamente isso que permite empilhá-los à vontade — dois, doze ou noventa e seis, sem mudar mais nada.

**A última dimensão muda de significado.** Ela começa como `8` (a dimensão do embedding, um espaço interno do modelo) e termina como `5` (o tamanho do vocabulário, um número por token possível).

**Há uma previsão por posição, não uma por sequência.** A saída `[2, 4, 5]` traz 4 distribuições para cada uma das 2 sequências. Cada posição prevê o seu próprio próximo token — é por isso que uma janela de 4 tokens rende 4 exemplos de treino, como a Etapa 7 §4 detalha.

> **Confira você mesmo.** Se você dobrar o número de blocos de 2 para 4, qual formato da tabela acima muda?
>
> <details><summary>Resposta</summary>
>
> **Nenhum.** Cada bloco recebe `[2,4,8]` e devolve `[2,4,8]`. Acrescentar blocos aumenta a profundidade do modelo e a contagem de parâmetros, sem tocar em nenhum formato. É exatamente essa propriedade que permite ao GPT-3 usar 96 blocos com a mesma arquitetura de um modelo de 2.
> </details>

---

## §3. As 19 etapas, mapeadas

O projeto se divide em quatro marcos.

### Marco 1 — Fundação matemática (Etapas 1 a 6)

Nada aqui é específico de modelos de linguagem. É o equivalente a construir o NumPy antes de construir o modelo. No repositório, isso vive no módulo `scalagrad`.

| Etapa | Constrói | Por que é necessária |
|---|---|---|
| 1 · Tensor | array N-dimensional, com `shape` e `strides` | tudo no modelo é um tensor |
| 2 · Autograd | derivadas automáticas pelo grafo | sem isso não há treino |
| 3 · Operações | as 12 operações com forward e backward | a caixa de ferramentas matemática |
| 4 · Gradient check | verificador numérico de gradientes | erros de gradiente são silenciosos |
| 5 · Ativações | ReLU, Sigmoid, Tanh, GELU | sem não-linearidade, profundidade não existe |
| 6 · Softmax | normalização em probabilidades | usado na atenção e na perda |

### Marco 2 — Peças do modelo (Etapas 7 a 15)

Aqui começa o que é específico de GPT. No repositório, o módulo `gpt`.

| Etapa | Constrói | Onde entra |
|---|---|---|
| 7 · Tokenização | texto ↔ números, e os lotes de treino | a entrada do modelo |
| 8 · Linear | `y = xW + b`, com pesos treináveis | a peça de que todo o resto é feito |
| 9 · Embedding | tabelas de token e de posição | estágio 2 do §2 |
| 10 · LayerNorm | normaliza cada vetor | estabiliza o treino entre blocos |
| 11 · Atenção | cada posição olha para as anteriores | estágio 3 — o coração do modelo |
| 12 · Multi-Head | várias atenções em paralelo | estágio 3, ampliado |
| 13 · MLP | rede densa por posição | estágio 4 |
| 14 · Bloco | junta atenção, MLP e conexões residuais | a unidade que se empilha |
| 15 · Modelo GPT | monta tudo e projeta para os logits | estágios 5 e 6 |

### Marco 3 — Treinamento (Etapas 16 a 18)

| Etapa | Constrói | Por que |
|---|---|---|
| 16 · Cross-Entropy | a função de perda | é o número que o treino minimiza |
| 17 · AdamW | o otimizador | decide como aplicar os gradientes |
| 18 · Laço de treino | o ciclo completo | onde tudo roda junto |

### Marco 4 — Uso (Etapa 19)

| Etapa | Constrói | Por que |
|---|---|---|
| 19 · Geração | amostragem autoregressiva | é onde o modelo finalmente escreve texto |

### Exemplo numérico: o tamanho do que você vai construir

Para uma configuração pequena mas realista — vocabulário de 95 caracteres, embeddings de dimensão 64, contexto de 128 tokens, 4 blocos e MLP de 256:

```
tabela de tokens (Etapa 9)            6.080
tabela de posições (Etapa 9)          8.192
um bloco (Etapas 10-14)              49.984
  × 4 blocos                        199.936
norm final + saída (Etapa 15)         6.303
                                    -------
TOTAL                               220.511 parâmetros
```

Duzentos e vinte mil números, todos aprendidos a partir do texto. É pequeno o bastante para treinar num laptop, e grande o bastante para gerar texto reconhecível.

Para escala: o GPT-3 tem 175 bilhões de parâmetros — cerca de **794 mil vezes** mais. A arquitetura, porém, é a mesma. Não há nenhuma peça no GPT-3 que este material não construa.

> **Confira você mesmo.** Olhando a tabela acima, onde está a maior parte dos parâmetros — e o que isso sugere sobre onde o modelo guarda o que aprendeu?
>
> <details><summary>Resposta</summary>
>
> Nos blocos: 199.936 dos 220.511, ou seja, 91%. E dentro de cada bloco, a maior fatia é o MLP (33.088 de 49.984), não a atenção. Isso sugere que a atenção é o mecanismo que **move informação** entre posições, enquanto o MLP é onde a maior parte do conhecimento fica **armazenada**. É uma intuição que pesquisa recente sustenta.
> </details>

---

## §4. Convenções de notação

Estas convenções valem em todos os capítulos. Vale ler agora, mesmo sem entender ainda o que cada símbolo faz.

| Símbolo | Significa |
|---|---|
| `A`, `B`, `C` | tensores no forward |
| `dA` | `∂L/∂A` — gradiente da **perda** em relação a `A` |
| `dC` | o gradiente que chega de cima, vindo da operação seguinte |
| `∂C/∂A` | a derivada **local** da operação, isolada do grafo |
| `L` | a perda |
| `N(μ, σ²)` | distribuição normal de média `μ` e **variância** `σ²` |

Quatro armadilhas de notação que este bloco resolve de uma vez.

**`dA` não é "a derivada de `A`".** É a derivada de **outra coisa** — a perda — em relação a `A`. Sempre que vir um `d` colado num nome, leia "quanto a perda responde a isto".

**`L` é um único número.** Não é um vetor nem uma matriz. É uma medida escalar de quão errado o modelo está, e é o que o treino minimiza. Que ela seja um número só é o que permite fazer uma única pergunta sobre cada parâmetro.

**O gradiente tem sempre o formato do tensor a que se refere.** Se `A` é uma matriz 3×4, `dA` também é 3×4. Nunca o formato da saída da operação.

**O segundo parâmetro de `N` é a variância, não o desvio padrão.** A literatura de aprendizado de máquina costuma escrever `N(0, 0.02)` quando quer dizer desvio padrão `0,02`. Este material sempre escreve `N(0, 0.02²)`, com o expoente explícito. A Etapa 1 §4 define a distribuição.

**Exemplo numérico.** Uma camada recebe `x` de formato `[10, 4]` e tem pesos `W` de formato `[4, 3]`:

```
y  = xW           formato [10, 3]
dy                formato [10, 3]      ← igual a y
dx                formato [10, 4]      ← igual a x
dW                formato [4, 3]       ← igual a W
```

Repare que `dW` **não** tem a dimensão 10 do lote. Os pesos são compartilhados por todos os exemplos, e o lote foi somado para dentro do gradiente. A Etapa 8 §3 detalha isso.

> **Confira você mesmo.** Alguém escreve "o gradiente `dx` tem o formato da saída, porque é de lá que ele vem". Onde está o erro?
>
> <details><summary>Resposta</summary>
>
> O gradiente **chega** vindo da saída, mas ele *é sobre* a entrada. `dx` responde "quanto a perda muda se eu mexer em cada posição de `x`?" — e existe uma resposta para cada posição de `x`, não para cada posição de `y`. Por isso `dx` tem o formato de `x`. A confusão é comum porque o gradiente viaja no sentido contrário ao do forward.
> </details>

---

## §5. Glossário

Todos os termos definidos ao longo do material, com o capítulo onde cada um aparece pela primeira vez.

### Fundamentos

| Termo | Significado | Onde |
|---|---|---|
| tensor | array de N dimensões | 1 §1 |
| rank | número de dimensões | 1 §1 |
| shape | tamanho de cada dimensão | 1 §1 |
| stride | quantas posições andar na memória ao incrementar uma dimensão | 1 §2 |
| row-major | última dimensão percorrida primeiro | 1 §2 |
| contíguo | os strides são os canônicos do formato | 1 §3 |
| view | compartilha os dados, muda os metadados | 1 §3 |
| distribuição normal (`N(μ, σ²)`) | curva em sino de média `μ` e variância `σ²` | 1 §4 |
| desvio padrão (`σ`) | raiz da variância — o espalhamento da curva | 1 §4 |

### Treinamento

| Termo | Significado | Onde |
|---|---|---|
| parâmetro (ou peso) | número ajustável dentro do modelo | 2 §1 |
| parâmetro treinável | parâmetro que persiste entre passadas e é atualizado | 8 |
| perda (`L`) | um único número que mede o erro do modelo | 2 §1 |
| taxa de aprendizado (`lr`) | tamanho do passo de ajuste | 2 §1 |
| autograd | aplicação automática da regra da cadeia | 2 §1 |
| DAG | grafo acíclico dirigido — a estrutura do autograd | 2 §3 |
| folha / raiz | tensores sem predecessores / o tensor final | 2 §3 |
| ordenação topológica | ordem em que todo nó vem depois de suas dependências | 2 §4 |
| derivada local | derivada da operação isolada do grafo | 3 |
| `noGrad` | bloco que pula a construção do grafo | 2 §7 |
| função de perda | recebe saída e resposta certa, devolve **um** número | 16 §1 |
| cross-entropy | `L = −log(p_alvo)`; a perda de classificação | 16 §2 |
| log-sum-exp | `log Σ eᶻ = max + log Σ e^(z−max)`, a identidade que evita estouro | 16 §3 |
| perplexidade | `exp(L)`; vale `V` num modelo aleatório | 16 §6 |
| otimizador | a regra que transforma gradientes em mudanças nos parâmetros | 17 §1 |
| SGD | *stochastic gradient descent*: `p ← p − lr·g` | 17 §1 |
| média móvel exponencial | média onde o valor novo entra com peso `1−β` e o passado encolhe por `β` | 17 §3 |
| primeiro momento (`m`) | média móvel do gradiente — guarda a direção | 17 §3 |
| segundo momento (`v`) | média móvel de `g²` — guarda a escala | 17 §4 |
| passo adaptativo | `m/√v`, adimensional: o `lr` decide o tamanho, o gradiente só o sentido | 17 §4 |
| correção de viés | dividir por `1−βᵗ` para compensar o arranque em zero | 17 §5 |
| weight decay | pressão constante que puxa todo parâmetro para zero | 17 §6 |
| regularização L2 | weight decay somado ao gradiente; dentro do Adam, decai errado | 17 §6 |
| hiperparâmetro | número escolhido antes do treino, que o gradiente não ajusta | 17 §7 |
| passo de treino | uma passada completa: lote entra, todo parâmetro se move uma vez | 18 §1 |
| norma global do gradiente | raiz da soma dos quadrados de **todos** os gradientes | 18 §2 |
| gradient clipping | reescalar todos os gradientes quando a norma global passa do teto | 18 §2 |
| schedule | a função que devolve a taxa de aprendizado de cada passo | 18 §3 |
| aquecimento (*warmup*) | rampa linear da taxa nos primeiros passos | 18 §3 |
| decaimento em cosseno | meia volta de cosseno de `lrMax` a `lrMin` | 18 §3 |
| conjunto de validação | fatia do corpus nunca usada para atualizar parâmetros | 18 §5 |
| overfitting | a perda de treino cai e a de validação sobe | 18 §5 |
| checkpoint | pesos, momentos e `t` salvos em disco, para retomar o treino | 18 §6 |

### Operações

| Termo | Significado | Onde |
|---|---|---|
| elemento a elemento | cada saída depende só da mesma posição da entrada | 3 §1 |
| redução | operação que colapsa vários valores num só | 3 §3 |
| fatia | elementos que compartilham todos os índices menos um | 3 §3 |
| broadcasting | operar formatos diferentes sem copiar dados | 3 §4 |
| `unbroadcast` | desfaz o broadcasting no backward, somando | 3 §4 |
| Jacobiano | matriz de todas as derivadas parciais | 6 §3 |

### Verificação

| Termo | Significado | Onde |
|---|---|---|
| diferença finita | aproximar derivada avaliando pontos próximos | 4 §1 |
| gradiente analítico | o que o `backward()` calculou | 4 §2 |
| gradiente numérico | a aproximação independente | 4 §2 |
| erro de truncamento | vem do `ε` finito; diminui com `ε` menor | 4 §1 |
| erro de arredondamento | vem da precisão do `Double`; aumenta com `ε` menor | 4 §1 |

### Rede neural

| Termo | Significado | Onde |
|---|---|---|
| camada linear | `f(x) = Wx + b` | 5 §1 |
| transformação afim | linear mais deslocamento | 8 §1 |
| neurônio | uma coluna de `W`; produz uma saída | 5 §2, 8 §1 |
| neurônio morto | neurônio preso na região de gradiente zero da ReLU | 5 §2 |
| problema da simetria | neurônios idênticos permanecem idênticos | 8 §4 |
| princípio da superposição | teste formal de linearidade | 5 §1 |
| inicialização de Kaiming | `Var(w) = 2/inputDim` | 8 §5 |
| ReLU, Sigmoid, Tanh, GELU | as quatro ativações | 5 |
| ativação | valor de saída de uma camada, para uma entrada específica | 10 §1 |
| normalizar | subtrair a média e dividir pelo desvio padrão | 10 §1 |
| Layer Normalization | normalizar cada vetor da última dimensão, isoladamente | 10 §2 |
| ganho e deslocamento (`γ`, `β`) | parâmetros treináveis que reescalam e reposicionam `x̂` | 10 §4 |
| BatchNorm | a alternativa que normaliza ao longo do lote | 10 §7 |
| pre-LN e post-LN | normalizar antes da subcamada ou depois da soma residual | 10 §8 |
| MLP (*feed-forward network*) | duas camadas lineares com uma GELU no meio | 13 |
| camada posição a posição | a mesma função, com os mesmos pesos, em cada vetor da última dimensão | 13 §1 |
| `dFF` | a largura da camada escondida do MLP, igual a `4 · dModel` | 13 §2 |
| fator de expansão | a razão `dFF / dModel`; vale 4 no transformer original e no GPT-2 | 13 §2 |
| memória associativa | guarda pares chave-valor e devolve o valor da chave mais parecida | 13 §3 |
| subcamada | cada metade do bloco: a atenção e o MLP | 14 §1 |
| conexão residual (*skip connection*) | somar a entrada da subcamada à saída dela, `y = x + f(x)` | 14 §2 |
| fluxo residual (*residual stream*) | o vetor que atravessa a pilha; subcamadas leem normalizado e escrevem somando | 14 §5 |
| bloco transformer | `X + MHA(LN(X))` seguido de `H + MLP(LN(H))` | 14 §1 |
| profundidade (`nLayers`) | quantos blocos a pilha tem | 15 §3 |
| cabeça de linguagem | a projeção final `[dModel, vocabSize]`, que produz os logits | 15 §1 |
| `ln_f` | o LayerNorm depois do último bloco, exigido pelo pre-LN | 15 §4 |
| equivariância a permutação | reordenar a entrada só reordena a saída | 15 §2 |
| pesos amarrados (*weight tying*) | a mesma matriz na entrada e, transposta, na saída | 15 §5 |

### Modelo de linguagem

| Termo | Significado | Onde |
|---|---|---|
| token | a menor unidade de texto que o modelo enxerga | 7 §1 |
| vocabulário | conjunto dos tokens distintos; tamanho `vocabSize` | 7 §1 |
| corpus | o texto de treino | 7 §1 |
| janela de contexto | quantos tokens o modelo vê de uma vez | 7 §4 |
| lote (*batch*) | conjunto de exemplos processados de uma vez | 7 §5 |
| logit | número bruto por token, antes do softmax | 6 §1, 0 §1 |
| softmax | converte logits em probabilidades que somam 1 | 6 §1 |
| embedding | vetor denso e treinável que representa um token | 9 §1 |
| vetor one-hot | vetor com `1` numa posição e `0` no resto | 9 §1 |
| gradiente esparso | só as linhas usadas recebem gradiente | 9 §3 |
| geração autoregressiva | prever, realimentar, repetir | 0 §1 |
| atenção | média ponderada das posições, com pesos vindos do conteúdo | 11 §1 |
| auto-atenção | atenção em que a sequência comparada e a misturada são a mesma | 11 §1 |
| query, key e value | o que um token procura, o que ele oferece, e o que ele entrega | 11 §2 |
| score de atenção | o alinhamento entre a query de `i` e a key de `j`, antes do softmax | 11 §3 |
| peso de atenção | o score já normalizado; cada linha soma 1 | 11 §6 |
| máscara causal | `-inf` nas posições futuras, somado **antes** do softmax | 11 §5 |
| cabeça (*head*) | um conjunto completo de `W_Q`, `W_K` e `W_V` | 11 §9 |
| `dHead` | o tamanho do subespaço de cada cabeça, igual a `dModel / H` | 12 §2 |
| concatenação de cabeças | empilhar as saídas das cabeças num vetor de tamanho `dModel` | 12 §5 |
| projeção de saída (`W_O`) | a matriz que mistura as cabeças entre si, depois da concatenação | 12 §6 |
| prompt | a sequência inicial de tokens que condiciona a geração | 19 §1 |
| janela deslizante | manter só os últimos `contextLength` tokens, cortando pela esquerda | 19 §2 |
| greedy decoding | escolher sempre o token de maior logit | 19 §3 |
| temperatura (`T`) | divisor aplicado aos logits antes do softmax | 19 §4 |
| top-k | manter só os `k` maiores logits e renormalizar entre eles | 19 §5 |
| CDF | soma acumulada das probabilidades, usada para amostrar | 19 §6 |
| KV cache | reaproveitar chaves e valores entre passos; não feito neste projeto | 19 §1 |

---

## §6. Por onde começar

Leia os capítulos em ordem. Cada um assume o anterior, e as dependências são reais — pular a Etapa 2 torna a 3 incompreensível.

Três conselhos de quem já percorreu o caminho.

**Refaça os exemplos numéricos.** Não todos, mas alguns por capítulo. Especialmente os de backward. Ler a derivação e reproduzi-la são experiências muito diferentes.

**Leia os blocos de Armadilha com atenção.** Eles não são curiosidades. Cada um marca um lugar onde a intuição falha de forma previsível, e a maioria descreve bugs **silenciosos** — que não travam, não avisam, e só produzem números errados.

**Não pule a Etapa 4.** Ela é curta e parece um desvio do caminho principal. É o oposto: sem o verificador de gradientes, todo backward escrito depois dela é um ato de fé.

Se em algum momento algo não fizer sentido, a causa mais provável é um pré-requisito não firmado, e não falta de capacidade. Volte um capítulo. O material foi escrito para ser lido em ordem, e cada peça só encaixa depois da anterior.

Comece pela [Etapa 1 — Tensor](../01-tensor/01-tensor.md).
