# Etapa 14 — Bloco Transformer

## Antes de começar

### O que você vai construir

A unidade que se repete `N` vezes num GPT. Quatro passos, dois deles somas.

| Componente | Formato | O que faz |
|---|---|---|
| `LayerNorm` 1 | `[dModel]` × 2 | normaliza o que entra na atenção |
| `MultiHeadAttention` | 6 tensores | mistura informação entre posições |
| soma residual 1 | — | devolve a entrada original ao resultado |
| `LayerNorm` 2 | `[dModel]` × 2 | normaliza o que entra no MLP |
| `MLP` | 4 tensores | processa cada posição isoladamente |
| soma residual 2 | — | devolve de novo |

Entrada `[B, T, dModel]`, saída `[B, T, dModel]`. Catorze tensores de parâmetro, e nenhuma operação nova.

### O que você precisa saber antes

**Da Etapa 10, em especial §8:** o LayerNorm, e a distinção entre pre-LN e post-LN. Esta etapa decide entre os dois e mostra a conta por trás da escolha.

**Da Etapa 12:** a `MultiHeadAttention` inteira, incluindo a máscara causal. O bloco herda a causalidade dela.

**Da Etapa 13:** o MLP, e a §6 sobre variância. Os números daquela seção reaparecem aqui, para explicar de onde vem o fator `1/√(2·nLayers)`.

**Da Etapa 3 §1:** a soma elemento a elemento e o seu backward. É a única operação nova do bloco, e o backward dela é o assunto da §3.

### Onde esta etapa se encaixa

Todas as peças existem. A atenção mistura posições, o MLP transforma cada uma, o LayerNorm mantém a escala sob controle.

Falta montá-las de um jeito que aguente ser empilhado dezenas de vezes.

Esse "aguente" não é força de expressão. Antes de 2015, redes muito profundas simplesmente não treinavam: o gradiente encolhia a cada camada e chegava ao início como ruído. A solução foi de uma simplicidade desconfortável — **somar a entrada de volta à saída**. Só isso.

Esta etapa é sobre essa soma. Ela não tem parâmetro, não tem hiperparâmetro, e é o que separa um modelo de 3 camadas de um de 96.

### Objetivos de aprendizagem

Ao terminar, você deve conseguir:

1. Escrever o bloco em quatro linhas e dizer o que cada uma preserva.
2. Derivar o backward de `y = x + f(x)` e apontar o termo que impede o gradiente de sumir.
3. Justificar pre-LN contra post-LN com uma conta, não com "é mais estável".
4. Explicar por que a variância do fluxo residual cresce com a profundidade, e de onde sai o `1/√(2·nLayers)`.
5. Escrever o teste que reprova um bloco em que falta uma das duas somas.

### Convenções de notação

| Símbolo | Significa |
|---|---|
| `X` | a entrada do bloco — `[B, T, dModel]` |
| `S₁` | a saída da atenção, `MHA(LN₁(X))` |
| `H` | o estado no meio do bloco, `X + S₁` |
| `S₂` | a saída do MLP, `MLP(LN₂(H))` |
| `Y` | a saída do bloco, `H + S₂` |
| `L` | o número de blocos empilhados — `nLayers` no código |
| `f` | uma subcamada genérica, quando o argumento vale para as duas |

A convenção `dX ≡ ∂L/∂X` do overview continua valendo. Cuidado com a colisão: `L` aqui é o número de blocos, e a perda continua sendo escrita por extenso quando aparecer.

> **Guia visual.** O bloco, o backward na bifurcação, pre-LN contra post-LN e o barramento residual: [`block.html`](block.html). **Exercícios (15 questões):** [`exercises.html`](exercises.html).

---

## §1. O bloco, em quatro passos

```
S₁ = MHA(LN₁(X))
H  = X + S₁
S₂ = MLP(LN₂(H))
Y  = H + S₂
```

É isso. Não há uma quinta operação escondida.

Repare no que **não** aparece nessas quatro linhas. Nenhuma matriz nova, nenhum hiperparâmetro novo, nenhuma dimensão nova. O bloco não acrescenta capacidade ao que a Etapa 12 e a Etapa 13 já construíram — ele acrescenta uma **forma de compor** as duas.

> **Definição — subcamada.** Cada uma das duas metades do bloco: a atenção e o MLP. As duas têm a mesma assinatura, `[B, T, dModel]` entra e sai, e as duas entram no bloco pelo mesmo padrão `x + Sub(LN(x))`.

Essa simetria não é decorativa. Ela é o que permite escrever o bloco como uma linha repetida duas vezes, e é o que a Etapa 15 vai repetir `L` vezes.

**O exemplo numérico.** Os formatos, com `B = 2`, `T = 5`, `dModel = 4` e `nHeads = 2`:

| passo | operação | formato de saída |
|---|---|---|
| — | `X` | `[2, 5, 4]` |
| 1 | `LN₁(X)` | `[2, 5, 4]` |
| 2 | `MHA(...)` = `S₁` | `[2, 5, 4]` |
| 3 | `X + S₁` = `H` | `[2, 5, 4]` |
| 4 | `LN₂(H)` | `[2, 5, 4]` |
| 5 | `MLP(...)` = `S₂` | `[2, 5, 4]` |
| 6 | `H + S₂` = `Y` | `[2, 5, 4]` |

Seis operações, um formato só. Essa coluna constante é a propriedade que a §5 vai chamar de barramento, e é o motivo de o bloco poder ser aplicado a si mesmo.

Contando tensores de parâmetro: 2 do `LN₁`, 6 da atenção, 2 do `LN₂` e 4 do MLP, num total de **14**. As duas somas não têm nenhum.

---

## §2. A conexão residual: somar a entrada de volta

> **Definição — conexão residual.** A saída de uma subcamada é somada à sua própria entrada: `y = x + f(x)`, em vez de `y = f(x)`. Também chamada de *skip connection*, porque cria um caminho que pula a subcamada.

A pergunta natural é por que somar, e não concatenar, ou multiplicar, ou simplesmente substituir.

**Substituir é o que quebra.** Com `y = f(x)`, a informação que chegou só sobrevive se `f` escolher preservá-la. Cada camada tem que reaprender a repassar o que já estava certo, e o gradiente da §3 mostra o preço disso.

**Concatenar mudaria o formato.** `[B, T, dModel]` viraria `[B, T, 2·dModel]`, e o bloco deixaria de poder ser empilhado. Para voltar ao tamanho original seria preciso uma matriz nova — parâmetros a mais, para fazer o que a soma faz de graça.

**Somar preserva as duas coisas.** O formato não muda, e `x` chega ao fim intacto. Isso muda o que a subcamada precisa aprender: ela não produz mais a resposta, e sim uma **correção** sobre o que já estava lá.

É por isso que o nome é residual. `f` aprende o resíduo — a diferença entre o que chegou e o que deveria sair.

**O exemplo numérico.** Use `x = [2, 3, 6, 9]`, o mesmo vetor da Etapa 10 §2, e suponha que a subcamada devolva `[0.4, -0.1, 0.2, -0.3]`:

```
sem residual:  y = f(x)      = [ 0.4, -0.1,  0.2, -0.3]
com residual:  y = x + f(x)  = [ 2.4,  2.9,  6.2,  8.7]
```

Na primeira linha, tudo o que `x` continha foi descartado. Na segunda, a subcamada moveu cada coordenada em menos de meia unidade, e o vetor original ainda está lá. Uma pilha de 12 blocos do primeiro tipo passa o sinal por 12 filtros seguidos; do segundo tipo, acumula 12 ajustes sobre o mesmo vetor.

> **Confira você mesmo.** Se a subcamada devolvesse exatamente zero, o que o bloco faria?
>
> <details><summary>Resposta</summary>
>
> Nada: `y = x + 0 = x`, a identidade. Isso é uma propriedade valiosa, e não um caso degenerado. Significa que um bloco pode "se desligar" aprendendo a produzir saída pequena, e que uma pilha profunda recém-inicializada está perto de ser a identidade em vez de ser um embaralhador. Sem o residual, a única forma de uma camada não atrapalhar é aprender a copiar a entrada, o que é bem mais difícil do que aprender a devolver zero.
> </details>

---

## §3. O backward: a rodovia que não some

A soma tem o backward mais simples da Etapa 3: ela **repassa o gradiente inteiro para os dois lados**. Aplicado a `y = x + f(x)`, isso dá:

```
dX = dY  +  dY · ∂f/∂X
     ↑          ↑
  caminho    caminho pela
  direto     subcamada
```

O primeiro termo é o ponto da etapa inteira. Ele não é multiplicado por Jacobiano nenhum. Seja o que for que aconteça dentro de `f`, uma cópia intacta de `dY` chega em `dX`.

**O exemplo numérico.** Uma subcamada de brinquedo, `f(x) = x·W`, com dois valores de `dY` bem separados:

```
W = [[0.5, -0.2]        x = [1, 2]        dY = [1, 10]
     [0.1,  0.3]]
```

O forward:

```
f(x) = [1·0.5 + 2·0.1,  1·(-0.2) + 2·0.3] = [0.7, 0.4]
y    = x + f(x)                            = [1.7, 2.4]
```

O backward, pelas duas rotas:

```
dY · Wᵀ = [1·0.5 + 10·(-0.2),  1·0.1 + 10·0.3] = [-1.5,  3.1]
dX      = dY + dY · Wᵀ = [1, 10] + [-1.5, 3.1]  = [-0.5, 13.1]
```

**A verificação independente.** Diferenças finitas centrais em `x`, com `ε = 1e-6`, sobre `Σ y ⊙ dY`, dão `[-0.5, 13.1]` — as duas rotas concordam.

Compare com o que aconteceria sem o residual: `dX` seria `[-1.5, 3.1]`. Na coordenada 1, o gradiente cairia de `13.1` para `3.1`; na coordenada 0, ele até troca de sinal. A subcamada encolheu o sinal, e é o termo direto que segura.

**Agora empilhe.** Com `N` blocos, o gradiente que chega ao primeiro é um produto de `N` fatores:

```
sem residual:  ∏ f'ₗ
com residual:  ∏ (1 + f'ₗ)
```

O primeiro produto morre. Se cada `f'ₗ` vale `0.8`, depois de 24 camadas sobra `0.8²⁴ = 0.0047` — menos de meio por cento. Se vale `0.25`, sobra `3.6e-15`.

O segundo não morre, e a razão é estrutural. Abrindo `∏(1 + f'ₗ)` você obtém `2²⁴` termos, um deles sendo exatamente `1` — o caminho que pula todas as subcamadas.

**Medido**, com 20 mil sorteios de `f'ₗ ~ N(0, 0.25²)` e 24 camadas:

| | mediana de \|gradiente\| | p10 | p90 | fração abaixo de `1e-6` |
|---|---|---|---|---|
| sem residual | `1.2e-21` | `7.3e-25` | `7.9e-19` | **100%** |
| com residual | `0.453` | `0.075` | `2.384` | **0%** |

Vinte e um zeros de diferença na mediana. E não é questão de sorte: em 20 mil execuções, o caso sem residual **nunca** ficou acima de `1e-6`, e o caso com residual **nunca** ficou abaixo.

> **Guia visual.** A queda do gradiente com a profundidade, em escala logarítmica: [`block.html`](block.html), FIG 3.

> **Confira você mesmo.** Se o termo `dY` chega intacto ao início da rede, por que ainda se fala em gradiente instável em transformers profundos?
>
> <details><summary>Resposta</summary>
>
> Porque a garantia é sobre o caminho residual, não sobre tudo. O gradiente de um **parâmetro** dentro da subcamada ainda passa pelo Jacobiano daquela subcamada, e pode ser grande ou pequeno. Além disso, a soma dos `2ᴺ` termos pode crescer: a tabela acima tem p90 em `2.384`, ou seja, casos em que o gradiente **aumentou**. O residual troca o desaparecimento garantido por um comportamento controlado — não por uma garantia de escala 1.
> </details>

---

## §4. Pre-LN: onde a normalização entra

A Etapa 10 §8 já definiu as duas posições. Vale reescrevê-las lado a lado:

```
pre-LN:   Y = X + Sub(LN(X))
post-LN:  Y = LN(X + Sub(X))
```

O paper original do transformer usa post-LN. O GPT-2 mudou para pre-LN, e é o que este projeto constrói.

A conta que decide é a da §3. Em pre-LN, o caminho direto é `X` puro: `∂Y/∂X` contém a identidade exata. Em post-LN, **todo** caminho atravessa uma normalização, inclusive o direto. O `LN` tem Jacobiano próprio, com fator de escala `1/σ`. Ele entra no produto de todas as camadas — exatamente o produto que a §3 mostrou morrer.

Dito de outro jeito: post-LN não tem rodovia. Tem uma estrada com um pedágio por bloco.

**O exemplo numérico.** Três blocos, começando de `x = [2, 3, 6, 9]`, com uma subcamada que devolve 10% do que recebe:

| depois de | pre-LN, desvio padrão | post-LN, desvio padrão |
|---|---|---|
| entrada | `2.7386` | `2.7386` |
| bloco 1 | `2.8386` | `1.0000` |
| bloco 2 | `2.9386` | `1.0000` |
| bloco 3 | `3.0386` | `1.0000` |

A coluna do post-LN é o ponto. Ela não é aproximadamente 1, é **exatamente** 1, nos três blocos — porque a última coisa que cada bloco faz é normalizar. A escala com que o sinal chegou foi descartada, e o bloco seguinte não tem como saber se o que veio era grande ou pequeno.

Na coluna do pre-LN, o desvio cresce `0.1` por bloco. Isso também é informação: o sinal está sendo acumulado, e é o assunto da §5.

O preço prático da diferença é conhecido. Post-LN treina, mas precisa de aquecimento cuidadoso da taxa de aprendizado nos primeiros passos, senão diverge. Pre-LN dispensa esse cuidado.

> **Confira você mesmo.** Em pre-LN, o `X` que sai do bloco nunca passou por normalização nenhuma. Isso não é um problema no fim da pilha?
>
> <details><summary>Resposta</summary>
>
> É, e a solução tem nome: um LayerNorm **final**, depois do último bloco e antes da projeção para o vocabulário. Todo GPT pre-LN tem esse `ln_f`, e ele existe justamente porque a saída da pilha acumulou escala sem nunca ser renormalizada. Você vai construí-lo na Etapa 15.
> </details>

---

## §5. O barramento residual, e de onde vem o `1/√(2·L)`

> **Definição — fluxo residual (*residual stream*).** O vetor de `dModel` números que atravessa a pilha inteira, de bloco em bloco. Cada subcamada **lê** dele uma versão normalizada, e **escreve** de volta somando um incremento. Ele nunca é substituído.

Essa leitura reorganiza o modelo inteiro. Não existe um "fluxo principal" que passa por dentro da atenção; existe um barramento, e as subcamadas são dispositivos pendurados nele. Uma pilha de 12 blocos tem 24 dispositivos, cada um lendo e escrevendo no mesmo lugar.

Isso tem uma consequência quantitativa direta. Se cada subcamada escreve um incremento, e os incrementos são aproximadamente independentes entre si, as **variâncias somam**:

```
Var(saída da pilha) ≈ Var(entrada) + Σ Var(incremento de cada subcamada)
```

A entrada de cada subcamada é `LN(x)`, que tem variância 1 por construção. Então a variância do incremento é uma propriedade da subcamada, e a Etapa 13 §6 já mediu a do MLP: `1.84`. Tomando a da atenção como `1`, para uma conta redonda:

| `L` blocos | Var da saída, sem escala | norma cresce |
|---|---|---|
| 2 | `5` | `×2.24` |
| 6 | `13` | `×3.61` |
| 12 | `25` | `×5.00` |
| 24 | `49` | `×7.00` |

A tabela usa incrementos de variância 1 nas duas subcamadas, para deixar a fórmula visível: `1 + 2L`. Com o `1.84` medido do MLP, `L = 12` dá `35.13`, ou norma `×5.93`.

O crescimento é o esperado, e não é um bug — é o que acontece quando 24 coisas escrevem no mesmo lugar. Mas ele significa que a saída da pilha nasce numa escala que depende da profundidade, e que os primeiros blocos operam numa escala diferente dos últimos.

**A correção, derivada.** Multiplique a saída de cada subcamada por `1/√(2L)`. A variância de cada incremento é multiplicada pelo quadrado disso, ou seja, por `1/(2L)`. Somando os `2L` incrementos:

```
Var(saída) = 1  +  2L · (1 / 2L)  =  2
```

Exatamente `2`, **para qualquer profundidade**. Norma `×1.41`, com `L = 2` ou com `L = 96`.

É daí que sai o fator `1/√(2·nLayers)` que o GPT-2 aplica às camadas que escrevem no fluxo residual — a `W_O` da atenção e a segunda `Linear` do MLP. Ele não é um truque empírico: é a escala que faz a soma telescopar para uma constante.

**A verificação independente.** Refazendo a conta com os números medidos em vez dos redondos — incremento `1.0` da atenção e `1.84429` do MLP, `L = 12`:

```
sem escala:   Var = 1 + 12·(1 + 1.84429) = 35.13     norma ×5.93
com 1/√(2L):  Var = 1 + 12·(1 + 1.84429)/24 = 2.42   norma ×1.56
```

O caso escalado não dá exatamente `2` porque os incrementos reais não têm variância 1 cada. A dependência com a profundidade, porém, sumiu — que era o objetivo.

Este projeto ainda **não** aplica esse fator. A pendência está aberta desde a Etapa 12, e agora tem endereço e fórmula: ela se resolve na Etapa 15, que é onde `nLayers` finalmente existe.

> **Confira você mesmo.** Por que o fator é `1/√(2L)` e não `1/(2L)`?
>
> <details><summary>Resposta</summary>
>
> Porque o que soma é a **variância**, não o desvio padrão. Multiplicar um vetor por `c` multiplica a variância dele por `c²`. Para que `2L` incrementos somem `1` no total, cada um precisa contribuir com `1/(2L)` de variância, e portanto ser multiplicado por `c = 1/√(2L)`. Usar `1/(2L)` daria variância total `1 + 2L/(2L)² = 1 + 1/(2L)`, que tende a 1 — os blocos praticamente não escreveriam nada.
> </details>

---

## §6. Contagem de parâmetros e o custo do bloco

Somando as três peças, com a contagem da Etapa 12 §9 para a atenção e a da Etapa 13 §7 para o MLP:

```
atenção:      4·dModel² + 2·dModel
MLP:          8·dModel² + 5·dModel
2 LayerNorm:              4·dModel
                    ------------------
bloco:       12·dModel² + 11·dModel
```

| `dModel` | atenção | MLP | 2 LN | bloco |
|---|---|---|---|---|
| 4 | 72 | 148 | 16 | 236 |
| 64 | 16.512 | 33.088 | 256 | 49.856 |
| 768 | 2.360.832 | 4.722.432 | 3.072 | 7.086.336 |

Duas proporções valem guardar. O MLP é **dois terços** do bloco, e a atenção o terço restante. Os dois LayerNorm juntos são `0.04%` — quatro décimos de um milésimo. A camada que mais aparece em discussão de estabilidade é a que menos pesa na conta.

**A verificação independente.** Doze blocos com `dModel = 768` dão `85.036.032` parâmetros. O GPT-2 small tem `85.054.464` na pilha. A diferença de `18.432` é a dos vieses `b_K` e `b_V`, que a Etapa 12 §8 provou serem mortos. São `1.536` por bloco, em doze blocos. A fórmula fecha com o modelo real.

---

## §7. Implementação e como testar

**A assinatura.** `TransformerBlock(dModel: Int, nHeads: Int, expansion: Int = 4)`, com quatro campos privados: `ln1`, `attention`, `ln2` e `mlp`. O `parameters` concatena os quatro, nessa ordem, e tem 14 entradas.

**Exponha as subcamadas.** `val attention`, `val mlp`, `val ln1`, `val ln2` públicos. Isso não é conveniência. É o que torna possível o teste central desta etapa: compor a referência a partir das peças já testadas, em vez de reimplementar atenção e MLP em Scala puro. O PyTorch faz o mesmo — `block.attn`, `block.mlp`.

**As validações.** `x.rank == 3` e `x.shape.last == dModel`, com `require`. A `MultiHeadAttention` já valida as duas, então há duplicação; ela se paga porque a mensagem passa a nomear a camada que o chamador de fato invocou. É a mesma decisão que a Etapa 13 tomou ao contrário, e o critério é o mesmo: onde o erro fica mais fácil de entender.

**O `parameters` posicional acabou.** Com 14 entradas, índices nomeados no topo da spec deixam de ser sustentáveis. A `Linear` ganhou `weights` e `bias` na Etapa 13, e o bloco deve seguir a mesma linha. O teste alcança `block.attention.parameters` ou `block.mlp.parameters`, em vez de contar posições numa lista achatada.

Oito frentes cobrem o bloco, que viraram 16 testes na suíte final.

**Formato preservado.** `[B, T, dModel]` entra e sai, com `B`, `T`, `dModel` e `nHeads` distintos entre si.

**Composição contra as peças.** Calcule `h = x + attention.forward(ln1.forward(x))` e depois `h + mlp.forward(ln2.forward(h))`, usando as subcamadas expostas, e compare posição a posição. Este é o teste que carrega a etapa.

**As duas somas existem.** Duas referências erradas, uma sem cada soma. As duas têm que diferir do bloco.

**Pre-LN, não post-LN.** Uma referência com `LN(x + Sub(x))` tem que diferir. Sem esse teste, trocar a posição da normalização passa: o formato é o mesmo, e o gradient check aprova.

**A ordem: atenção antes do MLP.** Uma referência com as duas subcamadas trocadas tem que diferir.

**Causalidade ponta a ponta.** Mude o último token da sequência e exija que todas as saídas anteriores fiquem idênticas. O bloco herda a máscara da Etapa 12, mas herança se testa: basta um `reshape` errado no meio para a propriedade sumir.

**Empilhável.** Aplique o bloco à própria saída, duas vezes, e confira que nada quebra. É a propriedade que a Etapa 15 depende.

**`parameters` e gradientes.** Catorze tensores, todos com `requiresGradient`, sem repetição — `toSet.size == 14` pega uma subcamada compartilhada por engano. Some os elementos e compare com `12·dModel² + 11·dModel`; essa soma é o único teste que separa um `LayerNorm` de verdade de um `Linear(dModel, dModel)` posto no lugar dele. Gradient check em `x`, nos 14 parâmetros e com entrada não contígua.

> **Armadilha.** Um campo derivado declarado depois de quem o usa.
>
> Aconteceu neste projeto, na Etapa 13. O `dFF` foi escrito abaixo dos dois `Linear` que o consomem, e o corpo da classe inicializa de cima para baixo — então ele valia `0` na construção, e as matrizes nasciam com zero coluna. O compilador não emite aviso.
>
> Por que é perigoso: sem uma validação de dimensão, o `forward` roda e devolve um tensor vazio. Nada estoura. O bloco desta etapa tem quatro campos que dependem dos argumentos do construtor, então a superfície para o mesmo erro é maior.
>
> **Lição geral:** num corpo de classe, ordem de declaração é ordem de execução. E dimensão não positiva não pode ser aceita em silêncio — o `require` que a `Linear` ganhou na Etapa 13 é o que transforma esse erro em falha alta.

> **Armadilha.** O gradient check aprovando um forward errado.
>
> Aconteceu neste projeto, na Etapa 12. O softmax normalizava o eixo das *queries* em vez do das *keys*. O formato não mudava, não havia `NaN`, a causalidade continuava valendo — e o gradient check passava, porque o backward estava coerente com o forward escrito.
>
> Por que é perigoso: esta etapa é composição pura, e o instinto é confiar no gradient check justamente porque nada de novo foi derivado. Mas trocar pre-LN por post-LN, inverter a ordem das subcamadas ou esquecer uma das somas produz, em todos os casos, um forward perfeitamente derivável.
>
> **Lição geral:** o gradient check confere coerência, nunca intenção. Num bloco que só compõe peças testadas, os únicos testes que valem alguma coisa são os que comparam **valores** contra uma referência montada de outro jeito.

**Mutações que a suíte precisa pegar:**

| mutação | testes que falharam |
|---|---|
| remover a primeira soma residual | 3 de 16 |
| remover a segunda soma residual | 3 de 16 |
| trocar para post-LN | 2 de 16 |
| trocar a ordem das subcamadas | 2 de 16 |
| `Linear(dModel, dModel)` no lugar de `LayerNorm(dModel)` | 2 de 16 |
| usar o mesmo `LayerNorm` nas duas posições | **1** de 16 |
| passar `x` no lugar de `h` ao MLP | **1** de 16 |

As duas últimas linhas são as finas, e vale saber por quê. O `LayerNorm` compartilhado só aparece em `toSet.size`: o formato não muda, os valores não mudam, e o gradient check **passa** — acumular gradiente de dois usos é o comportamento correto de um parâmetro compartilhado. O `x` no lugar do `h` só aparece na comparação contra a composição. Se um dia esses dois testes forem simplificados, a cobertura dessas mutações vai junto.

---

## §8. Para onde isso leva

O bloco é a peça repetível. A Etapa 15 empilha `L` deles e acrescenta três coisas em volta:

1. **Os embeddings na frente** — o de token da Etapa 9, mais o posicional, somados.
2. **O LayerNorm final**, que a §4 antecipou: a saída da pilha nunca foi normalizada, e a projeção final precisa de uma escala previsível.
3. **A projeção para o vocabulário**, transformando `[B, T, dModel]` em `[B, T, vocabSize]` — os logits.

É lá também que o `1/√(2L)` da §5 se decide, porque é lá que `nLayers` existe pela primeira vez.

Vale notar o que **não** falta. Depois da Etapa 15, o modelo estará completo e fará forward de ponta a ponta. Só não terá aprendido nada: ainda não existe perda nem otimizador, que são as Etapas 16 e 17.

---

## Cartão de referência

| Conceito | Resumo |
|---|---|
| o bloco | `H = X + MHA(LN₁(X))`, depois `Y = H + MLP(LN₂(H))` |
| conexão residual | `y = x + f(x)`; a subcamada aprende uma **correção**, não a resposta |
| por que somar | preserva o formato e a informação, sem custar parâmetro nenhum |
| backward | `dX = dY + dY·∂f/∂X`; o primeiro termo não passa por Jacobiano nenhum |
| sem residual, 24 camadas | mediana do gradiente em `1.2e-21`; com residual, `0.453` |
| pre-LN | `X + Sub(LN(X))` — o caminho direto fica intocado |
| post-LN | `LN(X + Sub(X))` — todo caminho paga um pedágio; a escala de entrada é descartada |
| fluxo residual | o barramento que atravessa a pilha; subcamadas leem normalizado e escrevem somando |
| variância na saída | `1 + 2L` sem escala; **exatamente 2** com `1/√(2L)`, para qualquer `L` |
| parâmetros do bloco | `12·dModel² + 11·dModel` — MLP 2/3, atenção 1/3, LayerNorm 0.04% |
| `dModel = 768` | `7.086.336` por bloco; `85.036.032` nos 12 |
| o teste que carrega a etapa | compor a referência a partir das subcamadas expostas |

### As quatro lições que se repetem

1. **A soma é a peça que faltava para a profundidade.** Ela não tem parâmetro, não tem hiperparâmetro, e é a diferença entre um gradiente de `1e-21` e um de `0.45` depois de 24 camadas.
2. **Onde a normalização entra decide se existe caminho direto.** Pre-LN mantém a identidade intacta; post-LN coloca um Jacobiano em cada bloco, inclusive no caminho que deveria ser livre.
3. **O que soma é a variância.** É por isso que a escala das camadas residuais é `1/√(2L)`, e não `1/(2L)` — a mesma razão pela qual o desvio padrão cresce com a raiz do número de incrementos.
4. **Compor peças testadas não dispensa testar a composição.** Todos os erros possíveis nesta etapa — soma faltando, LN no lugar errado, subcamadas trocadas — produzem forwards perfeitamente deriváveis, que o gradient check aprova sem hesitar.
