# Etapa 11 — Scaled Dot-Product Attention

## Antes de começar

### O que você vai construir

O mecanismo que deixa um token olhar para os outros tokens da sequência.

| Componente | Formato | O que faz |
|---|---|---|
| `W_Q`, `W_K`, `W_V` | `[dModel, dHead]` | três projeções treináveis da entrada |
| máscara causal | `[T, T]` | bloqueia as posições futuras |
| `forward(x)` | `[B, T, dModel]` → `[B, T, dHead]` | mistura os tokens permitidos, com pesos aprendidos |

### O que você precisa saber antes

**Da Etapa 3:** `matmul` em lote, `transpose(dim0, dim1)` e broadcasting. A etapa inteira é feita disso.

**Da Etapa 6:** o softmax por dimensão e o seu backward. Ele é o coração desta etapa.

**Da Etapa 8:** a camada `Linear`, que faz cada uma das três projeções.

**Da Etapa 9:** o formato `[B, T, dModel]` que sai do embedding. É a entrada daqui.

### Onde esta etapa se encaixa

A Etapa 9 deu a cada token um vetor. Esse vetor é o mesmo em qualquer frase. O embedding de `banco` não sabe se a frase fala de dinheiro ou de praça.

Falta um mecanismo que misture informação entre os tokens. Ele precisa de três propriedades. Precisa alcançar qualquer distância, porque o sujeito pode estar longe do verbo. Precisa ser aprendido, e não fixo. E precisa ser derivável, para caber no autograd da Etapa 2.

A atenção é esse mecanismo. É a peça que separa o transformer do que veio antes, e é o assunto do artigo que batizou a arquitetura.

### Objetivos de aprendizagem

Ao terminar, você deve conseguir:

1. Explicar por que existem três projeções da entrada, e não uma só.
2. Justificar a divisão por `√dHead` com o argumento de variância, não como um truque.
3. Explicar por que a máscara entra antes do softmax, e o que quebra se entrar depois.
4. Percorrer o backward da atenção e dizer de onde vem cada termo.
5. Escrever um teste que detecte um vazamento de informação do futuro.

### Convenções de notação

| Símbolo | Significa |
|---|---|
| `B`, `T` | tamanho do lote e número de tokens da sequência |
| `dModel` | tamanho do vetor de cada token na entrada |
| `dHead` | tamanho do vetor dentro da atenção |
| `Q`, `K`, `V` | as queries, keys e values — `[B, T, dHead]` |
| `S` | os scores já escalados — `[B, T, T]` |
| `P` | os pesos de atenção, depois do softmax — `[B, T, T]` |
| `Y` | a saída da atenção — `[B, T, dHead]` |

A convenção `dX ≡ ∂L/∂X` do overview continua valendo.

> **Guia visual.** O caminho completo com os formatos de cada passo: [`attention.html`](attention.html), FIG 1. **Exercícios (14 questões):** [`exercises.html`](exercises.html).

---

## §1. O problema: um vetor por token não basta

Depois da Etapa 9, uma frase de `T` tokens é uma matriz `[T, dModel]`. Cada linha é um token, isolada das demais.

Isso é pouco. O significado de uma palavra depende das outras palavras. Pronomes precisam de antecedente. Verbos precisam de sujeito. Nada disso cabe num vetor fixo por token.

Então precisamos de uma operação que produza um vetor novo para cada posição, misturando as outras. A forma mais simples de misturar é uma média ponderada:

```
yᵢ = Σⱼ pᵢⱼ · xⱼ           com  Σⱼ pᵢⱼ = 1
```

A pergunta toda desta etapa é de onde vêm os pesos `pᵢⱼ`.

Pesos fixos não servem. Eles seriam os mesmos para toda frase, e a relação entre duas palavras depende de quais palavras são. Os pesos precisam ser **calculados a partir do conteúdo**.

> **Definição — atenção.** Uma operação que, para cada posição, calcula uma média ponderada de todas as posições. Os pesos saem da comparação entre os próprios vetores, e não de parâmetros fixos por posição.

> **Definição — auto-atenção.** Atenção em que as posições comparadas e as posições misturadas são da mesma sequência. É o caso deste projeto. Existe também a atenção entre duas sequências diferentes, usada em tradução, que não vamos construir.

**Exemplo numérico.** Suponha três tokens com vetores `x₀ = [4, 2]`, `x₁ = [0, 3]` e `x₂ = [3, 1]`. Com pesos `p = [0.25, 0.60, 0.15]` para a posição 2:

```
y₂ = 0.25·[4, 2] + 0.60·[0, 3] + 0.15·[3, 1]
   = [1.00 + 0.00 + 0.45,  0.50 + 1.80 + 0.15]
   = [1.45, 2.45]
```

Repare onde `y₂` caiu. Cada coordenada está entre a menor e a maior das entradas: `1.45` está entre `0` e `4`, e `2.45` está entre `1` e `3`. Isso vale sempre, porque os pesos são positivos e somam `1`. Guarde essa propriedade — a §8 a transforma em teste.

---

## §2. Query, Key e Value: três papéis para o mesmo token

Os pesos precisam sair de uma comparação entre tokens. A tentação é comparar os vetores diretamente, com `xᵢ · xⱼ`. Isso não funciona bem, por dois motivos.

O primeiro é a simetria. `xᵢ · xⱼ` é igual a `xⱼ · xᵢ`, então o quanto A olha para B é forçosamente igual ao quanto B olha para A. Relações de linguagem não são assim. Um adjetivo precisa muito do seu substantivo, e o substantivo quase não precisa do adjetivo.

O segundo é a diagonal. `xᵢ · xᵢ` é o maior produto possível para vetores de mesmo tamanho. Todo token olharia sobretudo para si mesmo, sempre.

A solução é dar a cada token três papéis distintos, cada um com sua própria projeção treinável:

> **Definição — query, key e value.** A *query* de um token descreve o que ele procura. A *key* descreve o que ele oferece a quem procura. O *value* é o conteúdo que ele entrega quando é escolhido.

```
Q = X · W_Q        K = X · W_K        V = X · W_V
```

Vale a analogia da biblioteca. Você chega com uma pergunta, que é a query. Cada livro tem uma etiqueta na lombada, que é a key. A comparação entre pergunta e etiqueta decide quais livros valem a pena. O que você leva para casa é o texto do livro, que é o value.

As três matrizes têm formato `[dModel, dHead]` e são independentes. Com isso a simetria some, porque `qᵢ · kⱼ` não tem nenhuma relação com `qⱼ · kᵢ`. E o token deixa de ser obrigado a olhar para si mesmo, porque `qᵢ · kᵢ` não é mais um máximo garantido.

Separar o value dos outros dois também importa. O que decide **quanto** olhar não precisa ser o que é **transportado**. A key pode carregar "sou um verbo no passado" enquanto o value carrega o significado do verbo.

**Exemplo numérico.** Este é o exemplo que atravessa o capítulo inteiro. São três tokens, `dModel = 4` e `dHead = 2`:

```
      x₀ = [1, 0, 1, 0]          W_Q = [1 0]     W_K = [1 0]     W_V = [3 0]
X =   x₁ = [0, 2, 0, 1]                [0 1]           [0 1]           [0 1]
      x₂ = [1, 1, 0, 0]                [1 0]           [1 1]           [1 2]
                                       [0 1]           [2 0]           [0 1]
```

Multiplicando linha por matriz:

```
Q = [2  0]        K = [2  1]        V = [4  2]
    [0  3]            [2  2]            [0  3]
    [1  1]            [1  1]            [3  1]
```

Confira a primeira linha de `K` na mão. `x₀ = [1, 0, 1, 0]` seleciona as linhas 0 e 2 de `W_K`, que são `[1, 0]` e `[1, 1]`. A soma dá `[2, 1]`, como na tabela.

> **Confira você mesmo.** Se `W_Q` e `W_K` fossem a mesma matriz, a simetria voltaria?
>
> <details><summary>Resposta</summary>
>
> Sim, e completamente. Com `W_Q = W_K = W`, o score vira `xᵢW · xⱼW`, que é igual trocando `i` por `j`. O modelo perderia a capacidade de expressar relações de mão única. É por isso que as duas matrizes existem separadas, mesmo tendo o mesmo formato.
> </details>

---

## §3. Scores: o produto interno como medida de alinhamento

> **Guia visual.** A matriz de scores do exemplo, e a assimetria entre linha e coluna: [`attention.html`](attention.html), FIG 2.

Com `Q` e `K` prontos, o score de cada par é o produto interno da query de um com a key do outro:

```
sᵢⱼ = qᵢ · kⱼ = Σ_c qᵢc · kⱼc
```

> **Definição — score de atenção.** Um número por par de posições. Mede o quanto a query de `i` está alinhada com a key de `j`, antes de qualquer normalização.

O produto interno mede alinhamento por construção. Ele vale `|q|·|k|·cos θ`. Vetores apontando na mesma direção dão score alto. Perpendiculares dão zero. Opostos dão negativo.

Os `T²` scores saem de uma única multiplicação de matrizes:

```
S_bruto = Q · Kᵀ           [T, dHead] · [dHead, T] → [T, T]
```

Aqui aparece o primeiro `transpose` de verdade do projeto. A Etapa 3 construiu `transpose(dim0, dim1)` com o padrão nas duas últimas dimensões justamente para este momento. Num tensor `[B, T, dHead]`, a chamada sem argumentos troca `T` com `dHead` e deixa o lote em paz.

Duas coisas valem ser ditas sobre o formato `[T, T]`. A primeira é que a matriz de scores não é simétrica, pelo motivo da §2. A linha `i` diz para onde `i` olha; a coluna `j` diz quem olha para `j`. A segunda é que o custo cresce com `T²`. Dobrar o tamanho do contexto quadruplica esta parte da conta. É o gargalo conhecido do transformer.

**Exemplo numérico**, com o `Q` e o `K` da §2:

```
q₀ = [2, 0]        k₀ = [2, 1]      q₀·k₀ = 4     q₀·k₁ = 4     q₀·k₂ = 2
q₁ = [0, 3]        k₁ = [2, 2]      q₁·k₀ = 3     q₁·k₁ = 6     q₁·k₂ = 3
q₂ = [1, 1]        k₂ = [1, 1]      q₂·k₀ = 3     q₂·k₁ = 4     q₂·k₂ = 2

            [4  4  2]
S_bruto  =  [3  6  3]
            [3  4  2]
```

Confira a assimetria posição a posição. A entrada `(0,1)` vale `4` e a entrada `(1,0)` vale `3`. Nada obriga as duas a serem iguais, e é exatamente isso que se queria.

---

## §4. Por que dividir por `√dHead`

> **Guia visual.** Os 16 pesos com e sem a divisão, na mesma escala: [`attention.html`](attention.html), FIG 3.

Os scores ainda não estão prontos. Falta dividi-los por `√dHead`. Esse fator não é ajuste fino: sem ele o treino de modelos com `dHead` grande fica lento.

O argumento é de variância. Suponha que as coordenadas de `q` e `k` sejam independentes, com média `0` e variância `1`. O produto interno soma `dHead` parcelas independentes. Variâncias de parcelas independentes se somam:

```
Var(q · k) = Σ_c Var(q_c · k_c) = dHead · 1 = dHead

desvio padrão = √dHead
```

Ou seja, os scores crescem em escala com `√dHead`, só por causa do tamanho do vetor. Com `dHead = 64`, os scores se espalham numa faixa de largura oito. Dividir por `√dHead` devolve a variância para `1`, qualquer que seja a dimensão.

Isso importa por causa do softmax. Ele é sensível a **diferenças** entre os scores. Diferenças grandes fazem um peso ir para perto de `1` e os outros para perto de `0`. A Etapa 6 §4 mostrou que o gradiente que volta por um peso `p` é proporcional ao próprio `p`. Peso zero significa gradiente zero.

**Exemplo numérico.** Uma query contra 16 keys, com `dHead = 64` e coordenadas sorteadas de `N(0, 1)`. Os produtos internos vão de `-21.79` a `7.56`. A variância medida em 50 mil pares foi `64.03`, contra `64` da previsão:

```
                                  sem escala        com escala
maior peso                          0.724853          0.144914
menor peso                          1.3e-13           0.003696
pesos abaixo de 1e-6                 4 de 16            0 de 16
posições efetivamente olhadas          2.59             12.72
```

A última linha é `exp` da entropia dos pesos. Ela responde "sobre quantas posições esta distribuição está de fato espalhada".

Sem a escala, quatro das 16 posições recebem peso menor que `1e-6`. O gradiente que chega nelas é menor que `1e-6` do que deveria. Elas estão congeladas: não recebem correção e não têm como voltar à disputa.

Repare que a divisão é por `√dHead`, não por `dHead`. Queremos corrigir o **desvio padrão**, que cresce com a raiz. Dividir por `dHead` esmagaria todos os scores para perto de zero, e o softmax devolveria pesos quase uniformes — atenção nenhuma.

> **Confira você mesmo.** No exemplo corrente `dHead = 2`, e os scores brutos vão de `2` a `6`. A escala ainda faz diferença aqui?
>
> <details><summary>Resposta</summary>
>
> Faz pouca, e é o esperado. Com `dHead = 2` o desvio previsto é `√2 ≈ 1.41`, medido em `1.412`. Dividir por isso encolhe diferenças que já eram pequenas. A escala é um seguro que só cobra quando `dHead` cresce. Num GPT real ela é indispensável, com `dHead` de 64 ou 128.
> </details>

---

## §5. A máscara causal

> **Guia visual.** As duas ordens lado a lado, com a soma de cada linha: [`attention.html`](attention.html), FIG 4.

O modelo que estamos construindo prevê o próximo token. O treino aproveita isso: uma sequência de `T` tokens dá `T` previsões de uma vez, uma por posição.

Só que isso só é honesto se a posição `i` não puder ver as posições posteriores. Se ela puder, a resposta está na entrada, e o modelo aprende a copiá-la. A perda de treino despenca e a geração não funciona, porque na hora de gerar o futuro não existe.

> **Definição — máscara causal.** Uma matriz `[T, T]` que zera a influência de toda posição `j > i` sobre a posição `i`. Na prática, soma `-inf` a esses scores antes do softmax.

O nome vem de causalidade: efeito nenhum vem do que ainda não aconteceu. A matriz é triangular inferior, com `0` no que é permitido:

```
        j=0    j=1    j=2
i=0      0    -inf   -inf
i=1      0      0    -inf
i=2      0      0      0
```

O `-inf` funciona porque `exp(-inf) = 0`. A posição bloqueada sai do softmax com peso exatamente zero, e o denominador nem chega a contá-la.

**A ordem é obrigatória: máscara antes do softmax.** Zerar os pesos depois parece equivalente e não é. O softmax normaliza pelo que existia **naquele momento**. Se você zera depois, a linha deixa de somar `1`.

**Exemplo numérico**, com os scores escalados do exemplo corrente. Compare as duas ordens:

```
scores escalados        [2.8284  2.8284  1.4142]
                        [2.1213  4.2426  2.1213]
                        [2.1213  2.8284  1.4142]

CERTO — máscara, depois softmax          ERRADO — softmax, depois zerar
[1.000000  0        0       ] soma 1.0   [0.445808  0         0       ] soma 0.446
[0.107042  0.892958  0      ] soma 1.0   [0.096692  0.806617  0       ] soma 0.903
[0.283995  0.575975  0.140029] soma 1.0  [0.283995  0.575975  0.140029] soma 1.0
```

Na versão errada, a saída da primeira posição vira `0.446 · v₀` em vez de `v₀`. O vetor sai encolhido por um fator que muda de linha para linha. E a última linha fica intacta, o que torna o defeito ainda mais difícil de ver: o erro depende da posição.

```
CERTO                        ERRADO
y₀ = [4.000000  2.000000]    y₀ = [1.783233  0.891617]
y₁ = [0.428167  2.892958]    y₁ = [0.386767  2.613233]
y₂ = [1.556069  2.435946]    y₂ = [1.556069  2.435946]
```

Repare no `y₀` correto: ele é exatamente `v₀`. A primeira posição só pode olhar para si mesma, então seu peso é `1` e a média ponderada devolve o próprio value. Isso é uma boa sanidade a conferir em qualquer implementação.

**Um detalhe de `-inf` que vale conhecer.** Uma linha inteiramente `-inf` produziria `NaN`, porque o softmax subtrai o máximo da linha e `-inf - (-inf)` é indefinido. A máscara causal nunca faz isso, já que a diagonal está sempre liberada. O risco existe em máscaras de preenchimento, que este projeto não usa.

> **Confira você mesmo.** A primeira linha dos pesos é `[1, 0, 0]` quaisquer que sejam os scores. O que isso diz sobre o gradiente que chega em `q₀`?
>
> <details><summary>Resposta</summary>
>
> Que ele é zero. Se os pesos daquela linha não dependem dos scores, mudar `q₀` não muda nada da saída. A §7 confirma isso com números: a primeira linha de `dQ` é exatamente `[0, 0]`. É correto, e não um bug.
> </details>

---

## §6. Softmax e a média ponderada

Com os scores mascarados, o resto é curto. O softmax por linha transforma cada linha numa distribuição, e a multiplicação por `V` faz a mistura:

```
P = softmax(S + máscara, dim = -1)          [T, T]
Y = P · V                                    [T, dHead]
```

> **Definição — peso de atenção.** O elemento `pᵢⱼ`, entre `0` e `1`, com `Σⱼ pᵢⱼ = 1`. É a fração do value de `j` que entra na saída de `i`.

A dimensão do softmax é a última, e isso não é arbitrário. Cada linha responde a uma pergunta fechada: "dado que estou na posição `i`, como distribuo minha atenção entre as posições permitidas?". Somar `1` ao longo das colunas não teria significado nenhum.

**Exemplo numérico completo**, fechando o exemplo corrente:

```
mascarados            [2.8284    -inf      -inf  ]
                      [2.1213   4.2426     -inf  ]
                      [2.1213   2.8284    1.4142 ]

P                     [1.000000  0.000000  0.000000]
                      [0.107042  0.892958  0.000000]
                      [0.283995  0.575975  0.140029]

Y = P · V             [4.000000  2.000000]
                      [0.428167  2.892958]
                      [1.556069  2.435946]
```

Confira a linha 1 por dois caminhos. Pelo softmax: a diferença entre os dois scores é `4.2426 - 2.1213 = 2.1213`, e `exp(-2.1213) = 0.119873`. Daí `p₁₀ = 0.119873 / 1.119873 = 0.107042`, e o outro peso é o complemento.

Pela média ponderada, com `v₀ = [4, 2]` e `v₁ = [0, 3]`:

```
y₁ = 0.107042 · [4, 2] + 0.892958 · [0, 3]
   = [0.428167,  0.214084 + 2.678874]
   = [0.428167,  2.892958]
```

Os dois caminhos batem. E `y₁` obedece à propriedade da §1: a primeira coordenada ficou entre `0` e `4`, a segunda entre `2` e `3`.

---

## §7. Backward

A boa notícia primeiro. Se o forward for escrito com as operações que já existem, não há backward nenhum para escrever. `matmul`, `transpose`, `+`, `/` e `softmax` já têm o seu desde as Etapas 3 e 6, e a regra da cadeia da Etapa 2 costura tudo.

Ainda assim, vale percorrer o caminho. Ele explica os testes da §8 e reaparece na Etapa 12.

**Passo 1 — de `Y` para `P` e `V`.** É um `matmul`, com a regra da Etapa 3 §6:

```
dV = Pᵀ · dY              dP = dY · Vᵀ
```

**Passo 2 — pelo softmax.** Linha a linha, com a fórmula da Etapa 6 §4:

```
dSᵢⱼ = pᵢⱼ · ( dpᵢⱼ - Σₖ pᵢₖ · dpᵢₖ )
```

Duas consequências caem daqui. A primeira: onde `pᵢⱼ = 0`, o gradiente é exatamente zero. As posições futuras não recebem gradiente nenhum, o que é a versão em backward da máscara. A segunda: cada linha de `dS` soma zero, porque o softmax ignora deslocamentos comuns da linha inteira.

**Passo 3 — pela escala e pelas projeções.** Dividir por `√dHead` é linear, então o gradiente é dividido pelo mesmo número. Depois, o `matmul` de novo:

```
dQ = dS · K / √dHead              dK = dSᵀ · Q / √dHead
```

A transposta em `dK` tem uma leitura direta. No forward, `kⱼ` foi comparado com todas as queries. Então o gradiente dele recolhe contribuições da **coluna** `j` inteira de `S`, não da linha.

**Exemplo numérico**, com um gradiente de saída de valores distintos:

```
dY = [ 1.0   0.5]
     [-2.0   1.0]
     [ 0.5   3.0]

dV = [ 0.927914   1.459028]      dP = [ 5.0   1.5   3.5]
     [-1.497929   2.620884]           [-6.0   3.0  -5.0]
     [ 0.070015   0.420088]           [ 8.0   9.0   4.5]

dS (após a escala) = [ 0.000000   0.000000   0.000000]
                     [-0.608292   0.608292   0.000000]
                     [-0.017245   0.372302  -0.355057]

dQ = [ 0.000000   0.000000]      dK = [-0.017245  -1.842120]
     [ 0.000000   0.608292]           [ 0.372302   2.197178]
     [ 0.355057   0.372302]           [-0.355057  -0.355057]
```

Três leituras confirmam que isso está certo, antes mesmo de conferir número por número.

A primeira linha de `dS` é toda zero. É a previsão do "Confira você mesmo" da §5: aquela linha tem um peso fixo em `1`, logo nenhuma dependência dos scores. A primeira linha de `dQ` herda o zero.

O triângulo superior de `dS` é exatamente zero, não aproximadamente. Nenhum gradiente atravessou a máscara.

Cada linha de `dS` soma zero. Medido: `0`, `-3.3e-16` e `2.2e-16`.

**Verificação independente.** Perturbando cada entrada de `Q`, `K` e `V` por `±1e-6` e medindo a variação da perda pela diferença central da Etapa 4 §1:

```
erro máximo em dQ:  7.8e-10
erro máximo em dK:  1.1e-09
erro máximo em dV:  1.6e-09
```

> **Confira você mesmo.** `dP` na posição `(0,1)` vale `1.5`, e não zero. Isso contradiz a máscara?
>
> <details><summary>Resposta</summary>
>
> Não. `dP` é o gradiente em relação aos **pesos**, calculado por `dY · Vᵀ` sem saber que existe máscara. O bloqueio acontece um passo depois: o backward do softmax multiplica por `pᵢⱼ`, que ali vale zero. O produto zera. Vale prestar atenção nisso ao depurar — um valor não nulo em `dP` não é sinal de vazamento.
> </details>

---

## §8. Implementação: formatos, a projeção em lote, e como testar

**O rastreio de formatos**, do começo ao fim:

| Passo | Operação | Formato |
|---|---|---|
| entrada | — | `[B, T, dModel]` |
| projeções | `x.matmul(W_*)` | `[B, T, dHead]` |
| scores | `Q.matmul(K.transpose())` | `[B, T, T]` |
| escala | `/ √dHead` | `[B, T, T]` |
| máscara | `+ mask` | `[B, T, T]` |
| pesos | `softmax(dim = 2)` | `[B, T, T]` |
| saída | `P.matmul(V)` | `[B, T, dHead]` |

Um erro de formato aqui costuma explodir na hora, e isso é sorte. O caso perigoso é `T == dHead`, quando um `transpose` esquecido ainda produz formatos compatíveis. Prefira testes com `T ≠ dHead`.

**As projeções, e uma limitação que esta etapa derrubou.** Até aqui, o `matmul` aceitava só os pares de rank `(2,2)` e `(3,3)`. A atenção precisa de `[B, T, dModel]` contra `[dModel, dHead]`, que é `(3,2)` — e `Linear.forward` é exatamente `x.matmul(W) + b`.

A limitação passou dois capítulos sem incomodar porque nada tinha exercitado o `Linear` com entrada rank 3. Ela caiu na Etapa 11: o antigo `matmul3D` virou `matmulBatched` e cobre os dois casos no mesmo corpo. A matemática é a da Etapa 3 §6, na subseção do operando compartilhado — mesmo forward, e o `dW` somando sobre o lote.

Havia três rotas possíveis, e vale saber por que a escolhida foi essa:

1. **`reshape` para `[B·T, dModel]`, `matmul` 2D, `reshape` de volta.** Nenhum código novo no núcleo, e é o que o PyTorch faz por baixo. Custa uma cópia por projeção, porque `reshape` chama `contiguous`.
2. **Estender o `matmul`** — a escolhida. A Etapa 12 repete esse padrão quatro vezes por bloco, então o custo de mexer no núcleo se paga rápido. E evita replicar dado.
3. **`broadcastTo` em `W` seguido de `matmul` em lote.** Não vá por aí. O `broadcastTo` compartilha o objeto `gradient` do tensor original, enquanto o backward em lote acumula por índice canônico do formato novo. Os índices não se correspondem.

O efeito prático é que o `Linear` da Etapa 8 atende a atenção **sem uma linha de mudança**. O `b` de formato `[dHead]` broadcasta sobre lote e sequência, e o `unbroadcast` da Etapa 3 §4 soma o gradiente de volta. Uma camada que não sabe que sequências existem funciona dentro delas.

**A máscara, na prática.** O roadmap fala em pré-computar a máscara para `contextLength`. Hoje isso esbarra numa limitação: não existe operação de recorte em `scalagrad`. O `indexSelect` da Etapa 9 seleciona linhas de um tensor rank 2, então recortar `[C, C]` para `[T, T]` exigiria dois `indexSelect` com um `transpose` no meio. Construir a máscara `[T, T]` direto, por `Tensor.make`, é mais simples. Ela não tem gradiente e é barata. Se o custo incomodar, guarde-a num mapa indexado por `T`.

> **Armadilha.** Ler o dado de um tensor por `t.data(i)` em vez de `t.get(...)`.
>
> Aconteceu neste projeto, no forward do softmax da Etapa 6. `data` é o array físico; `get` passa pelas strides. Para um tensor contíguo os dois dão o mesmo resultado, e o teste passa.
>
> Por que é perigoso: o primeiro tensor não contíguo do projeto aparece **aqui**, em `K.transpose()`. O bug ficou dormindo duas etapas. Foi corrigido durante a revisão da Etapa 6, com esta etapa citada por nome como o cenário que o acordaria.
>
> **Lição geral:** dentro de uma operação, leia sempre pela API que respeita as strides. Um teste com entrada transposta custa duas linhas e cobre a classe inteira desses erros.

**Como testar.** Além do gradient check da Etapa 4 §3 em `Q`, `K` e `V`, quatro testes valem por muito:

**Causalidade.** Mude o último token da entrada e confira que as saídas anteriores não mudam. É o teste que pega vazamento, e nenhum gradient check pega. Com o exemplo corrente, trocando `x₂ = [1, 1, 0, 0]` por `[5, -3, 7, 2]`:

```
linhas 0 e 1:  diferença máxima 0.0e+00       ← intactas, como exigido
linha 2:       [1.556069  2.435946] → [22.0  13.0]
```

**Primeira linha.** `y₀` tem que ser exatamente `v₀`, sem tolerância.

**Envelope convexo.** Cada saída fica entre o mínimo e o máximo dos values permitidos, coordenada a coordenada. É a propriedade da §1, e ela pega tanto peso negativo quanto linha que não soma `1`.

**Soma das linhas de `P`.** Cada uma vale `1`, incluindo as mascaradas. Este é o teste que separa a ordem certa da errada da §5.

> **Armadilha.** Usar `P.sum` como perda no gradient check.
>
> A soma dos pesos de atenção é sempre `T`, uma para cada linha. É uma função constante da entrada, e o gradiente verdadeiro dela é zero. O mesmo já foi medido no softmax da Etapa 6 e no LayerNorm da Etapa 10 §6.
>
> Por que é perigoso: o veredito não fala sobre o backward. Ele compara ruído de arredondamento contra zero, e o erro relativo do `Gradcheck` chega a reprovar código correto.
>
> **Lição geral:** a perda do gradient check precisa depender de verdade de toda saída. Multiplique por pesos distintos antes de somar.

---

## §9. Para onde isso leva

Esta etapa construiu uma cabeça de atenção.

> **Definição — cabeça (*head*).** Um conjunto completo de `W_Q`, `W_K` e `W_V`, com a sua própria matriz de pesos `[T, T]`.

Uma cabeça aprende um tipo de relação. A Etapa 12 roda várias em paralelo, cada uma num pedaço do vetor, e concatena os resultados. O mecanismo é idêntico ao daqui; o que muda é a contabilidade de formatos, com um eixo a mais entre o lote e a sequência.

Depois disso, a Etapa 13 traz o MLP, que processa cada posição isoladamente. A Etapa 14 junta os dois com o LayerNorm da Etapa 10 e as conexões residuais, formando o bloco que se repete.

Vale registrar o que já está fechado. A atenção mistura posições; o MLP processa cada uma; o LayerNorm mantém a escala. São as três peças de um bloco de transformer, e a única que faltava era esta.

---

## Cartão de referência

| Conceito | Resumo |
|---|---|
| ideia central | média ponderada das posições, com pesos calculados do próprio conteúdo |
| `Q`, `K`, `V` | três projeções de `X`, com `W_*` de formato `[dModel, dHead]` |
| por que três | quebram a simetria de `xᵢ·xⱼ` e separam "quanto olhar" de "o que transportar" |
| scores | `S = Q·Kᵀ / √dHead`, formato `[T, T]`, não simétrico |
| a escala | corrige o desvio padrão `√dHead` do produto interno |
| máscara causal | `-inf` no triângulo superior, somada **antes** do softmax |
| pesos | `softmax(dim = -1)`; cada linha soma `1` |
| saída | `Y = P·V`, formato `[T, dHead]` |
| `dV`, `dP` | `Pᵀ·dY` e `dY·Vᵀ` |
| `dQ`, `dK` | `dS·K/√dHead` e `dSᵀ·Q/√dHead` |
| conferências | `y₀ = v₀`; linhas de `P` somam `1`; triângulo superior de `dS` zerado |
| custo | `T²` em memória e em tempo |

### As quatro lições que se repetem

1. **Quebrar simetria é uma decisão de projeto.** Três projeções em vez de uma existem para tornar a relação entre dois tokens direcional.
2. **Fatores de escala vêm de contas de variância.** `√dHead` sai do mesmo raciocínio que deu `2/inputDim` na Etapa 8 §5.
3. **A ordem entre normalizar e mascarar não é comutativa.** Quem normaliza depois de zerar perde a soma `1`, e o erro varia com a posição.
4. **Propriedades estruturais dão os melhores testes.** Causalidade, soma `1` e envelope convexo pegam bugs que o gradient check não vê.
