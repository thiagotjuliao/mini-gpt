# Etapa 4 — Verificação Numérica de Gradientes

## Antes de começar

### O que você vai construir

Uma função que **audita** qualquer backward que você escrever:

```scala
Gradcheck.run(input, eps = 1e-5)(f: Tensor => Tensor): Double
```

Ela devolve um número: o maior erro relativo encontrado entre o gradiente que o seu código calculou e uma aproximação numérica obtida por um caminho totalmente independente. Perto de zero significa que o backward está certo. Grande significa que há bug.

É a etapa mais curta do projeto, e a que dá confiança para todas as outras.

### O que você precisa saber antes

**Da Etapa 2:** o que `backward()` faz, e que o gradiente vive em `input.gradient` depois dele.

**Da Etapa 3:** as operações já implementadas — elas serão as primeiras cobaias.

**De cálculo:** a definição de derivada como limite. Só isso.

### Onde esta etapa se encaixa

Até aqui, cada backward foi validado à mão. Deriva-se a fórmula no papel, monta-se um exemplo numérico, confere-se posição a posição.

Isso funcionou para doze operações. **Não vai funcionar daqui em diante.** Ainda faltam `softmax`, `layerNorm`, `attention` e cada camada nova do transformer — e refazer a conferência manual a cada uma é inviável.

Pior: você já viu três vezes na Etapa 3 que erros de gradiente são **silenciosos**. O `exp` usando a entrada, a redução acumulando no índice errado, o `reshape` escrevendo num tensor órfão. Nenhum deles trava. O forward continua perfeito e o modelo só treina mal.

Esta etapa constrói o detector.

### Objetivos de aprendizagem

Ao terminar, você deve conseguir:

1. Explicar por que a diferença central é melhor que a progressiva, usando a expansão de Taylor.
2. Justificar por que `ε = 1e-5`, e por que diminuir mais **piora** o resultado.
3. Explicar por que o erro precisa ser relativo, e não absoluto.
4. Usar `Gradcheck.run` em operações que não devolvem escalar.

> **Guia visual.** Secante contra tangente, e as faixas de erro: [`finite-difference.html`](finite-difference.html). **Exercícios (10 questões):** [`exercises.html`](exercises.html).

---

## §1. Diferença central

### A ideia

A derivada é um limite:

```
f'(x) = lim(ε→0) [f(x+ε) - f(x)] / ε
```

Um computador não calcula limites. Mas dá para **aproximar**, usando um `ε` pequeno em vez de zero.

> **Definição — diferença finita.** Aproximar uma derivada avaliando a função em pontos próximos, em vez de calcular o limite.

A forma mais direta usa um ponto à frente:

```
diferença progressiva:  [f(x+ε) - f(x)] / ε
```

Geometricamente, isso é a inclinação da **secante** entre `(x, f(x))` e `(x+ε, f(x+ε))`. Ela se parece com a tangente — a derivada de verdade —, mas não é igual.

A alternativa pega um ponto de **cada lado**:

```
diferença central:  [f(x+ε) - f(x-ε)] / (2ε)
```

Agora a secante está **centrada** em `x`, em vez de deslocada para frente. Essa mudança aparentemente pequena melhora o resultado de forma dramática.

### Por que a central é muito melhor

A explicação sai da expansão de Taylor. Escrevendo os dois pontos:

```
f(x+ε) = f(x) + f'(x)ε + f''(x)ε²/2 + f'''(x)ε³/6 + ...
f(x-ε) = f(x) - f'(x)ε + f''(x)ε²/2 - f'''(x)ε³/6 + ...
```

Subtraindo a segunda da primeira, veja o que acontece. O termo `f(x)` cancela. O termo `f'(x)ε` **dobra**. E o termo de segunda ordem, `f''(x)ε²/2`, aparece com o mesmo sinal nos dois — então ele **cancela exatamente**.

```
f(x+ε) - f(x-ε) = 2f'(x)ε + (termos de ordem ε³)
```

Dividindo por `2ε`, sobra `f'(x)` mais um resto proporcional a `ε²`.

Compare com a diferença progressiva, onde o termo de segunda ordem **não** cancela — ali o erro é proporcional a `ε`.

```
progressiva:  erro ∝ ε      →  com ε=1e-5, erro da ordem de 1e-5
central:      erro ∝ ε²     →  com ε=1e-5, erro da ordem de 1e-10
```

**Exemplo numérico**, com `f(x) = x³` em `x = 2`. A derivada exata é `f'(x) = 3x² = 12`.

Expandindo os dois pontos algebricamente:

```
f(x+ε) = 8 + 12ε + 6ε² + ε³
f(x-ε) = 8 - 12ε + 6ε² - ε³
```

Agora as duas aproximações:

```
central     = [(8+12ε+6ε²+ε³) - (8-12ε+6ε²-ε³)] / (2ε) = [24ε + 2ε³]/(2ε) = 12 + ε²
progressiva = [(8+12ε+6ε²+ε³) - 8] / ε                  = 12 + 6ε + ε²
```

Substituindo `ε = 1e-5`:

```
central     = 12.0000000001      erro teórico = 1e-10
progressiva = 12.00006           erro teórico = 6e-5
```

O erro da progressiva é **seiscentas mil vezes maior**. Mesmo `ε`, mesma função, mesmo custo de duas avaliações. É por isso que usamos a central.

### Por que `ε = 1e-5`, e não menor

Se o erro é proporcional a `ε²`, a tentação é óbvia: usar um `ε` minúsculo, como `1e-12`, e ter um erro de `1e-24`.

Isso **não funciona**, e o motivo é instrutivo.

Existem dois erros em jogo, e eles puxam para lados opostos.

> **Definição — erro de truncamento.** Vem de usar um `ε` finito em vez do limite. É o `ε²` da conta acima. **Diminui** quando `ε` diminui.

> **Definição — erro de arredondamento.** Vem de o `Double` ter precisão finita. Ao subtrair dois números quase iguais — `f(x+ε)` e `f(x-ε)` —, os dígitos significativos se cancelam e sobra ruído. **Aumenta** quando `ε` diminui.

Medindo o erro real, para a mesma `f(x) = x³` em `x = 2`:

```
     ε        erro medido      ε² (teoria)
  1e-02         1.000e-04         1e-04
  1e-03         1.000e-06         1e-06
  1e-04         1.001e-08         1e-08
  1e-05         2.118e-10         1e-10      ← melhor ponto
  1e-06         7.892e-10         1e-12      ← já piorou
  1e-08         1.173e-07         1e-16
  1e-10         9.929e-07         1e-20
  1e-12         1.067e-03         1e-24
```

Acompanhe a coluna do meio. Até `ε = 1e-5`, o erro medido segue a teoria de perto — o truncamento domina. A partir de `1e-6`, ele **para de cair e volta a subir**, cada vez mais rápido. Em `ε = 1e-12`, o resultado é pior do que com `ε = 1e-2`.

O ponto ótimo, em análise numérica, fica em torno de `(ε_máquina)^(1/3)`. Para `Double`, isso dá `(2.2e-16)^(1/3) ≈ 6.1e-6`.

Ou seja: `1e-5` não é um número mágico. É praticamente o melhor valor possível para diferença central em precisão dupla.

> **Confira você mesmo.** Alguém propõe usar `ε = 1e-15` para "ter mais precisão". O que acontece de fato?
>
> <details><summary>Resposta</summary>
>
> O resultado fica muito **pior**. Com `ε` tão pequeno, `f(x+ε)` e `f(x-ε)` são praticamente o mesmo número em `Double`. A subtração cancela quase todos os dígitos significativos e sobra ruído de arredondamento, que é então dividido por um `2ε` minúsculo — amplificando o ruído. É o extremo direito da tabela acima.
> </details>

---

## §2. Erro relativo

Com os dois gradientes em mãos, falta decidir quando eles são "iguais o suficiente".

> **Definição — gradiente analítico.** O que o seu `backward()` calculou, seguindo as fórmulas derivadas nas Etapas 2 e 3.

> **Definição — gradiente numérico.** A aproximação da seção §1, obtida perturbando a entrada. Ele não conhece o seu código de backward — é um juiz independente.

Comparar por diferença **absoluta** não serve. Um desvio de `0.001` é enorme se o gradiente vale `0.0001`, e irrelevante se ele vale `1000`. A escala importa.

```
erro = |analítico - numérico| / max(|analítico|, |numérico|, 1e-8)
```

O `1e-8` no denominador existe apenas para evitar divisão por zero quando ambos os gradientes são minúsculos. Nesse caso, qualquer diferença absoluta pequena já basta para considerá-los iguais.

As três faixas de interpretação:

| Erro relativo | Interpretação |
|---|---|
| `< 1e-5` | correto |
| `1e-5` a `1e-3` | suspeito — conferir com cuidado, pode ser instabilidade numérica |
| `> 1e-3` | bug |

**Exemplo numérico.** Reaproveitando `f(x) = x³` em `x = 2`, com `analítico = 12` e `numérico = 12.0000000001`:

```
erro = |12 - 12.0000000001| / max(12, 12.0000000001, 1e-8)
     = 1e-10 / 12
     ≈ 8.3e-12
```

Muito abaixo de `1e-5`. Backward aprovado.

Agora um caso com bug. Suponha um backward com o **sinal trocado**, devolvendo `-12`:

```
erro = |-12 - 12.0000000001| / max(12, 12.0000000001, 1e-8)
     = 24.0000000001 / 12
     ≈ 2.0
```

Erro relativo de `2.0`, contra um limite de `1e-3`. O verificador pega o bug na hora — sem que ninguém tenha escrito um teste específico para esse caso.

Erro de sinal é o tipo mais comum e mais silencioso de bug de gradiente. O forward continua perfeito, e você vai ver exatamente esse erro acontecer de verdade na GELU, na Etapa 5.

---

## §3. `Gradcheck.run` — perturbando uma posição por vez

A seção §1 usou um `x` escalar. Tensores têm muitas posições, e cada uma precisa da sua própria verificação.

A assinatura real, em curry para permitir a sintaxe de bloco:

```scala
Gradcheck.run(input, eps = 1e-5) { x => ... }
```

O algoritmo tem três partes.

**Parte 1 — o lado analítico.** Roda o forward e o backward uma vez:

```scala
val output = f(input)
output.backward()
val gradA = input.gradient
```

Duas sutilezas aqui.

O gradiente que interessa está em `input.gradient`, **não** em `output.gradient`. O `output` é a raiz escalar: seu gradiente é sempre `1.0` depois da semeadura. O que a regra da cadeia produz de útil fica na entrada.

E `f(input)` precisa devolver um tensor **escalar**, porque `backward()` exige isso. Isso muda como você usa o verificador em operações que não são escalares por natureza. Para testar `matmul`, passe `f = x => x.matmul(w).sum` — compondo com `.sum` antes, e não o `matmul` sozinho.

**Parte 2 — o lado numérico.** Para cada posição `i` do tensor:

1. Guardar o valor original.
2. Substituir por `orig + ε` e rodar só o forward.
3. Substituir por `orig - ε` e rodar de novo.
4. Aplicar a diferença central.
5. **Restaurar o valor original** antes de passar à próxima posição.

O passo 5 não é opcional. Sem ele, as perturbações vazam de uma posição para a seguinte, e todas as medições depois da primeira ficam erradas.

Os dois forwards rodam dentro de `Tensor.noGrad` — não há motivo para construir grafo em algo que será descartado.

**Parte 3 — a comparação.** Aplica o erro relativo do §2 posição a posição, e devolve o **maior** valor encontrado. Reportar o pior caso é o que garante que um bug numa única posição não se dilua na média.

**Exemplo numérico completo.** Vamos rodar o algoritmo à mão num tensor de dois elementos.

```
f(x) = (x · x).sum()          soma dos quadrados
x    = [3.0, -2.0]
```

**Analítico.** Como `f(x) = x₀² + x₁²`, temos `∂f/∂x₀ = 2x₀ = 6` e `∂f/∂x₁ = 2x₁ = -4`. O `backward()` chega nesses valores compondo as regras de `mul` e `sum`, já implementadas na Etapa 3.

```
grad_analítico = [6, -4]
```

**Numérico, posição 0** — perturbando apenas `x₀`:

```
f(3.00001, -2.0) = 9.0000600001 + 4 = 13.0000600001
f(2.99999, -2.0) = 8.9999400001 + 4 = 12.9999400001

grad_numérico[0] = (13.0000600001 - 12.9999400001) / (2 · 1e-5)
                 = 0.00012 / 0.00002
                 = 6.0
```

**Numérico, posição 1** — perturbando apenas `x₁`:

```
f(3.0, -1.99999) = 9 + 3.9999600001 = 12.9999600001
f(3.0, -2.00001) = 9 + 4.0000400001 = 13.0000400001

grad_numérico[1] = (12.9999600001 - 13.0000400001) / (2 · 1e-5)
                 = -0.00008 / 0.00002
                 = -4.0
```

Os dois vetores batem: `[6, -4]` contra `[6.0, -4.0]`. As regras de `mul` e `sum`, escritas à mão na Etapa 3, ficam confirmadas por um caminho que não usa nenhuma delas.

> **Armadilha.** A primeira versão deste verificador teve **cinco** bugs. Vale a lista inteira, porque cada um representa uma categoria diferente de erro.
>
> **Perturbava todas as posições ao mesmo tempo**, com um único forward. Isso mede a sensibilidade combinada de todos os parâmetros juntos, não o gradiente de cada um.
>
> **Lia o gradiente de `output` em vez de `input`.** Como o `output` é a raiz, seu gradiente é sempre `1.0`.
>
> **Precedência de operador:** escreveu `(a - b) / 2 * eps`, que calcula `((a-b)/2)·ε` em vez de `(a-b)/(2ε)`.
>
> **Devolvia o valor errado:** usou `maxBy(...)._2`, que retorna o gradiente numérico do par de maior erro, e não o erro em si.
>
> **Piso do denominador errado:** usou `eps` (`1e-5`) no lugar de `1e-8`, tornando o critério bem menos sensível para gradientes pequenos.
>
> **Lição geral:** o verificador também é código, e também tem bugs. Por isso ele precisa de um teste próprio — com uma operação **deliberadamente quebrada**, confirmando que o erro reportado fica acima do limite. Um verificador que aprova tudo é pior que nenhum.

> **Armadilha.** Esta apareceu só quando a suíte inteira rodou junta, e é a mais instrutiva do projeto.
>
> Depois de escrever dezesseis verificações cobrindo todas as operações da Etapa 3, elas passavam isoladamente. Mas ao rodar **todas as suítes de uma vez**, quarenta e cinco testes de outros arquivos começaram a falhar de forma imprevisível, com gradientes zerados.
>
> A causa: o `gradEnabled` da Etapa 2 era uma variável global compartilhada, e o ScalaTest roda suítes em paralelo. Isso nunca tinha dado problema porque nenhuma operação chamava `noGrad` com frequência. O verificador muda isso — ele chama `noGrad` duas vezes por posição perturbada. Um único tensor `(2,3)` já dispara doze alternâncias da flag global, e uma suíte vizinha era pega no meio, com o autograd desligado na hora errada.
>
> A correção foi trocar a variável global por uma `DynamicVariable`, em que cada thread enxerga o próprio valor.
>
> **Lição geral:** uma varredura ampla encontra bugs que testes isolados nunca encontram — e nem sempre no código que está sendo testado. Aqui, o alvo era o backward das operações; o bug estava no autograd, escrito duas etapas antes.

> **Confira você mesmo.** Por que `Gradcheck.run(x)(x => x.softmax(0))` seria uma chamada inválida?
>
> <details><summary>Resposta</summary>
>
> Porque `softmax` devolve um tensor do mesmo tamanho da entrada, e não um escalar — e `backward()` só pode ser chamado numa raiz escalar. É preciso compor: `x => (x.softmax(0) * pesos).sum`. Por que multiplicar por pesos, e não apenas somar, é um detalhe sutil que a Etapa 6 explica.
> </details>

---

## §4. Para onde isso leva

A partir daqui, toda operação nova ganha uma forma barata de auditoria. Gere um tensor de formato aleatório, rode `Gradcheck.run`, confirme que o erro fica abaixo de `1e-5`.

Isso muda o custo de errar. Sem o verificador, um bug de gradiente sobrevive até o modelo inteiro estar montado — e aí a única evidência é "o treino não converge", sem nenhuma pista de qual das quinze camadas está errada. Com ele, o bug é pego na operação isolada, minutos depois de ter sido escrito.

A Etapa 5 é a primeira cliente. Ela implementa quatro funções de ativação, e uma delas — a GELU — tem a derivada mais complicada do projeto inteiro. Você vai ver o verificador ganhar o seu sustento lá.

---

## Cartão de referência

| Conceito | Resumo |
|---|---|
| diferença progressiva | `[f(x+ε) - f(x)] / ε` — erro ∝ `ε` |
| diferença central | `[f(x+ε) - f(x-ε)] / (2ε)` — erro ∝ `ε²` |
| por que a central é melhor | o termo de segunda ordem de Taylor cancela |
| `ε = 1e-5` | ponto ótimo entre truncamento e arredondamento |
| erro de truncamento | do `ε` finito; **diminui** com `ε` menor |
| erro de arredondamento | da precisão do `Double`; **aumenta** com `ε` menor |
| erro relativo | `\|a - n\| / max(\|a\|, \|n\|, 1e-8)` |
| limites | `< 1e-5` correto · `1e-5` a `1e-3` suspeito · `> 1e-3` bug |
| gradiente analítico | vem de `input.gradient`, nunca de `output.gradient` |
| `f` deve devolver | um **escalar** — componha com `.sum` quando preciso |
| restaurar após perturbar | obrigatório, senão vaza para a próxima posição |
| resultado devolvido | o **maior** erro, não a média |

### As quatro lições que se repetem

1. **Menor nem sempre é melhor.** Diminuir `ε` além do ponto ótimo piora o resultado, porque troca erro de truncamento por erro de arredondamento.
2. **Compare em escala relativa.** Erro absoluto não significa nada sem saber a magnitude do gradiente.
3. **Um verificador precisa ser verificado.** Teste-o com uma operação propositalmente quebrada e confirme que ele reprova.
4. **Rode tudo junto, de vez em quando.** O bug de concorrência deste capítulo só existia quando várias suítes rodavam ao mesmo tempo — e estava em código escrito duas etapas antes.
