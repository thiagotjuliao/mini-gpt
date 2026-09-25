# HISTORY

Log resumido de checkpoints do projeto mini-gpt. Um registro por marco relevante (etapa concluída, decisão importante, bug significativo resolvido). Não é changelog de commits — é para dar contexto rápido de "onde paramos" em conversas futuras.

---

## 2026-07-13 — Setup inicial do projeto

- Criado `PLAN.md` com as 19 etapas do roteiro (Tensor → Autograd → ... → Geração de texto).
- Criada a pasta `checklists/` com um arquivo de checklist por etapa, espelhando as subtarefas do `PLAN.md`.
- Criado `CLAUDE.md` definindo o modo de colaboração (Claude guia, usuário implementa) e a estrutura de diretórios planejada para `src/`.
- Nenhum código Scala escrito ainda. Próximo passo: Etapa 1 — Tensor (`checklists/01-tensor.md`).

---

## 2026-07-16 — Etapa 1 concluída: Tensor

- Implementado `Tensor` (`src/main/scala/minigpt/tensor/Tensor.scala`): construtor privado no companion object com `apply` sobrecarregado, representação `data`/`shape`/`strides` (row-major), `rank`/`size`, `get`/`index`/`unravelIndex`, `reshape`, `transpose(dim0, dim1)` (default nas duas últimas dimensões), `contiguous`, e inicializações `zeros`/`ones`/`fill`/`arange`/`randn` (Box-Muller via `LazyList` infinita de amostras gaussianas).
- Decisão de design: `Tensor` é `class` comum (não `case class`) por causa de igualdade por referência necessária para os nós do grafo de autograd (Etapa 2); documentado em `CLAUDE.md`.
- Adicionado `ScalaTest` como framework de testes (`build.sbt`), com abordagem de TDD pragmática (test-first para operações com backward, validação manual para código estrutural) documentada em `CLAUDE.md`.
- Criada suíte `TensorSpec.scala` (14 testes, em inglês) cobrindo construção, `index`/`unravelIndex`, `reshape`, `transpose`, `contiguous`, `get`. Processo revelou e ajudou a corrigir 3 bugs reais ao longo da implementação: cálculo de strides incorreto para rank ≥ 4, `transpose` mutando o tensor original em vez de retornar view nova, e `unravelIndex`/`contiguous` usando strides errados (do tensor atual em vez de canônicos do shape) — todos corrigidos, suíte passa 14/14.
- Validação manual adicional: `randn` com 100k amostras deu média ≈ -0.0037 e variância ≈ 0.999 (consistente com N(0,1)); `toString` verificado visualmente com truncamento (`edgeItems=3`) para tensores grandes.
- Todos os itens de `checklists/01-tensor.md` marcados como concluídos. Próximo passo: Etapa 2 — Autograd (`checklists/02-autograd.md`).

---

## 2026-07-16 — Etapa 2 em andamento: `topologicalSort` validado isoladamente

- Antes de mexer no `Tensor`, foi criado um harness mínimo em `src/test/scala/minigpt/tensor/TopologicalSortSpec.scala`: uma classe `Node` (igualdade por referência, mesmo raciocínio do `Tensor`) com só `name`/`previous`, pra desenvolver e validar `topologicalSort` isolado da complexidade do `Tensor` (shape, strides, data, gradient).
- TDD revelou 2 bugs reais numa primeira versão tail-recursive do algoritmo (pilha de trabalho explícita simulando DFS):
  1. **Loop infinito garantido**: a recursão concatenava `parents ::: nodes` em vez de `parents ::: nodes.tail`, então o nó no topo da pilha nunca era removido — travava em qualquer grafo com pelo menos uma folha (ou seja, todos).
  2. **Duplicação/loop infinito com dependência não-folha compartilhada**: numa segunda versão (com sets `visited`/`added` separados), o ramo que emitia um nó "pronto" (`parents.forall(visited.contains)`) esquecia de marcar esse nó como `visited`/`added`. Funcionava para folhas compartilhadas (testado no grafo diamante clássico), mas quando o nó compartilhado por dois consumidores era ele mesmo um nó intermediário (não-folha) — padrão comum em conexões residuais (Etapa 14) — o algoritmo reemitia esse nó e seus consumidores indefinidamente, sem nunca satisfazer a condição de prontidão do nó raiz.
- Fix: unificar os dois ramos (folha é só o caso particular de `parents.forall(...)` vacuamente verdadeiro numa lista vazia) e aplicar a mesma lógica de dedup (`added`) e marcação (`visited`) em ambos.
- Suíte final: 7/7 testes verdes, incluindo checagem de stack-safety com uma cadeia de 100.000 nós (a versão tail-recursive não estoura a pilha).
- Próximo passo: portar essa lógica validada para dentro de `object Tensor` (trocando `Node` por `Tensor`), adicionar os demais campos de autograd faltantes (`_prev` e `_backward` já existem desde a Etapa 1; falta o campo `grad` propriamente dito — hoje `gradient` já existe mas não é usado por nenhuma operação), implementar as primeiras operações (`+`, `*`) registrando `_prev`/`_backward`, e então `backward()`, `zeroGrad()` e `noGrad`.

---

## 2026-07-16 — Etapa 2 em andamento: `add`/`mul` e `backward()` funcionando

- `topologicalSort` portado de `Node`/`TopoSort` (harness de teste) para `object Tensor`, usando `Tensor` no lugar de `Node` diretamente.
- Construtor do `Tensor` mudou de `private[tensor]` para `private[minigpt]`, permitindo que o novo pacote `ops` (`src/main/scala/minigpt/ops/TensorOps.scala`) construa tensores de saída com `previous`/`_backward` customizados sem precisar de uma fábrica extra. `previous`/`_backward` continuam `private[tensor]` — só leitura de dentro do pacote `tensor`, então `ops` só constrói, nunca inspeciona o grafo de um tensor existente.
- Implementadas as extensions `+` e `*` (element-wise, mesma shape) em `TensorOps.scala`. Auto-referência do tensor de saída dentro da própria closure de `_backward` resolvida capturando o array `grad` local (criado antes do `Tensor(...)`) em vez do tensor em si — evita precisar de `lazy val`.
- `backward()` implementado como método de instância em `Tensor`: valida `size == 1` (só raiz escalar), semeia `this.gradient` com `1.0`, roda `topologicalSort(this).reverse` chamando `_backward()` em cada nó.
- TDD (seguindo a convenção do `CLAUDE.md` de testar forward+backward antes de dar por pronto) revelou 3 bugs reais, todos corrigidos:
  1. `add`/`mul` iterando `(0 until t.rank)` em vez de `(0 until t.size)` no backward — só propagava gradiente pras primeiras `rank` posições do tensor, não pra todas as `size` posições.
  2. Uma fábrica intermediária (`Tensor.make` com overload de 3 args) não repassava `previous`, deixando o tensor resultante sem ancestrais no grafo — quebraria qualquer encadeamento de operações (`(a+b)+c`), embora passasse despercebido num teste de um só nível. Resolvido chamando o construtor de `Tensor` direto com `previous` explícito.
  3. `backward()` inicial não semeava `this.gradient` com `1.0` antes de propagar — todo `_backward()` corria sobre um array de gradiente zerado, então nada se propagava (bug silencioso: o código rodava sem erro, só não fazia nada). Corrigido com `this.gradient.mapInPlace(_ => 1.0)`.
- Validado manualmente (grafo `c = a+b`, `d = c*a`, reusando `a`): `d.backward()` produz `a.gradient = 7.0` (soma da contribuição direta via `mul` com a contribuição via `add`→`c`), `b.gradient = 2.0`, `c.gradient = 2.0` — bate com a regra da cadeia calculada à mão. É exatamente a validação de "grafo com reuso" pedida pelo checklist da Etapa 2.
- Falta pra fechar a Etapa 2: `zeroGrad()` e o mecanismo `noGrad`/`gradEnabled`. `previous`/`_backward` de `reshape`/`transpose`/`contiguous` ainda repassam os do tensor original sem criar um nó novo no grafo — não é problema agora (nenhuma operação real os encadeia ainda), mas vale revisitar antes da Etapa 3 usar `transpose` de verdade (matmul em batch).

---

## 2026-07-16 — Etapa 2 concluída: `zeroGrad`, `noGrad`/`gradEnabled`

- `zeroGrad()` implementado como método de instância: percorre `topologicalSort(this)` (ordem não importa aqui) e zera (`mapInPlace(_ => 0.0)`) o `gradient` de todo tensor com `requiresGradient = true`. Validado com um grafo `a + b` onde só `a` tem `requiresGradient = true`: depois de `backward()` os dois tinham gradiente `1.0`; depois de `zeroGrad()`, só `a` foi zerado, `b` continuou `1.0` — confirma que o filtro funciona.
- Notado en passant: nesse ponto ainda não existia API pública pra criar um tensor com `requiresGradient = true` (`Tensor.make` não expunha esse parâmetro) — só dava pra fazer via `new Tensor(...)` direto. Resolvido pouco depois com um overload `Tensor.make(data, shape, requiresGradient)`.
- `gradEnabled: Boolean` (var global em `object Tensor`) e `noGrad[T](block: => T): T` implementados. `noGrad` salva o valor anterior de `gradEnabled`, desliga, roda o bloco (parâmetro por nome — crucial, porque por valor o bloco rodaria antes da flag ser desligada) dentro de um `try`, e restaura o valor anterior num `finally` — suporta aninhamento corretamente.
- TDD revelou 2 bugs reais nessa parte também:
  1. `gradEnabled` nasceu com default `false` em vez de `true` — como não existia (e ainda não existe) um jeito de ligar a flag além de `noGrad` (que só desliga), o autograd inteiro ficava morto por padrão, fora de qualquer bloco especial. Confirmado rodando um `a + b` comum: `previous` saía vazio mesmo sem nenhum `noGrad` por perto. Corrigido trocando o default pra `true`.
  2. Em `add`/`mul`, `val reqGrad = t1.requiresGradient || t2.requiresGradient && Tensor.gradEnabled` — `&&` tem precedência maior que `||` em Scala, então isso lia como `t1.requiresGradient || (t2.requiresGradient && gradEnabled)`: se `t1` já tivesse `requiresGradient = true`, o resultado saía `true` mesmo dentro de um `noGrad`, ignorando a flag global. Corrigido com parênteses explícitos: `(t1.requiresGradient || t2.requiresGradient) && Tensor.gradEnabled`.
- Validado: fora de `noGrad`, `a + b` constrói grafo e propaga normalmente; dentro de `noGrad`, mesmo com um input `requiresGradient = true`, o resultado sai com `previous` vazio e `requiresGradient = false`; depois do bloco, `gradEnabled` volta pro valor anterior e o comportamento normal retorna.
- Todos os itens de `checklists/02-autograd.md` marcados como concluídos. Etapa 2 fechada. Próximo passo: Etapa 3 — Operações Elementares (`checklists/03-operacoes-elementares.md`) — `add`/`mul` já existem como MVP validado, falta expandir pra `sub`/`div`/`pow`/`neg`/`exp`/`log`/reduções (`sum`/`mean`/`max`)/broadcasting/`transpose`+`reshape` com backward de verdade/`matmul`/`clamp`. Vale revisitar ali o ponto notado na Etapa 2 sobre `reshape`/`transpose` não criarem nó próprio no grafo.
- Criada a suíte permanente `src/test/scala/minigpt/ops/TensorOpsSpec.scala` (12 testes), cobrindo `add`/`mul` (forward, backward, rejeição de shapes diferentes), `backward()` com reuso de tensor, `zeroGrad()` e `noGrad`/`gradEnabled` (incluindo restauração do estado mesmo com exceção no meio do bloco). Até aqui essas operações só tinham sido validadas com scripts descartáveis durante a revisão — não ficavam como regressão. Segue a convenção de TDD do `CLAUDE.md` (teste obrigatório pra qualquer operação com forward *e* backward) e o padrão "espelha `main`" pra estrutura de `src/test`. Suíte completa: 33 testes (14 Tensor + 7 TopologicalSort + 12 TensorOps), todos verdes.

---

## 2026-07-19 — Etapa 3 em andamento: `sub`/`div`/`neg`/`pow`

- Usuário implementou as 4 operações element-wise restantes em `TensorOps.scala`: `sub` (`da += grad`, `db -= grad`), `div` (regra do quociente: `da += grad/b`, `db += grad·(-a/b²)`), `neg` (`da += -grad`) e `pow(n)` (regra da potência: `da += grad·n·a^(n-1)`). Revisão confirmou a matemática de todas as 4 corretas, incluindo precedência de operadores em `div`.
- Discussão sobre divisão por zero: decisão de **não** adicionar guard/epsilon no `div` genérico — deixar a semântica IEEE 754 de `Double` fluir naturalmente (`x/0.0 = Infinity`, `0.0/0.0 = NaN`), mesmo comportamento de PyTorch/NumPy pra divisão elementar. Estabilidade numérica fica reservada pra onde already faz sentido no roadmap: `log` (Etapa 3, clipping/log-sum-exp) e `LayerNorm` (Etapa 10, `ε=1e-5` somado à variância).
- Adicionados 11 testes em `TensorOpsSpec.scala` cobrindo forward/backward/rejeição-de-shape de `sub`/`div`/`neg`/`pow`, incluindo um teste explícito de divisão por zero (`Infinity`/`NaN` documentados como comportamento intencional, não bug). Suíte completa: 23 testes em `TensorOpsSpec`, todos verdes.
- Novo item permanente em `CLAUDE.md` (seção Testes): sempre criar e rodar os testes imediatamente ao validar uma funcionalidade nova, mesmo fora de fluxo test-first — motivado por essas 4 operações terem sido implementadas antes de ganhar cobertura.
- `checklists/03-operacoes-elementares.md`: seção "Element-wise" completa (`add`/`sub`/`mul`/`div`/`pow`/`neg`), status "Em andamento". Falta: funções transcendentais (`exp`/`log`), reduções (`sum`/`mean`/`max`), broadcasting, `transpose`/`reshape` com backward próprio, `matmul` (2D e batch), `clamp`, e o gradient check manual final da etapa.

---

## 2026-07-20 — Etapa 3 em andamento: funções transcendentais e reduções escalares

- Usuário implementou `exp`/`log` (funções transcendentais) e `sum`/`mean`/`max` (reduções escalares, sem suporte a `dim` ainda) em `TensorOps.scala`.
- Revisão encontrou 3 bugs reais antes de qualquer teste existir pra essas operações:
  1. **`sum`/`mean`/`max` construíam o tensor de saída com `t1.shape`/`t1.strides` (shape do tensor de *entrada*) em vez de `Array(1)`/`Array(1)`** — como o `data` de saída dessas reduções tem sempre tamanho 1, isso violava o assert `data.length == shape.product` do construtor de `Tensor` e estourava exceção pra qualquer entrada com mais de um elemento. Só não quebrava em teste manual porque ninguém tinha testado ainda com tensores de tamanho > 1.
  2. **`max` indexava `grad(i)` no backward em vez de `grad(0)`** — como o `grad` do tensor escalar de saída tem tamanho 1, `grad(i)` com `i == maxIdx != 0` estourava `ArrayIndexOutOfBoundsException`. `sum`/`mean` já usavam `grad(0)` corretamente; foi inconsistência isolada do `max`.
  3. **`exp` usava `t1.data(i)` (a entrada `x`) no backward em vez de `data(i)` (a saída `eˣ`)** — bug matemático silencioso: a derivada de `eˣ` é o próprio resultado, não a entrada. Coincidentemente bateria só em `x = 1` (onde `e¹ = 1`), o que dificultaria notar em teste manual pontual. Motivo exato do comentário do checklist ("guardar `c.data` para reuso no backward").
- Todos os 3 corrigidos pelo usuário e revalidados. `log` já estava correto (`1/x` depende da entrada mesmo, diferente de `exp`).
- Criados 16 testes novos em `TensorOpsSpec.scala` cobrindo forward/backward de `exp`/`log`/`sum`/`mean`/`max`, incluindo: regressão explícita de `sum` com tensor rank > 1 (pra travar o bug #1 caso reapareça) e um teste de empate no `max` confirmando a convenção decidida em conversa anterior (gradiente vai só pro primeiro índice em caso de empate, mesma semântica do PyTorch). Suíte completa: 56 testes (14 Tensor + 7 TopologicalSort + 35 TensorOps), todos verdes.
- `checklists/03-operacoes-elementares.md`: seções "Funções transcendentais" e as versões escalares de "Reduções" (`sum`/`mean`/`max`) marcadas como concluídas. Falta ainda: `sum(a, dim)`/`mean(a, dim)` (redução por dimensão), broadcasting (`unbroadcast`), `transpose`/`reshape` com backward próprio, `matmul` (2D e batch), `clamp`, e o gradient check manual final da etapa.

---

## 2026-07-20 — Etapa 3 em andamento: `broadcastTo`, e reestruturação `Shape`/`Strides` pra `final class`

- Usuário implementou `broadcastTo(newShape)` em `Tensor` em 3 iterações, cada uma revisada e corrigida antes da próxima:
  1. Primeira versão só recalculava strides canônicas do shape alvo (`getCanonicalStrides(shapeB)`) — ignorava que `data` continua do tamanho *original* (menor); estourava o assert de tamanho do `Tensor` pra qualquer broadcast real.
  2. Segunda versão fazia padding de rank (`leftPad` no shape com `1`, nas strides com `0`) mas esquecia de *esticar* o tamanho das dimensões que já eram `1` (novas ou pré-existentes) — ficava um no-op quando não havia diferença de rank (ex.: `(1,3).broadcastTo((2,3))` não mudava nada).
  3. Versão final: depois do padding de rank, percorre `shapeB` comparando com o shape paddado e, onde o tamanho é `1` mas `shapeB` pede mais, troca o tamanho e zera a stride — cobrindo os dois casos (dimensão nova do padding e dimensão pré-existente).
- Isso expôs que o invariante original do construtor de `Tensor` (`data.length == shape.product`) não dava conta de views com stride `0` (que têm *menos* dado físico que `shape.product` sugere). Trocado por um invariante geral compatível com qualquer view strided: `Σᵢ (shape[i]-1)·strides[i] < data.length` (maior índice de memória alcançável precisa caber em `data`) — cobre `reshape`/`transpose`/`contiguous` (onde sempre valia por construção) e `broadcastTo` (onde é o ponto principal) com a mesma checagem.
- Assinatura de `unbroadcast(grad: Array[Double], gradShape: Shape, targetShape: Shape): Array[Double]` decidida (ainda não implementada, corpo `???`) — função pura sobre array+shapes (não sobre `Tensor`), já que é puramente um detalhe de implementação do backward dos operadores binários.
- **Tentativa com `opaque type Shape`/`Strides = Array[Int]` abandonada por bug real de resolução de extension methods no Scala 3.** Ao introduzir os dois tipos opacos (pra parar de passar `Array[Int]` cru e evitar trocar shape por strides por engano), qualquer extension method com o mesmo nome (`apply`, `updated`, `leftPad`, `toArray`) definido tanto para `Shape` quanto pra `Strides` resolvia de forma não-confiável — às vezes acertava o tipo certo, às vezes silenciosamente tentava aplicar a extension do *outro* tipo e falhava a compilar, mesmo com os dois tipos em arquivos separados (o que se imaginava resolver o escopo de transparência). Foi reproduzido de forma consistente: dar nomes distintos (`raw` em vez de `toArray` só pra `Strides`) resolvia sempre; nomes compartilhados falhavam sempre. Conclusão prática: **não usar `opaque type` quando dois tipos opacos com a mesma representação subjacente (`Array[Int]`) vão precisar de métodos com nomes iguais** — o custo de descobrir/testar quais nomes colidem não compensa o ganho de zero-cost.
- Substituído por `final class Shape private[minigpt] (values: Array[Int])` e `final class Strides private[minigpt] (values: Array[Int])` de verdade (não `case class` — mesmo motivo do `Tensor`: `equals`/`hashCode` estruturais com campo `Array` não fazem o que parecem fazer). Com tipos nominais reais, despacho de método é resolução normal (por tipo do receptor), sem ambiguidade nenhuma — o mesmo nome (`apply`, `updated`, `leftPad`, `toArray`) em `Shape` e `Strides` funciona sem problema.
- Aproveitado pra reestruturar `src/main/scala/minigpt/`: `tensor/Tensor.scala` → `core/Tensor.scala`, junto com os novos `core/Shape.scala` e `core/Strides.scala` (Tensor e os tipos que ele usa diretamente ficam co-localizados, sem `Tensor` precisar importar de `ops` pra sua própria mecânica de shape/stride). `ops/` continua só com as *operações* (`TensorOps.scala`, `BroadcastOps.scala`) que usam os tipos de `core/`. `ops/ArrayOps.scala` (o `leftPad` genérico usado só pela primeira versão de `Shape`/`Strides`) removido — `leftPad` virou método nativo de cada uma. Testes movidos de `test/.../tensor/` pra `test/.../core/` pra espelhar `main`. `CLAUDE.md` atualizado (estrutura de diretórios e nova convenção de design documentando `Shape`/`Strides` como `final class`).
- `Shape`/`Strides` ganharam `toString` customizado (`Shape(2, 3)`, `Strides(3, 1)`) — aparece automaticamente em mensagens de assert (ex.: erro de broadcast incompatível agora imprime `Shapes Shape(2) and Shape(3) cannot be broadcasted.`).
- Validado manualmente via console (`broadcastTo` não tem suíte própria ainda): `(1,3)→(2,3)` e `(3,)→(2,3)` continuam lendo os dados corretamente repetidos após a reestruturação, e o caso incompatível continua rejeitado. Suíte automatizada (56 testes) segue toda verde.
- Pendente: testes permanentes pra `broadcastTo`/`Shape.broadcast`/`Shape`/`Strides`, corpo do `unbroadcast`, e então finalmente usar isso dentro de `add`/`sub`/`mul`/`div`.

---

## 2026-07-20 — Etapa 3 em andamento: `Gradient` como wrapper dedicado

- Discussão sobre se valia a pena o campo `gradient` (hoje `Array[Double]` cru, mutado in-place por `+=`) ganhar seu próprio tipo, no mesmo espírito de `Shape`/`Strides`. Diferença chave em relação àquele caso: o problema ali não é confundir dois `Array[Double]` (não há dois campos adjacentes do mesmo tipo com risco real de troca), é que nada impedia código de *sobrescrever* um gradiente (`t1.gradient(i) = x`) em vez de *acumular* (`+=`) — bug silencioso que quebraria a soma de contribuições em grafos com tensores reusados (o cenário coberto pelo teste "should sum contributions when a tensor is reused across the graph").
- Criado `core/Gradient.scala`: `final class Gradient private[minigpt] (values: Array[Double])`, mesmo padrão de construção restrita de `Shape`/`Strides` — mas deliberadamente mutável (é a exceção já documentada no `CLAUDE.md`). API restrita de propósito a só 3 operações: `accumulate(i, delta)` (a única forma de escrever, sempre soma), `zero()` (usado por `zeroGrad`) e `seed()` (semear `1.0` na raiz, usado por `backward`) — sem `update`/`:=` genérico, tornando a sobrescrita acidental impossível de escrever, não só malvista por convenção.
- `Tensor.gradient` mudou de `Array[Double]` pra `Gradient`; `Tensor.make`/`zeros`/`ones`/etc. passam a construir via `Gradient.zeros(n)`. Todos os 12 operadores/reduções em `TensorOps.scala` trocaram `t.gradient(i) += x` por `t.gradient.accumulate(i, x)` (e o caso de `sub`, que fazia `t2.gradient(i) -= grad(i)`, virou `t2.gradient.accumulate(i, -grad(i))` — `Gradient` não tem `-=`, só soma, então subtrair é acumular o negativo).
- Ao contrário da saga de `Shape`/`Strides` (que passou por `opaque type` → bug de resolução de extension → `final class`), essa foi direto pra `final class` e compilou sem nenhum problema já na primeira tentativa — confirma que o problema anterior era mesmo específico de `opaque type` com dois tipos de mesma representação subjacente, não de introduzir tipos wrapper em geral.
- Suíte completa (56 testes) segue verde sem nenhuma mudança nos testes — a API pública relevante pros testes (`gradient(i)`, `gradient.toList`) permaneceu idêntica via `apply`/`toList` no novo tipo.

---

## 2026-07-20 — Etapa 3 em andamento: `unbroadcast` implementado, `Shape.index`, e suíte permanente pra toda a infra de broadcasting

- `unbroadcast(grad, gradShape, targetShape)` implementado em `BroadcastOps.scala`: pra cada posição `i` de `grad`, desenrola o multi-índice via `gradShape.unravelIndex`, alinha `targetShape` ao rank de `gradShape` (mesmo left-padding do `broadcastTo`), zera a coordenada em toda dimensão onde o shape alinhado é `1` (a dimensão foi broadcastada), e reindexação no `targetShape` original pra acumular. Validado manualmente contra os dois exemplos numéricos do `teoria/03-operacoes-elementares.md` (`M(3,4)+v(4,)` → soma e remove dimensão; `M(3,4)+b(3,1)` → soma e mantém) mais um caso de identidade (sem broadcasting).
- Nesse processo, extraído `Shape.index(dims*)` (inverso de `unravelIndex`, usando strides *canônicas* — diferente do `Tensor.index`, que usa as strides *reais* daquele tensor, podem não ser canônicas após `transpose`/`broadcastTo`) — simplificou o cálculo de índice linear dentro do `unbroadcast`, que antes fazia isso manualmente com um `foldLeft`.
- Usuário pediu duas mudanças permanentes na forma de colaborar (registradas em `CLAUDE.md` e na memória entre sessões): (1) sinalizar proativamente pontos de legibilidade/idiomação durante revisão, não só bugs — foi exatamente o caso do `Shape.index`/`unravelIndex`; (2) escrever testes permanentes de qualquer funcionalidade nova proativamente, sem esperar pedido, só avisando antes o quê e por quê.
- Primeira aplicação dessa segunda autorização: criadas 4 suítes novas — `ShapeSpec` (17 testes: indexação, `updated`/`leftPad`/`zip`, `canonicalStrides`, `unravelIndex`/`index` como inversos, `Shape.broadcast` válido/inválido), `StridesSpec` (5 testes, básico equivalente), `GradientSpec` (6 testes: `accumulate` soma em vez de sobrescrever, `zero`, `seed`), `BroadcastOpsSpec` (4 testes: os 3 casos numéricos validados manualmente + um caso de colapso total pra escalar). Mais 3 testes de `broadcastTo` adicionados em `TensorSpec`. Nenhum teste pré-existente precisou mudar. Suíte completa: 83 testes, todos verdes.
- `checklists/03-operacoes-elementares.md`: item de `unbroadcast` marcado como concluído. Falta: ligar o broadcasting dentro de `add`/`sub`/`mul`/`div` (hoje ainda rejeitam shapes diferentes), `sum(a,dim)`/`mean(a,dim)`, `transpose`/`reshape` com backward próprio, `matmul`, `clamp`, gradient check manual.

---

## 2026-07-21 — Etapa 3 em andamento: broadcasting ligado em `add` (primeira operação completa)

- Usuário implementou o broadcasting dentro de `+` em 3 iterações, cada uma revisada:
  1. Primeira versão calculava `shape = Shape.broadcast(t1.shape, t2.shape)` mas o forward continuava fazendo `t1.data.zip(t2.data)` nos arrays crus — mesmo bug de sempre, ignorando o shape calculado. Estourava o invariante de `Tensor` assim que os shapes fossem realmente diferentes.
  2. Segunda versão passou a usar `t1.broadcastTo(shape)`/`t2.broadcastTo(shape)`, mas lia `t1B.data(linearIdx)` direto — indexando o array cru (menor, compartilhado) com o índice linear do shape de *saída*. Bug sutil: `shape.index(shape.unravelIndex(i))` sempre devolve `i` de volta (são inversos por definição, testado no `ShapeSpec`), então essa "conversão" não fazia nada — o problema real era ler `data` diretamente em vez de `t1B.get(multiIdx*)`, que é o método que de fato aplica as strides (incluindo os `0`s) da view broadcastada.
  3. Versão final: forward usa `t1B.get(multiIdx*) + t2B.get(multiIdx*)` (lê corretamente através da view); backward chama `Broadcast.unbroadcast(grad, shape, t1.shape)`/`(..., t2.shape)` e acumula cada resultado com o próprio tamanho do operando (`t1.size`/`t2.size`), não mais o tamanho da saída.
- Validado manualmente e depois com testes permanentes novos em `TensorOpsSpec` (3 casos): broadcasting com diferença de rank (`(3,)` com `(2,3)`), sem diferença de rank mas com dimensão `1` (`(1,3)` com `(2,3)`), e o backward correspondente (`a` usado 2x → gradiente `[2,2,2]`; `b`, sem broadcasting, gradiente tudo `1`). Suíte completa: 93 testes, todos verdes.
- `sub`/`mul`/`div` ainda não foram atualizados — continuam com o assert estrito de shapes iguais. Mesmo padrão de 3 mudanças (`Shape.broadcast` pro shape de saída, `broadcastTo` + `.get` no forward, `unbroadcast` no backward) deve se repetir pra elas.

---

## 2026-07-21 — `teoria/` reestruturada em pastas por etapa, com guias visuais (`.html`)

- Usuário pediu documentação teórica mais detalhada e ilustrativa, no estilo de um guia visual com diagramas que havia sido gerado como Artifact durante a conversa (explicando broadcasting/`unbroadcast`). Decisão: `teoria/` deixa de ser um `.md` por etapa e vira uma **pasta** por etapa, permitindo acompanhar a prosa de um `.md` com `.html` autocontidos (SVG inline, sem dependências externas) pros assuntos mais visuais/complexos — convenção documentada no `CLAUDE.md`.
- Estrutura resultante:
  - `teoria/01-tensor/01-tensor.md` + `strides-e-memoria.html` (layout de memória row-major, fórmula de stride, transpose como troca de metadados sem copiar dado, por que a transposta deixa de ser contígua).
  - `teoria/02-autograd/02-autograd.md` + `grafo-computacional.html` (o grafo como DAG, por que a ordenação topológica é necessária — grafo diamante —, convergência/soma de gradiente em nós reusados).
  - `teoria/03-operacoes-elementares/03-operacoes-elementares.md` + `broadcasting.html` (já existente, só movido pra dentro do projeto) + `matmul.html` novo (produto escalar linha·coluna, derivação de `dA=dC@Bᵀ`/`dB=Aᵀ@dC`, batch matmul) — essa etapa acumulou os dois `.html` por ser a mais densa/visual até agora.
- Os `.md` de cada etapa ganharam uma linha no topo (ou na seção relevante) apontando pro `.html` companheiro — a prosa não duplica o conteúdo visual, só referencia.
- Os três `.html` compartilham o mesmo sistema de design (paleta "blueprint/blueline" — navio escuro com linhas claras no dark mode, papel claro com tinta azul no light mode, tipografia mono pra rótulos/dados e serifada pro corpo de texto), pra ficar consistente como uma "coleção" de referência, não documentos avulsos.

---

## 2026-07-21 — Etapa 3: broadcasting ligado em `mul`/`sub`/`div` (todas as 4 operações binárias concluídas)

- Usuário generalizou o padrão já validado em `add` pras outras 3 operações binárias, extraindo um helper compartilhado `forward(t1, t2, targetShape)(op)` (usado pelos 4) que constrói `t1.broadcastTo`/`t2.broadcastTo` e aplica `op` posição a posição via `.get(multiIdx*)`.
- **`sub`**: como a derivada local é constante (`∂(a-b)/∂a=1`, `∂(a-b)/∂b=-1`), não precisa reler os operandos — só roteia `grad` por `unbroadcast` duas vezes e nega o resultado de `t2` (negação é linear, `-(Σxᵢ) = Σ(-xᵢ)`, então negar antes ou depois do `unbroadcast` dá o mesmo resultado).
- **`mul`/`div`**: como a derivada local depende do *valor* do outro operando (`∂(a·b)/∂a=b`, `∂(a/b)/∂a=1/b`, `∂(a/b)/∂b=-a/b²`), o backward primeiro calcula "derivada local × grad" inteiramente em espaço de *saída* (lendo `t1B`/`t2B` — as views broadcastadas — em vez de `t1.data`/`t2.data` crus, senão os índices não bateriam pra shapes diferentes) e só depois chama `unbroadcast` pra colapsar cada resultado pro shape original do respectivo operando. Essa foi uma dúvida real do usuário durante a implementação — registrada porque não é óbvia: o "quando" broadcastar/desbroadcastar importa (multiplicar em espaço de saída, resumir depois — nunca o contrário).
- `Gradient` ganhou `get` (devolve uma cópia do array via `.clone()` — nunca a referência interna, preservando a garantia de que só `accumulate` escreve) e `accumulateAll(deltas)` (soma um array inteiro de uma vez, por baixo dos panos só chama `accumulate` em loop — não abre brecha nova pra sobrescrita).
- `Shape` também ganhou um ajuste de nomenclatura: `length`/`product` (métodos) viraram `rank`/`size` (`val`s calculados uma vez na construção) — agora consistente com os nomes que `Tensor` já usava (`Tensor.rank`/`Tensor.size`).
- Criados 6 testes novos (2 cada em `*`, `-`, `/`) cobrindo forward com broadcasting genuíno e o backward correspondente, com valores calculados à mão e conferidos batendo exatamente. Suíte completa: 99 testes, todos verdes.
- Com isso, as 4 operações binárias (`+`/`-`/`*`/`/`) suportam broadcasting completo (forward e backward). Falta ainda: `sum(a,dim)`/`mean(a,dim)`, `transpose`/`reshape` com backward próprio, `matmul`, `clamp`, gradient check manual.

---

## 2026-07-21 — Primeira revisão periódica de padrão de código (nova autorização permanente)

- Usuário pediu uma revisão rigorosa de legibilidade/idiomação/enxugamento do código acumulado até aqui, e autorizou Claude a fazer esse tipo de revisão periodicamente daqui pra frente sem esperar pedido (registrado como autorização permanente no `CLAUDE.md`, seção "Como colaborar").
- Executada via 4 agentes em paralelo (reuso, simplificação, eficiência, altitude), cada um revisando os 6 arquivos de `core`/`ops` de forma independente — sem repositório git aqui, então a revisão foi por arquivo inteiro, não por diff.
- Aplicado (achados com convergência entre vários agentes = prioridade alta):
  - `Shape.canonicalStrides` virou `lazy val` (era recalculado do zero em todo `unravelIndex`/`index`, chamados por elemento em todo forward/backward de toda operação) + implementação trocada pra `scanRight` (mais direta que `reverse`+`foldLeft`+`prepend`).
  - `Shape.linearIndex(dims, strides)` extraído, compartilhado por `Tensor.index` (usa as strides *reais*, possivelmente não-canônicas após `transpose`/`broadcastTo`) e `Shape.index` (sempre `canonicalStrides`) — eliminou duplicação de validação.
  - `Shape`/`Strides` ganharam `padTo(targetRank, value)`, consolidando a lógica de "calcular padding de rank e aplicar left-pad" que estava reimplementada em `Shape.broadcast`, `Tensor.broadcastTo` e `BroadcastOps.unbroadcast`.
  - `Tensor.broadcastTo`: o `foldLeft` com tupla acumuladora foi trocado por dois `.map` independentes (cada dimensão do broadcast é computada isoladamente, não precisa de estado encadeado).
  - `Tensor.contiguous`: `Array.ofDim`+`foreach`+`.update` imperativo virou `Array.tabulate` computando o índice remapeado uma vez (reaproveitado pra `data` e `gradient`).
  - `Tensor.make`: dois overloads viraram um com parâmetro default; `zeros`/`ones` passaram a delegar pra `fill`.
  - Maior mudança: `TensorOps.scala` tinha ~190 linhas de scaffolding quase idêntico repetido em `+`/`-`/`*`/`/`, `neg`/`pow`/`exp`/`log`, e `sum`/`mean`/`max`. Extraídas 3 fábricas privadas — `binary` (recebe `op` + duas funções `(a,b,upstream) => delta` pra cada operando), `unary` (recebe `op` + `(x, fx) => delta`) e `reduce` (recebe o valor reduzido + `i => contribuição`) — os 11 operadores viraram one-liners. De bônus, isso resolveu um achado de eficiência separado: `*`/`/` reconstruíam as views broadcastadas (`t1B`/`t2B`) no backward mesmo já tendo sido criadas no forward; agora a closure de `binary` captura as mesmas instâncias.
  - `max`: fazia `.max` seguido de `.indexOf` (2 passadas no array); virou 1 `foldLeft` só, preservando o desempate "primeiro índice vence".
- Deixado de fora deliberadamente (achados reais, mas custo/risco não compensava agora — critério do próprio processo de revisão, que autoriza pular achados que mudariam comportamento ou não valem a complexidade):
  - Reescrever `topologicalSort` pra DFS recursivo simples (sugestão de um agente) — a versão atual é deliberadamente stack-safe via pilha explícita/tail-recursion, comprovado por um teste com cadeia de 100k nós; a troca reintroduziria risco real de stack overflow.
  - Substituir os `zip`/`map`/`sum` por `while` loops imperativos — o projeto usa estilo funcional de coleções consistentemente, e o ganho de performance é irrelevante na escala deste projeto de estudo (tensores pequenos).
  - Fundir as duas chamadas de `unbroadcast` por operação binária numa única passada — adicionaria superfície de API nova (retornar dois `Gradient`s ou aceitar dois alvos) por um ganho marginal.
  - Classe-base compartilhada pra `Shape`/`Strides` (ambas são wrappers sobre `Array[Int]` com `apply`/`toArray`/`mkString`/`updated`/`leftPad` idênticos) — a duplicação restante é ~5 one-liners cada, e `Shape` tem bem mais API própria (`canonicalStrides`/`index`/`unravelIndex`/`rank`/`size`/...) que `Strides`; não justifica introduzir hierarquia de herança pra isso ainda.
- Suíte completa (99 testes) validada sem nenhuma mudança de teste necessária — todos os refactors preservaram o comportamento exato (matemática de gradiente, mensagens de assert, semântica de desempate do `max`, stack-safety do `topologicalSort`).

---

## 2026-07-21 — `teoria/` enriquecida com exemplos numéricos variados e novo mecanismo de sincronização

- Revisão de qualidade pedida pelo usuário sobre `teoria/` (etapas 1-3): matemática conferida linha a linha (Box-Muller, regra da cadeia, `dA=dC@Bᵀ`/`dB=Aᵀ@dC`, `unbroadcast`) sem nenhum erro encontrado, mas duas seções de `03-operacoes-elementares.md` (Transpose/Reshape e Clamp) destoavam do resto do documento por não fecharem com exemplo numérico verificado, ao contrário de todo o resto do arquivo.
- Seção de Transpose/Reshape ganhou exemplos numéricos completos (reshape e transpose, valores distintos, verificação posição-a-posição) mais link direto pro `strides-e-memoria.html` da Etapa 1 (mesmo mecanismo de view sem cópia). Seção de Clamp ganhou exemplo cobrindo os três regimes (dentro do intervalo, saturado, exatamente na borda).
- `sum(a, dim)`/`mean(a, dim)` também ganharam exemplo teórico próprio nessa revisão, com valores distintos (`A=[[1,4,2],[5,0,3]]`, `dC=[10,100]`) propositalmente diferentes entre si — motivado pela constatação de que os exemplos de broadcasting anteriores usavam `dC` todo `1`, o que esconde que cada posição de saída carrega um valor específico.
- Duas mudanças permanentes registradas no `CLAUDE.md`: (1) toda subseção de `teoria/*.md` agora exige exemplo numérico verificado, com preferência por valores distintos entre si (não uniformes); (2) nova autorização permanente pra sincronizar `teoria/` sempre que uma refatoração relevante acontecer ou uma seção "aspiracional" (escrita antes do código) for finalmente implementada — mesmo padrão de proatividade das outras autorizações do projeto.

---

## 2026-07-21 — Etapa 3 em andamento: `sum(a, dim)`/`mean(a, dim)` — `sum` concluído, `mean` pendente

- Usuário implementou `sum(dim: Int, keepDim: Boolean = false)` em `TensorOps.scala`, seguindo o design discutido em conversa (scatter-add: percorrer `t1` por índice lógico via `t1.shape.unravelIndex`/`t1.get`, remover ou zerar a coordenada `dim` pra achar a posição de saída via `Shape.crop`/`Shape.updated` + `Shape.index`, e acumular).
- Revisão encontrou 2 bugs reais na primeira versão, ambos corrigidos pelo usuário e revalidados:
  1. **Backward acumulava no índice de saída em vez do índice de entrada** (`t1.gradient.accumulate(outLinIdx, grad(outLinIdx))` em vez de `accumulate(i, grad(outLinIdx))`) — como `shape.size` (saída) é bem menor que `t1.size` (entrada), isso não estourava exceção, só corrompia silenciosamente: várias posições de entrada escreviam repetidamente nos mesmos poucos slots iniciais de `t1.gradient`, enquanto o resto do array ficava intocado (zero). Confirmado com trace numérico (`A=[[1,4,2],[5,0,3]]`, `dim=1`, `grad=[10,100]`): versão com bug dava `t1.gradient=[30,300,0,0,0,0]`; versão corrigida deu `[10,10,10,100,100,100]`, batendo com a teoria.
  2. **`reqGrad` usava `||` em vez de `&&`** (`t1.requiresGradient || Tensor.gradEnabled`) — reincidência exata do bug de precedência de operadores já documentado e corrigido em `add`/`mul` na Etapa 2: dentro de `noGrad`, se o input já tivesse `requiresGradient=true`, o resultado saía `true` mesmo assim, ignorando a flag global.
  3. Faltava o guard `if Tensor.gradEnabled then` ao redor do corpo do backward (presente em `binary`/`unary`/`reduce`, ausente aqui) — usuário adicionou por consistência; sem ele, um tensor criado dentro de `noGrad` ainda vazaria gradiente pra `t1` se `.backward()` fosse chamado nele diretamente, já que a closure captura `t1` direto, independente de `previous`.
- Pendência de limpeza (não bloqueante, não corrigida ainda): o backward recalcula `val a = t1.get(inMultiIdx*)` sem nunca usar o valor (a derivada local de `sum` é constante `1`) — sobra do copy-paste do forward. Vale remover quando `mean(dim, keepDim)` for implementado, já que provavelmente vai reaproveitar/generalizar esse mesmo bloco.
- Criados 5 testes novos em `TensorOpsSpec.scala` (`sum(dim)` com `dim=1` e `dim=0`, `keepDim=true`, e 2 testes de backward usando pesos distintos por fatia — `weights=[10,100]` multiplicados pelo resultado antes do `.sum` final — pra confirmar que cada posição de entrada recebe o gradiente da *sua própria* fatia de saída, não uma constante genérica). Suíte completa: 86 testes, todos verdes.
- `checklists/03-operacoes-elementares.md`: item `sum(a, dim) + backward` marcado como concluído. Falta: `mean(a, dim)` (ainda `???`, deve reaproveitar a mesma lógica de mapeamento de índice, só dividindo por `t1.shape(dim)`), `transpose`/`reshape` com backward próprio, `matmul`, `clamp`, gradient check manual final da etapa.

---

## 2026-07-21 — Etapa 3 concluída (reduções por dimensão): `reduceDim` compartilhada, `mean(a, dim)` fechado

- A pedido explícito do usuário (única exceção à regra de "Claude não escreve `.scala`" — o design já tinha sido combinado em conversa e só faltava materializar), Claude implementou a fábrica privada `reduceDim(t1, dim, keepDim)(scale)` em `TensorOps.scala`, generalizando o `sum(dim, keepDim)` já validado: um único fator `scale` (`1.0` para `sum`, `1.0 / t1.shape(dim)` para `mean`) escala tanto a soma bruta no forward quanto o gradiente local no backward — consequência direta de `mean = sum/n` e da regra da cadeia (`d(mean)/dx = (1/n)·d(sum)/dx`). O mapeamento de índice de entrada → índice de saída (`outLinIdx`, remove ou zera a coordenada `dim`) virou uma função local reusada pelos dois passes, eliminando a duplicação que existia entre forward/backward do `sum(dim)` anterior — e de brinde removeu a sobra de código morto (`val a` não usado) apontada na revisão anterior.
- `sum(dim, keepDim)` e `mean(dim, keepDim)` viraram one-liners: `reduceDim(t1, dim, keepDim)(1.0)` e `reduceDim(t1, dim, keepDim)(1.0 / t1.shape(dim))`.
- Criados 5 testes novos em `TensorOpsSpec.scala` pra `mean(dim)`, espelhando a suíte de `sum(dim)` (forward em `dim=1`/`dim=0`, `keepDim=true`, e 2 testes de backward com pesos distintos por fatia confirmando o fator `1/n` — ex.: linha de 3 elementos com peso upstream `10` propaga `10/3` pra cada posição). Suíte completa: 91 testes, todos verdes (`sbt test` reexecutou só `TensorOpsSpec`, já que as demais suítes não foram tocadas por essa mudança).
- `checklists/03-operacoes-elementares.md`: seção "Reduções" 100% concluída (`sum`/`mean` escalares e por dimensão, `max`). Falta pra fechar a Etapa 3: `transpose`/`reshape` com backward próprio, `matmul` (2D e batch), `clamp`, gradient check manual final (add/mul/matmul) antes da Etapa 4.

---

## 2026-07-21 — Convenção nova: comentários em `.scala` apontam pra `teoria/` em vez de reexplicar

- Usuário propôs comentar as partes mais complexas do código, mas com receio de poluir os arquivos se a explicação ficasse extensa, e sem saber onde colocar uma explicação mais longa quando necessário. Decisão: não criar um tipo de documento novo — `teoria/` já é o lugar certo (prosa + exemplo numérico + `.html` visual pros casos complexos), e duplicar explicação em dois arquivos é o mesmo risco de divergência que já motivou a autorização de sincronização de `teoria/` registrada antes. Convenção formalizada no `CLAUDE.md`: comentários em `.scala` ficam curtos (só o "porquê" estrutural local) e apontam pra seção específica de `teoria/*.md` quando a complexidade for matemática/conceitual, em vez de reescrever a explicação ali.
- Ponteiros adicionados nos 6 pontos do código já identificados como complexos o bastante (todos já tinham cobertura em `teoria/`, só faltava a referência de volta):
  - `Tensor.broadcastTo` → `03-operacoes-elementares.md` §4 (mecanismo de view por stride 0)
  - `Tensor.getGaussianSamples` (Box-Muller) → `01-tensor.md` §4
  - `Tensor.topologicalSort` → `02-autograd.md` §4 (por que a ordem importa, caso do grafo diamante)
  - `Broadcast.unbroadcast` → `03-operacoes-elementares.md` §4 (os dois casos: dimensão nova vs. já existente)
  - `TensorOps.binary` (fábrica de `+`/`-`/`*`/`/`) → `03-operacoes-elementares.md` §1 (derivadas locais) e §4 (broadcasting)
  - `TensorOps.reduceDim` (fábrica de `sum(dim)`/`mean(dim)`) → `03-operacoes-elementares.md` §3
- Mudança só de doc comments (`/** ... */`), sem alterar nenhuma lógica — suíte completa revalidada (75 testes reexecutados nas suítes tocadas, resto em cache), todos verdes.

---

## 2026-07-21 — Etapa 3: `transpose`/`reshape` com backward próprio (fecha a pendência aberta desde a Etapa 2)

- Usuário implementou backward de verdade pra `transpose`/`reshape` em `Tensor.scala`, fechando a pendência registrada no checkpoint de 2026-07-16 ("`previous`/`_backward` de `reshape`/`transpose`/`contiguous` ainda repassam os do tensor original sem criar um nó novo no grafo... vale revisitar antes da Etapa 3 usar `transpose` de verdade"). Design discutido em conversa: como o resto do código já assume implicitamente que `gradient` de um tensor é sempre indexado canonicamente pelo seu `shape` (nunca pelas `strides` reais), o backward de `transpose` desfaz a troca de eixos via `Shape.unravelIndex`/`Shape.index` (nunca mexendo em `strides`), e `reshape` aproveita que ele não reordena elementos — mesma posição canônica nos dois shapes.
- Revisão encontrou 3 bugs reais na primeira versão, todos corrigidos pelo usuário e revalidados:
  1. **`reshape` calculava `t = this.contiguous` (corrigindo o caso de reshape sobre tensor não-contíguo, ex. depois de um `transpose`) mas nunca usava `t.data`** — o `Tensor(...)` de saída continuava construído com o `data` físico bruto do tensor original. Confirmado com trace: `A=[[1,2,3],[4,5,6]]` transposto e depois `reshape(6)` dava `[1,2,3,4,5,6]` (o array físico, errado) em vez de `[1,4,2,5,3,6]` (a ordem lógica real). Fix: usar `t.data`.
  2. **Backward de `reshape` acumulava em `t.gradient` (o array do tensor `contiguous` auxiliar, órfão — nunca alcançado por `topologicalSort`) em vez de `this.gradient`** — qualquer gradiente que chegasse num `reshape` era descartado silenciosamente, nunca propagando pro tensor real. Fix: acumular em `gradient` (= `this.gradient`).
  3. **`reqGrad` nos dois métodos não multiplicava por `Tensor.gradEnabled`** (`prev` já tinha sido corrigido pra respeitar a flag, mas o `requiresGradient` passado pro construtor continuava cru) — reincidência do mesmo bug de precedência/composição já visto em `add`/`mul` (Etapa 2) e no `sum(dim)` original.
- `transpose` já nasceu matematicamente correto na primeira versão (só com o bug #3 de `reqGrad`) — a lógica de desfazer a troca de eixos bateu exatamente com o design discutido e com o exemplo da teoria (`dA[i,j]=dC[j,i]`).
- Criados 6 testes novos em `TensorSpec.scala`: `reshape` sobre tensor não-contíguo (regressão do bug #1, com `A.transpose().reshape(6)`), backward de `reshape` com valores distintos por posição (seedando `r.gradient` diretamente e chamando `r._backward()`, sem precisar de `ops` — mantém `TensorSpec` testando só `core`), backward de `transpose` (mesmo `A=[[1,4,2],[5,0,3]]` da teoria, `dC=[[10,20],[30,40],[50,60]]` → `dA=[[10,30,50],[20,40,60]]`, batendo com a seção 5 de `teoria/03-operacoes-elementares.md`), e `noGrad` pros dois (`requiresGradient=false` e `previous` vazio). Suíte completa: 97 testes, todos verdes.
- `checklists/03-operacoes-elementares.md`: seção "Transpose e Reshape" concluída. Falta pra fechar a Etapa 3: `matmul` (2D e batch), `clamp`, gradient check manual final (add/mul/matmul) antes da Etapa 4.

---

## 2026-07-22 — `teoria/`: `exercicios.html` (quiz de múltipla escolha) pras Etapas 1-3

- Usuário propôs complementar `teoria/` com exercícios interativos de múltipla escolha (níveis fácil/médio/difícil/desafio, feedback imediato), pra reforçar o entendimento além da leitura passiva do `.md`/`.html` visual. Formato combinado: um `exercicios.html` autocontido por pasta de `teoria/` (mesma paleta blueprint/blueline, sem dependências externas), baseline de 10 questões por etapa, mais em etapas mais densas. Nova autorização permanente registrada em `CLAUDE.md`: gerar `exercicios.html` junto sempre que uma pasta nova de `teoria/` for escrita, sem esperar pedido.
- Etapa 1 (Tensor) usada como piloto e validada pelo usuário antes de replicar: 10 questões cobrindo rank/shape, fórmula de stride, indexação física, `reshape`/`transpose` como view sem cópia, contiguidade e Box-Muller — incluindo uma questão de raciocínio geral (`B.index(j,i) == A.index(i,j)` após transpose) que espelha o item de validação manual do checklist da Etapa 1.
- Etapa 2 (Autograd): 10 questões cobrindo regra da cadeia, `previous`/`_backward`, por que acumular (`+=`) em vez de sobrescrever, ordenação topológica (incluindo o caso do grafo diamante com nó compartilhado não-folha), `zeroGrad`/`noGrad`, e duas questões numéricas (uma reaproveitando o grafo `c=a+b, d=c*a` do `HISTORY.md` com valores diferentes, `a=3, b=4`, pra evitar memorização; outra generalizando a soma de contribuições pra 3 caminhos em vez de 2).
- Etapa 3 (Operações Elementares): a mais densa até aqui (8 subseções no `.md`), por isso 20 questões em vez de 10, agrupadas pelas mesmas seções do `.md` (element-wise, transcendentais, reduções, broadcasting, transpose/reshape backward, matmul, batch matmul, clamp) — incluindo questões numéricas verificadas (`sum(dim)`, `transpose` backward, `matmul` 2D com `dA=dC@Bᵀ`, `unbroadcast` dos dois casos, `clamp` nos três regimes) e uma questão conceitual ligando a convenção de `transpose(dim0, dim1)` (default nas últimas duas dimensões) documentada no `CLAUDE.md` ao motivo de bastar pra batch matmul sem tocar a dimensão de batch.
- Todos os três `.md` correspondentes ganharam uma linha no topo referenciando o `exercicios.html`, junto com o(s) `.html` visual(is) já existentes.

---

## 2026-07-22 — Reestruturação: projeto dividido em módulos sbt `scalagrad`/`mini-gpt`

- Usuário propôs separar o que é mecânica genérica de tensores/autodiferenciação (hoje `core`/`ops`) do que é específico da arquitetura GPT (`nn`/`train`/etc., ainda não implementados) em dois projetos Scala isolados dentro do mesmo diretório — no espírito de como PyTorch/NumPy existem como bibliotecas separadas de qualquer modelo construído em cima. Motivação adicional: abrir espaço pra evoluir o núcleo de tensores no futuro (ex.: um backend GPU) sem acoplar isso ao código do modelo. Decisão: usar o suporte nativo de multi-projeto do sbt (dois subprojetos no mesmo `build.sbt`, cada um com suas próprias `libraryDependencies`) em vez de repositórios git separados — mesmo diretório, build isolado.
- Nome do núcleo escolhido pelo usuário: **`scalagrad`**. Timing decidido em conversa: fazer a divisão agora (logo após a Etapa 3, antes do `nn/` começar) em vez de esperar, já que `core`/`ops` ainda não tinham nenhum acoplamento com lógica específica de GPT — esperar mais teria tornado o corte mais caro.
- Escopo do módulo `scalagrad` (pacote `scalagrad.*`): `core/` (`Tensor`, `Shape`, `Strides`, `Gradient`) e `ops/` (`TensorOps`, `BroadcastOps`) — todo o código já existente até a Etapa 3 — mais `gradcheck/` (Etapa 4, ainda não implementado, mas conceitualmente genérico: verificação numérica de qualquer backward, não específico de GPT). O módulo `mini-gpt` (pacote `minigpt.*`) fica reservado pra `data/`/`nn/`/`loss/`/`optim/`/`train/`/`generate/` (Etapas 7+), com `dependsOn(scalagrad)` — ainda sem nenhum arquivo, criado incrementalmente quando a Etapa 7 começar (mesma filosofia incremental já usada em `teoria/`).
- Migração mecânica: os 13 arquivos existentes (6 `main` + 7 `test`) movidos de `src/main|test/scala/minigpt/{core,ops}/` pra `scalagrad/src/main|test/scala/scalagrad/{core,ops}/`, com `package minigpt.*` → `package scalagrad.*`, imports `minigpt.core.*`/`minigpt.ops.*` → `scalagrad.core.*`/`scalagrad.ops.*`, e o qualificador `private[minigpt]` (construtores restritos de `Tensor`/`Shape`/`Strides`/`Gradient`, ver `CLAUDE.md`) → `private[scalagrad]`. O qualificador `private[core]` (`previous`/`_backward` em `Tensor`) não precisou mudar — continua resolvendo pro pacote `core` mais próximo (`scalagrad.core`) sem ambiguidade. Nenhuma lógica/matemática foi tocada, só nomes de pacote — sem `git` neste repositório (decisão consciente do usuário, que usa backup via OneDrive), a pasta antiga foi renomeada pra `src_old_pre_scalagrad_migration/` em vez de apagada, como rede de segurança até o usuário confirmar que está tudo certo e poder removê-la manualmente.
- `build.sbt` reescrito: `scalagrad` e `miniGpt` como projetos independentes (`project.in(file(...))`, `miniGpt.dependsOn(scalagrad)`), e `root` (via `rootProject`, sintaxe do sbt 2.x já usada no `build.sbt` original) só agregando os dois (`aggregate`), sem código próprio — `sbt test`/`sbt compile` na raiz constroem/rodam os dois módulos juntos.
- Validado com `sbt test` rodando na raiz depois da migração: 114 testes, todos verdes (mesma suíte de antes — 97 + os 17 testes adicionados entre a última entrada do `HISTORY.md` e esta migração não foram tocados, só reorganizados).
- `CLAUDE.md` atualizado: nova nota no topo explicando a divisão em dois módulos, convenções de `Shape`/`Strides`/`Gradient` atualizadas pra `private[scalagrad]`/pacote `scalagrad.core`, seção de Testes explicando `sbt scalagrad/test`/`sbt miniGpt/test`, e "Estrutura de diretórios do código" reescrita mostrando as duas árvores (`scalagrad/` e `mini-gpt/`, esta última ainda planejada/vazia).
- Não tocado: `.vscode/launch.json` (referencia `mainClass: "minigpt.ops.Test"`, uma classe que não existe hoje — já estava desatualizado antes desta migração) e as pastas geradas por IDE/build tooling (`.idea/`, `.bloop/`, `.bsp/`, `.metals/`, `target/`) — devem se auto-regenerar na próxima vez que o sbt/Metals recarregar o projeto.

---

## 2026-07-22 — Ajuste de nome: submódulo `mini-gpt` → `gpt`

- Usuário notou, logo depois da migração acima, que o submódulo `mini-gpt/` duplicava o nome da própria pasta raiz do repositório (`mini-gpt/mini-gpt/...`), confuso. Discutido em conversa: renomear a pasta raiz seria bem mais disruptivo (caminho absoluto usado por esta sessão, IDE, OneDrive) do que renomear só o submódulo — decisão de mexer só no submódulo.
- Nome novo do submódulo: **`gpt`** (pacote `gpt.*`, projeto sbt `gpt`, pasta `gpt/`). Descartada a opção `transformer`: conferido `PLAN.md` em conversa — o roadmap (Etapas 8-15) constrói especificamente um GPT decoder-only autoregressivo (Etapa 15 é literalmente "Modelo GPT Completo", Etapa 19 é geração autoregressiva), sem nenhum plano documentado de variantes encoder-only/encoder-decoder. `transformer` seria um nome mais amplo do que o escopo real hoje; `gpt` fica mais honesto, e migra fácil se o escopo mudar no futuro.
- Como o módulo ainda não tinha nenhum arquivo fonte (Etapa 7 nem começou), o ajuste foi só: pasta vazia `mini-gpt/` renomeada pra `gpt/` (`rmdir`+`mkdir`, sem nada pra migrar), `build.sbt` (`lazy val miniGpt` → `gpt`, `.in(file("mini-gpt"))` → `.in(file("gpt"))`, `name := "mini-gpt"` → `name := "gpt"`), e `CLAUDE.md` atualizado (pacote, nome do módulo, árvore de diretórios, comando `sbt gpt/test`). Validado com `sbt compile` na raiz — sem erros (módulo continua sem fontes).

---

## 2026-07-22 — Diretórios e arquivos renomeados pra inglês (consistência com código já 100% inglês)

- Usuário notou que classes/objects/métodos Scala já eram inteiramente em inglês (`Tensor`, `Shape`, `reshape`, `broadcastTo`, etc.), mas nomes de diretório/arquivo ainda tinham palavras em português (`teoria/`, `exercicios.html`, `03-operacoes-elementares.md`, ...) — pediu alinhar isso por consistência. Escopo combinado em conversa: diretórios/arquivos/identificadores de código em inglês; comentários `.scala` e conteúdo dos documentos (prosa dos `.md`/`.html`, títulos como "Teoria — Etapa X") continuam em português — só os *nomes*, não o conteúdo.
- Renomeações: `teoria/` → `theory/`; dentro dela, `03-operacoes-elementares/` → `03-elementary-operations/` (e o `.md` correspondente), `strides-e-memoria.html` → `strides-and-memory.html`, `grafo-computacional.html` → `computational-graph.html`, os 3 `exercicios.html` → `exercises.html`. Em `roadmap/` e `checklists/` (mesma numeração nas duas pastas), 9 arquivos com palavra em português no nome: `03-operacoes-elementares` → `03-elementary-operations`, `05-ativacoes` → `05-activations`, `07-tokenizacao` → `07-tokenization`, `11-atencao` → `11-attention`, `14-bloco-transformer` → `14-transformer-block`, `15-modelo-gpt` → `15-gpt-model`, `17-otimizador-adamw` → `17-adamw-optimizer`, `18-loop-treinamento` → `18-training-loop`, `19-inferencia-geracao` → `19-inference-generation`.
- Referências cruzadas atualizadas em: `PLAN.md` (9 links do índice), `CLAUDE.md` (convenções, exemplos de nome de arquivo, autorizações permanentes que citavam `teoria/`/`exercicios.html`), os 3 `.md` de `theory/` (links pro próprio `.html` visual, pro `exercises.html`, e pro `roadmap/` correspondente — incluindo o link cruzado de `03-elementary-operations.md` pro `strides-and-memory.html` da Etapa 1), os 3 `exercises.html` (links de volta pro `.md`/`.html` companheiros, citados tanto no HTML quanto no texto da mensagem final de resultado), e os 6 comentários `.scala` em `scalagrad` (`Tensor.scala` ×3, `TensorOps.scala` ×2, `BroadcastOps.scala` ×1) que apontavam `// ver teoria/...`.
- Deliberadamente não tocado: `HISTORY.md` (este arquivo) — é log histórico, entradas antigas continuam citando os nomes que existiam *na época* (ex.: `teoria/03-operacoes-elementares.md` em checkpoints de 2026-07-20/21); reescrevê-las seria falsificar o registro. Título das páginas/documentos (`# Teoria — Etapa 1: Tensor`, `<title>Strides & Memória`) também não mudou — é conteúdo/prosa, fora do escopo combinado (só nomes de arquivo/diretório).
- Validado com `sbt test` na raiz: as 3 suítes cujas dependências mudaram (`TensorSpec`, `TensorOpsSpec`, `BroadcastOpsSpec` — únicas que importam os 3 arquivos `.scala` com comentário editado) recompilaram e passaram 100%; as outras 4 suítes (`ShapeSpec`, `StridesSpec`, `GradientSpec`, `TopologicalSortSpec`) não tiveram nenhuma dependência alterada, então o sbt manteve o resultado em cache do run anterior — suíte completa (114 testes) seguindo verde.

---

## 2026-07-26 — Etapa 3: `matMul2D` (forward + backward)

- Decisão de escopo discutida antes de implementar: `matmul` fica restrito a 2D e 3D (batch), sem generalizar pra N dimensões — o roadmap já não pede N-D, e o único uso futuro conhecido de tensores de rank maior (Multi-Head Attention, Etapa 12, `[B, nHeads, T, dHead]`) vai colapsar as dimensões de batch via `reshape` antes de chamar `matmul`, em vez do `matmul` entender rank arbitrário. Pelo mesmo motivo, o batch matmul (ainda não implementado) vai exigir os dois tensores com o mesmo batch — sem broadcasting da dimensão de batch (ex.: `(12,4,5) @ (1,5,4)` não vai ser aceito, mesmo sendo tecnicamente possível via a mesma view de stride 0 já usada no broadcasting elementwise).
- Também decidido: `matMul2D`/`matMul3D` ficam privados dentro de `object tensor`, com um `matmul` público único despachando pelo rank dos dois tensores (`(2,2) => matMul2D`, `(3,3) => matMul3D`, outros ranks => erro) — mesmo padrão já usado por `binary`/`unary`/`reduce`/`reduceDim` no arquivo (fábricas privadas, extension method público como única porta de entrada). `matmul` público e `matMul3D` ainda não implementados.
- Usuário implementou `matMul2D` ([TensorOps.scala](scalagrad/src/main/scala/scalagrad/ops/TensorOps.scala)) com forward `C[i,j] = Σₖ A[i,k]·B[k,j]` e backward `dA = dC @ Bᵀ`/`dB = Aᵀ @ dC` calculado via laços manuais (sem reusar `transpose`/`matMul2D` recursivamente). Revisão encontrou e corrigiu, em rodadas sucessivas:
  1. Shape de saída errada (`Shape(t1.shape(1), t1.shape(1))`, depois `Shape(t1.shape(0), t1.shape(1))` — ambas as tentativas ignoravam `t2.shape(1)`) — corrigido pelo usuário pra `Shape(t1.shape(0), t2.shape(1))` (`M × N`, não `K × K` nem `M × K`). Bug ficava mascarado sempre que `K == N` por coincidência.
  2. Forward inicial sem somar sobre `k` (`t1.get(i,j) * t2.get(j,i)`, uma multiplicação só, com `j` fazendo dois papéis incompatíveis) — corrigido pra somar `t1.get(i,k) * t2.get(k,j)` sobre `0 until t1.shape(1)`.
  3. Backward vazio na primeira versão — implementado depois em duas rodadas: usuário escreveu a primeira versão (`dA`) sozinho a partir da derivação explicada em conversa, e o `dB` simétrico; revisão trocou `for {...} yield {...}` (usado só pelo efeito colateral de `accumulate`, construindo e descartando uma coleção de `Unit` à toa) por `.foreach` aninhado, consistente com o resto do arquivo, e renomeou `dBik` → `dBkj` (nome não batia com as variáveis do loop, que são `k,j`).
- Criados 4 testes novos em `TensorOpsSpec.scala`: forward com shapes retangulares não-quadradas (`(2,3) @ (3,2)`, pra pegar regressão do bug #1 — não teria pegado com shapes quadradas), os dois `assert`s de validação (rank e dimensão interna), e backward com `dC` não-uniforme por posição (`[[1,2],[3,4]]`, obtido multiplicando `C` por um tensor de pesos antes do `sum` — técnica já usada em outros testes do arquivo pra evitar expor `_backward`, que é `private[core]` e não acessível de `scalagrad.ops`). Valores conferidos contra o exemplo numérico de `theory/03-elementary-operations/03-elementary-operations.md` §6. Suíte completa: 58 testes em `TensorOpsSpec`, todos verdes; suíte geral do módulo `scalagrad` também revalidada.
- `checklists/03-elementary-operations.md`: seção "Matrix Multiplication" com `matmul` 2D (forward + backward) marcado. Falta pra fechar a Etapa 3: `matmul` em batch (3D), `clamp`, gradient check manual final (add/mul/matmul) antes da Etapa 4.

---

## 2026-07-26 — Etapa 3: `matmul3D` + dispatcher público `matmul` (fecha "Matrix Multiplication")

- Usuário implementou `matmul3D` (forward + backward em batch, exigindo `t1.shape(0) == t2.shape(0)` — sem broadcast de batch, conforme decidido) e o dispatcher público `matmul` (`case (2,2) => matmul2D`, `case (3,3) => matmul3D`, resto lança `AssertionError`), com `matmul2D`/`matmul3D` agora privados dentro de `object tensor` — só o `matmul` é a porta de entrada pública, mesmo padrão já usado por `binary`/`unary`/`reduce`/`reduceDim`. Diferente da primeira versão do `matmul2D` (que precisou de 3 rodadas de correção), o `matmul3D` já nasceu correto — mesma fórmula do 2D aplicada por fatia de batch, com os nomes das variáveis (`dAbik`/`dBbkj`) já batendo com os índices do loop desde a primeira versão (reincidência do problema anterior evitada).
- Rename `matMul2D` → `matmul2D` (privado) quebrou a compilação da suíte de testes inteira, já que os 4 testes criados na revisão anterior chamavam `a.matMul2D(b)` diretamente — `private` num `object` não é visível nem de outros arquivos do mesmo pacote. Corrigido: os 4 testes trocados para chamar `a.matmul(b)` (API pública), reagrupados sob um único bloco `"matmul"`, e 3 testes novos adicionados: forward batched (`(2,2,2) @ (2,2,2)` com valores distintos por fatia), backward batched (pesos distintos *entre* as duas fatias de batch, não só entre posições dentro da mesma fatia, pra pegar um eventual vazamento de índice de uma fatia pra outra), rejeição de batch desalinhado sem broadcast, e rejeição de combinação de ranks não suportada pelo dispatcher.
- Suíte completa revalidada: 61 testes em `TensorOpsSpec` (54 anteriores + 7 de `matmul`), suíte geral do módulo `scalagrad` verde.
- `checklists/03-elementary-operations.md`: seção "Matrix Multiplication" completa. Falta pra fechar a Etapa 3: `clamp`, gradient check manual final (add/mul/matmul) antes da Etapa 4.

---

## 2026-07-26 — Etapa 3: `clamp` (fecha "Clamp / Clip")

- Usuário implementou `clamp(min, max)` reaproveitando a fábrica privada `unary` já existente (mesma usada por `neg`/`pow`/`exp`/`log`) — sem precisar de laços manuais, já que `clamp` não muda shape nem mistura posições, igual as outras operações unárias. Forward satura fora do intervalo; backward passa gradiente só nas posições estritamente dentro de `(min, max)`, zerando fora e nas duas bordas exatas (`a == min`/`a == max`), pela convenção do "bico" descrita em `theory/03-elementary-operations/03-elementary-operations.md` §7. Já nasceu correto — sem bugs encontrados na revisão.
- Dois ajustes de organização, aplicados na revisão: `clamp` estava posicionado entre `log` e `+` (quebrando a ordem que o arquivo vinha seguindo até então — operações agrupadas pela fábrica compartilhada, sempre anexadas no fim conforme a Etapa 3 avançava); movido pro fim do bloco `extension`, depois do `matmul`, batendo com a ordem do roadmap (`Clamp/Clip` é a última seção, logo depois de `Matrix Multiplication`). Também removido trailing whitespace recorrente (mesmo hábito já visto no `matMul2D`).
- Criados 2 testes novos em `TensorOpsSpec.scala`: forward cobrindo os três regimes num único tensor (abaixo do min, dentro, acima do max, e as duas bordas exatas — mesmos valores do exemplo de `theory/`), e backward com pesos distintos por posição (mesma técnica dos testes de `matmul`, pra evitar `dC` uniforme escondendo erro de indexação) confirmando que só as posições estritamente dentro do intervalo recebem gradiente. Suíte completa: 63 testes em `TensorOpsSpec`, todos verdes.
- `checklists/03-elementary-operations.md`: seção "Clamp / Clip" completa. **Etapa 3 fica só com um item pendente:** gradient check manual (visual) em add/mul/matmul, reservado pra rodar junto com a implementação da Etapa 4 (Gradient Check), já que a função `gradCheck` genérica que a Etapa 4 pede automatiza exatamente essa verificação — não faz sentido fazer na mão agora e depois refazer automatizado.

---

## 2026-07-26 — `theory/04-gradient-check/` escrita antes da implementação (nova convenção de fluxo)

- Usuário pediu que, daqui pra frente, a teoria de cada etapa seja sempre escrita **antes** de começar a implementação (não depois, como vinha acontecendo até aqui) — facilita entender o que precisa ser codado antes de codar. Primeira etapa a seguir esse fluxo: a 4 (Gradient Check).
- Criada a pasta `theory/04-gradient-check/` com os três arquivos de sempre: `04-gradient-check.md` (diferença central deduzida via série de Taylor mostrando por que o erro é ordem `ε²` e não `ε`; erro relativo com os três limiares; o algoritmo do `gradCheck` descrito passo a passo, incluindo o cuidado de restaurar `input.data(i)` entre perturbações e a exigência de `f` devolver escalar; cada seção fechada com exemplo numérico verificado — `f(x)=x³` em `x=2` pra diferença central/erro relativo, `f(x)=(x·x).sum()` em `x=[3.0,-2.0]` pro `gradCheck` completo), `finite-difference.html` (guia visual: secante vs. tangente, os três limiares de erro como faixas coloridas, fluxograma do algoritmo de perturbação), e `exercises.html` (10 questões fácil→desafio, cobrindo diferença central, erro relativo, e os cuidados de implementação do `gradCheck`).
- `checklists/04-gradient-check.md`: status marcado "Em andamento" — teoria pronta, implementação (`gradCheck` em `scalagrad/src/main/scala/scalagrad/gradcheck/`, conforme a estrutura já reservada no `CLAUDE.md`) ainda não começou.

---

## 2026-07-26 — Etapa 4: `Gradcheck.run` implementado

- Usuário implementou `Gradcheck.run(input, eps=1e-5)(f: Tensor => Tensor): Double` em `scalagrad/src/main/scala/scalagrad/gradcheck/Gradcheck.scala`, no pacote reservado no `CLAUDE.md` (separado de `core`, mantendo a fronteira entre mecânica de tensor e ferramentas de verificação). Decisões de design discutidas antes de implementar: retorno é só o erro relativo máximo (`Double`, não a `GradCheckResult(maxError, worstIndex)` cogitada em conversa — simplificação deliberada, mais perto do que o roadmap pede literalmente); assinatura em curry com `f` no último grupo de parâmetros, permitindo sintaxe de lambda à direita (`Gradcheck.run(input) { x => ... }`).
- Revisão encontrou 5 bugs na primeira versão (implementada dentro de `object Tensor`, antes de mover pro pacote próprio), todos corrigidos pelo usuário:
  1. **Perturbava todas as posições do tensor simultaneamente** (`input.data.map(_ + eps)`, um único forward) em vez de uma posição por vez — media a sensibilidade combinada de todos os parâmetros juntos, não o gradiente por posição. Fix: loop `(0 until input.size)`, perturbando/restaurando `input.data(i)` individualmente a cada iteração (mutação direta do array, dentro de `Tensor.noGrad`, já que `data` é `private[scalagrad]` e o pacote `gradcheck` tem acesso).
  2. **Gradiente analítico lido de `output.gradient` em vez de `input.gradient`** — `output` é a raiz escalar, sempre `[1.0]` após `seed()`; o gradiente que interessa é o que a regra da cadeia escreve em `input.gradient` durante o `backward()`.
  3. **Precedência de operador**: `(a - b) / 2 * eps` calculava `((a-b)/2)·ε` em vez de `(a-b)/(2ε)`.
  4. **`maxBy(...)._2`** devolvia o gradiente numérico do par de maior erro, não o valor do erro em si — trocado por `.map(erro).max`.
  5. **Piso do denominador usava `eps` (`1e-5`) em vez de `1e-8`** (valor fixo do roadmap/teoria) — deixava o critério de erro relativo bem menos sensível pra gradientes pequenos.
- Criados 4 testes em `GradcheckSpec.scala`: soma dos quadrados (`x=[3.0,-2.0]`, mesmo exemplo de `theory/04-gradient-check/04-gradient-check.md` §3), `matmul` composto com `sum`, `clamp` (com valores estritamente fora dos pontos exatos `min`/`max`, já que ali o gradiente numérico diverge do analítico por construção — não é bug, é limite conhecido de diferença finita perto de não-diferenciabilidades), e um teste com uma op deliberadamente quebrada (backward que ignora `2x` e devolve só o upstream) — confirma que `Gradcheck.run` realmente **denuncia** um backward incorreto (erro `≈0.83`, acima de `1e-3`), não só "carimba" qualquer coisa como correta. Suíte completa revalidada, todos verdes.
- `checklists/04-gradient-check.md`: "Gradiente numérico", "Erro relativo" e "Função genérica" completos. Falta só "Aplicação": rodar `Gradcheck.run` em todas as ops da Etapa 3 com inputs aleatórios de shapes variadas (hoje só foi validado com valores fixos em 3 operações) — item reservado pra decidir com o usuário se vale fazer agora ou seguir pra Etapa 5.

---

## 2026-07-26 — Etapa 4: varredura "Aplicação" + bug de concorrência descoberto e corrigido em `Tensor.gradEnabled`

- Criados 16 testes em `GradcheckSweepSpec.scala`, rodando `Gradcheck.run` em todas as ops da Etapa 3 (`add`/`sub`/`mul`/`div`/`pow`/`neg`/`exp`/`log`/`sum`/`mean`/`max`/`sum(dim)`/`mean(dim)`/`matmul` 2D e 3D/`clamp`) com inputs de um `Random` de seed fixa (42) — "aleatório" o suficiente pra não depender de valores escolhidos a dedo, mas determinístico entre execuções. Cuidados de domínio por operação: divisor de `div` e input de `log` mantidos longe de zero; `exp` com magnitude pequena (evita estourar a precisão da diferença central); `clamp` com faixa larga o suficiente pra cobrir região saturada e região livre, sem cair exatamente nos pontos `min`/`max` (onde o próprio método numérico diverge do analítico por causa do "bico", não por bug — já discutido na teoria). Todos os 16 passaram isoladamente (`testOnly` de uma única suíte).
- **Ao rodar a suíte inteira de uma vez** (`testOnly *`, todas as suítes juntas), 45 testes de outras suítes (`TensorSpec`, `TensorOpsSpec`) começaram a falhar de forma não determinística — gradientes vindo zerados em testes que sempre passaram antes. Isolado o problema: `TensorSpec`+`TensorOpsSpec` sozinhas, 85/85 verdes; assim que as suítes de `gradcheck` entravam na mesma rodada, as falhas apareciam.
- **Causa raiz**: `Tensor.gradEnabled` ([Tensor.scala](scalagrad/src/main/scala/scalagrad/core/Tensor.scala)) era um `var` mutável **compartilhado globalmente** no companion object, e `noGrad` funcionava ligando/desligando essa flag temporariamente (`try/finally`). O ScalaTest roda suítes diferentes em paralelo (threads distintas) por padrão — isso nunca deu problema antes porque nenhuma operação chamava `noGrad` com frequência suficiente pra colidir com outra suíte rodando ao mesmo tempo. `Gradcheck.run` muda isso: chama `noGrad` duas vezes por posição perturbada do tensor, então uma única chamada num tensor `(2,3)` já dispara 12 toggles da flag global — o suficiente pra uma suíte rodando em paralelo (ex. `TensorSpec`) ter seu `backward()` "pego no meio" com `gradEnabled=false` no momento errado, zerando o gradiente silenciosamente. Confirmado forçando `scalagrad / Test / parallelExecution := false`: as 143 voltavam a passar 100% — prova de que era mesmo condição de corrida, não bug de lógica no `Gradcheck`.
- **Correção aplicada** (discutida em conversa: a alternativa mais simples seria só desligar paralelismo dos testes no `build.sbt`, mas o usuário preferiu resolver a causa raiz, cogitando que operações futuras — ex. `matmul3D` paralelizado pelo laço de batch, já que cada fatia é independente — também vão precisar de `gradEnabled` thread-safe de verdade): `gradEnabled` trocado de `var` global pra `scala.util.DynamicVariable[Boolean]` — cada thread enxerga seu próprio valor (herdado da thread que a criou), então `noGrad` numa thread nunca interfere no `gradEnabled` de outra rodando em paralelo. `noGrad` reescrito como `gradEnabledVar.withValue(false)(block)`, que já cobre o `try/finally` internamente (não precisou mais ser escrito à mão). Nenhum call site mudou (`Tensor.gradEnabled` continua legível como antes em todo `core`/`ops`, agora como `def` delegando pro `DynamicVariable`, não mais um `var` direto).
- Validado com 8 execuções seguidas da suíte completa (`scalagrad/testOnly *`, paralelismo normal do ScalaTest, sem nenhum override) — 143/143 verdes em todas.
- `checklists/04-gradient-check.md`: **Etapa 4 concluída** (todos os itens, incluindo "Aplicação"). Achado de concorrência não estava em nenhum item do checklist original — foi um efeito colateral descoberto só por causa da varredura ampla, reforçando por que o roadmap pede "inputs aleatórios de shapes variadas" e rodar tudo junto, não isolado.
- **Ideia registrada pra revisitar mais tarde, não implementada agora:** paralelizar `matmul3D` pelo laço de batch (`0 until t1.shape(0)`, via `ParSeq`/`.par`) — discutido em conversa depois do fix do `gradEnabled`. É seguro fazer isso (cada `(b,i,k)`/`(b,k,j)` no backward escreve num índice único de `gradient`, sem overlap entre threads — só ficou seguro por causa da correção do `DynamicVariable` acima), mas depende de adicionar o módulo `scala-parallel-collections` (não é mais parte da standard library desde 2.13) e só compensa pra tensores grandes o suficiente pra pagar o overhead de despachar threads — otimização prematura pro estágio atual, mesmo espírito da decisão de não generalizar `matmul` além de 2D/3D. Revisitar quando shapes de batch/seq real tornarem isso um gargalo medido.

---

## 2026-07-26 — `theory/05-activations/` escrita antes da implementação (Etapa 5)

- Seguindo o novo fluxo combinado ([[feedback_theory_before_implementation]] — teoria antes do código), criada `theory/05-activations/` com os três arquivos: `05-activations.md` (por que não-linearidade importa, com prova numérica via princípio da superposição — `g(-1)+g(1)=6 ≠ g(0)=0` quando uma ReLU quebra a linearidade de duas camadas compostas; ReLU e o "neurônio morto"; Sigmoid e Tanh com backward derivado via regra do quociente/identidade padrão; GELU com a derivação completa do backward — regra do produto combinada com regra da cadeia através do `tanh`, incluindo o detalhe de que o coeficiente `0.134145` é `3×0.044715` por causa da regra da potência ao derivar o termo cúbico — conferida contra a fórmula de referência usada no GPT-2/Hugging Face antes de escrever, e batendo; cada seção fechando com exemplo numérico verificado), `activations.html` (duas figuras: ReLU vs. GELU mostrando a atenuação suave da GELU na região negativa em vez do corte abrupto da ReLU, e Sigmoid vs. Tanh mostrando a diferença de centro/simetria), e `exercises.html` (12 questões, um pouco acima do baseline de 10 por causa da densidade extra da derivação de GELU — inclui duas questões ligando de volta ao `Gradcheck` da Etapa 4, reforçando por que GELU é a candidata mais importante pra rodar a verificação numérica).
- `checklists/05-activations.md`: status marcado "Em andamento" — teoria pronta, implementação (ReLU/GELU/Sigmoid/Tanh em `TensorOps.scala`) ainda não começou.

---

## 2026-07-26 — `ops/` reorganizado por categoria, antes de implementar as ativações

- Usuário pediu, e Claude implementou a pedido explícito, uma quebra de `TensorOps.scala` (que já estava em ~280 linhas depois do `matmul`/`clamp`) em arquivos por categoria: `UnaryOps.scala` (fábrica `unary` + `neg`/`pow`/`exp`/`log`/`clamp` — destino natural das 4 ativações da Etapa 5, todas cabendo na mesma fábrica), `BinaryOps.scala` (fábrica `binary` + `+`/`-`/`*`/`/`), `ReduceOps.scala` (fábricas `reduce`/`reduceDim` + `sum`/`mean`/`max`/`sum(dim)`/`mean(dim)`), `MatmulOps.scala` (`matmul2D`/`matmul3D`/`matmul`). `TensorOps.scala` virou só a composição: `object tensor extends UnaryOps with BinaryOps with ReduceOps with MatmulOps`.
- Detalhe técnico que motivou usar `trait` em vez de `object` em cada arquivo novo: `extension` methods em `object`s separados exigiriam um `import` por arquivo pra quem for usar a API (`import scalagrad.ops.unary.*, scalagrad.ops.binary.*, ...`), perdendo a ergonomia de hoje (`import scalagrad.ops.tensor.*` só). Com `trait` + mixin no `object tensor` final, todos os `extension` de todos os arquivos ficam disponíveis num import só, sem duplicar nada — cada `trait` marcado `private[ops]` (implementação interna do pacote, não deveria ser importado/misturado individualmente por fora).
- Mudança 100% mecânica (mover código, sem alterar nenhuma lógica) — confirmado com `sbt compile` limpo e 3 execuções seguidas de `scalagrad/testOnly *` (143/143 verdes em todas), mesmo hábito de revalidar múltiplas vezes adotado depois do achado de concorrência da Etapa 4.
- Decisão de escopo registrada em conversa: a Etapa 6 (softmax) deve ganhar seu próprio `SoftmaxOps.scala` quando chegar lá, já que normalizar um vetor inteiro não se encaixa nem em `unary` nem em `reduce`/`reduceDim` — mantém a mesma convenção sem inchar nenhum arquivo existente. Segundo o `CLAUDE.md`, `ops/` para de crescer depois da Etapa 6 (Etapas 7+ vão pro módulo `gpt/`).

---

## 2026-07-26 — Etapa 5 concluída: ReLU, Sigmoid, Tanh, GELU

- Usuário implementou as 4 ativações em `UnaryOps.scala`, todas reaproveitando a fábrica `unary` já existente (sem precisar de fábrica nova, confirmando a expectativa levantada antes de começar).
- `relu`: revisão encontrou a borda `x=0` usando `x >= 0` (gradiente `1` no zero exato) em vez de `x > 0` — divergia da convenção documentada em `theory/05-activations/05-activations.md` §2 (mesma do `clamp`, Etapa 3) e do padrão do PyTorch (`input > 0`, estrito). Corrigido pelo usuário.
- `sigmoid`/`tanh`: implementados corretos já na primeira versão (backward reusando o valor de saída — `fx*(1-fx)` e `1-fx²` — igual `exp` já fazia desde a Etapa 3).
- `gelu`: revisão encontrou um bug real no backward — `(1 + Math.pow(t, 2))` em vez de `(1 - Math.pow(t, 2))` no termo que vem da derivada de `tanh` (`d/dx tanh(u) = (1-tanh(u)²)·u'(x)`). Sinal trocado, silencioso (roda sem erro, só produz gradiente numericamente errado — no exemplo `x=1` da teoria, `≈1.504` em vez de `≈1.083`). Exatamente o cenário que a seção §5 da teoria previu ao recomendar rodar `Gradcheck` em `gelu`. Corrigido pelo usuário; `uPrime` (nome da variável, antes `u_x`) também ajustado por consistência de estilo (camelCase).
- Criados 8 testes em `TensorOpsSpec.scala` (forward+backward de cada uma das 4, valores calculados via `Math.exp`/`Math.tanh`/`Math.pow` diretamente no teste — não reusando a mesma expressão do código sob teste, servindo de oráculo independente — mais os valores de referência do exemplo numérico da teoria pro `gelu`) e 4 testes novos em `GradcheckSweepSpec.scala` (estendendo a varredura da Etapa 4 pras 4 ativações, com inputs aleatórios de shapes variadas — item explícito do checklist desta etapa, "especialmente GELU"). `gelu` passou no `Gradcheck` só depois da correção do sinal, confirmando de forma independente que o fix estava certo. Suíte completa: 155 testes, revalidada 3x seguidas, todas verdes.
- `checklists/05-activations.md`: **Etapa 5 concluída** (todos os itens, incluindo gradient check em cada ativação).

---

## 2026-07-26 — `theory/06-softmax/` escrita antes da implementação (Etapa 6)

- Seguindo o fluxo combinado ([[feedback_theory_before_implementation]]), criada `theory/06-softmax/` com os três arquivos: `06-softmax.md` (estabilidade numérica via subtração do máximo, com prova de que é matematicamente idêntica ao original — o fator `e⁻ᵐ` cancela — e a propriedade de invariância por deslocamento `softmax(x+c)=softmax(x)`; derivação completa do Jacobiano `∂sᵢ/∂xⱼ = sᵢ(δᵢⱼ-sⱼ)` nos dois casos, diagonal e fora dela; simplificação algébrica do backward de `O(N²)` pra `O(N)` — `dx = s·(dOut-(dOut·s).sum())` — verificada numericamente batendo com o produto pelo Jacobiano completo; log-softmax com sua própria derivação e backward mais simples, `dx = dOut - s·dOut.sum()`; mesmo exemplo numérico reusado em todas as seções, `x=[1,2,3]`, `dOut=[0.1,0.2,0.3]`, pra permitir comparar os resultados entre seções diferentes), `softmax.html` (diagrama de dependência total — cada saída depende de toda entrada, ao contrário das ativações da Etapa 5 — e a matriz do Jacobiano completa com os valores numéricos do exemplo), e `exercises.html` (12 questões, incluindo uma sobre por que o roadmap recomenda gradient check com logits de magnitudes bem diferentes especificamente).
- `checklists/06-softmax.md`: status marcado "Em andamento" — teoria pronta, implementação (`softmax`/`log_softmax` em `ops/`, provavelmente um `SoftmaxOps.scala` próprio conforme decidido na reorganização anterior) ainda não começou. Esta etapa fecha o milestone "o autograd está completo" quando implementada.

---

## 2026-07-26 — Etapa 6 em andamento: `SoftmaxOps.scala` criado, forward (sem `dim`) revisado, `Shape.groupIndex` extraído

- Usuário criou `scalagrad/src/main/scala/scalagrad/ops/SoftmaxOps.scala` e implementou o forward de `softmax` **sem suporte a `dim`** ainda (normaliza o tensor inteiro achatado), como primeiro passo deliberado pra fixar a matemática antes de generalizar — `TensorOps.scala` atualizado pra `... with SoftmaxOps`. Revisão confirmou o forward correto (`m = max`, `d = exp(x-m)`, `s = d.sum`, `data = d/s`), validado numericamente contra o exemplo de `theory/06-softmax/06-softmax.md` §1 (`x=[1,2,3] → ≈[0.09003,0.24473,0.66524]`). Backward ainda vazio, `logSoftmax` ainda `???` — esperado nesse estágio.
- Discutido em conversa como generalizar pra `softmax(a, dim)`: como o `max`/soma internos são só estabilidade numérica (não precisam passar pelo autograd — `softmax` é implementado como nó opaco só, com backward manual via fórmula fechada, igual `matmul`/`clamp`), a generalização reusa o mesmo padrão "gather por grupo" que `reduceDim` (Etapa 3) já usa: agrupar posições por um índice que colapsa a dimensão `dim`, calcular `max`/soma por grupo em vez de globalmente, e espalhar de volta.
- Isso expôs duplicação: o mapeamento de índice usado por `reduceDim` (antes um `outLinIdx` local, `private`, só dentro de `ReduceOps`) seria reimplementado quase idêntico dentro de `SoftmaxOps`. A pedido do usuário, extraído pra `Shape.groupIndex(multiIdx, dim)` (`Shape.scala`) — método público, já que é puramente uma questão de indexação de shape, não de tensor (mesmo raciocínio já usado pra `index`/`unravelIndex` morarem em `Shape`, não em `Tensor`). Insight que simplificou a extração: indexar em `crop(dim)` (dimensão removida) ou em `updated(dim,1)` (dimensão zerada, mesmo rank) sempre dá o **mesmo** índice linear — reduzir uma dimensão pra tamanho 1 não muda as strides relativas das demais — então `groupIndex` não precisa de parâmetro `keepDim` nenhum, ao contrário do `outLinIdx` antigo (que tinha um `if keepDim then ... else ...` a mais, agora removido).
- `ReduceOps.reduceDim` refatorado pra usar `t1.shape.groupIndex(_, dim)` em vez do `outLinIdx` local — mudança mecânica, comportamento idêntico. Criados 4 testes novos em `ShapeSpec.scala`: agrupamento por linha/coluna em casos simples 2D, e dois testes de equivalência (`groupIndex` bate com indexar `crop(dim)` diretamente, e bate com indexar a versão `updated(dim,1)` com a coordenada zerada — confirmando a prova usada na extração). Suíte completa: 159 testes (155 anteriores + 4 de `groupIndex`), revalidada 3x seguidas, todas verdes.
- Próximo passo: usar `Shape.groupIndex` pra generalizar `softmax`/implementar `logSoftmax` com suporte a `dim`, e então o backward de ambos.

---

## 2026-07-26 — Etapa 6 em andamento: forward de `softmax(dim)` concluído e correto

- Usuário generalizou `softmax` pra receber `dim`, usando `Shape.groupIndex` (extraído mais cedo) pra agrupar posições por fatia. Revisão encontrou um bug real na primeira versão: as duas passadas necessárias (achar o `max` de cada grupo; depois calcular `exp(x-max)`) estavam fundidas num laço só, usando `maxes(idx)` como "máximo visto até agora" em vez do máximo final do grupo — a soma final ainda dava `≈1` (mascarando o problema numa checagem só de "soma bate"), mas a distribuição *relativa* dentro de cada fatia saía errada. Contra-exemplo usado na revisão: `x=[1.0,5.0,3.0]` (um grupo só) dava `≈[0.468,0.468,0.063]` em vez do correto `≈[0.0159,0.8668,0.1173]`.
- Corrigido pelo usuário: duas passadas de verdade (a primeira só preenche `maxes` por completo; a segunda, já com o máximo definitivo, calcula `exp`/soma). Revisão também pegou dois problemas de leitura: `t.data(i)` usado direto em vez de `t.get(multiIdx*)` (só funciona se `t` for contíguo — quebraria com um tensor vindo de `transpose()`, cenário provável na Etapa 11/attention), e a saída construída com `t.strides` em vez de `t.shape.canonicalStrides`. Ambos corrigidos — a primeira passada (que só tinha sido ajustada parcialmente numa rodada intermediária) ficou consistente com a segunda, as duas lendo via `.get()`.
- Forward validado numericamente contra o contra-exemplo acima (bate com o valor correto depois do fix) e contra o exemplo de `theory/06-softmax/06-softmax.md` §1. `sbt compile` limpo.
- Pendente pra fechar a Etapa 6: backward de `softmax` (`dx = s·(dOut-(dOut·s).sum())`, reusando `groupIndex` do mesmo jeito), `logSoftmax` (forward + backward), testes permanentes, e gradient check com logits de magnitudes bem diferentes (item explícito do checklist).

---

## 2026-08-04 — Etapa 6 concluída: backward do `softmax`, `logSoftmax` completo, e bug real descoberto em `Shape.canonicalStrides`

- Usuário implementou o backward de `softmax(dim)` (`dx = s·(dOut - Σgrupo(dOut·s))`). Revisão encontrou 1 bug real na primeira versão: a passada que acumula `Σgrupo(dOut·s)` lia `data(idx)*grad(idx)` (indexando os arrays de tamanho `t.size` pelo índice do *grupo*, não do *elemento*) em vez de `data(i)*grad(i)` — traço numérico com um tensor 1D de 3 elementos (um grupo só, `idx=0` sempre) mostrou que a soma dava `3·data(0)·grad(0)` em vez de `Σᵢdata(i)·grad(i)`, ignorando as posições 1 e 2 inteiramente. Corrigido pelo usuário trocando pra `data(i)*grad(i)` (a segunda passada do backward já usava a indexação certa, só a primeira tinha o bug).
- Usuário implementou `logSoftmax(dim)` (assinatura corrigida de `logSoftmax: Tensor` sem parâmetro pra `logSoftmax(dim: Int): Tensor`, alinhando com `softmax(dim)` e com o checklist) — forward via log-sum-exp (`(x-m) - log(S)`, mesmas duas passadas de `maxes`/`sums` já usadas por `softmax`) e backward (`dx = dOut - s·Σgrupo(dOut)`, reaproveitando `s = exp(data(i))` já que `data(i) = log(s_i)`, sem precisar guardar a probabilidade separada). Nasceu correto já na primeira versão, sem bugs — validado contra o exemplo numérico de `theory/06-softmax/06-softmax.md` §5 (`x=[1,2,3]`, `dOut=[0.1,0.2,0.3]` → `dx≈[0.04598,0.05316,-0.09914]`).
- Criados 10 testes novos em `TensorOpsSpec.scala` (forward/backward de `softmax`/`logSoftmax`, incluindo: invariância por deslocamento implícita no teste de estabilidade numérica com logits que estourariam `exp` sem o truque do máximo — `[700,800,900,750]` —, e um teste de independência entre grupos usando pesos/`dOut` distintos por linha, técnica que teria pego o bug do backward acima se já existisse antes da correção). Estendido `GradcheckSweepSpec.scala` com o item explícito do checklist ("logits de magnitudes bem diferentes") — usando o mesmo tensor `[700,800,900,750]`. Detalhe de design descoberto ao escrever esses testes: `(a.softmax(dim) * pesos).sum` é obrigatório (não dá pra testar com `a.softmax(dim).sum` puro) — a soma de probabilidades dentro de um grupo é sempre `1`, uma função *constante* de `x`, então gradiente analítico e numérico dariam `~0` os dois mesmo com um backward quebrado, "passando" por acidente; multiplicar por pesos fixos antes do `.sum` quebra essa degenerescência, mesma técnica já usada nos testes unitários de backward.
- **Bug real descoberto pelos próprios testes, fora do código que o usuário tinha acabado de escrever**: `softmax(0)`/`logSoftmax(0)` sobre um tensor rank 1 (o caso mais elementar — um vetor simples, `x=[1,2,3]` da teoria) estourava `UnsupportedOperationException: tail of empty array`. Causa raiz em `Shape.canonicalStrides` ([Shape.scala](scalagrad/src/main/scala/scalagrad/core/Shape.scala)): `Shape.groupIndex` colapsa a única dimensão de um shape rank 1 via `crop(dim)`, produzindo um `Shape` rank 0 (`values = Array()`); `canonicalStrides` fazia `values.tail.scanRight(...)`, e `.tail` num array **já vazio** lança exceção em Scala (diferente de `.tail` de um array de 1 elemento). Nunca tinha aparecido antes porque nenhum uso anterior de `groupIndex`/`reduceDim` (Etapa 3) testou colapsar a única dimensão de um tensor rank 1 — sempre foram tensores rank 2+. Corrigido com um guard (`if values.isEmpty then Strides(Array.empty[Int]) else ...`) — um shape rank 0 (escalar) legitimamente não tem stride nenhuma. Regressão coberta com 2 testes novos em `ShapeSpec.scala` (`canonicalStrides` vazio pra shape rank 0; `groupIndex` colapsando a única dimensão de um shape rank 1 pro grupo `0`).
- A pedido explícito do usuário (mesma exceção de sempre à regra de "Claude não escreve `.scala`" — refatoração já discutida e aprovada em conversa), Claude extraiu a fábrica privada `logSumExpStats(t, dim)` em `SoftmaxOps.scala`, compartilhando as duas passadas (achar `maxes` por grupo, somar `exp(x-max)` por grupo) que `softmax`/`logSoftmax` duplicavam quase que literalmente — mesmo espírito de `unary`/`binary`/`reduce`/`reduceDim` (Etapa 3). `expData` (usado só por `softmax`) continua sendo calculado dentro da fábrica compartilhada e ignorado por `logSoftmax` via `_`, evitando uma segunda passada separada. De bônus, `trait SoftmaxOps` ganhou o modificador `private[ops]` que já estava presente em `UnaryOps`/`BinaryOps`/`ReduceOps`/`MatmulOps` mas tinha ficado de fora quando o arquivo foi criado.
- Suíte completa revalidada 2x seguidas: 175 testes (165 anteriores + 10 de softmax/logSoftmax), todos verdes.
- `checklists/06-softmax.md`: **Etapa 6 concluída** — todos os itens, incluindo gradient check com logits de magnitudes bem diferentes. **MILESTONE alcançado: "o autograd está completo"** — todas as primitivas necessárias para o transformer têm forward+backward testados. Próximo passo: Etapa 7 — Tokenização (`checklists/07-tokenization.md`), primeira etapa do módulo `gpt/` (ainda sem nenhum arquivo fonte).

---

## 2026-08-04 — `theory/07-tokenization/` escrita antes da implementação (Etapa 7)

- Seguindo o fluxo combinado ([[feedback_theory_before_implementation]]), criada `theory/07-tokenization/` com os três arquivos: `07-tokenization.md` (por que character-level em vez de BPE/palavra-inteira; construção do vocabulário via `charToIdx`/`idxToChar` com ênfase em por que a ordenação alfabética importa — determinismo entre execuções, não estética; `encode`/`decode` como inversos, com o porquê de `decode(encode(text))==text` ser uma checagem real de bug silencioso, não formalidade; a parte mais sutil da etapa — o `target` é a *mesma* janela do `input`, deslocada uma posição no corpus original, não uma janela nova, com o cuidado do intervalo válido de `start` exigir `contextLength+1` tokens consecutivos (`0` até `corpusLength-contextLength-1`, não `corpusLength-contextLength`); montagem de um batch de `B` janelas independentes; cada seção fechando com o mesmo exemplo numérico reusado ao longo do arquivo, `encode("abacate")=[0,1,0,2,0,4,3]`, permitindo comparar seções entre si), `tokenization.html` (tabela de vocabulário, diagrama da janela deslizante com o deslocamento input→target, e o batch de duas janelas empilhadas), e `exercises.html` (10 questões fácil→desafio, incluindo duas numéricas sobre o exemplo do `.md` e uma ligando à Etapa 9/Embedding — por que um índice bruto ainda não é algo que a rede processa).
- `checklists/07-tokenization.md`: status marcado "Em andamento" — teoria pronta, implementação (`Tokenizer`/`encode`/`decode`/amostragem de batches, em `gpt/src/main/scala/gpt/data/`, primeiro arquivo fonte do módulo `gpt/`) ainda não começou.

---

## 2026-08-07 — Etapa 7 em andamento: `Tokenizer` (vocabulário + `encode`/`decode`) — primeiro código do módulo `gpt/`

- Duas decisões de design discutidas antes de implementar: (1) `charToIdx`/`idxToChar`/`vocabSize` são construídos **uma única vez**, a partir do corpus que dimensiona o modelo, e a mesma instância de `Tokenizer` é reusada pra qualquer texto futuro (validação, prompt de geração na Etapa 19) — nunca reconstruída por corpus, já que `vocabSize` fica embutido na tabela de embedding (Etapa 9) e na camada de saída (Etapas 15-16); um caractere fora do vocabulário em `encode`, ou um índice fora do vocabulário em `decode`, falha explicitamente (`NoSuchElementException` com a posição/índice na mensagem) em vez de introduzir um token `UNK` — fora do escopo deste tokenizer character-level simples. (2) `object Tokenizer` fica com um único factory (`charLevel`), não uma coleção de tokenizers alternativos — não há nenhuma segunda variante planejada no roadmap, então manter só o ponto de injeção (`mapper: String => Map[Char, Int]` no construtor) sem construir infraestrutura pra variantes hipotéticas.
- Usuário criou `gpt/src/main/scala/gpt/data/Tokenizer.scala` (primeiro arquivo fonte do módulo `gpt/`, que só tinha a pasta reservada desde a divisão em módulos). Revisão encontrou e corrigiu, em rodadas sucessivas:
  1. **Faltava a declaração `package gpt.data` no topo do arquivo** — sem ela, o qualificador `private[data]` não tinha nenhum pacote chamado `data` no escopo pra resolver, gerando erro de compilação de scope. Único arquivo do projeto até agora que nasceu sem a declaração de pacote already presente (todo `scalagrad` já tinha esse hábito desde a Etapa 1).
  2. **`private[data]` na classe era estreito demais**: o `object Tokenizer` (público) tem um factory (`charLevel`) que devolve o tipo `Tokenizer` — como o tipo só é nomeável dentro do pacote `gpt.data`, qualquer código fora dele (ex. `gpt.nn` na Etapa 9, precisando de `vocabSize`) não compilaria ao tentar tipar o retorno de `charLevel`. Trocado para `private[gpt]` — visível em todo o módulo `gpt`, mas não fora dele.
  3. **`vocabSize` estava `private`**, embora seja um item explícito do checklist e vá precisar ser consultado de fora (Etapa 9, dimensionar a tabela de embedding) — trocado pra público.
  4. **`encode` reimplementava `zipWithIndex` manualmente via `foldLeft`** com uma tupla `(i, acc)`, recalculando `input(i)` a cada iteração em vez de usar o `c` que o próprio fold já entregava (redundante, não bug — os dois sempre coincidiam), e reconstruindo o array inteiro a cada caractere via `.appended` (O(n²) no tamanho do texto). Simplificado pra `input.toArray.zipWithIndex.map { case (c, i) => charToIdx.getOrElse(c, throw ...) }` — O(n), sem index tracking manual.
- Implementação final: `charToIdx`/`idxToChar` (inversos, construídos de `charToIdx.map((k,v) => (v,k))`), `vocabSize`, `encode`/`decode` simétricos (ambos falham com `NoSuchElementException` incluindo a posição/índice problemático na mensagem), e `Tokenizer.charLevel(corpus)` usando `corpus.distinct.sorted.zipWithIndex.toMap` (ordem alfabética determinística, conforme `theory/07-tokenization/07-tokenization.md`).
- `checklists/07-tokenization.md`: "Coletar caracteres únicos e ordenar", "Construir charToIdx/idxToChar", "vocabSize", "encode", "decode" marcados concluídos. Falta pra fechar a Etapa 7: "Ler corpus de texto" (hoje `charLevel` recebe a `String` já pronta, sem ler arquivo), testes permanentes de `encode`/`decode`/`charLevel` (incluindo a verificação `decode(encode(text)) == text` pedida pelo checklist — ainda não criados, pendência explícita pra próxima sessão), e toda a seção "Mini-batches" (amostragem de janelas `contextLength`, input/target deslocados, batch de `B` janelas).

---

## 2026-08-08 — Etapa 7 em andamento: suíte permanente do `Tokenizer`

- Seguindo a autorização permanente de testes ([[feedback_proactive_test_writing]]), criada `gpt/src/test/scala/gpt/data/TokenizerSpec.scala` (12 testes, `AnyFlatSpec`/`Matchers`, mesmo padrão de `scalagrad`), reusando o exemplo numérico já verificado em `theory/07-tokenization/07-tokenization.md` §2-3 (`corpus="abacate"` → `charToIdx={a:0,b:1,c:2,e:3,t:4}`, `encode("abacate")=[0,1,0,2,0,4,3]`) em vez de inventar valores novos.
- Cobertura: `vocabSize` (contagem de distintos, incluindo colapso de repetidos), ordem alfabética das idx (`"abacate"` tem `t` antes de `e` na ordem de aparição, mas `e < t` alfabeticamente — teste escolhido especificamente pra distinguir as duas ordens), `encode`/`decode` batendo com o exemplo verificado, string/array vazios, os dois casos de fronteira (`encode` com char fora do vocabulário — incluindo checagem de que a posição reportada na mensagem está correta quando o char inválido não é o primeiro — e `decode` com índice fora do vocabulário, incluindo índice negativo), e round-trip `decode(encode(text)) == text` para várias amostras do alfabeto do corpus. Nenhum bug encontrado — implementação já revisada anteriormente se manteve correta. Suíte nova: 12/12 verdes; suíte completa dos dois módulos revalidada sem regressão.
- `checklists/07-tokenization.md`: item "Verificar `decode(encode(text)) == text`" marcado concluído. Falta pra fechar a Etapa 7: "Ler corpus de texto" (hoje `charLevel` recebe a `String` já pronta) e toda a seção "Mini-batches".

---

## 2026-08-08 — Etapa 7 em andamento: `charLevel(file)`, `BatchSampler`, e convenção nova `require` vs. `assert`

- Usuário implementou `Tokenizer.charLevel(file: File)`, lendo o corpus de um arquivo real (`gpt/src/main/scala/gpt/data/Tokenizer.scala`). Revisão encontrou 2 bugs reais, ambos corrigidos: (1) `Source.fromFile(file).getLines().mkString` descartava as quebras de linha de cada linha lida (`getLines()` já remove o `\n`, e `mkString` sem separador não recoloca nada), colando a última palavra de uma linha com a primeira da próxima — corrigido pra `Using.resource(Source.fromFile(file))(_.mkString)`, que preserva o texto exatamente como está no arquivo; (2) o `Source` nunca era fechado (vazamento de recurso) — o mesmo `Using.resource` resolve os dois problemas de uma vez, fechando o arquivo mesmo se `mkString` lançar.
- Design discutido antes de implementar a seção de mini-batches: mesma calibragem já usada pro `Tokenizer` (Etapa 7 anterior) — o roadmap só descreve uma estratégia de amostragem (janelas aleatórias, independentes, com reposição), sem nenhuma segunda variante prevista, então `BatchSampler` fica uma **classe concreta única** (`BatchSampler(corpus: Array[Int], contextLength: Int, randomizer: Random = new Random())`), sem `trait`/hierarquia pra variantes hipotéticas — mesmo raciocínio já aplicado à decisão de não ter múltiplos `Tokenizer`s.
- Usuário implementou `BatchSampler.sample(batchSize): (Tensor, Tensor)` — **sem bugs encontrados na revisão**, incomum pro projeto (quase toda operação nova até aqui teve pelo menos um off-by-one). Conferido explicitamente: `randomizer.nextInt(corpus.size - contextLength)` produz exatamente o intervalo válido de `start` que a teoria pede (`[0, corpusLength-contextLength-1]`, batendo com o exemplo numérico de `theory/07-tokenization/07-tokenization.md` §4-5); o `flatMap` sobre `starts` (ordem preservada) concatena as janelas no layout row-major que `Tensor.make` espera pra shape `(B, contextLength)`, sem precisar de reshape extra; `requiresGradient=false` (default) é a escolha certa, já que os índices de entrada não são parâmetros diferenciáveis (o gradiente flui pela embedding, Etapa 9, não pelos índices).
- **Nova convenção formalizada, a pedido do usuário**: `require` (lança `IllegalArgumentException`) para pré-condição de argumento de método/factory público — o chamador passou algo inválido; `assert` (lança `AssertionError`, pode ser desabilitado via flag da JVM) só para invariante interna inalcançável por um chamador externo válido. Antes disso, todo o projeto usava `assert` uniformemente pra ambos os casos.
- A pedido explícito do usuário (mesma exceção de sempre à regra de "Claude não escreve `.scala`" — mudança mecânica já combinada em conversa), Claude levantou todos os 13 usos de `assert` em código de produção (`Tensor.scala`, `Shape.scala`, `MatmulOps.scala`) e converteu 10 pra `require`: `Tensor.reshape` (`newShape.product == size`), `Tensor.transpose` (dimensões válidas), `Tensor.backward` (`size == 1`), `Shape.linearIndex` (contagem e limites de dimensões — usado por `Tensor.index`/`Shape.index`), `Shape.unravelIndex` (índice dentro do intervalo), `Shape.broadcast` (shapes compatíveis), e dentro de `matmul2D`/`matmul3D` as checagens de dimensão interna e de batch size (que validam os *valores* dos tensores que o chamador passou pro `matmul` público, mesmo estando fisicamente dentro de um método privado). O `throw new AssertionError` manual no dispatcher `matmul` (combinação de rank não suportada) também virou `throw new IllegalArgumentException`, mesmo raciocínio.
- Ficaram como `assert` (invariante interna, não pré-condição de chamador): o `maxReachableIndex < data.length` no construtor de `Tensor` (checagem estrutural de baixo nível, atravessada por toda construção interna, não só por um argumento externo isolado) e as checagens de rank dentro de `matmul2D`/`matmul3D` (`t1.rank == 2/3 && t2.rank == 2/3` — já garantidas pelo dispatcher público `matmul` antes de despachar pra esses métodos privados, portanto inalcançáveis com rank errado por qualquer caminho válido).
- Consequência: 15 testes existentes que verificavam `an[AssertionError]` precisaram trocar pra `an[IllegalArgumentException]` (7 em `TensorOpsSpec` — `add`/`sub`/`mul`/`div` com shapes incompatíveis e as 3 rejeições de `matmul`; 4 em `ShapeSpec` — `index` e `broadcast`; 4 em `TensorSpec` — `index`, `unravelIndex`, `reshape`, `broadcastTo`). Só 1 teste manteve `AssertionError` (`Tensor.make` com `data`/`shape` incompatíveis, único caso que ainda passa pelo invariante do construtor mantido como `assert`). Suíte completa revalidada: 175 testes em `scalagrad` + 12 em `gpt`, todos verdes.
- Convenção documentada em `CLAUDE.md` (seção "Convenções e decisões de design").
- `checklists/07-tokenization.md`: falta só a seção "Mini-batches" marcar os itens (amostragem de janelas, input/target deslocados, batch de `B`) — implementação pronta, mas ainda sem testes permanentes de `BatchSampler` (pendência pra próxima sessão, junto com marcar o checklist).

---

## 2026-08-08 — Etapa 7 concluída: suíte permanente do `BatchSampler`

- Criada `gpt/src/test/scala/gpt/data/BatchSamplerSpec.scala` (5 testes, mesmo padrão `AnyFlatSpec`/`Matchers` do resto do projeto). Como `Tensor.shape` é `private[scalagrad]` (não acessível de `gpt`), os testes leem os tensores via a API pública (`rank`/`size`/`get(dims*)`) em vez do `.data.toList` usado dentro do próprio `scalagrad` — um helper local `row(t, r, len)` monta cada linha como `List[Double]` pra comparação.
- Pra reproduzir exatamente o exemplo verificado de `theory/07-tokenization/07-tokenization.md` §5 (`start=0` e `start=2`) sem depender do algoritmo interno do `java.util.Random`, criada uma subclasse de teste `FixedRandom(values*)` que devolve, em ordem, uma sequência fixa em `nextInt` — abordagem preferida a uma seed numérica fixa (usada em `GradcheckSweepSpec`), já que aqui o objetivo é controlar exatamente *quais* `start` saem, não só ter reprodutibilidade.
- Cobertura: o exemplo numérico exato da teoria (`encode("abacate")`, `B=2`, `start=0,2`); shape `(batchSize, contextLength)` genérico; a invariante `target[i] == input[i+1]` dentro da mesma janela (testada com `Random` de seed fixa, já que vale pra qualquer `start` sorteado, sem precisar saber qual foi); o caso de borda `corpusLength == contextLength + 1` (só existe um `start` válido — testado repetindo a amostragem 10x com `Random` sem seed, confirmando que a mesma única janela sai sempre); e uma checagem de que janelas diferentes do mesmo batch não são todas iguais (amostragem independente, não uma janela repetida). Nenhum bug encontrado — implementação já revisada anteriormente se manteve correta. Suíte nova: 5/5 verdes; suíte completa dos dois módulos revalidada: 175 (`scalagrad`) + 17 (`gpt`), todas verdes.
- `checklists/07-tokenization.md`: **Etapa 7 concluída** — todos os itens (vocabulário, encode/decode, mini-batches). Próximo passo: Etapa 8 — Linear Layer (`checklists/08-linear.md`), primeira camada de rede neural do módulo `gpt/nn/` (ainda sem nenhum arquivo).

---

## 2026-08-09 — `theory/08-linear/` escrita antes da implementação (Etapa 8)

- Seguindo o fluxo combinado ([[feedback_theory_before_implementation]]), criada `theory/08-linear/` com os três arquivos: `08-linear.md` (transformação afim `y=xW+b`, cada coluna de `W` como um neurônio; backward tratado como pura composição de `matmul`+`add` já provados na Etapa 3, com `dx=dy@Wᵀ`/`dW=xᵀ@dy`/`db=dy.sum(dim=0)` reaproveitando diretamente as fórmulas de `03-elementary-operations.md` §4/§6 em vez de rederivar; o problema da simetria do zero-init demonstrado com exemplo numérico — duas colunas idênticas de `W` recebem gradiente idêntico pra sempre; derivação de Kaiming/He via preservação de variância (`Var(w)=2/inputDim` compensando os ~50% de ativações que a ReLU zera), contrastada com o `N(0,0.02)` fixo do GPT-2 original; por que o bias pode ser zero mesmo `W` não podendo; papel de `parameters(): List[Tensor]` pro otimizador da Etapa 17; fechando com um exemplo numérico completo de forward+backward com batch, `dW`/`db`/`dx` calculados e cruzados com uma verificação independente), `linear.html` (forward como produto escalar por neurônio com uma coluna de `W` destacada, o backward como composição de duas fórmulas já conhecidas, e o ciclo de "sempre iguais" do zero-init), e `exercises.html` (10 questões fácil→desafio, incluindo duas numéricas reusando o exemplo de batch do `.md` §8 pra `db`/`dW[0,0]`).
- Duas figuras de `linear.html` (`FIG1`/`FIG3`) tiveram as coordenadas do SVG desenhadas à mão e verificadas programaticamente (sem preview visual disponível no ambiente desta sessão): um script injetado via `javascript_tool` mediu o `getBBox()` de todo `rect`/`circle`/`text` de cada figura contra o `viewBox` declarado. Encontrado e corrigido 1 problema real: o rótulo "gradiente vindo de cima" da `FIG2` (backward) ficava com `text-anchor` padrão (início à esquerda) numa posição perto da borda direita, estourando o `viewBox` em ~45px — corrigido trocando pra `text-anchor="end"` ancorado mais à esquerda. Revalidado depois do fix: nenhum elemento fora dos limites nas 3 figuras.
- `checklists/08-linear.md`: status marcado "Em andamento" — teoria pronta, implementação (`Linear` em `gpt/src/main/scala/gpt/nn/`, primeiro arquivo do pacote `nn`) ainda não começou.

---

## 2026-08-09 — Etapa 8 concluída: `Linear` (primeiro arquivo de `gpt/nn/`)

- Usuário implementou `gpt/src/main/scala/gpt/nn/Linear.scala`. Revisão encontrou 5 bugs reais na primeira versão, todos corrigidos pelo usuário e revalidados:
  1. **`std = Math.sqrt(2 / inputDim)` com divisão inteira** — `2` e `inputDim` são os dois `Int`, então `2/inputDim` truncava pra `0` sempre que `inputDim > 2`, deixando `std = 0`. Corrigido pra `2.0 / inputDim`.
  2. **Pesos preenchidos com `Array.fill(n)(std)`** — todo peso recebia o mesmo valor constante (`std`, não uma amostra de `N(0,std²)`), reproduzindo exatamente o problema da simetria descrito em `theory/08-linear/08-linear.md` §4 (colunas de `W` idênticas, "gêmeas" pra sempre). Combinado com o bug #1, `W` nascia inteiramente zero — o pior caso possível. Corrigido pra `Array.fill(n)(std * Random.nextGaussian())` (o parâmetro por-nome de `Array.fill` chama `Random.nextGaussian()` uma vez por posição, gerando amostras independentes).
  3. **`b` sem `requiresGradient = true`** — nascia fora do grafo de autograd; o otimizador (Etapa 17) nunca teria como atualizá-lo. Corrigido.
  4. **`forward` fazia `x.broadcastTo(Shape(batchSize, inputDim))` com `batchSize` fixado no construtor**, antes do `matmul` — além de ser desnecessário (é só o *bias* que precisa de broadcast, e o `+` já faz isso sozinho desde a Etapa 3), acoplava a camada a um único tamanho de batch, quebrando o reuso previsto pra Etapa 18 (batches variáveis) e Etapa 19 (geração, tipicamente `B=1`). Corrigido removendo `batchSize` do construtor inteiramente e simplificando `forward` pra `x.matmul(W) + b` — o batch size passa a vir livre de `x.shape(0)` a cada chamada.
  5. **`backward()` vazio** — a teoria (§3) é explícita que `Linear` não precisa de backward manual (pura composição de `matmul`+`add`, já testados na Etapa 3); o método vazio não fazia nada e sugeria falsamente que faltava implementar algo ali. Removido.
- Também sinalizados e corrigidos dois ajustes de limpeza (não-bugs): renomear a classe de `LinearLayer` pra `Linear` (consistência com o nome das próximas camadas — Embedding, LayerNorm — que não devem ganhar sufixo `Layer`), e remover o import de `Shape` que ficou órfão depois do fix #4.
- `parameters` implementado como `val parameters: List[Tensor] = List(W, b)` (em vez de `def parameters()`) — funciona idêntico do lado de quem chama e evita reconstruir a lista a cada acesso, já que `W`/`b` nunca trocam de referência.
- Criada `gpt/src/test/scala/gpt/nn/LinearSpec.scala` (9 testes). Cobertura: forward comparado contra uma implementação independente em Scala puro (lê `W`/`b` via `parameters` e `.get`, sem precisar controlar a inicialização aleatória); shape de saída pra várias combinações de `inputDim`/`outputDim`/`batchSize`; `parameters` com exatamente `[W, b]` e shapes corretas; regressão do bug #3 (`requiresGradient` de ambos); regressão do bug #2 (pesos de `W` não são todos o mesmo valor); um teste estatístico de variância (`inputDim=50`, `outputDim=200`, 10000 pesos, tolerância de 35% — folgada o bastante pra não ser instável, já que o erro-padrão da variância amostral nesse `N` é da ordem de 1-2%) confirmando `Var(W) ≈ 2/inputDim`; e gradient check separado pra `x`, `W` e `b` usando `Gradcheck.run` da Etapa 4.
  - Técnica notável usada nos 3 testes de gradient check: como `W`/`b` são `private` dentro de `Linear`, mas `parameters` expõe as mesmas instâncias publicamente, dá pra chamar `Gradcheck.run(linear.parameters(0))(_ => linear.forward(x).sum)` — a closure ignora o parâmetro recebido e usa `linear`/`x` do escopo externo, mas como `linear.parameters(0)` É o mesmo objeto `W` usado internamente pelo `forward`, a perturbação que o `Gradcheck` aplica em `input.data` (mutação in-place) afeta exatamente os pesos que o `forward` lê. Cuidado aplicado: como `Gradient` só acumula (nunca sobrescreve), cada um dos 3 gradient checks usa uma instância nova de `Linear`/`x` — reusar a mesma teria somado o gradiente de checagens anteriores por cima, corrompendo a leitura do gradiente analítico.
- Suíte nova: 9/9 verdes, revalidada 3x seguidas (sem flakiness nos testes estatístico/gradient-check). Suíte completa dos dois módulos: 175 (`scalagrad`) + 26 (`gpt`) = 201 testes, todos verdes.
- `checklists/08-linear.md`: **Etapa 8 concluída** — todos os itens (parâmetros com `requiresGrad`, inicialização Kaiming, forward, `parameters()`, gradient check). Próximo passo: Etapa 9 — Embedding (`checklists/09-embedding.md`), segunda camada do módulo `gpt/nn/` — teoria ainda não escrita.

---

## 2026-08-09 — `theory/09-embedding/` escrita antes da implementação (Etapa 9)

- Seguindo o fluxo combinado ([[feedback_theory_before_implementation]]), criada `theory/09-embedding/` com os três arquivos: `09-embedding.md` (a equivalência `onehot(t) @ Table = Table[t]`, provada a partir da própria definição de `matmul` — fecha o gancho deixado por `theory/08-linear/08-linear.md` §9, "um índice é uma forma degenerada de camada linear"; backward do Token Embedding **derivado**, não postulado, a partir do `dW=xᵀ@dy` já provado na Etapa 8, mostrando que a regra "acumular só nas linhas usadas" do roadmap é o mesmo `matmul` de sempre com o operando esquerdo esparso; Positional Embedding e por que a atenção precisa de ordem explícita; backward posicional como soma sobre o batch (mesmo mecanismo do bias, Etapa 8 §3); a soma token+posição dividindo o gradiente igualmente pros dois ramos (regra do `+`, Etapa 3 §1); por que não sinusoidal; e uma seção de nota de implementação discutindo — sem decidir — a tensão entre lookup eficiente e `onehot×matmul`, cada exemplo numérico fechando com valores conferidos e cruzáveis entre seções), `embedding.html` (a equivalência one-hot/matmul como seleção de linha, o scatter-add do backward em ação com um token repetido, e o broadcast posicional dividindo em duas sequências do mesmo batch), e `exercises.html` (12 questões fácil→desafio, densidade um pouco acima do baseline por causa das duas tabelas + a nota de implementação, incluindo duas numéricas reusando os exemplos do `.md` e uma sobre a restrição arquitetural do `Tensor` privado).
- **Achado de design relevante, registrado como seção própria do `.md` (§8) em vez de decidido unilateralmente**: ao raciocinar sobre como `embed` seria implementado, ficou claro que um lookup eficiente de verdade (`O(1)` por token) não pode ser escrito inteiramente dentro de `gpt.nn`, do jeito que `Linear` foi — o construtor de `Tensor` é `private[scalagrad]` e `Tensor.make` (única fábrica pública) não aceita `previous`/`_backward` customizados, então só código dentro do pacote `scalagrad` consegue criar um tensor com backward próprio. Isso tensiona com a decisão já registrada em 2026-07-26 ("`ops/` para de crescer depois da Etapa 6, Etapas 7+ vão pro módulo `gpt/`"). A alternativa que não exige tocar `scalagrad` (`onehot(token) @ Table`, usando só `Tensor.make` e `.matmul` já públicos) funciona com zero mudança de infraestrutura, ao custo da ineficiência que o próprio `roadmap/09-embedding.md` justifica evitar. Fica como decisão em aberto pra próxima sessão, quando a implementação começar.
- Duas figuras de `embedding.html` precisaram de ajuste depois da checagem programática de `getBBox()` contra o `viewBox` (mesmo processo de validação sem preview visual adotado na Etapa 8): a legenda de duas linhas da FIG3 estourava a largura do `viewBox` em ~77px de cada lado — corrigido quebrando em duas linhas mais curtas e aumentando a altura do `viewBox` de 220 pra 230. Revalidado: nenhum elemento fora dos limites nas 3 figuras.
- `checklists/09-embedding.md`: status marcado "Em andamento" — teoria pronta (incluindo a decisão de implementação em aberto), implementação (`Embedding` em `gpt/src/main/scala/gpt/nn/`) ainda não começou.

---

## 2026-08-09 — Etapa 9 em andamento: decisão do lookup eficiente e `Tensor.indexSelect` (novo primitivo em `scalagrad`)

- Decisão da tensão registrada em `theory/09-embedding/09-embedding.md` §8: usuário optou pelo **lookup eficiente**, reabrindo `scalagrad.ops` pra um primitivo novo (`ops/` tinha "parado de crescer" desde a Etapa 6, ver `HISTORY.md` 2026-07-26) em vez do caminho `onehot(token) @ Table`.
- Antes de implementar, conversa passo a passo sobre o algoritmo (forward: copiar `table.get(indices(i), j)` linha a linha; backward: `table.gradient.accumulate` por posição, o scatter-add já derivado na teoria a partir de `matmul`) e sobre a mensagem de erro do `require` de índice fora do intervalo — decidido incluir o índice inválido, a posição no array e o tamanho da tabela na mensagem, seguindo o padrão já usado em `Tokenizer.decode` (Etapa 7).
- Usuário criou `scalagrad/src/main/scala/scalagrad/ops/IndexOps.scala` (`indexSelect(indices: Array[Int])`, mesmo padrão de fábrica privada misturada em `object tensor` via `with IndexOps`). Revisão encontrou 1 bug real, corrigido pelo próprio Claude a pedido explícito do usuário (exceção já autorizada em `CLAUDE.md`):
  - **`val numRows = t.shape(0)` / `val rowDim = t.shape(1)` rodavam *antes* do `require(t.rank == 2, ...)`** — pra um `t` de rank 0 ou 1, `t.shape(1)` (ou já `t.shape(0)` no caso rank 0) estourava um `ArrayIndexOutOfBoundsException` cru do Scala antes do `require` ter qualquer chance de rodar, escondendo a mensagem cuidadosamente desenhada ("indexSelect expects a rank-2 table...") atrás de um erro sem contexto nenhum. Corrigido movendo o `require` de rank pra antes do acesso a `t.shape(1)` — mesma ordem que `matmul2D`/`matmul3D` já seguem. Também corrigidos 2 nits de estilo (espaço faltando antes de `=>`, espaço sobrando no fim de linha).
- Criados 6 testes em `TensorOpsSpec.scala` (reusando a mesma `Table` de 5 linhas/`embeddingDim=2` de `theory/09-embedding/09-embedding.md` §1/§3, pra poder cruzar os valores): forward com índice repetido (`indices=[1,0,1]`); backward com pesos distintos por posição de saída confirmando o scatter-add (linha 0 recebe só uma contribuição, linha 1 soma duas, linhas 2-4 ficam em zero); regressão explícita do bug de ordem do `require` (tabela rank-1); rejeição de índice além do limite e de índice negativo; e `requiresGradient` do resultado seguindo o da tabela. Suíte completa: 181 testes em `scalagrad` (175 + 6), revalidada junto com os 26 de `gpt` — 207 no total, todos verdes.
- Falta pra fechar a Etapa 9: `gpt/src/main/scala/gpt/nn/Embedding.scala` em si (tabelas de token/posição, `embed`, `parameters`), incluindo decidir a assinatura de `embed` (`Array[Int]` cru vs. o `Tensor` que `BatchSampler` já produz — pendência registrada na conversa anterior) e os testes correspondentes.

---

## 2026-08-10 — Auditoria didática de `theory/`, novo padrão de escrita, e reescrita dos 9 capítulos

- Usuário pediu uma auditoria crítica do material de `theory/` sob a ótica de um iniciante em ML/IA — o material serve de apoio teórico pra implementação, então precisa ser aprendível por quem não sabe o assunto. Auditoria feita sobre os 9 documentos (~14.500 palavras, 106 exercícios), com verificação de **todos** os exemplos numéricos por script.
- **Achados principais.** A matemática estava correta (todos os exemplos conferidos batem) e as derivações são honestas — o material deriva em vez de postular (matmul, simplificação `O(N²)→O(N)` do softmax, GELU, embedding a partir de matmul), o que é o seu maior valor. O problema era de **público-alvo**: texto escrito por quem já sabe ML, pra quem já sabe ML. Quatro travas estruturais:
  1. **Sem mapa do território** — `transformer` aparecia na Etapa 2 e `atenção` na Etapa 5, nenhum dos dois definido em lugar nenhum.
  2. **Termos usados antes de definidos** — `neurônio` (usado na Etapa 5, definido na 8), `logit` (usado na 6, nunca definido), `lr`, e principalmente `L`: `∂L/∂A` aparecia na Etapa 3 sem o leitor saber que a perda é *um único número*.
  3. **Convenção `dX ≡ ∂L/∂X` nunca declarada** — estava num parêntese da Etapa 3 §6, e a notação variava entre documentos (`a.grad`, `dC`, `dy`, `dTable`).
  4. **Nenhum passo de treino completo** — o aluno nunca via a perda cair.
- **Erros concretos encontrados e corrigidos:** razão de erro na Etapa 4 §1 ("seiscentas vezes" quando é seiscentas **mil**); 4 assinaturas defasadas em relação ao código (`gradCheck` vs `Gradcheck.run`, `assert` que virou `require`, `unbroadcast` com 2 vs 3 parâmetros, `keepdim`/`keepDim`); Etapa 9 §8 descrevendo como decisão em aberto algo já decidido e implementado; Etapa 3 §4 descrevendo `unbroadcast` como `sum`+`squeeze` quando a implementação desenrola índice por posição (induzia a implementar errado); desempate do `max` não documentado.
- **Métricas de prosa medidas** (a raiz do problema de legibilidade): média de **31,2 palavras por frase** contra 15-20 de prosa técnica confortável, p90 entre 50 e 63, e 52 parênteses por mil palavras. A tendência estava **piorando** (26 na Etapa 1 → 34 na Etapa 9). Diagnóstico: o problema não era complexidade do assunto (inerente, deve ficar) e sim complexidade sintática — três ideias empilhadas por período. A correção não é simplificar, é desempacotar.
- **Piloto aprovado.** Etapa 3 (a mais longa e densa) reescrita em formato de livro-texto e validada pelo usuário antes de replicar: média caiu de 28,8 pra 13,0, p90 de 58 pra 22, parênteses de 53 pra 12.
- **Criado `theory/STYLE-GUIDE.md`** — o padrão, escrito pra ser checável e não conselho vago: leitor-alvo explícito, esqueleto obrigatório com seções ★, os quatro blocos destacados com formato exato, metas numéricas, notação canônica, regras de exemplo numérico, checklist de 20 itens em cinco grupos, e script de medição. Duas seções que valem destaque: a regra de que a numeração `§N` é **contrato** (comentários `// ver theory/...` no código apontam pra lá, então material novo entra como seção não numerada antes do `§1`), e o `§10 "O que não mudar"`, que preserva o que a auditoria identificou como já bom.
- **Autorização permanente registrada no `CLAUDE.md`:** ler o `STYLE-GUIDE.md` antes de escrever/reescrever/revisar qualquer documento de `theory/`.
- **Os 9 capítulos reescritos**, todos aprovados no checklist completo (média 11,7 a 15,8 · p90 19 a 24 · parênteses 5,8 a 18,8 · 62 Definições, 27 Armadilhas, 25 "Confira você mesmo" no total). Originais preservados em `theory/_pre-textbook-backup/` (mesmo precedente do `src_old_pre_scalagrad_migration`), podem ser apagados quando o usuário confirmar.
- **Ganhos de conteúdo que não existiam antes:**
  - **Etapa 1:** mapa das 19 etapas em quatro marcos; Box-Muller com exemplo numérico pela primeira vez; e o contraste que prova a mecânica de strides — o mesmo array `[1,2,3,4,5,6]` produzindo `[[1,4],[2,5],[3,6]]` via `transpose` e `[[1,2],[3,4],[5,6]]` via `reshape`, mesma memória e mesmo formato final.
  - **Etapa 2:** o **passo de treino completo** (aprender `y=2x` com um peso, 3 iterações, perda caindo `20,25 → 0,2025 → 0,002025`); `L`/`lr`/parâmetro definidos; e o `+=` vs `=` com números (`a.grad = 2` em vez de `7`).
  - **Etapa 4:** a constante `1e-5` deixou de ser mágica. Tabela medida mostrando que o erro cai como `ε²` até `1e-5` e depois **volta a subir** (arredondamento do `Double` domina), com o ótimo teórico `(ε_máquina)^(1/3) ≈ 6.1e-6` justificando a escolha. Nuance descoberta na verificação: a razão teórica dos erros é 600.000×, mas a medida real é ~283.000×, justamente porque o erro da central não chega a `1e-10`.
  - **Etapa 5:** "neurônio" definido; bug de sinal da GELU com os dois valores lado a lado (`1.08296` contra `1.50434`). Recalculando, o original tinha erro de arredondamento propagado (`0.68227` em vez de `0.68238`) — a versão nova bate melhor com a referência do GPT-2.
  - **Etapa 6:** "logit" definido; e a melhor armadilha do conjunto — a versão com "máximo corrente" produz `[0.468, 0.468, 0.063]` em vez de `[0.016, 0.867, 0.117]`, **e a soma dá 1 nos dois casos**, então qualquer teste de "soma 1" aprova o bug. Junto, a sutileza de que `softmax(x).sum` é função constante e por isso aprova qualquer backward.
  - **Etapas 7, 8 e 9:** as armadilhas são os bugs reais desta sessão e das anteriores — `getLines().mkString` colando palavras entre linhas, divisão inteira `2 / inputDim` zerando o desvio padrão, `Array.fill(n)(std)` produzindo neurônios gêmeos, `b` sem `requiresGradient`, e o `require` de rank do `indexSelect` rodando depois do acesso que deveria proteger.
- **Correção no próprio medidor:** o script inicial removia blocos de código, colando o texto antes e depois numa frase falsa e gigante — foi isso que inflou o p90 da Etapa 6 pra 27 (as "frases longas" eram fragmentos costurados). Corrigido no `STYLE-GUIDE.md` pra tratar blocos como separador, com nota de que `REV` isolado pede conferência e não reescrita automática. Piso da média afrouxado pra 11.
- Nenhum código `.scala` foi tocado nesta sessão — a suíte segue em 207 testes (181 `scalagrad` + 26 `gpt`).
- **Pendência conhecida, não resolvida:** o achado A1 da auditoria (falta um mapa geral) foi só **parcialmente** resolvido, com o mapa dos quatro marcos na abertura da Etapa 1. A recomendação original era um `theory/00-overview/` dedicado, com glossário, pré-requisitos e o diagrama do GPT completo. Vale fazer antes de a Etapa 10 começar. *(Resolvido no checkpoint seguinte.)*

---

## 2026-08-10 — `theory/00-overview/` criada: fecha o achado A1 da auditoria

- Criada a pasta de entrada do material, resolvendo a pendência do checkpoint anterior. Três arquivos, no padrão do `STYLE-GUIDE.md`:
  - **`00-overview.md`** — o que é um modelo de linguagem (a ideia de "prever o próximo token" e por que isso basta, via geração autoregressiva); o caminho de um texto pelo modelo em seis estágios, com os formatos de tensor acompanhados de ponta a ponta; as 19 etapas mapeadas nos quatro marcos, com o que cada uma constrói e onde entra; as convenções de notação canônica; um **glossário consolidado** com os 58 termos definidos ao longo dos capítulos, agrupados em seis categorias e com o ponteiro pro capítulo de origem; e uma seção final de conselhos de leitura.
  - **`gpt-architecture.html`** — três figuras: o laço autoregressivo (prever → escolher → realimentar), o caminho completo com os formatos e a etapa responsável por cada estágio, e os quatro marcos empilhados.
  - **`exercises.html`** — 10 questões de orientação, testando o mapa geral e não os detalhes.
- **Exemplos numéricos novos**, todos verificados por script: a distribuição sobre o vocabulário de `"abacate"` dado o contexto `"aba"` (o modelo aposta 72% em `'c'`, que é a resposta certa no corpus); o rastreamento de formatos `[2,4] → [2,4,8] → [2,4,8] → [2,4,5]`; e a **contagem de parâmetros** de uma configuração pequena (vocabulário 95, embedding 64, contexto 128, 4 blocos) — **220.511 parâmetros**, contra os 175 bilhões do GPT-3, ou seja ~794 mil vezes menor com a mesma arquitetura. Desse total, 91% vivem nos blocos, o que virou um exercício sobre onde o modelo guarda o que aprendeu.
- **Bug de renderização encontrado e corrigido na validação do SVG**, e vale registrar porque é uma pegadinha genérica: a classe CSS `.ct { text-anchor: middle }` **sobrescreve** o atributo de apresentação `text-anchor="start"` — em SVG, CSS vence atributo de apresentação. Os oito rótulos da FIG 3 que deveriam ficar alinhados à esquerda saíam centralizados, estourando o `viewBox` em até 148px à esquerda. Corrigido com uma classe dedicada (`.ct.start`), com comentário no CSS explicando o porquê. Um segundo rótulo, na FIG 1, estourava 12px à direita e foi reposicionado. Revalidado: nada fora dos limites nas três figuras.
- **Consolidação de duplicação:** o bloco "Como ler este capítulo" (a descrição dos quatro blocos destacados) estava na Etapa 1 e passou pro overview, que agora é o ponto de entrada. A Etapa 1 ganhou no lugar um ponteiro curto pro overview. O `STYLE-GUIDE.md` foi atualizado em dois pontos: removeu "Como ler" do esqueleto obrigatório dos capítulos de etapa, e ganhou uma nota no topo explicando que o overview é o ponto de entrada, que a notação canônica e a descrição dos blocos moram lá, e que **todo termo novo definido num capítulo deve ser acrescentado ao glossário do overview**.
- Verificação final: **10/10 documentos** aprovados no checklist completo (média 12,0 a 15,8 · p90 19 a 24 · parênteses 5,8 a 18,8). Total do material: 63 Definições, 27 Armadilhas, 30 blocos "Confira você mesmo", mais 116 questões nos 10 `exercises.html`.

---

## 2026-08-11 — Etapa 9 concluída: `Embedding`, mais duas aberturas de API em `scalagrad`

- **`Tensor.shape` virou público.** A camada precisava de `batchSize`/`seqLen` pra achatar os tokens e pra fazer o `reshape` de volta pra `[B, T, D]`, e `shape` era `private[scalagrad]` — de dentro de `gpt` só `rank`, `size`, `get` e `index` eram alcançáveis. Não é restrição específica desta etapa: LayerNorm (10) precisa da última dimensão, Attention (11/12) precisa de `B`/`T`/`C`, e o `GPT` (15) precisa do `seqLen`. O sintoma já estava no `LinearSpec`, cujo teste de shape só conseguia checar `rank` e `size`. Junto com a abertura, `Shape.toArray` passou a devolver `values.clone()` em vez do array interno — enquanto tudo era `private[scalagrad]` o aliasing era tolerável, mas com `shape` público qualquer código de fora poderia mutar o array e quebrar a imutabilidade que o `CLAUDE.md` define.
- **`Tensor.randn` ganhou `std` e `requiresGradient`** (`randn(shape, std = 1.0, requiresGradient = false)`, defaults preservando os chamadores antigos). Surgiu de uma duplicação sinalizada em revisão: o `Linear` rolava o gaussiano na mão (`Array.fill(n)(std * Random.nextGaussian())`) e o `Embedding` ia repetir o mesmo padrão. `Linear.initializeWeights` foi refatorado pra usar a fábrica nova e o import órfão de `Random` saiu.
- **`gpt/src/main/scala/gpt/nn/Embedding.scala`** implementado pelo usuário: as duas tabelas (`[vocabSize, embeddingDim]` e `[contextLength, embeddingDim]`, ambas `N(0, 0.02²)` — sem Kaiming, porque não há fan-in: a tabela é lida, não multiplica nada), `parameters: List[Tensor]`, e `embed(tokens: Tensor)` recebendo o `[batchSize, seqLen]` que o `BatchSampler` já produz. O forward achata os tokens num `Array[Int]` na ordem canônica (batch por fora, posição por dentro, lendo via `get(b, t).toInt` pra respeitar strides reais), faz `tokenTable.indexSelect(flat).reshape([B, T, D])`, e soma `positionTable.indexSelect(0 until seqLen)` de shape `[T, D]` — deixando o broadcast com left-pad espalhar sobre o batch. Nenhum backward manual: o `unbroadcast` do `+` (`BinaryOps.binary`) soma o gradiente posicional sobre o lote sozinho, que é exatamente a §5 da teoria.
- **Bug real encontrado na revisão:** `positionTable` nascia com `Array(embeddingDim, vocabSize)` — as duas dimensões erradas, quando o correto é `Array(contextLength, embeddingDim)`. O delator foi `contextLength` não aparecer em lugar nenhum da classe. Vale registrar **como falharia**, porque é o caso silencioso que o `roadmap/09-embedding.md` chama de "indexar a tabela errada": o `indexSelect` valida os índices contra o número de linhas, então `Array.range(0, seqLen)` passaria sem reclamar sempre que `embeddingDim >= seqLen` (cenário comum, ex.: `embeddingDim=64`, `seqLen=8`), a saída viria `[seqLen, vocabSize]`, e o erro só apareceria na soma final como broadcast incompatível — longe da causa. Também sinalizados e corrigidos: `parameters` sem anotação de tipo, mensagens de `require` sem os valores ofensores (convenção já firmada no `Tokenizer.decode` e no `indexSelect`), espaço em branco no fim de linha, e dois métodos `initialize*` que deixaram de se pagar depois do `randn` (viraram campos diretos).
- **`gpt/src/test/scala/gpt/nn/EmbeddingSpec.scala` criada, 11 testes.** Os três que carregam o peso: o do **mesmo token em duas posições** (se o passo do positional embedding sumisse, todos os outros continuariam passando); o de **reuso das linhas posicionais entre as sequências do batch**, que subtrai a linha do token de cada saída e confirma que o resíduo é idêntico nas duas sequências — o broadcast da §5 checado de fato; e os dois **gradient checks**, um na tabela de tokens com `[[1,3,1],[0,2,1]]` (o token 1 aparece três vezes, então a linha 1 tem que somar três contribuições, e a linha 5 nunca é usada e tem que ficar zerada) e outro na posicional com `batchSize=2`. Mais shapes, forward cruzado contra a leitura direta das tabelas, os três `require` (rank, `seqLen > contextLength`, índice fora do vocabulário — este confirmando que a validação do `indexSelect` continua alcançável através da camada), e um estatístico de `std ≈ 0,02`. Suíte completa: **218 testes** (181 `scalagrad` + 37 `gpt`), revalidada 3x sem flakiness.
- **`Tensor.gradient` também virou público**, pelo mesmo tipo de motivo do `shape`. A restrição apareceu ao escrever a `EmbeddingSpec`: com `gradient` sendo `private[scalagrad]`, testes e código em `gpt` não conseguiam ler gradiente nenhum, e o item do checklist "conferir que o gradiente esparso só afeta as linhas presentes" teve que ser cumprido via `Gradcheck` (que compara analítico contra numérico posição por posição da tabela inteira, então um vazamento pra linha não usada apareceria) em vez de asserção direta — mesma razão pela qual o `LinearSpec` já validava gradiente só assim. A abertura seria exigida de qualquer forma na Etapa 17: o AdamW mora em `gpt/optim` e precisa ler o gradiente de cada parâmetro pra atualizá-lo. A `Gradient` exposta continua segura por construção: sua API só oferece `accumulate`/`zero`/`seed` e leituras (`apply`, `get` com `clone`), sem `update` genérico — a decisão registrada na Etapa 2. **Consequência aproveitada no mesmo dia:** a `EmbeddingSpec` ganhou dois testes que afirmam a esparsidade **diretamente**, lendo `tokenTable.gradient(tokenTable.index(row, d))` — um exigindo gradiente exatamente zero nas linhas 4 e 5 (tokens ausentes do lote) e não-zero nas presentes, outro conferindo, posição por posição, que cada linha soma exatamente as contribuições que lhe cabem. A perda usada é `(embed(tokens) * weights).sum` com pesos todos distintos, e não `.sum` puro: gradiente uniforme daria o mesmo resultado com as contribuições trocadas de posição, escondendo justamente o erro de indexação que o teste procura — mesmo princípio que o `STYLE-GUIDE.md` §5 impõe aos exemplos numéricos. Com isso o item do checklist é cumprido de forma literal, e não mais por inferência a partir do `Gradcheck`. Suíte: **220 testes** (181 + 39).
- **Pendência de teoria resolvida no fim da etapa** (a notação `N(0, 0.02)` era ambígua e o usuário esbarrou nela em conversa): a Etapa 1 §4 teve o bloco Definição reescrito de `N(0,1)` pra `N(μ, σ²)` geral, explicitando que o segundo parâmetro é a **variância** e que a normal padrão é o caso `N(0,1)`; junto, a nota de que a literatura de ML escreve `N(0, 0.02)` querendo dizer desvio padrão, quando ao pé da letra isso daria desvio `√0,02 ≈ 0,141`, sete vezes maior. O `theory/00-overview/` ganhou a linha `N(μ, σ²)` na tabela de notação, uma quarta armadilha de notação, e duas entradas de glossário (distribuição normal e desvio padrão). Todas as ocorrências passaram pra `N(0, 0.02²)` com expoente explícito (`theory/08-linear`, `theory/09-embedding`, `roadmap/08-linear`, `roadmap/09-embedding`, `checklists/08-linear`), e o `STYLE-GUIDE.md` §7 ganhou a mesma linha na tabela canônica. Os 10 capítulos remedidos depois da edição: todos ainda `OK` no script do guia.
- **Sincronia `roadmap`/`checklist`:** os dois descreviam `embed(tokens: Array[Int])`; a assinatura real ficou `embed(tokens: Tensor)`, pra receber direto o que o `BatchSampler` produz e o que o `GPT.forward` vai passar na Etapa 15. Corrigido nos dois.
- `checklists/09-embedding.md`: **Etapa 9 concluída**. Próximo passo: Etapa 10 — LayerNorm (`checklists/10-layer-norm.md`), primeira etapa do bloco que ainda não tem teoria escrita; pelo fluxo combinado, `theory/10-layer-norm/` vem antes da implementação.

---

## 2026-08-11 — scalafmt adotado e projeto reformatado

- Criado `.scalafmt.conf` na raiz (scalafmt 3.9.4). As escolhas que não são default: `runner.dialect = scala3` (sem isso o formatador não entende `extension`, `if ... then` nem indentação significativa, e pode reindentar um bloco pra dentro ou pra fora de um `if` — risco real neste código, ver abaixo); `maxColumn = 100`, medido contra a distribuição real do projeto (3.042 das 3.371 linhas já cabiam em 80, 62 ficavam entre 101 e 120); `docstrings.wrap = no`, porque os scaladoc daqui são prosa em português quebrada à mão com ponteiros pra `theory/*.md`, e o reflow automático embaralharia esse texto; `align.preset = none`; e `RedundantBraces` deliberadamente **fora** do `rewrite.rules`, porque há chaves com intenção no código (`{ for {...} yield ... }.toArray` em `Embedding.embed`).
- **Projeto inteiro reformatado:** 20 dos 30 arquivos, 1.267 das 3.371 linhas. A massa é reindentação de 4 pra 2 espaços em `scalagrad/ops/` (`MatmulOps` sozinho, 232 linhas) — o repositório estava dividido, com `core/` e todos os specs em 2 espaços e `ops/`, `gpt/data/`, `gpt/nn/` em 4. Suíte revalidada depois: **218 testes verdes**, sem warnings novos. Backup do estado anterior guardado no scratchpad da sessão.
- **Risco checado antes de aplicar, e vale registrar por que:** o backward do `matmul2D` tem dois `foreach` irmãos sob um único `if Tensor.gradEnabled then`, e eles só estão dentro do `if` por causa da indentação. Um formatador sem o dialeto correto poderia movê-los pra fora, o que compilaria igual e faria o gradiente de B acumular mesmo dentro de `noGrad` — bug silencioso do tipo que este projeto já sabe ser o pior. Antes de tocar no código real, a configuração foi rodada sobre um clone completo no scratchpad e a suíte executada lá; o `if/then` sobreviveu intacto.
- **Sintaxe sem chaves (`class Foo:` em vez de `class Foo {`) avaliada e adiada, não descartada.** Testada de verdade (`rewrite.scala3.removeOptionalBraces = true`): remove ~108 linhas de `}` e passa os 218 testes. Dois motivos pra esperar. Primeiro, lambdas passadas como argumento (`.foreach { i => ... }`, `Tensor(...) { () => ... }`) **continuam com chaves** de qualquer forma, então o resultado não é um arquivo sem chaves — é um com chaves só nos pontos mais aninhados, que são justamente os backwards. Segundo, o modo braceless amplia a superfície do risco descrito acima bem na véspera das Etapas 11-12 (atenção multi-cabeça), onde os backwards ficam mais aninhados: com chaves, mover uma sentença pra dentro ou fora de um `if` exige editar a chave; sem chaves, dois espaços a menos compilam e mudam o significado. Se for adotado no futuro, é uma linha no `.scalafmt.conf`.
- Formatação no editor: o Metals lê o `.scalafmt.conf` e baixa a versão declarada sozinho, sem plugin no `build.sbt`. Formatar ao salvar é configuração do editor (`"[scala]": { "editor.formatOnSave": true }` no VS Code), não do Metals.

---

## 2026-08-11 — `theory/10-layer-norm/` escrita antes da implementação (Etapa 10)

- Seguindo o fluxo combinado ([[feedback_theory_before_implementation]]), criada `theory/10-layer-norm/` com os três arquivos do padrão. O capítulo tem 9 seções: o problema da escala das ativações; a fórmula passo a passo; o papel do `ε`; `γ`/`β`; a derivação do backward de `x`; as duas implementações possíveis; o contraste com BatchNorm; pre-LN contra post-LN; e a ligação com a Etapa 11.
- **Exemplo numérico único atravessando o capítulo inteiro:** `x = [2, 3, 6, 9]`, com `μ = 5`, `σ² = 7.5`, `s = 2.738615` e `x̂ = [-1.0954, -0.7303, 0.3651, 1.4606]`, mais `γ = [1, 0.5, 2, 1.5]` e `β = [0, 1, -1, 0.5]`. Todos os valores foram gerados e conferidos por script antes de virarem texto — inclusive os do backward, que reusa o mesmo vetor.
- **A derivação de `dx` é o miolo do capítulo**, e foi escrita como derivação, não como fórmula entregue pronta: os três caminhos de `xⱼ` até a saída (direto, pela média, pela variância), cada um virando um termo de `dxᵢ = (dx̂ᵢ - m₁ - x̂ᵢ·m₂)/s`. A leitura que amarra tudo é que os dois termos subtraídos removem exatamente as direções que o forward ignora — deslocamento e escala —, o que torna a fórmula previsível em vez de decorada.
- **Duas verificações independentes do backward**, ambas rodadas: contra diferenças finitas (`±1e-6`, erro máximo `1.3e-9` contra a fórmula fechada) e geometricamente (`Σ dx = -1.4e-16` e `Σ dx·x̂ = 3.8e-6`). O segundo resíduo não é zero exato por causa do próprio `ε`; refazendo a conta com `ε = 0`, cai pra `5.3e-16`. Isso virou conteúdo do capítulo, não nota de rodapé: são duas conferências que o usuário pode usar como teste da implementação.
- **`ε` ganhou tratamento além do "evita divisão por zero".** Com `x = [7, 7, 7, 7]` a variância é exatamente zero e sai `NaN`; mas o caso interessante é o quase-constante `x = [7, 7, 7, 7.001]`, onde sem `ε` uma diferença de um milésimo é amplificada até um desvio de `1.73`, e com `ε` fica em `0.235`. Também explica por que ele fica **dentro** da raiz.
- **Ganchos fechados com etapas anteriores:** o contraste com o problema da simetria da Etapa 8 (por que `γ = 1` em toda posição não é o mesmo erro que `W` constante — `γᵢ` só enxerga a posição `i`, enquanto duas colunas de `W` viam a mesma entrada); `dγ`/`dβ` como o mesmo desbroadcast do bias (Etapa 8 §3); e a promessa de que o padrão da derivação se repete no softmax (Etapa 6) e na atenção (Etapa 11).
- **As duas Armadilhas são bugs reais deste projeto**, como o `STYLE-GUIDE.md` exige: a divisão inteira `1 / H` (o mesmo `2 / inputDim` que zerou os pesos na Etapa 8) e um parâmetro criado sem `requiresGradient` (o bias `b` do `Linear`, também Etapa 8).
- **`layer-norm.html`** com três figuras: a normalização como duas operações geométricas numa reta numérica (subtrair a média move a origem, os pontos não saem do lugar; dividir por `s` contrai a régua), os três caminhos do backward mapeados um a um nos termos da fórmula, e o contraste LayerNorm/BatchNorm com o lote `[2,3,6,9]` contra `[200,300,600,900]` — onde BatchNorm transforma a linha A em `[-1,-1,-1,-1]` por causa da linha B.
- **Validação dos SVG sem preview visual** (mesmo processo das Etapas 8 e 9, e a captura de tela segue indisponível neste ambiente): `getBBox()` de todo elemento contra o `viewBox`, mais uma checagem nova de **colisão entre textos**, que a das etapas anteriores não fazia. Nada fora dos limites nas três figuras; a checagem de colisão pegou 1 problema real — o rótulo `0.37`, escalonado pra cima na FIG 1, encostava no título da linha por 3px. Corrigido alinhando-o aos vizinhos.
- **`exercises.html` com 12 questões** (fácil → desafio), validadas por script no navegador: as 12 com 4 opções, gabarito existente, letra da explicação batendo com o `data-answer`, e o caminho de resposta errada revelando a correta. A validação pegou um problema que passaria despercebido numa leitura: o gabarito tinha saído no padrão cíclico `c,b,d,a` repetido três vezes, o que permitiria acertar tudo sem ler as perguntas. Quatro questões tiveram as opções reordenadas; o gabarito agora é `cbbacddaabca`.
- Glossário do `theory/00-overview/` atualizado com os 6 termos novos (ativação, normalizar, Layer Normalization, ganho e deslocamento, BatchNorm, pre-LN/post-LN). Os 11 documentos remedidos no script do guia: todos `OK` (média 12,0 a 15,8 · p90 19 a 24 · parênteses 3,9 a 18,8).
- `checklists/10-layer-norm.md`: status "Em andamento" — teoria pronta, implementação (`LayerNorm` em `gpt/nn/`) ainda não começou. O roadmap recomenda começar pela Opção A (composição das ops existentes, com o autograd cuidando do backward); a Opção B, se vier depois, já tem a fórmula derivada e conferida na §5.

---

## 2026-08-12 — Etapa 10 concluída: `LayerNorm` e uma correção no que o projeto acreditava sobre testes de gradiente

- **`gpt/src/main/scala/gpt/nn/LayerNorm.scala`** implementado pelo usuário pela Opção A do roadmap — composição de `mean(dim, keepDim)`, `pow`, `sub` e `div`, sem backward manual. Construtor `(dim, eps = 1e-5)`, `γ = ones` e `β = zeros` treináveis, e `forward` validando rank antes de ler `shape.last`, no padrão já firmado. A revisão não encontrou bug: os dois `keepDim = true` no lugar, o `ε` dentro da raiz, e a ordem dos `require` correta.
- **Decisão de projeto registrada: a camada não exige rank 3.** Ela usa apenas `rank - 1` como eixo, então rank 1, 2 ou 3 funcionam igual. Exigir `[B, T, H]` acoplaria a camada a um formato que ela não precisa conhecer — mesmo raciocínio que tirou o `batchSize` do construtor do `Linear` na Etapa 8. Quem precisa restringir é o bloco que chama, não a camada. Efeito colateral bom: os testes puderam usar rank 2, que é o formato natural pra conferir contra a conta na mão.
- **Três apontamentos de revisão, todos corrigidos:** o tensor de `ε` estava sendo criado com o shape de `x` inteiro (`Tensor.fill(x.shape.toArray, eps)`) em vez de `Array(1)` — numericamente idêntico, porque `variance` broadcastaria de qualquer jeito, mas fazia a subárvore `(variance + ε).pow(0.5)` rodar sobre `H` vezes mais elementos e alocar um tensor de constantes do tamanho da entrada a cada passada; `Shape.last` estava implementado como `toArray.last`, que desde a mudança de visibilidade de hoje **clona** o array inteiro pra ler uma posição (virou `values(rank - 1)`); e a mensagem do segundo `require` não dizia o valor encontrado. Junto, `Tensor.zeros`/`ones`/`fill` ganharam `requiresGradient`, completando o padrão iniciado no `randn`.
- **`LayerNormSpec` com 16 testes.** Forward contra referência em Scala puro; os valores da §2 da teoria; média zero e variância um por linha; a invariância a escala e deslocamento da §1; vetor constante sem `NaN`; rank 1, 2 e 3; normalização independente por linha num lote rank-3 com escalas muito diferentes; os dois `require`; gradient check em `x`, `γ` e `β`; e as duas identidades da §5 (`Σ dx = 0` e `Σ dx·x̂ = 0`), agora possíveis de afirmar diretamente porque `Tensor.gradient` virou público na Etapa 9. Suíte: **236 testes** (181 `scalagrad` + 55 `gpt`).
- **A suíte foi submetida a mutação, não só rodada.** Num clone do projeto no scratchpad, `pow(0.5)` virou `pow(0.25)` (6 testes falharam) e um `keepDim = true` virou `false` (10 testes falharam). Vale registrar o que a primeira mutação revelou: **os gradient checks continuaram passando**, e corretamente — o autograd deriva com precisão qualquer forward, inclusive um errado. Gradient check valida a coerência entre forward e backward, nunca se o forward é o pretendido. Quem pega forward errado é a referência independente.
- **Correção de uma crença errada que estava registrada na `theory/`, descoberta ao medir.** Ao escrever a suíte, afirmei que `forward(x).sum` seria um "falso verde" — que aprovaria qualquer backward, por comparar zero analítico com zero numérico. Medido, é diferente: o gradiente analítico dá **exatamente** `0.0`, o numérico é ruído de arredondamento (~`2e-11`), e o erro relativo do `Gradcheck` divide esse ruído pelo piso de `1e-8`, resultando em `2.2e-3` — **acima** do limite de `1e-5`. Ou seja, o teste degenerado não aprova em silêncio: ele reprova código correto. O veredito dele, em qualquer direção, é ruído e não fala sobre o backward. Confirmado que o softmax degenera igual (`1.7e-3` contra `4.6e-10` com pesos). A conclusão prática não muda — a perda precisa ser ponderada com pesos distintos —, mas o motivo estava errado.
  - Corrigidos os três pontos de `theory/06-softmax/06-softmax.md` que diziam "aprova qualquer backward"/"aprova qualquer coisa" (§5, o bloco "Confira você mesmo" e a lição 4 do cartão), agora com os números medidos. A mesma nota entrou em `theory/10-layer-norm/10-layer-norm.md` §6, com os números do LayerNorm.
- `checklists/10-layer-norm.md`: **Etapa 10 concluída**, com a Opção B registrada como pendência opcional (a fórmula fechada já está derivada e conferida na §5). Próximo passo: Etapa 11 — Self-Attention (`checklists/11-attention.md`), o mecanismo central do transformer; teoria antes, pelo fluxo combinado.

---

## 2026-08-13 — `theory/11-attention/` escrita antes da implementação (Etapa 11)

- **`theory/11-attention/11-attention.md`** escrita no padrão do `STYLE-GUIDE.md`, em 9 seções: o problema da mistura entre tokens (§1), Q/K/V (§2), scores (§3), a escala `√dHead` (§4), a máscara causal (§5), softmax e média ponderada (§6), backward (§7), implementação e testes (§8), e a ligação com a Etapa 12 (§9). Métrica de prosa: média 12,2 · p90 21 · parênteses 4,8/1k — dentro das três metas do guia.
- **Um exemplo corrente atravessa o capítulo inteiro**, com 3 tokens, `dModel = 4` e `dHead = 2`. As matrizes `W_*` foram escolhidas pra que `Q`, `K` e `V` saiam inteiros (`Q = [[2,0],[0,3],[1,1]]`, `K = [[2,1],[2,2],[1,1]]`, `V = [[4,2],[0,3],[3,1]]`), o que deixa o leitor refazer cada passo na mão. Todos os números do capítulo saíram de um script em Python puro, incluindo forward, backward e a verificação por diferenças centrais (erro máximo `7.8e-10` em `dQ`, `1.1e-9` em `dK`, `1.6e-9` em `dV`).
- **A escala `√dHead` ganhou números, não só o argumento de variância.** Uma query contra 16 keys com `dHead = 64`: sem a divisão, uma posição fica com 72% do peso e quatro caem abaixo de `1e-6`; com ela, o menor peso sobe de `1.3e-13` pra `3.7e-3`. A medida de "posições efetivamente olhadas" (`exp` da entropia) vai de 2,59 pra 12,72 de 16. A variância empírica do produto interno em 50 mil pares bateu com a previsão (`64.03` contra `64`).
- **A ordem máscara/softmax virou contraste numérico.** Zerar depois do softmax deixa as linhas somando `0.446`, `0.903` e `1.000` no exemplo — e a última linha fica intacta, o que explica por que o defeito escapa de um teste que só olhe o fim da sequência.
- **Duas Armadilhas, as duas de bugs reais deste projeto:** ler `t.data(i)` em vez de `t.get(...)` (bug corrigido na revisão da Etapa 6, com esta etapa citada por nome como o cenário que o acordaria — `K.transpose()` é o primeiro tensor não contíguo do projeto); e usar `P.sum` como perda do gradient check, pela degenerescência já medida na Etapa 6 e recorrigida na Etapa 10.
- **Achado de implementação registrado antes de a implementação começar: `matmul` não aceita rank `(3,2)`.** `Linear.forward` faz `x.matmul(W)`, e o dispatcher só cobre `(2,2)` e `(3,3)`; a atenção quer `[B,T,dModel]` contra `[dModel,dHead]`. Ficou invisível até aqui porque `LinearSpec` só exercitou rank 2. As três saídas estão na §8 do capítulo e no checklist, com a terceira (`broadcastTo` + `matmul3D`) marcada como inviável — o `broadcastTo` compartilha o objeto `gradient` do tensor original, enquanto o backward do `matmul3D` acumula por índice canônico do formato novo. Também registrado que a máscara não é recortável hoje (não há operação de slice; `indexSelect` só seleciona linhas de rank 2), então construí-la por `Tensor.make` a cada forward é o caminho simples.
- **`attention.html` com 4 figuras:** o caminho completo com os formatos de cada passo e o value entrando só no fim; a matriz de scores do exemplo com duas células destacadas mostrando `4 ≠ 3` (a assimetria que `W_Q ≠ W_K` compra); os 16 pesos com e sem a divisão por `√dHead`, na mesma escala; e as duas ordens de máscara lado a lado, com a soma de cada linha.
- **Validação dos SVG sem preview visual** (mesmo processo das Etapas 8 a 10; a captura de tela segue indisponível): `getBBox()` de todo elemento contra o `viewBox`, colisão texto-contra-texto, e uma checagem nova de **texto atravessando a borda de uma caixa**. Essa terceira pegou um defeito real e não óbvio: `text-anchor="end"` como atributo de apresentação **perde** para a regra CSS `.cell-text { text-anchor: middle }`, então os rótulos do eixo vertical estavam centrados em vez de alinhados à direita, e invadiam a primeira barra em 3,8px. Corrigido com `style="text-anchor:end"`, que tem precedência. O mesmo padrão existe nos `.html` das etapas anteriores, sem colisão visível — vale conferir numa próxima passada.
- **`exercises.html` com 14 questões** (fácil → desafio), validadas por script no navegador: 4 opções cada, gabarito existente, letra da explicação batendo com o `data-answer`, badge de dificuldade presente, e resposta errada revelando a correta. O gabarito (`bdcacbdadbcadb`) foi conferido contra padrão cíclico de período 2, 3 e 4 — nenhum —, com distribuição 3/4/3/4 entre as quatro letras.
- Glossário do `theory/00-overview/` atualizado com os 7 termos novos (atenção, auto-atenção, query/key/value, score de atenção, peso de atenção, máscara causal, cabeça). Isso fecha a dívida apontada na auditoria de 2026-08-10, que listava `atenção` como termo usado e nunca definido.
- `checklists/11-attention.md`: status "Em andamento" — teoria pronta, implementação não começou. O checklist ganhou os quatro testes de propriedade da §8 (causalidade, `y₀ = v₀`, linhas somando 1, envelope convexo) e as duas notas de implementação acima. Próximo passo: decidir a rota do `matmul` rank `(3,2)` e escrever a camada de atenção em `gpt/nn/`.

---

## 2026-08-13 — `matmul` passa a aceitar rank `(3,2)`: uma matriz compartilhada por todo o lote

- **`matmul3D` virou `matmulBatched`** ([MatmulOps.scala](scalagrad/src/main/scala/scalagrad/ops/MatmulOps.scala)), aceitando `(B,M,K) x (B,K,N)` e `(B,M,K) x (K,N)` no mesmo corpo. Implementado pelo usuário. A diferença entre os dois casos colapsa em dois acessores (`if sharedRhs then t2.get(k, n) else t2.get(b, k, n)`, e o mesmo para o índice do gradiente), com `sharedRhs` içado pra fora dos laços. O dispatcher manda `(3,2)` e `(3,3)` pro mesmo lugar.
- **A soma sobre o lote sai de graça.** Com o laço de `b` por fora e o índice do gradiente colapsando pra `(k, n)` quando o operando é rank 2, todas as fatias acumulam na mesma posição de `W` — e `Gradient` só sabe somar, nunca sobrescrever (decisão de design da Etapa 2, registrada no `CLAUDE.md`). O `dW[k,n] = Σ_b Σ_m dC[b,m,n]·A[b,m,k]` é consequência estrutural da API, não um caso especial no código.
- **Dois bugs pegos na revisão da primeira versão, os dois invisíveis pra suíte de então:** o `require` de batch estava escrito como `t1.shape(0) == t2.shape(0) && t2.rank == 3`, exigindo rank 3 **sempre** e bloqueando o caso novo inteiro; e o backward tinha ficado idêntico ao antigo, lendo `t2.get(b, k, j)` (três índices contra um shape rank 2) e tirando os limites dos laços de `t2.shape(0/1/2)` em vez de `B`/`M`/`K`/`N`. Os 115 testes do `scalagrad` continuavam verdes com o caminho `(3,2)` 100% quebrado — nada na suíte o tocava.
- **`Linear` não precisou de mudança nenhuma.** Com o `matmul` cobrindo `(3,2)`, `x.matmul(W) + b` funciona com `[B, T, inputDim]` direto, e o `b` de formato `[outputDim]` broadcasta sobre lote e sequência com o `unbroadcast` da Etapa 3 somando de volta. A camada da Etapa 8 atende a Etapa 11 sem saber que a sequência existe.
- A pedido explícito do usuário (mesma exceção de sempre à regra de "Claude não escreve `.scala`" — mudanças cosméticas combinadas em conversa), Claude reescreveu as mensagens dos `assert`/`require` dos dois métodos (agora dizendo os ranks/dimensões encontrados, no padrão firmado na Etapa 10) e renomeou os índices de `matmul2D` de `i`/`j` pra `m`/`n`, alinhando com a notação nova do `matmulBatched` — `m`, `n` e `k` indexam `M`, `N` e `K`, então o nome do índice passa a dizer de qual dimensão ele veio. Também entrou um comentário curto explicando por que o `require` de batch tem a cláusula `t2.rank == 2`.
- **8 testes novos em `TensorOpsSpec` e 2 no `GradcheckSweepSpec`.** Cobertura: forward com `B`, `M`, `K` e `N` todos diferentes (a armadilha de shapes quadradas da Etapa 3 §6); equivalência fatia a fatia contra o `matmul` 2D; backward completo contra valores conferidos por diferenças centrais fora do projeto; **duas fatias idênticas dando exatamente o dobro do `dB` do teste 2D** — é o único teste que pega um `Σ_b` faltando, e ele exige `dC` distinto dentro da fatia e idêntico entre fatias; operando não contíguo (`x.transpose(1,2).matmul(w)`, o cenário do `K.transpose()` da Etapa 11); rejeição de dimensão interna incompatível; e gradient check nos dois operandos, incluindo o caso não contíguo.
- **3 testes novos em `LinearSpec`, fechando o buraco de cobertura que escondeu a limitação:** forward com entrada `[B, T, inputDim]` conferido posição a posição contra Scala puro, uma verificação de que cada posição recebe o mesmo resultado que receberia sozinha (mistura entre tokens é trabalho da atenção, nunca do `Linear`), e gradient check em `x`, `W` e `b` com entrada rank 3. Suíte: **247 testes** (189 `scalagrad` + 58 `gpt`).
- Nota de ferramenta: o `sbt test` no sbt 2.0.2 usa cache de build e reroda só as suítes afetadas. Pra forçar a suíte inteira, `sbt "scalagrad/Test/testOnly *" "gpt/Test/testOnly *"`.
- Pendências de sincronia da `theory/` abertas por esta mudança: `theory/03-elementary-operations` §6 descreve o lote só como `[B,M,K] @ [B,K,N]`, e `theory/11-attention` §8 ainda apresenta as três rotas como decisão em aberto.

---

## 2026-08-13 — `theory/` sincronizada com o `matmul` de operando compartilhado

- **`theory/03-elementary-operations` §6 ganhou a subseção "Um operando compartilhado pelo lote"**, cobrindo `[B,M,K] × [K,N]`: forward igual ao do lote com o índice `b` removido do segundo operando, `dA` fatia a fatia, e `dW` com a soma sobre `b`. O texto liga esse `Σ_b` ao desbroadcast da §4 — quem foi compartilhado no forward soma no backward —, para que não seja lido como regra nova. Exemplo numérico com `B=2, M=2, K=2, N=3`, conferido por dois caminhos: as contribuições parciais das duas fatias somam o `dW` total posição a posição, e todos os valores batem por diferenças centrais.
- **A subseção "Multiplicação em lote" também ganhou exemplo numérico**, que ela não tinha — o guia exige um por subseção. A fatia 0 é o mesmo exemplo 2D de cima, o que transforma o exemplo num teste: se um índice vazasse entre fatias, `C[0]` mudaria.
- **`matmul.html` ganhou a FIG. 4** (o operando compartilhado, com `W` como a única matriz sem "sombra" de fatias, e a seta do backward saindo de `dC[0] + dC[1]` para o mesmo `dW`), mais uma linha na referência rápida. As seções foram renumeradas (§5 novo, referência rápida virou §6). Validação por script no navegador: nada fora do `viewBox`, sem colisão de texto, sem texto atravessando caixa. Também corrigido o rodapé, que ainda dizia "matmul ainda não implementado" desde antes da Etapa 3 fechar.
- **`theory/11-attention` §8 reescrita na parte que apresentava as três rotas como decisão em aberto** — o guia proíbe seção descrevendo decisão já tomada. O texto agora registra que a limitação caiu nesta etapa, por que a rota 2 venceu, e o efeito prático: o `Linear` da Etapa 8 atende a atenção sem uma linha de mudança. As três rotas continuam listadas, mas como histórico da escolha. O título da seção deixou de falar em "um buraco no `matmul`".
- **2 exercícios novos no quiz da Etapa 3** (agora 22), sobre o formato `[B,M,K] × [K,N]` e sobre o que muda no backward. Uma das alternativas erradas é "`dW` vira a média das fatias" — vale como distrator porque a regra da cadeia soma caminhos, nunca os promedia.
- **Achado ao validar os quizzes: os três primeiros têm gabarito desbalanceado.** Medido em todos os 12: Etapas 1, 2 e 3 nunca têm `d` como resposta certa, e a Etapa 3 tem `a` em 13 das 22 questões (59%). Da Etapa 4 em diante a distribuição é equilibrada — os quizzes antigos são anteriores à lição registrada na Etapa 10, quando um gabarito cíclico foi pego por script. Não corrigido ainda: reordenar as opções exige reescrever as explicações que citam a letra da alternativa.
- Prosa remedida nos 12 capítulos, todos `OK` (média 12,0 a 15,8 · p90 19 a 24 · parênteses 4,2 a 18,8).

---

## 2026-08-18 — Etapa 11 em andamento: forward da `Attention`, e padronização de nomes para o padrão da literatura

- **`gpt/nn/Attention.scala` implementado pelo usuário**: três `Linear(dModel, dHead)` para `Q`/`K`/`V`, `causalMask(seqLen)` construída por `Tensor.make` a cada forward, e o caminho `q.matmul(k.transpose()) / scale` → `+ mask` → `softmax(rank - 1)` → `.matmul(v)`. O `transpose()` sem argumentos entrega `[B, dHead, T]` direto, que é exatamente o motivo do default `(rank-2, rank-1)` decidido na Etapa 3.
- **Forward verificado por 5 checagens descartáveis** (apagadas em seguida; a suíte permanente é o próximo passo), com `B=2, T=3, dModel=4, dHead=2`, todos diferentes entre si: formato `[B, T, dHead]`; `parameters` com 6 tensores treináveis; **`y₀ == v₀` com tolerância `1e-15`**; causalidade exata (perturbar `+3.0` a última posição não move nenhuma saída anterior em nenhum bit); e envelope convexo. As cinco passaram de primeira.
- **Nota de teste que vale reusar na suíte permanente:** as quatro propriedades da §8 são todas alcançáveis pela superfície pública, sem expor os `Linear` internos — `parameters` é `[Wq, bq, Wk, bk, Wv, bv]`, então o teste reconstrói `v = x.matmul(parameters(4)) + parameters(5)` sozinho.
- **A ordem escala-antes-da-máscara não é estética, é o que evita um `NaN`.** O backward de `/` em `BinaryOps` é `(a, b, g) => g * (-a / (b * b))` — ele lê o valor do operando. Com a máscara aplicada antes da divisão, `a` seria `-inf` nas posições bloqueadas e `g` seria `0` (o softmax zera ali), e `0 * inf = NaN` iria parar no gradiente do `scale`. O backward de `+` ignora `a`/`b` e só repassa `g`, então a máscara é segura depois. Registrado como comentário curto no próprio arquivo, já que é detalhe de implementação e não está na `theory/`.
- **Três defeitos corrigidos nas mensagens de `require`** (a pedido do usuário, mesma exceção de sempre à regra de "Claude não escreve `.scala`"): `s"...$x.rank"` interpolava **`x`** seguido do literal `".rank"` — como `Tensor.toString` imprime o tensor formatado em várias linhas, a mensagem de erro virava um despejo do tensor inteiro; o formato anunciado era `(B, T, T)` em vez de `(B, T, dModel)`; e a segunda mensagem chamava `dModel` de "number of input tokens". Reescritas no padrão do `LayerNorm`.
- **Legibilidade, no mesmo pedido:** `dHeadT` → `scale` (o campo guarda `sqrt(dHead)`, não `dHead`), `createMasking(seqTokenLen)` → `causalMask(seqLen)`, e `val T` → `val seqLen`. O `T` maiúsculo não é só estética: identificador maiúsculo em Scala lê como parâmetro de tipo, e um erro de tipo perto dele fez o compilador responder *"No ClassTag available for T, where T is a type variable"* durante a verificação.

### Padronização de nomes: `embeddingDim` → `dModel`, `Embedding.embed` → `forward`

Decisão do usuário, disparada por uma inconsistência apontada na conversa: `Embedding` chamava de `embeddingDim` a mesma grandeza que `Attention` chama de `dModel`. Dois nomes para o mesmo número dentro do mesmo modelo cobram imposto de leitura, e a Etapa 15 vai ligar as duas camadas. Critério firmado: **seguir sempre o nome da literatura**.

- **`embeddingDim` → `dModel`** em `Embedding.scala`, `EmbeddingSpec.scala`, `checklists/09-embedding.md`, `roadmap/09-embedding.md`, `theory/09-embedding/` (capítulo, `embedding.html` e `exercises.html`) e `theory/10-layer-norm/10-layer-norm.md`.
- **`Embedding.embed(tokens)` → `Embedding.forward(tokens)`**, pelo mesmo critério: `Linear`, `LayerNorm` e `Attention` já expunham `forward`, e o `forward` é o nome do PyTorch. Uma camada com nome de método diferente das outras quebraria a uniformidade justamente na Etapa 14/15, onde os blocos são compostos. A mensagem de `require` que dizia `"embed expects..."` também foi atualizada.
- **Não mudados, deliberadamente:** `Linear(inputDim, outputDim)` e `LayerNorm(dim)` — as duas camadas são genéricas e não sabem que `dModel` existe; amarrá-las ao vocabulário do transformer seria estreitar a API sem ganho. `contextLength` — é o termo da literatura ("context length"), e `blockSize` do nanoGPT é o apelido de uma implementação, não o padrão. O nome da classe `Attention` — coerente com os títulos das próprias Etapas 11 e 12 do `PLAN.md` (`Attention` e, depois, `MultiHeadAttention`).
- **`HISTORY.md` não foi reescrito**: as entradas antigas continuam dizendo `embeddingDim`/`embed`, porque um log registra o que aconteceu na época. Esta entrada é o ponto de virada.
- Suíte completa rodada depois de tudo: **247 testes verdes** (189 `scalagrad` + 58 `gpt`).

---

## 2026-08-18 — Etapa 11 concluída: suíte da `Attention`, mutação como controle, e `bk` provado morto

- **`AttentionSpec` com 18 testes**, escritos por Claude sob a autorização permanente de testes. Cobertura: comparação posição a posição contra uma **referência do bloco inteiro em Scala puro** (não usa nenhuma operação de `scalagrad`, só lê os 6 parâmetros); formatos variados, incluindo `(1,1,1,1)` e casos com `T ≠ dHead`; entrada não contígua; formatos e `requiresGradient` dos 6 parâmetros; as quatro propriedades da §8; e gradient check.
- **Dois testes que não estavam no checklist original e valem o espaço:**
  - **Contraprova da causalidade.** O teste de causalidade sozinho é passado por uma máscara que bloqueie *tudo* fora da diagonal. O par novo exige que mexer no primeiro token **mude** a última saída.
  - **Regressão de projeção compartilhada.** Reusar uma única `Linear` para `Q`, `K` e `V` deixa os formatos certos, o forward roda e o gradient check passa — só a assimetria de `scores` denuncia. O teste compara os três pesos e exige três conjuntos distintos.
  - **Sequência constante.** Com todos os tokens iguais, toda saída tem que valer `v₀` — mas só se cada linha de `P` somar 1. É a verificação externa da soma das linhas, que não é observável pela API pública de outra forma.
- **A suíte foi validada por mutação**, precaução herdada da lição do `matmulBatched` (115 testes verdes com o caminho `(3,2)` totalmente quebrado). Três mutações no `Attention.scala`, cada uma revertida em seguida: remover a divisão por `√dHead` → 1 teste falha (só a referência em Scala puro; esperado, a escala preserva todas as propriedades estruturais); inverter a máscara para triangular superior → **5 falham**; fazer `key` reusar a projeção de `query` → 1 falha, exatamente o teste escrito para isso.
- **Achado real: `bk` tem gradiente exatamente zero, sempre.** O gradient check reprovava o parâmetro 3 em 8 de 10 execuções, com erro relativo `1.1e-3`. Não era bug: somar uma constante a **todas** as keys desloca a linha inteira de scores pelo mesmo valor (o termo extra é `q_i · bk`, que não depende de `j`), e o softmax é invariante a isso. Medido: `maxAbs(dbk) = 2.2e-16` contra `O(1)` nos outros cinco. É o mesmo fato de que o gradiente que atravessa o softmax soma zero ao longo da dimensão normalizada, visto por outro ângulo.
  - **Com `bq` não acontece**, e a assimetria é instrutiva: deslocar as queries acrescenta `bq · k_j`, que **varia** com `j` e portanto muda a distribuição. Medido `8.6e-1`.
  - **Por que o `Gradcheck` reprova um backward correto aqui:** a métrica é relativa, `|a-n| / max(|a|, |n|, 1e-8)`. Com o gradiente verdadeiro zero, `a ≈ 1e-16` e o ruído de diferenças finitas dá `n ≈ 1e-11`; o piso `1e-8` do denominador transforma isso em `~1e-3`. É a primeira vez no projeto que o `Gradcheck` encontra um gradiente legitimamente nulo — vale lembrar disso quando a Etapa 12 ou a 16 reprovarem algo com erro na casa de `1e-3` e o forward parecer certo.
  - **Solução adotada:** `bk` sai do laço de gradient check (que roda nos outros cinco) e ganha um teste direto, `grad == 0` com tolerância `1e-14`, mais a contraprova de que `bq` no mesmo backward é não trivial. O teste direto é mais forte e mais informativo que o check numérico.
  - **Decisão registrada em aberto:** `bk` é parâmetro morto que o otimizador da Etapa 17 vai carregar para sempre. Um `useBias: Boolean` no `Linear` resolveria, e modelos recentes (LLaMA) dispensam viés nas três projeções. Não mexido agora para não alterar a camada da Etapa 8 no meio da 11 — reavaliar na Etapa 12.
- **Estabilidade conferida:** 10 execuções seguidas da suíte, 18/18 em todas. Antes da correção do `bk`, 8 de 10 falhavam.
- Suíte total: **265 testes** (189 `scalagrad` + 76 `gpt`). `checklists/11-attention.md` marcado como concluído, com o achado do `bk` e a nota do `slice` (que deve nascer na Etapa 19, pelo `logits[-1]`).
- Próximo passo: `theory/12-multi-head-attention/` antes da implementação, como manda a convenção. A lacuna de `scalagrad` que a Etapa 12 vai abrir já está identificada: o roadmap divide as cabeças por `reshape` para `[B,T,nHeads,dHead]` e `transpose(1,2)`, o que exige **`matmul` com duas dimensões de lote** — o `matmulBatched` de hoje cobre uma só.


---

## 2026-08-21 — `sbt test` não roda a suíte inteira (armadilha de falso verde)

Descoberto durante a validação do `useBias` no `Linear` (Etapa 12, item 1). O projeto está no **sbt 2.0.2**, e nessa versão a task `test` é cache-aware: ela delega pro `testQuick`, que roda **apenas as suítes afetadas** desde a última execução. O log é explícito — `No tests to run for gpt / Test / testQuick`.

- **O sintoma é um falso verde.** Depois de mexer só no `Linear.scala`, um `sbt test` rodou **30 testes** (`LinearSpec` + `AttentionSpec`, as duas suítes afetadas) e imprimiu `All tests passed`. A leitura natural desse output é "os 265 passaram".
- **`clean` antes não resolve.** Um `sbt clean test` rodou **zero** testes e ainda terminou em `[success]`: o cache de disco restaurou tudo (`cache 100%, 88 disk cache hits`) e o `testQuick` concluiu que nada mudou.
- **Vale nos dois níveis.** `sbt gpt/test` tem o mesmo comportamento — verificado logo após uma execução completa: `Passed: Total 0`.
- **O comando correto é `sbt "testOnly *"`** (raiz), ou `sbt "scalagrad/testOnly *"` / `sbt "gpt/testOnly *"` por módulo. Verificado: 189 + 76 = **265 testes**, e o módulo sozinho roda as 6 suítes / 76 testes.
- **Por que isso é grave neste projeto especificamente:** a suíte é a única rede de proteção contra bugs de gradiente, que são silenciosos por natureza. É o mesmo gênero de defeito da lição do `matmulBatched` (115 testes verdes com o caminho `(3,2)` totalmente quebrado) e da validação por mutação adotada na Etapa 11 — só que aqui a mentira não está na cobertura dos testes, está no **relatório de execução**.
- **Consequência retroativa, registrada por honestidade:** as contagens de "N testes verdes" anotadas neste log depois de mudanças recentes podem ter sido execuções parciais, dependendo de quando o sbt 2 passou a ser usado. As contagens continuam válidas como *tamanho* da suíte (foram conferidas em execuções completas em vários pontos), mas "rodei e passou" nas entradas recentes merece essa ressalva.
- **`CLAUDE.md` corrigido** na seção "Testes": a linha que dizia que `sbt test` na raiz roda a suíte dos dois módulos foi substituída por um item destacado com o comando certo e o motivo. A frase "rodar a suíte", usada em várias autorizações permanentes deste projeto, passa a significar `testOnly *`.

### E tem um segundo andar, pior: o action cache serve `.class` obsoleto

Descoberto minutos depois, ao validar o `useBias = false` na `Attention`. O teste novo dizia `6 was not equal to 5` — a camada continuava expondo 6 parâmetros com o fonte já corrigido no disco.

- **Não era o teste, era o bytecode.** `javap -p -c` no `Attention.class` mostrou as três projeções chamando `Linear$.$lessinit$greater$default$3()`, o *default* do terceiro parâmetro — inclusive a `key`, que no fonte passa `useBias = false` explicitamente. A classe em execução tinha sido compilada de uma versão anterior do arquivo.
- **`clean` não resolve. `touch` não resolve. `rm -rf target/` não resolve.** Todos foram tentados, nessa ordem, e a cada execução o `.class` velho voltava. O motivo: `target/out/.../classes/` é um conjunto de **symlinks** pra um armazenamento content-addressed global em `~/AppData/Local/sbt/v2/cas`, e o `~/AppData/Local/sbt/v2/ac` (action cache, 1.893 entradas na hora) devolvia "hit" para a compilação do módulo `gpt` mesmo com o fonte alterado.
- **A correção foi `rm -rf ~/AppData/Local/sbt/v2/ac`**, preservando os 50 MB do `cas`. Na execução seguinte o sbt saiu de `cache 100%` para `cache 46%, 25 onsite tasks`, e a contagem de parâmetros virou 5, como esperado.
- **Por que isso é mais grave que o `testQuick`:** lá a mentira estava no relatório de execução; aqui está no *código que roda*. O fluxo natural — editar, rodar a suíte, ver verde — testa a versão anterior do arquivo e não dá nenhum sinal.
- **Método de diagnóstico que vale guardar:** quando uma mudança parece não ter efeito, comparar fonte com bytecode (`javap -p -c -classpath target/out/jvm/scala-3.8.4/gpt/classes gpt.nn.<Classe>`) separa "meu raciocínio está errado" de "não é o meu código que está rodando". Foi o que fechou o caso; antes disso, três hipóteses plausíveis sobre a semântica de `Option.when` e de argumentos nomeados foram levantadas e descartadas por especulação pura.
- `CLAUDE.md` ganhou um item na seção "Testes" com o sintoma, a correção e o comando de `javap`.

---

## 2026-08-21 — Etapa 12, item 1: `useBias` no `Linear` e o fim do parâmetro morto

Fecha a pendência deixada em aberto na Etapa 11 ("`bk` é parâmetro morto que o otimizador da Etapa 17 vai carregar para sempre — reavaliar na Etapa 12").

- **`Linear` ganhou `useBias: Boolean = true`** (implementado pelo usuário). O viés virou `Option`: `Option.when(useBias)(initializeBias())`, `parameters = W :: b.toList`, e `forward` fecha com `b.fold(z)(z + _)`. O default preserva as Etapas 8/9/10 inteiras — nenhuma das 16 construções existentes de `Linear` mudou de comportamento.
- **`Attention` passou a construir `key` com `useBias = false`.** `parameters` caiu de 6 para 5: `[Wq, bq, Wk, Wv, bv]`.

### O que foi decidido, e com base em quê

O critério adotado foi **desligar o viés só onde é provadamente grátis**, não onde a convenção manda:

- **`b_K` sai.** Gradiente exatamente zero, sempre — somar constante a todas as keys desloca a linha inteira de scores pelo mesmo valor (o termo extra é `q_i · b_K`, independente de `j`) e o softmax é invariante a isso. Provado e medido na Etapa 11 (`2.2e-16` contra `O(1)` nos outros cinco).
- **`b_Q` fica.** Deslocar as queries acrescenta `b_Q · k_j`, que **varia** com `j` e portanto muda a distribuição de atenção. É um viés funcional e não redundante — um "prior global de atenção". Tirá-lo (estilo LLaMA) seria redução real de capacidade decidida por convenção, não por prova.
- **`b_V` fica agora, mas sai na MHA.** Derivação nova, feita nesta conversa: como cada linha de `P` soma 1, `P·(X·W_V + 1·b_V) = P·X·W_V + 1·b_V` — o viés atravessa a atenção intacto e vira uma constante somada a toda posição. Na Etapa 12, com `W_O` em cena, a saída depende de `b_V` **apenas** através de `c·W_O + b_O` (com `c` a concatenação dos `b_V` das cabeças): `b_V` é **exatamente redundante com `b_O`**, uma direção plana da loss. Na `Attention` de uma cabeça só não há `W_O`, então lá ele é o único viés de saída e permanece.
- **Os dois casos mortos têm a mesma origem**, o que vale como intuição: são as duas propriedades estruturais da linha do softmax — invariância a shift constante da entrada (mata `b_K`) e linhas somando 1 (torna `b_V` absorvível por `b_O`).
- **Opção descartada conscientemente:** GPT-2 mantém todos os vieses, e o nanoGPT tem um flag `bias` com default `True` justamente para reproduzi-lo. Não mexer seria defensável. O que pesou a favor foi o item "contar parâmetros (~`4·dModel²`)" do próprio `checklists/12-multi-head-attention.md` — a conta da literatura pressupõe ausência de viés — e o desaparecimento do caso especial no laço de gradient check.

### Suítes

Escritas por Claude sob a autorização permanente de testes. Total: **270 testes** (189 `scalagrad` + 81 `gpt`), com 6 execuções seguidas do módulo `gpt` sem instabilidade.

- **`AttentionSpec` renumerado, com índices nomeados.** A lista `parameters` já tinha renumerado duas vezes, e `att.parameters(4)` cru não denuncia nada quando o layout muda. Agora há `qWeight`/`qBias`/`kWeight`/`vWeight`/`vBias` como constantes no topo da spec. O helper `project` passou a receber `b: Option[Tensor] = None`, espelhando o `Option` do próprio `Linear`.
- **O teste `bk.grad == 0` foi removido** — o parâmetro não existe mais. No lugar ficou `"give bq a non-trivial gradient"`, que documenta a assimetria pelo lado que sobrou: `b_Q` aprende, um viés em `key` nunca aprenderia. A explicação matemática migrou para o comentário do teste de `parameters`, que é onde a decisão estrutural mora.
- **O laço de gradient check perdeu a exceção.** Antes rodava em `Seq(0,1,2,4,5)` pulando `bk`; agora roda nos cinco parâmetros sem ressalva, porque todos têm gradiente não trivial.
- **`LinearSpec` ganhou 5 testes de `useBias = false`**: `parameters` com um único elemento, `y = xW` sem termo constante, entrada zero mapeando para saída exatamente zero, entrada rank 3 (o caminho de produção da atenção hoje), e gradient check em `x` e `W`.
- **Limitação registrada dentro da própria spec, para não dar falsa confiança:** `initializeBias` preenche o viés com zeros, então no instante da construção uma camada com viés e outra sem produzem o **mesmo** forward, bit a bit. Nenhum teste de saída distingue as duas. A diferença observável é `parameters` — e é ela que importa, porque é a lista que o otimizador da Etapa 17 vai percorrer. Os testes de forward cobrem outra coisa: que o ramo `b.fold(z)(z + _)` devolve `z` intacto.
- **Validação por mutação, obtida de graça pelo bug do cache.** Enquanto o `.class` obsoleto ainda rodava (`key` **com** viés, suítes novas já escritas), **5 testes falharam** — os quatro que leem `v` ou a referência completa do bloco, mais o de `parameters`. É exatamente o experimento que a Etapa 11 fez à mão: a suíte nova de fato detecta o viés voltando.

### Pendências abertas

- **`b_V` na Etapa 12:** quando a `MultiHeadAttention` for escrita, as projeções `value` das cabeças devem nascer com `useBias = false`, e o viés fica só em `W_O`. A derivação está acima; falta aplicar.
- **`parameters` posicional continua frágil.** Os índices nomeados na spec resolvem o sintoma, não a causa. Se doer uma terceira vez, o caminho é expor `weight`/`bias` no `Linear` (é o que o PyTorch faz) em vez de indexar a lista.

---

## 2026-08-21 — `theory/12-multi-head-attention/` escrita antes da implementação (Etapa 12)

Segue a convenção firmada na Etapa 4: o capítulo nasce antes do código. Três arquivos, todos no padrão do `theory/STYLE-GUIDE.md`, que foi relido antes de escrever.

- **`12-multi-head-attention.md`** — 10 seções numeradas, mais abertura e cartão de referência. O fio condutor é que multi-head **não custa nada**: como `H · dHead = dModel`, `H` cabeças têm exatamente o mesmo número de pesos que uma cabeça de tamanho `dModel`. O ganho vem de ter `H` distribuições de atenção em vez de uma.
- **`heads.html`** — quatro figuras. FIG 1, a divisão em dois passos, com o rótulo `10t + d` em cada célula para dar rastreabilidade; FIG 2, o resultado do `reshape` sozinho, com as cores mostrando o embaralhamento; FIG 3, o caminho de volta e a única cópia obrigatória do bloco; FIG 4, as setas de `W_O` atravessando a fronteira entre cabeças.
- **`exercises.html`** — 14 questões (4 fácil, 5 médio, 4 difícil, 1 desafio), mesma identidade visual e mesmo motor de placar dos capítulos anteriores.

### Duas contribuições que não estavam no roadmap

**O argumento de por que uma cabeça é limitada.** O roadmap diz que cada cabeça "aprende relações diferentes", o que é verdadeiro mas não explica nada. O capítulo troca isso por um argumento estrutural: em `y_i = Σⱼ p_ij · v_j`, o peso `p_ij` **não tem índice de coordenada**. A decisão de para onde olhar é tomada uma vez por posição e vale para o vetor inteiro. Demonstrado com o menor exemplo possível — dois tokens, `v₀ = [7,2]`, `v₁ = [3,9]`, alvo `[7,9]`. Uma cabeça erra em qualquer `p`, com mínimo `12.06154` em `p = 32/130`; duas cabeças com `dHead = 1` acertam exato. O mínimo foi confirmado por varredura numérica de 200 mil pontos, como verificação independente.

**A redundância `b_V` ↔ `b_O`, derivada nesta conversa.** Já estava registrada na entrada do `useBias` de hoje; o capítulo a apresenta como §8, junto com o `b_K`. O ponto que amarra os dois casos e que vale guardar: **ambos saem das duas propriedades estruturais da linha do softmax** — invariância a deslocamento constante (mata `b_K`) e linhas que somam 1 (faz `b_V` ser absorvido por `b_O`). São o mesmo fato visto de dois ângulos.

### Verificação

- **Todos os exemplos numéricos conferidos por script** antes de entrar no texto, com asserções (dois scripts no scratchpad da sessão). Inclui os pesos de atenção das duas cabeças, a contagem de parâmetros para `dModel = 4` e `768`, a aritmética de índice linear do `reshape`, `dC = dY·W_Oᵀ` conferido posição a posição, e as duas provas de viés morto.
- **Prosa medida** com o script do §9 do guia: média `11.8` palavras por frase, p90 `19`, `4.3` parênteses por mil palavras. Dentro das três metas, e alinhado com os 12 capítulos anteriores.
- **Os quatro `.html` validados no navegador.** Duas figuras estouravam o `viewBox` à direita (13.4 px e 31.1 px) e foram corrigidas para `700` de largura; as quatro agora cabem inteiras. Console sem erros. O quiz foi exercitado por script: 14 questões, 4 alternativas cada, toda `data-answer` apontando para uma alternativa existente, placar e explicações respondendo ao clique.
- **Glossário do `00-overview` atualizado** com os três termos novos: `dHead`, concatenação de cabeças e projeção de saída (`W_O`).

### Blocos de armadilha, e de onde vieram

Todos correspondem a bugs reais deste log, como exige o guia:

- **§3** — o `Shape.canonicalStrides` quebrando com shape de rank 0 (Etapa 6). Lição transferida: código de shape falha nas pontas, e esta etapa cria pontas novas (`H = 1`, `T = 1`, rank 4).
- **§4** — os 115 testes verdes com o caminho `(3,2)` do `matmul` quebrado (Etapa 11). Direto ao ponto: a Etapa 12 abre o par `(4,4)`, que nasce sem cobertura.
- **§8** — o gradient check reprovando um backward correto por gradiente verdadeiro zero (Etapa 11).
- **§9** — a suíte testando a versão anterior do arquivo por causa do action cache do sbt (hoje). Entrou porque esta etapa mexe muito em formato, e formato errado costuma dar erro explícito — o que torna especialmente confuso um erro que não muda quando o código é corrigido.

### O que o capítulo deixa encaminhado para a implementação

- **A lacuna do núcleo, com as duas rotas pesadas.** O `matmul` aceita hoje `(2,2)`, `(3,2)` e `(3,3)`; a etapa precisa de `(4,4)`. O capítulo recomenda generalizar o laço de lote em vez de achatar para `[B·H, T, dHead]`, pelo mesmo critério da Etapa 11 — estender o núcleo em vez de replicar dado, já que a Etapa 14 vai repetir o bloco.
- **Quatro testes específicos da divisão em cabeças:** ida e volta com igualdade exata (pega a troca de ordem entre `reshape` e `transpose`), cabeças produzindo distribuições diferentes, `H = 1` reproduzindo a Etapa 11, e formatos com `T ≠ dHead` e `H ≠ dHead`.
- **A pendência do `b_V`** está no texto como decisão já tomada: `useBias = false` em `W_K` e `W_V`, viés mantido em `W_Q` e `W_O`.

Próximo passo: estender o `matmul` para o par `(4,4)` em `scalagrad`, com os testes do caminho novo escritos antes do uso — exatamente a lição do bloco de armadilha da §4.

---

## 2026-08-21 — `matmul` unificado: um só caminho, para lote de rank qualquer

O usuário estendeu o `matmul` para multi-batch achatando as dimensões da frente (`Shape.stack` novo) e delegando ao `matmulBatched`. A revisão encontrou os valores e os gradientes **exatamente certos** nos três caminhos — `(4,4)`, `(4,2)` e entrada não contígua, com diferença `0.0` no forward e erro na casa de `1e-9` no gradient check. Os defeitos eram de contorno:

- **O resultado nunca era desachatado.** `[2,3,4,5] · [2,3,5,6]` devolvia `[6,4,6]` em vez de `[2,3,4,6]`. As dimensões de lote se perdiam na volta.
- **`rank % 3` no lugar de `rank - 3`.** As duas coincidem para rank 3, 4 e 5, e divergem no 6: `6 % 3 = 0` vira um `stack(0)` que não faz nada, e o `assert` do `matmulBatched` estourava.
- **A mensagem de erro de lote citava dimensões achatadas.** `[2,3,4,5]` contra `[2,5,5,6]` dizia *"got 6 and 10"* — números que o chamador nunca escreveu.
- **`Shape.stack` nasceu sem teste**, exatamente a armadilha da §4 do capítulo escrito horas antes.

### A decisão: eliminar o dispatcher em vez de remendá-lo

A pedido do usuário, que queria um `matmul` único. `matmul2D` e `matmulBatched` eram quase o mesmo corpo escrito duas vezes, e o dispatcher acrescentava uma terceira. A versão nova tem **uma** regra: as duas últimas dimensões são a matriz, todas as anteriores são lote. O laço percorre as posições da saída e lê os operandos pelas strides.

- **Não há teto de rank.** Nada no código conta dimensões de lote. Rank 6 funciona, testado.
- **Entrada não contígua não é copiada.** É o caminho da Etapa 12: `Q` e `K` chegam recém-transpostos.
- **`MatmulOps.scala` caiu de 159 para 93 linhas**, com os dois helpers privados e o dispatcher eliminados.
- **Broadcasting de lote continua recusado**, deliberadamente. O PyTorch broadcasta; este projeto exige lotes iguais, ou um segundo operando rank 2 compartilhado. A validação agora acontece antes de qualquer achatamento, então a mensagem cita os shapes originais.

### Sobre desempenho, que era a dúvida que motivou a mudança

A rota do achatamento **é** a do PyTorch — a §9 do capítulo, escrita horas antes, a tinha descartado pelo motivo errado. O que a torna barata lá não está no `matmul`: é que o `reshape` do PyTorch devolve uma *view* quando as strides permitem.

E o caso da divisão em cabeças é justamente um em que a view não é possível. Fundir as duas primeiras dimensões de `[B, H, T, dHead]` exigiria `stride(0) = shape(1)·stride(1)`; depois do `transpose` isso dá `T·H·dHead` contra `H·dHead`, que só batem quando `T = 1`. Achatar copiaria os dois operandos a cada multiplicação. É por isso que o MHA do nanoGPT escreve `.contiguous()` explícito na concatenação.

Ler pelas strides evita a cópia inteira. O custo por elemento também caiu — ver o adendo no fim desta entrada, onde as três formulações do laço foram medidas.

### `contiguous` ganhou atalho, e uma ressalva

O usuário acrescentou `if strides == shape.canonicalStrides then this`. Medido, o atalho dispara em todos os caminhos normais — `make`, `zeros`, `arange`, resultado de `matmul`, de `reshape` e de soma.

**Mas a comparação é por referência**, porque `Strides` não tem `equals` (decisão registrada no `CLAUDE.md`). Isso dá um falso negativo real: um tensor transposto **duas vezes** tem strides canônicas de fato e mesmo assim é copiado, porque a instância é outra. E transpor duas vezes é exatamente o ida-e-volta da divisão em cabeças. Falso positivo não é possível — só um tensor canônico compartilharia a instância —, então é seguro, só perde otimização. Vale trocar por comparação de valor (`sameElements`) ou por um `isContiguous` dedicado.

### Testes

Nove testes permanentes, escritos sob a autorização permanente. Estruturais e de valor em `TensorOpsSpec`; gradient checks em `GradcheckSweepSpec`, que é onde eles moram neste projeto.

Cobertura: produto com dois eixos de lote conferido posição a posição contra cálculo manual; matriz rank 2 compartilhada sob dois eixos; ausência de teto de rank (rank 6); operando rank 4 não contíguo; rejeição de lotes diferentes, exigindo que a mensagem cite os shapes originais; gradient check nos dois operandos, no compartilhado e no não contíguo; e a soma do gradiente da matriz compartilhada sobre todas as `B·H` fatias, com `dC` distinto por posição.

**Suíte validada por mutação**, precaução herdada da Etapa 11. Quatro mutações, cada uma revertida em seguida:

| mutação | testes que falharam |
|---|---|
| forward lê o eixo errado do operando esquerdo | 20 |
| backward lê o operando errado (bug silencioso de gradiente) | 11 |
| aceita lotes de tamanhos diferentes | 1 |
| `rhs` compartilhado com `k` e `n` trocados | 8 |

Total: **279 testes** (198 `scalagrad` + 81 `gpt`).

### `theory/12` sincronizada

A §9 descrevia um dispatcher que não existe mais, e recomendava a rota errada pelo motivo errado. Reescrita como "O núcleo já atende", com a rota do achatamento apresentada como descartada e a conta das strides mostrando por que a view falha. A armadilha da §4 e o cartão de referência também foram atualizados. Prosa remedida: média `12.0`, p90 `19`, `5.2` parênteses por mil palavras.

### Achado separado, ainda aberto

Durante a revisão apareceu um **bug pré-existente em `UnaryOps`**, sem relação com o `matmul`. O backward de `unary` mistura indexação física e canônica, e o gradiente sai errado quando a entrada é não contígua — forward perfeito, gradiente errado. Medido: `exp(transpose(x))` dá erro `0.632` no gradient check, contra `8.5e-11` do mesmo `exp` sobre entrada contígua. Shape quadrado não mascara (`0.918`). Registrado como tarefa separada.

### Adendo, mesma sessão: a versão imperativa foi substituída por uma funcional — e mais rápida

A primeira versão do `matmul` unificado usava `while` com `var acc`, e mutava dois arrays de multi-índice ao longo da soma em `k`. O usuário questionou: não deveria ser funcional, ou isso agregaria complexidade demais?

A crítica procedia por dois motivos. O `matmul2D` e o `matmulBatched` antigos eram feitos de `for`/`map`/`.sum`, então a versão imperativa destoava do resto do projeto. E o problema maior nem era o `var acc`: era **mutar arrays compartilhados** dentro do laço, um padrão em que a ordem das linhas passa a importar de um jeito que só se percebe lendo com lupa.

Três versões foram escritas e medidas, no mesmo shape (`[4,4,32,16] · [4,4,16,32]`, forward mais backward, mediana de 12 execuções após aquecimento da JVM):

| versão | mediana | `var`? | muta estado? |
|---|---|---|---|
| funcional ingênua — `updated` a cada multiplicação | 288,4 ms | não | não |
| imperativa — `while` + `var`, índice reusado | 126,2 ms | sim | sim |
| **funcional com aritmética de deslocamento** | **57,7 ms** | não | não |

**A versão escolhida é a mais funcional e a mais rápida das três**, o que desfaz o falso dilema. A chave é notar que, dentro da soma em `k`, só uma dimensão de cada operando muda — então o deslocamento em `data` avança por um **passo constante**. Calcular a base uma vez por posição da saída troca o índice multidimensional por aritmética inteira, e o laço interno vira um `foldLeft` sem alocação nenhuma.

As strides canônicas de quem varia são conhecidas de antemão, o que evita recalculá-las: `1` para a última dimensão do operando esquerdo, `N` para a penúltima do direito.

- **A versão ingênua é a mais lenta por um motivo instrutivo:** `updated` aloca um array novo a cada multiplicação. O ganho da imperativa vinha de reusar o array, não do `while` — e a variante C elimina o array do laço interno de vez, em vez de reusá-lo.
- **Custo honesto:** a variante C lê `t1.data(...)` direto, em vez de `t1.get(...)`. É o padrão por trás de um bug real deste projeto (Etapa 6, ver a armadilha em `theory/11-attention` §8). Aqui é legítimo porque o deslocamento físico é calculado das strides — que é exatamente o que o `get` faz por dentro —, e os testes de operando não contíguo cobrem o caso. Ficou um comentário no código explicando o porquê.
- **`MatmulOps.scala` não tem mais nenhum `var` nem `while`.** Segue com 93 linhas.
- Um efeito colateral bem-vindo: sem mutação, o `clone()` do multi-índice direito deixou de ser necessário. Os dois operandos compartilham o mesmo array base, porque `updated` é puro.

**Revalidado por mutação** — a suíte anterior foi validada contra a implementação antiga, e essa garantia não se transfere. Cinco mutações, cada uma revertida em seguida:

| mutação | testes que falharam |
|---|---|
| `leftStep` lendo o eixo errado | 22 |
| backward lendo o próprio operando em vez do outro | 11 |
| stride canônica errada no gradiente do segundo operando | 10 |
| aceita lotes de tamanhos diferentes | 1 |
| base sem zerar o eixo da soma | 22 |

Suíte total inalterada: **279 testes** (198 `scalagrad` + 81 `gpt`).

**Nota de infraestrutura, terceira manifestação do mesmo problema:** a primeira rodada de mutações não produziu saída nenhuma. O action cache do sbt guardou uma **falha de compilação** e passou a repeti-la (`1 cached-failure cache hit`) em todas as execuções seguintes, inclusive depois do fonte voltar ao original. A saída foi limpar `~/AppData/Local/sbt/v2/ac` antes de cada mutação. Vale a regra geral: qualquer processo que troque arquivos de código repetidamente neste projeto precisa invalidar o action cache entre as trocas.

---

## 2026-08-21 — Duas indexações convivendo: bug de gradiente em todas as ops unárias

Encontrado durante a revisão do `matmul`, ao conferir como o array de gradiente é indexado. Não tinha relação com a mudança revisada — era pré-existente, e atingia **as nove** operações unárias: `neg`, `pow`, `exp`, `log`, `clamp`, `relu`, `sigmoid`, `tanh` e `gelu`.

### A causa: duas convenções de índice no mesmo laço

O projeto indexa `data` **fisicamente**, pelas strides reais (`Tensor.index`/`Tensor.get`), e indexa `gradient` **canonicamente** — todas as ops acumulam via `shape.index(...)` ou por laço em ordem canônica. Num tensor contíguo as duas coincidem, e a diferença é invisível.

O backward de `unary` misturava as duas:

```scala
t1.gradient.accumulate(i, grad(i) * localGrad(t1.data(i), data(i)))
//                     ↑ canônico        ↑ físico      ↑ físico
```

Com entrada não contígua, a derivada local era pareada com o gradiente de cima de **outra posição**. O forward continuava perfeito, porque `data = t1.data.map(op)` preserva a ordem física e o tensor de saída herdava `t1.strides`. Só o gradiente saía errado — o tipo de defeito que este projeto trata como o pior.

### Medições, com controles

Limiar do gradient check é `1e-5`:

| caso | erro |
|---|---|
| `exp(x)` com `x` contíguo | `8.5e-11` |
| `transpose(x)` sem op unária | `1.7e-10` |
| **`exp(transpose(x))`, shape 2×3** | **`0.632`** |
| **`exp(transpose(x))`, shape 3×3 quadrado** | **`0.918`** |
| `log(transpose(x))` | `0.333` |
| `exp(transpose(x).reshape(...))`, que força `contiguous` | `9.2e-10` |

Os controles isolam a não contiguidade como causa, e a última linha confirma: forçar ordem canônica conserta. **Shape quadrado não mascara** — o `transpose` troca as strides mesmo com dimensões iguais.

### A correção

`unary` passou a construir a saída em ordem canônica, como todas as outras ops do projeto já faziam. O mapa de índice canônico para posição física é calculado uma vez e reusado no backward, o que também evita recalcular o desenrolamento a cada elemento.

Dois efeitos colaterais bons: o tensor de saída deixou de herdar `t1.strides` e agora é sempre contíguo; e `Gradient.zeros(data.length)` virou `Gradient.zeros(shape.size)`, fechando uma inconsistência de dimensionamento que existia para tensores cujo array físico é maior que o shape lógico.

### O irmão latente, em `Tensor.contiguous`

Mesma confusão, no mesmo dia: `contGrad = mapping.map(i => gradient(i))` aplicava o mapeamento **físico** ao array de gradiente, que é canônico. Ninguém consumia esse array — o único chamador, `reshape`, só usa `.data` da cópia —, então era uma armadilha armada para o próximo chamador em vez de um bug ativo. Corrigido para copiar sem permutar (`gradient.get`), com comentário explicando por que o `mapping` só vale para `data`.

### Testes

- **`GradcheckSweepSpec`**: gradient check das nove ops unárias sobre entrada não contígua, em três shapes — incluindo `3×3` quadrado, de propósito. Mais uma contraprova direta: a mesma função montada por dois caminhos que só diferem no layout de memória tem que dar o mesmo gradiente, posição a posição.
- **`TensorSpec`**: `contiguous` copia o gradiente sem permutar, com valores em potências de 10 para que qualquer troca de posição salte aos olhos.
- **Validação**: revertendo cada correção, os testes novos falham (2 no `unary`, 1 no `contiguous`) e **nenhum dos 198 testes anteriores acusava nada**. É a medida exata do buraco de cobertura: as ops unárias nunca tinham sido exercitadas com entrada não contígua.

Suíte total: **282 testes** (201 `scalagrad` + 81 `gpt`).

### Por que ninguém tinha esbarrado nisso

O primeiro tensor não contíguo do projeto nasceu na Etapa 11, em `K.transpose()`, e ele só alimenta `matmul` — que lê pelas strides e está correto. As ops unárias sempre tinham recebido entrada contígua. A Etapa 12 muda isso: `transpose(1, 2)` passa a estar em todo lugar, e a Etapa 13 vai rodar `gelu` no MLP. O bug estava a uma etapa de acordar.

É o mesmo erro do bloco de armadilha de `theory/11-attention` §8 — ler pelo índice errado dentro de uma operação — em outro arquivo. Vale considerar um bloco de armadilha em `theory/03-elementary-operations` ou `theory/05-activations`, onde essas ops moram.

### `CLAUDE.md`: três convenções registradas a partir do que esta sessão descobriu

- **Estilo funcional sempre que possível**, sem `var`/`while`/mutação de estado compartilhado, com os três tempos do `matmul` como evidência de que a versão mais funcional foi também a mais rápida. Pedido do usuário, depois da revisão do laço imperativo.
- **O invariante de indexação:** `data` é físico, `gradient` é canônico, nunca use o mesmo `i` nos dois. É a causa-raiz do bug das ops unárias, e o corolário ficou junto: toda op nova precisa de teste com entrada não contígua, e shape quadrado não protege.
- **Correção de uma afirmação que envelheceu hoje.** O item sobre `Shape`/`Strides` dizia que nada no código compara os dois por valor. O atalho novo de `Tensor.contiguous` compara — por referência, já que não há `equals`. Registrado que funciona, que falso positivo é impossível, e qual é o falso negativo concreto: um tensor transposto duas vezes tem strides canônicas e mesmo assim é copiado, que é justamente o ida-e-volta da divisão em cabeças.

---

## 2026-08-21 — Auditoria do repositório, e a varredura de correções que saiu dela

Pedido do usuário: varredura completa das APIs, com nota por categoria e rigor deliberado. O relatório completo saiu como artifact (&ldquo;Raio-X do mini-gpt&rdquo;). Notas no momento da auditoria: documentação 9, testes 8, desenho de API 7, estilo funcional 7, erros 7, correção 6, consistência 6, custo 5 — geral **6,9/10**, contra a régua de &ldquo;biblioteca da qual outros vão depender&rdquo;, não de projeto de estudo.

Em seguida o usuário pediu para executar tudo. O que segue é o que mudou.

### Três bugs, todos da mesma família do `UnaryOps`

O invariante das duas indexações (`data` físico, `gradient` canônico) estava violado em mais três lugares. Os três confirmados por execução antes e depois:

| Onde | Antes | Depois |
|---|---|---|
| `ReduceOps.max` sobre transposto | gradiente na posição 4, máximo na 3 | posição correta |
| `ReduceOps.sum` sobre view broadcastada | `6.0` onde o certo é `12.0` | `12.0` |
| `Gradcheck` com entrada não contígua | erro `0.99` contra limiar `1e-5` | passa |

- **`ReduceOps`** ganhou `canonicalValues(t) = t.contiguous.data`, que devolve o próprio tensor quando as strides já são canônicas — o caso comum não copia nada. `mean` passou a dividir por `t.size` em vez de `t.data.length`, que são grandezas diferentes numa view esticada.
- **`Gradcheck`** passou a mapear índice canônico para posição física antes de perturbar. De quebra ganhou duas proteções: `require` de que a perda é escalar (antes lia `.data(0)` de um tensor qualquer e devolvia um número plausível e sem sentido), e `input.gradient.zero()` no início, que torna `run` idempotente — as suítes não precisam mais criar uma instância nova por parâmetro para contornar a acumulação.
- **Validação por mutação:** revertendo cada correção, 2 e 1 testes falham respectivamente. Seis testes de regressão novos.

### Desempenho: o ganho foi maior que o esperado

`Shape.groupIndex` construía um `Shape` descartável por chamada, via `crop(dim)` — e `softmax` chama isso cinco vezes por elemento. Trocado por Horner sobre as dimensões mantidas, sem alocar nada. `Shape.linearIndex` perdeu o `zip`/`map`/`sum` por um `foldLeft`, e é o caminho de todo acesso a elemento.

Medido, mediana de 9 execuções após aquecimento:

| Operação | Antes | Depois | Ganho |
|---|---|---|---|
| `softmax(3)` fwd+bwd em `[4,4,48,48]` | 173,7 ms | 30,2 ms | **5,8×** |
| soma elementar fwd+bwd em `[64,256]` | 21,0 ms | 5,6 ms | **3,8×** |
| `sum(dim=1)` fwd+bwd em `[32,64,32]` | 70,9 ms | 7,5 ms | **9,5×** |

Nenhuma das três mudanças custou legibilidade — as duas versões novas são mais curtas que as antigas.

### Desenho de API

- **`Gradient.zeros(shape: Shape)`** como sobrecarga preferida. O invariante é que o gradiente tem uma posição por elemento; a assinatura antiga aceitava qualquer `Int` e já tinha sido chamada com `data.length` em dois lugares. Todos os 14 pontos de construção migrados.
- **`Gradient.get` → `toArray`**, alinhando os três wrappers irmãos. E `accumulateAll` ganhou sobrecarga que recebe `Gradient` direto, eliminando um `clone()` por operando em toda operação binária.
- **`accumulateAll` valida o comprimento** antes de escrever: antes acumulava até a metade e só então lançava, deixando o gradiente em estado parcial.
- **`Strides.toArray` passou a clonar**, como `Shape.toArray` sempre fez. O acesso cru, que existe por causa do caminho quente, virou `unsafeValues` — o nome agora diz o que faz.
- **`broadcastTo` fechado em `private[scalagrad]`.** A view compartilha o objeto `gradient` do original, que tem menos posições que o shape anunciado; era uma armadilha pública guardada em comentário. Continua acessível a quem precisa dela dentro do núcleo.
- **`Tensor.randn` aceita um `Random` injetável**, com default, no mesmo desenho que o `BatchSampler` já usava. Sem isso nenhuma inicialização de modelo era reproduzível — e a partir da Etapa 18 &ldquo;rodei de novo e deu diferente&rdquo; seria indistinguível de &ldquo;mudei alguma coisa&rdquo;.
- **`Tokenizer` perdeu o parâmetro de estratégia.** A segunda lista recebia um `mapper` que as duas únicas fábricas preenchiam identicamente, e que vazava o `Map` para a assinatura pública. Agora o construtor é privado e recebe o vocabulário pronto.
- **`Shape.stack` apagado.** Ficou órfão quando o `matmul` foi unificado, nunca teve teste, e o nome significa o oposto do que fazia — em bibliotecas de tensor, *stack* empilha tensores criando uma dimensão.

### Estilo e consistência

- **Dez `(0 until n).map { ... }.toArray` viraram `Array.tabulate`**, eliminando a coleção intermediária de `Double` encaixotados. `MatmulOps`, `BinaryOps`, `SoftmaxOps`, `BroadcastOps`, `Shape`, `Tensor` e `Embedding`.
- **`Tensor.toString` reescrito em sintaxe Scala 3**, era o único ponto do arquivo com `if (...) { } else { }`.
- **`_backward` → `backwardStep`.** Sublinhado inicial é convenção de Python; o modificador de acesso já faz esse trabalho.
- **`isContiguous` virou `lazy val` e passou a comparar por valor.** Como `Strides` não tem `equals`, o `==` era comparação por referência, e um tensor **transposto duas vezes** — que é o ida-e-volta da divisão em cabeças da Etapa 12 — tinha strides canônicas de fato e mesmo assim era copiado.
- **`maxReachableIndex`** trocou `map(...).sum` por `foldLeft`, saindo do caminho de construção de todo tensor.
- **Mensagens de erro** que não ecoavam o valor recebido, em `LayerNorm`, `BatchSampler` e `Shape.linearIndex` — esta última agora nomeia qual dimensão estourou e qual era o limite.
- **`Array.size` → `Array.length`** em `BatchSampler` e `IndexOps`.
- **`Embedding`** passou a exigir que os índices de token sejam inteiros. Como eles viajam como `Double`, um `3.7` virava `3` em silêncio.
- **`LayerNorm.epsT`** virou campo: era um tensor alocado a cada forward, enquanto o `scale` da `Attention` — mesmo papel — já era campo.

### Estado final

**291 testes** (210 `scalagrad` + 81 `gpt`), verdes em 5 execuções seguidas. Nenhum `var` e nenhum `while` no código de implementação. Todas as linhas dentro de 100 colunas, exceto mensagens de erro pré-existentes que o `scalafmt` não quebra.

### O que ficou de fora, e por quê

- **Separar `BroadcastView` de `Tensor`.** É arquitetura, não limpeza. Fechar a visibilidade resolve o risco imediato; o tipo próprio continua sendo a resposta certa se o incômodo voltar.
- **Índices de token como `Double`.** Consequência de `Tensor` ser só de `Double`. O `require` novo no `Embedding` fecha o sintoma; a decisão de fundo é da arquitetura e não muda hoje.

### A lição que atravessa tudo

Os quatro bugs da família de indexação não vieram de desatenção, e sim de um invariante que existia só na cabeça de quem escreveu. Ele valeu em nove lugares e falhou em quatro. Foi para o `CLAUDE.md` hoje junto com o corolário de teste — **toda op nova precisa de um caso com entrada transposta** — e é isso que muda o futuro do projeto mais do que qualquer correção individual.

---

## 2026-08-22 — Etapa 12 concluída: `MultiHeadAttention` e os dois bugs de eixo

O usuário escreveu a camada; a revisão encontrou dois bugs, ambos do mesmo deslize. Suíte de **316 testes** (210 `scalagrad` + 106 `gpt`), sendo 25 novos.

### Antes da camada: o `Masks` extraído

`causalMask` era `private` dentro de `Attention.scala`, e a MHA precisava do mesmo. Virou `gpt.nn.Masks`, com as duas camadas chamando de lá — em vez da terceira cópia, que a Etapa 14 pediria de novo.

### Os dois bugs, e a causa comum

O `attentionWeights` nasceu recebendo `scores` num parâmetro chamado `x`. Mas `x` significa a entrada `[B, T, dModel]` em todo o resto da classe, e os dois tensores têm rank e eixos diferentes. Duas linhas erraram junto:

**1. `softmax(x.rank - 1)` normalizava o eixo errado.** Com `x` sendo `scores`, rank 4, a expressão dava 3 e funcionava — por coincidência. Na primeira versão, com `x` sendo a entrada rank 3, dava 2: o eixo das *queries* em vez do das *keys*. As colunas de `P` somavam 1, não as linhas.

Medido, `B=2, T=3, H=2, dHead=2`:

| | soma das linhas |
|---|---|
| `softmax(2)` | `0,3507`  `0,8344`  `1,8148` |
| `softmax(3)` | `1,0000`  `1,0000`  `1,0000` |

E a primeira linha saía `[0,3507, 0, 0]` em vez de `[1, 0, 0]` — ou seja, `y₀ = 0,3507·v₀`.

**Por que é do tipo mais perigoso:** os eixos 2 e 3 têm ambos tamanho `T`, então o shape não muda. Nenhuma fatia normalizada é toda `-inf`, então não há `NaN`. As posições mascaradas continuam com peso zero, então **a causalidade e a contraprova passam**. E forward e backward continuam coerentes entre si, então **o gradient check passa** — ele valida coerência, não intenção. Toda a proteção herdada da Etapa 11 passa batido; só a soma das linhas e o `y₀ == v₀` denunciam.

**2. `Masks.causalMask(x.shape(1))` construía a máscara com `nHeads`.** Sobre `scores` de shape `[B, H, T, T]`, `shape(1)` é `H`, não `T`. Falha alta e imediata:

```
T = 3, H = 2  →  Shapes Shape(2,2,3,3) and Shape(2,2) cannot be broadcasted
T = 2, H = 2  →  passa, por coincidência aritmética
```

O segundo caso é o alerta: um primeiro teste escrito com `T = nHeads` teria dado verde num código errado. É a razão de a spec fixar `B=3, T=5, dModel=8, nHeads=4, dHead=2`, com os cinco números distintos entre si.

**A correção** foi de desenho, não pontual: `attentionWeights` passou a receber a entrada de verdade, calcular as projeções `Q`/`K` e derivar cada índice do tensor a que ele pertence — `x.shape(1)` para `T`, `scores.rank - 1` para o eixo do softmax. O `forward` chama esse método e calcula só `V` por fora, então nada é recalculado. Um `private def validate` cobre as duas portas públicas.

Isso também consertou uma sugestão minha que estava errada: eu tinha proposto expor `attentionWeights` como janela de diagnóstico, mas do jeito que ela nasceu o chamador precisaria produzir `scores` sozinho, o que exige as projeções e o split, ambos privados. A janela não abria.

### A lição, que é a mesma de 2026-08-21 em outra roupa

Ontem foi `data` físico contra `gradient` canônico. Hoje é o eixo de um tensor lido do rank de outro. Nos dois casos o invariante existia só na cabeça de quem escreveu, e o nome da variável não o carregava. **Derive todo índice do tensor a que ele pertence** — e um parâmetro chamado `x` numa classe onde `x` já significa outra coisa é o convite para o erro.

### A suíte: 25 testes

Cobertura, na ordem do arquivo: comparação posição a posição contra uma referência do bloco inteiro em Scala puro (que fatia as cabeças por `c / dHead`, e portanto denuncia troca de ordem entre `reshape` e `transpose`); `nHeads = 1` contra uma segunda referência **sem aritmética de fatiamento nenhuma**, que é a regressão contra a Etapa 11; preservação de formato em quatro configurações, incluindo as pontas `H = 1`, `T = 1` e `dHead = 1`; entrada não contígua; ida e volta da divisão em cabeças com igualdade exata, mais a contraprova de que a ordem inversa produz shapes compatíveis e valores diferentes; cabeças com distribuições distintas; soma das linhas e zeros nas posições mascaradas; `P[0,0] == 1` exato; causalidade e contraprova ponta a ponta; os 6 parâmetros com shapes e `requiresGradient`; contagem `4·dModel² + 2·dModel` para `H ∈ {1,2,4,8}`, mais a linha `dModel=4, H=2 → 72` conferida contra a tabela da teoria §9; gradient check em `x`, nos 6 parâmetros e com entrada não contígua; e as quatro validações de entrada.

Semente fixa no `Random` da spec — os testes de valor comparam números, e sem semente uma falha não é reproduzível. É o primeiro uso real do `Random` injetável que a auditoria de ontem acrescentou.

### Validação por mutação

| mutação | testes que falharam |
|---|---|
| `softmax` no eixo das queries (o bug original) | 7 de 25 |
| split com `transpose` antes do `reshape` | 14 de 25 |
| merge sem o `transpose` de volta | 2 de 25 |
| máscara depois do softmax | 10 de 25 |
| sem a divisão por `√dHead` | 3 de 25 |

As cinco pegas. A mais fina é o merge, com 2 — e as duas são as comparações contra a referência em Scala puro, que é o teste que carrega a etapa. Vale saber disso: se um dia essa referência for simplificada, a cobertura do caminho de volta cai junto.

### Pendências abertas

- **Inicialização.** `Linear` usa He (`σ = √(2/inputDim)`), cujo fator 2 existe para compensar a ReLU zerando metade do sinal. Nas quatro projeções da MHA não há não-linearidade nenhuma entre elas, então cada `Linear` **dobra** a variância e o bloco a quadruplica. O LayerNorm renormaliza na fronteira dos blocos, então não explode com a profundidade, mas o principiado aqui seria Xavier (`σ = √(1/inputDim)`). Junto disso, o GPT-2 escala as projeções residuais (`W_O` e o segundo `Linear` do MLP) por `1/√(2·nLayers)`. Revisitar na Etapa 15, quando houver profundidade de verdade para medir.
- **`MultiHeadAttention` não aceita um `Random`.** `Tensor.randn` já aceita desde ontem, mas a camada não repassa. A spec contorna lendo os pesos de `parameters`, o que basta para testes de valor; a partir da Etapa 18, "rodei de novo e deu diferente" precisa ser distinguível de "mudei alguma coisa".
- **`parameters` posicional segue frágil.** Terceira etapa consecutiva com índices nomeados no topo da spec. Se doer de novo, o caminho é expor `weight`/`bias` no `Linear`, como o PyTorch faz.

### Próximo passo

Etapa 13, o MLP: `Linear(dModel, 4·dModel) → GELU → Linear(4·dModel, dModel)`. Como manda a convenção da Etapa 4, `theory/13-mlp/` antes do código. Vale notar que a `gelu` vai receber entrada não contígua pela primeira vez em produção — o bug das nove ops unárias de ontem estava a exatamente uma etapa de acordar.

---

## 2026-08-30 — `theory/13-mlp/` escrita antes da implementação (Etapa 13)

Segue a convenção da Etapa 4. Três arquivos, no padrão do `theory/STYLE-GUIDE.md`, relido antes de escrever.

- **`13-mlp.md`** — 9 seções numeradas, mais abertura e cartão de referência. O fio condutor é a divisão de trabalho do bloco: a atenção move informação entre posições, o MLP transforma o que chegou, e nenhuma das duas substitui a outra.
- **`mlp.html`** — cinco figuras. FIG 1, três posições atravessando as mesmas caixas de peso, com duas faixas destacadas por receberem o mesmo token; FIG 2, as três saídas de um MLP estreito confinadas ao plano gerado por `v₀` e `v₁`; FIG 3, os quatro neurônios lidos como pares chave-valor, com a ativação em barra na mesma escala; FIG 4, a curva da GELU com os quatro `Z` marcados; FIG 5, o backward, separando quem soma sobre o lote de quem tem uma cópia por posição.
- **`exercises.html`** — 14 questões (4 fácil, 5 médio, 3 difícil, 2 desafio), mesmo motor de placar dos capítulos anteriores.

### Quatro contribuições que não estavam no roadmap

**O que a largura compra, dito de forma verificável.** O roadmap justifica a expansão 4x como "gargalo expandido", o que não é falso mas não prova nada. O capítulo troca isso por um argumento estrutural: como `Y[t] = Σᵢ A[t,i]·vᵢ + b2`, a saída vive sempre no espaço gerado pelos `dFF` vetores de valor. Com `dModel = 3` e `dFF = 2`, três saídas quaisquer são coplanares — determinante `-5.6e-16`, contra `-259.029` com `dFF = 4`. A verificação independente reescreve cada linha como combinação de `v₀` e `v₁`, e os coeficientes saem sendo exatamente as ativações.

**A leitura chave-valor, conferida contra a conta original.** Coluna `i` de `W1` é a chave, linha `i` de `W2` é o valor, `−b1[i]` é o limiar. A soma das quatro contribuições, agrupada por neurônio, reproduz `[1.59224, −4.08314]` — a mesma linha que os dois `matmul` produzem agrupando por coordenada. Não é metáfora, é a mesma soma reordenada.

**Onde o fator 2 do Kaiming cabe, e onde não.** Ele compensa uma perda **já ocorrida**, então pertence à camada cuja *entrada* passou pela ativação. No MLP isso é a segunda `Linear`. A primeira recebe o fluxo residual, que não passou por corte nenhum, e ali o 2 dobra a variância. Medido por quadratura numérica: `Var(x)=1 → Var(Z)=2 → E[A²]=0.92214 → Var(Y)=1.84429`, com o excesso todo vindo do primeiro passo. Com ReLU no lugar da GELU, `E[relu²]` vale exatamente `1.0` e a segunda camada preserva de forma exata — que é como Kaiming derivou o fator. Isso generaliza a pendência que a Etapa 12 registrou para as quatro projeções da MHA, e reforça que a decisão certa é única, na Etapa 15.

**O gradient check aprova um MLP sem GELU.** Ele confere se o backward é a derivada do forward que você escreveu, e `Linear → Linear` é perfeitamente derivável. O teste de formato passa. O de independência entre posições também. O capítulo fecha a brecha com uma contraprova de linearidade (`f(2x) ≠ 2·f(x)`), e o `checklists/13-mlp.md` a registra como item obrigatório, com mais seis mutações para validar a suíte.

### Uma previsão da entrada anterior que estava errada

A entrada de 2026-08-22 dizia que a `gelu` receberia entrada não contígua pela primeira vez em produção. **Não recebe.** No caminho normal ela é aplicada ao resultado de `matmul` seguido de soma, e as duas operações constroem a saída com `shape.canonicalStrides`. O tensor que chega à `gelu` é sempre contíguo.

O teste com entrada transposta continua obrigatório, por outro motivo: o `x` que o chamador passa ao `forward` pode ter qualquer stride, e o `matmul` o lê direto, sem cópia. O capítulo diz isso explicitamente na §8, em vez de repetir a previsão errada.

### Verificação

- **Todos os exemplos numéricos conferidos por script** antes de entrar no texto. O backward completo foi confrontado com diferenças finitas centrais (`ε = 1e-6`) em `X` e em `W1`: os doze valores batem nas cinco casas decimais.
- **A contagem de parâmetros conferida contra o GPT-2 small.** A fórmula `8·dModel² + 5·dModel` dá `4.722.432` para `dModel = 768`, idêntico ao MLP do GPT-2 na casa das unidades, e a soma das peças fecha em `124.439.808` — o "124M" dos papers. A nossa implementação daria `18.432` a menos, e a diferença tem endereço: `b_K` e `b_V`, removidos na Etapa 12 por serem parâmetros mortos.
- **Prosa medida** com o script do §9 do guia: média `13.7` palavras por frase, p90 `22`, `7.1` parênteses por mil palavras. Dentro das três metas.
- **Os dois `.html` validados no navegador.** A figura do backward estourava o `viewBox` em 22 px à direita, e a largura foi corrigida para `920`; as cinco agora cabem inteiras, conferidas por `getBBox` contra o `viewBox` de cada uma. Console sem erros, e o quiz responde ao clique com placar e explicação.
- **Glossário do `00-overview` atualizado** com cinco termos: MLP, camada posição a posição, `dFF`, fator de expansão e memória associativa.

### Blocos de armadilha, e de onde vieram

- **§5** — o backward das nove ops unárias lendo `data` por índice canônico (2026-08-21). Entrou aqui porque a `gelu` era uma das nove, e porque a lição transferível é o corolário de teste que a etapa precisa aplicar.
- **§8** — o `(1 + t²)` no lugar de `(1 − t²)` na derivada da GELU (Etapa 5). Entrou junto da discussão de testes: é o caso em que o gradient check pagou o próprio custo, e serve de contraste com o caso em que ele é cego.

### Próximo passo

Implementar `gpt/nn/MLP.scala`, reusando `Linear`. A suíte da etapa está especificada na §8 do capítulo e listada no checklist — sete testes, com a contraprova de linearidade como o item que carrega a etapa.

---

## 2026-08-30 — Etapa 13 concluída: o MLP, e dois erros que o forward não vê

O usuário escreveu a camada; a revisão e a validação por mutação encontraram dois problemas, nenhum deles visível no resultado do forward. Suíte de **338 testes** (210 `scalagrad` + 128 `gpt`), sendo 22 novos.

A camada em si é a menor da etapa: `down.forward(up.forward(x).gelu)`, com duas `Linear` reusadas, três `require` e nada de álgebra nova. Tudo o que segue é sobre o que quase passou batido.

### O primeiro erro: uma dimensão que valia zero

O `dFF` foi declarado **depois** dos dois `Linear` que o usam:

```scala
private val up = Linear(dModel, dFF)     // dFF ainda vale 0 aqui
private val down = Linear(dFF, dModel)
val dFF = dModel * expansion
```

O corpo da classe inicializa de cima para baixo, então `dFF` tinha o valor default do campo `Int`. As duas matrizes nasciam com zero coluna. **O compilador não emite aviso nenhum** — conferido, zero warnings.

O que transformou isso em falha alta foi o `require` novo na `Linear`, escrito na mesma sessão por outro motivo: `inputDim >= 1` e `outputDim >= 1`. Sem ele, o `forward` rodava e devolvia tensor vazio. Com ele, 16 dos 17 testes acusaram na hora, todos com a mesma mensagem — `Output dimension must be at least 1, but got 0`. O único que passava era o que valida `expansion`, que estoura antes de chegar na `Linear`.

A decisão de onde a checagem mora vale registrar: **na `Linear`, uma vez**, e não em cada camada que a usa. `LayerNorm` e `Embedding` criam os próprios tensores e continuam sem validação de dimensão; se doer, o conserto é lá, pelo mesmo critério.

### O segundo: o viés somado depois da GELU, e por que nenhum forward vê

A mutação "somar `b1` depois da `gelu`" passou em **17 de 17** na primeira rodada da suíte. Não é falta de teste, é uma propriedade: `b1` nasce zerado, e somar zero antes ou depois da ativação dá o mesmo número. As duas versões são a **mesma função** numa camada recém-construída.

Preencher o viés por fora também não é opção: `data` é `private[scalagrad]`, e a tentativa de escrever nele do módulo `gpt` não compila.

Quem separa as duas é o backward, e sem preparação nenhuma:

| ordem | `db1` |
|---|---|
| viés antes da ativação (certo) | `Σ dA ⊙ gelu'(Z)` |
| viés depois da ativação | `Σ dA` |

A diferença está na regra da cadeia, não no valor do viés, então aparece mesmo com `b1 = 0`. O teste novo calcula `db1` pela fórmula da §5 em Scala puro e compara com o que o `backward()` acumulou — e é o único dos 18 que reprova essa mutação.

Isso também corrigiu uma previsão errada do próprio capítulo, que dava a referência em Scala puro como quem pegaria. A §8 foi reescrita com os números medidos.

### A suíte: 18 testes no `MLPSpec`, 4 no `LinearSpec`

`B=3, T=5, dModel=4, expansion=2, dFF=8` — os cinco distintos entre si, porque com `dFF == dModel` a troca de `W2` por `W2ᵀ` continuaria produzindo shapes compatíveis.

Cobertura: referência em Scala puro posição a posição; formato preservado em cinco configurações, incluindo `dModel=1`, `T=1` e `expansion=1`; entrada rank 2 aceita, que é a diferença deliberada em relação à `MultiHeadAttention`; entrada não contígua no forward e no gradient check; token repetido com igualdade exata e permutação das posições; contraprova de linearidade; GELU comparada contra duas referências erradas, com a ativação na entrada e na saída; `db1` contra a fórmula; `parameters` com os quatro formatos e `requiresGradient`; contagem `8·dModel² + 5·dModel`; gradient check em `x` e nos quatro parâmetros; e as três validações.

No `LinearSpec`, quatro testes novos: os dois `require` de dimensão, os acessores `weights`/`bias` conferidos com `theSameInstanceAs` — o otimizador da Etapa 17 precisa atualizar *a* instância, não uma cópia — e o `Random` injetável, com mesma semente dando pesos idênticos.

Duas decisões de teste que valem para as próximas etapas:

- **A contraprova de linearidade usa `f(2x) − 2f(x) + f(0)`**, e não `f(2x) ≠ 2f(x)`. Para qualquer mapa afim a primeira expressão é exatamente zero, seja qual for o viés. A segunda só funcionaria enquanto os vieses nascem zerados, e passaria a dar falso positivo depois do primeiro passo do otimizador.
- **As referências reescrevem `gelu` e `gelu'` a partir da fórmula.** Uma referência que chama o código sob teste não é caminho independente.

### Validação por mutação

| mutação | testes que falharam |
|---|---|
| remover a `gelu` | 3 de 18 |
| `gelu → Linear → Linear` | 3 de 18 |
| `gelu` depois da segunda `Linear` | 3 de 18 |
| `dFF = dModel` | 3 de 18 |
| somar `b1` depois da `gelu` | 1 de 18 |
| `W2ᵀ` no lugar de `W2` | 12 de 18 |

As seis pegas. Antes do teste de gradiente, a linha do `b1` era zero.

### Sincronização da teoria

`theory/13-mlp/13-mlp.md` §8 foi atualizada com o que a medição mostrou: o oitavo teste, a tabela de mutações com os números reais, a correção da linha do `b1`, e uma armadilha nova sobre a ordem de inicialização no corpo da classe. O cartão de referência ganhou a quinta lição — **parâmetro que nasce zerado esconde erro de ordem**. O `exercises.html` passou a 15 questões, com a nova sendo justamente essa.

### Pendência aberta

**Inicialização**, agora com endereço na Etapa 15: o fator 2 do Kaiming cabe na segunda `Linear`, cuja entrada veio da GELU, e não na primeira, que recebe o fluxo residual. Junto com o `1/√(2·nLayers)` das camadas que escrevem no residual. É a mesma pendência que a Etapa 12 registrou para as projeções da MHA.

### Próximo passo

Etapa 14, o bloco transformer: `LayerNorm → MHA → soma residual → LayerNorm → MLP → soma residual`. Como manda a convenção, `theory/14-transformer-block/` antes do código. A peça conceitualmente nova é a conexão residual; o resto é composição de camadas já testadas. Vale notar que o `parameters` do bloco passa de uma dezena de entradas, e a lista posicional — incômodo recorrente desde a Etapa 11 — agora tem saída pronta: os acessores `weights`/`bias` que a `Linear` ganhou hoje.

---

## 2026-08-30 — `theory/14-transformer-block/` escrita antes da implementação (Etapa 14)

Segue a convenção da Etapa 4. Três arquivos, no padrão do `theory/STYLE-GUIDE.md`.

- **`14-transformer-block.md`** — 8 seções numeradas. O fio condutor é a soma residual: ela não tem parâmetro, não tem hiperparâmetro, e é a diferença entre um gradiente de `1e-21` e um de `0.45` depois de 24 camadas.
- **`block.html`** — cinco figuras. FIG 1, o bloco pre-LN com o fluxo residual como linha vertical; FIG 2, o backward na bifurcação, com os números do exemplo; FIG 3, a queda do gradiente com a profundidade em escala logarítmica; FIG 4, pre-LN e post-LN lado a lado, mostrando onde o caminho direto é interrompido; FIG 5, o barramento residual com `2L` dispositivos lendo e escrevendo.
- **`exercises.html`** — 15 questões (4 fácil, 5 médio, 4 difícil, 2 desafio).

### A contribuição que não estava no roadmap: o `1/√(2L)` derivado

O roadmap cita o fator de escala do GPT-2 como prática. A §5 o **deriva**, e a derivação fecha a pendência que o projeto carrega desde a Etapa 12.

O argumento tem três passos. Cada subcamada lê `LN(x)`, que tem variância 1 por construção, e escreve um incremento. Como os incrementos são aproximadamente independentes, as variâncias somam: uma pilha de `L` blocos tem `2L` deles, e a saída fica com variância `1 + 2L`. Com `L = 12` isso é 25, ou norma `×5`.

Multiplicando a saída de cada subcamada por `c`, a variância de cada incremento é multiplicada por `c²`. Para que os `2L` incrementos somem 1 no total, cada um precisa contribuir com `1/(2L)`, logo `c = 1/√(2L)`. A conta fecha exata:

```
Var(saída) = 1 + 2L · (1/2L) = 2       para qualquer profundidade
```

É por isso que a raiz está lá: **o que soma é a variância, não o desvio padrão**. Com `1/(2L)` a variância total tenderia a 1, e os blocos praticamente não escreveriam nada.

Refazendo com os números medidos em vez dos redondos — incremento `1.0` da atenção e `1.84429` do MLP, da Etapa 13 §6 — `L = 12` dá `35.13` sem escala e `2.42` com. A dependência com a profundidade some, que era o objetivo.

### O que foi medido para o capítulo

**A rodovia do gradiente, com números.** 20 mil sorteios de `f' ~ N(0, 0.25²)` em 24 camadas:

| | mediana | p10 | p90 | fração abaixo de `1e-6` |
|---|---|---|---|---|
| sem residual | `1.2e-21` | `7.3e-25` | `7.9e-19` | 100% |
| com residual | `0.453` | `0.075` | `2.384` | 0% |

Vinte e um zeros de diferença, e nenhuma sobreposição entre as duas distribuições. A explicação estrutural entra junto: `∏(1 + f'ₗ)` abre em `2²⁴` termos, um dos quais é exatamente `1` — o caminho que pula todas as subcamadas.

**Pre-LN contra post-LN, em três blocos.** Começando de `[2,3,6,9]`, com uma subcamada que devolve 10% do que recebe, o desvio padrão vai de `2.7386` para `3.0386` em pre-LN, e fica em `1.0000` **exato** nos três blocos em post-LN. O exato é o ponto: a última operação de cada bloco post-LN é normalizar, então a escala de entrada é descartada, e o bloco seguinte não sabe quão grande era o sinal.

**O backward da bifurcação**, conferido por diferenças finitas: com `f(x) = x·W`, `x = [1,2]` e `dY = [1,10]`, o gradiente é `[-0.5, 13.1]` com residual e `[-1.5, 3.1]` sem. Na coordenada 0 ele chega a trocar de sinal.

**A contagem do bloco**, `12·dModel² + 11·dModel`, conferida contra o modelo real: 12 blocos de `dModel = 768` dão `85.036.032`, contra `85.054.464` do GPT-2 small — e a diferença de `18.432` é exatamente a dos vieses `b_K` e `b_V` que a Etapa 12 removeu.

### Verificação

- **Todos os exemplos numéricos conferidos por script** antes de entrar no texto, incluindo o backward contra diferenças finitas e as duas simulações estatísticas.
- **Prosa medida** com o script do §9 do guia: média `14.5` palavras por frase, p90 `25`, `9.1` parênteses por mil palavras. Cinco frases longas foram quebradas para o p90 entrar na meta.
- **Os dois `.html` validados no navegador**, com `getBBox` comparado ao `viewBox` de cada figura. A FIG 4 estourava 116 px à direita por causa de uma linha de texto longa, corrigida quebrando a linha e ampliando o `viewBox` para `700`. Console sem erros; o quiz confere 15 questões, 4 alternativas cada, toda `data-answer` apontando para uma alternativa existente.
- **Glossário do `00-overview` atualizado** com cinco termos: subcamada, conexão residual, fluxo residual, bloco transformer e a distinção já existente de pre-LN.

### Blocos de armadilha, e de onde vieram

- **§7** — o `val dFF` declarado depois de quem o usa (Etapa 13, hoje). Entra aqui porque o bloco tem quatro campos derivados dos argumentos do construtor, e a superfície para o mesmo erro é maior.
- **§7** — o gradient check aprovando um forward errado (Etapa 12, o softmax no eixo das queries). É a armadilha central desta etapa: soma faltando, LN no lugar errado e subcamadas trocadas produzem, todos, forwards perfeitamente deriváveis.

### O que o capítulo deixa encaminhado para a implementação

- **Expor `ln1`, `attention`, `ln2` e `mlp`.** Não é conveniência: é o que torna possível o teste central, que compõe a referência a partir das peças já testadas em vez de reimplementar atenção e MLP em Scala puro. Com 14 parâmetros, a lista posicional deixou de ser sustentável — e a `Linear` já abriu esse caminho na Etapa 13 com `weights`/`bias`.
- **Oito testes**, sendo o de composição o que carrega a etapa, mais causalidade ponta a ponta e o de empilhamento que a Etapa 15 depende.
- **Seis mutações** listadas na §7, com a expectativa de quem deve pegar cada uma.
- **Dropout fica fora do escopo**, como o próprio roadmap permite.

Próximo passo: implementar `gpt/nn/TransformerBlock.scala`.

---

## 2026-08-31 — Etapa 14 concluída: o bloco, e um `Linear` disfarçado de `LayerNorm`

O usuário escreveu a camada; a revisão encontrou um erro, e a suíte encontrou outro — meu. Suíte de **354 testes** (210 `scalagrad` + 144 `gpt`), sendo 16 novos.

O bloco em si é uma composição de quatro linhas, com as duas somas residuais e nenhuma operação nova. Tudo o que segue é sobre os dois erros.

### O erro do código: os dois `LayerNorm` eram `Linear`

```scala
val ln1: Linear = Linear(dModel, dModel)   // deveria ser LayerNorm(dModel)
val ln2: Linear = Linear(dModel, dModel)
```

O bloco virou `x + MHA(Wx + b)`: uma projeção afim aprendida no lugar da normalização. Compila, roda, devolve `[B, T, dModel]`, e não normaliza nada.

Dos 16 testes, **2** reprovaram:

```
ln1: Array(8, 8) was not equal to Array(8)     ← formato dos parametros do LayerNorm
dModel=2 nHeads=1: 74 was not equal to 70      ← 14d² + 9d em vez de 12d² + 11d
```

O que **passou** é o mais instrutivo. Passou a comparação contra a composição das subcamadas, porque a referência usa `block.ln1` e portanto concorda com o erro. Passou o gradient check nos 14 parâmetros. Passou a causalidade, passou o empilhamento, e passou o teste que distingue pre-LN de post-LN.

É exatamente o que a §7 do capítulo previu, escrita no dia anterior: **o gradient check confere coerência, nunca intenção**. A lição prática que fica é sobre o formato do teste, não sobre o erro: uma referência montada a partir dos campos do objeto sob teste herda os erros que estão nos campos. Só uma asserção sobre o **formato dos parâmetros** e outra sobre a **contagem total** olham para o objeto de fora.

### O meu erro: 12 tensores onde são 14

A §1 do capítulo dizia "2 do `LN₁`, 6 da atenção, 2 do `LN₂` e 4 do MLP, num total de 12". A soma dá **14**. O engano se propagou para o guia visual, os exercícios, o checklist e a entrada de ontem no `HISTORY`; corrigido nos cinco lugares.

Vale separar as duas contagens, porque só uma estava errada. A de **tensores** é 14. A de **números**, `12·dModel² + 11·dModel`, sempre esteve certa — e é justamente ela que pegou o `Linear` disfarçado.

### A suíte: 16 testes

Cobertura, na ordem do arquivo: comparação contra a composição das subcamadas expostas; formato preservado em quatro configurações, incluindo `dModel=2, nHeads=1, expansion=1`; as duas somas residuais, cada uma contra uma referência sem ela; pre-LN contra post-LN; a ordem das subcamadas contra a inversa; entrada não contígua; causalidade ponta a ponta **com contraprova** — a última posição tem que mudar, senão o teste passaria num bloco que ignora a entrada; empilhamento; `parameters` com 14 tensores distintos, na ordem certa, conferidos fatia por fatia contra `ln1`/`attention`/`ln2`/`mlp`; formato `[dModel]` dos parâmetros de cada `LayerNorm`; contagem `12·dModel² + 11·dModel` em quatro configurações; gradient check em `x`, nos 14 parâmetros e com entrada não contígua; e as duas validações.

Uma decisão de teste que vale para a Etapa 15: **a referência não reimplementa atenção nem MLP em Scala puro**. As duas já foram conferidas contra referências em laço nas Etapas 12 e 13, e reescrevê-las aqui duplicaria o risco em vez de reduzi-lo. O preço é o ponto cego que o `Linear` explorou, e ele se cobre com asserções estruturais sobre os parâmetros.

### Validação por mutação

| mutação | testes que falharam |
|---|---|
| remover a primeira soma residual | 3 de 16 |
| remover a segunda soma residual | 3 de 16 |
| trocar para post-LN | 2 de 16 |
| trocar a ordem das subcamadas | 2 de 16 |
| `Linear` no lugar de `LayerNorm` | 2 de 16 |
| mesmo `LayerNorm` nas duas posições | 1 de 16 |
| passar `x` no lugar de `h` ao MLP | 1 de 16 |

As sete pegas. As duas finas dependem de um teste só cada: `toSet.size == 14` para o `LayerNorm` compartilhado, e a comparação contra a composição para o `x` no lugar do `h`. O compartilhamento é o caso mais silencioso de todos — formato igual, valores plausíveis, e gradient check aprovando, porque acumular gradiente de dois usos é o comportamento **correto** de um parâmetro compartilhado.

### Pendência aberta

**`1/√(2·nLayers)`**, agora com fórmula derivada na §5 do capítulo e endereço na Etapa 15.

### Próximo passo

Etapa 15, o modelo GPT: empilhar `L` blocos, somar embedding de token e posicional na frente, LayerNorm final, e a projeção para o vocabulário. Como manda a convenção, `theory/15-gpt-model/` antes do código. É lá que a pendência da escala residual se decide, porque é lá que `nLayers` existe pela primeira vez.

---

## 2026-08-31 — `theory/15-gpt-model/` escrita antes da implementação (Etapa 15)

Segue a convenção da Etapa 4. Três arquivos, no padrão do `theory/STYLE-GUIDE.md`.

- **`15-gpt-model.md`** — 9 seções numeradas. É a etapa de integração: nenhuma matemática nova, e o capítulo aproveita isso para fechar duas pendências antigas e provar uma propriedade que estava só afirmada desde a Etapa 9.
- **`gpt.html`** — quatro figuras. FIG 1, o caminho completo de inteiros a logits, com as duas tabelas somadas na entrada; FIG 2, a prova da permutação, com as saídas idênticas lado a lado; FIG 3, a mesma matriz nas duas pontas do modelo; FIG 4, a proporção dos parâmetros no tiny e no GPT-2 small.
- **`exercises.html`** — 15 questões (4 fácil, 5 médio, 4 difícil, 2 desafio).

### A contribuição principal: por que a posição precisa entrar, demonstrado

A Etapa 9 construiu a tabela posicional e afirmou que ela é necessária. O capítulo agora **prova**, com um experimento que também vira teste.

Numa atenção causal sem informação de posição, a saída em `t` depende do *conjunto* de tokens visíveis, não da ordem deles. Rodando uma cabeça com `[a, b, c]` e depois com `[b, a, c]`:

```
saída em t=2, com [a, b, c]:  [-1.873728, 0.438537, -1.909045, 1.416224]
saída em t=2, com [b, a, c]:  [-1.873728, 0.438537, -1.909045, 1.416224]

diferença máxima em t=2:  0.0        exatamente zero
diferença máxima em t=0:  4.69444    ali o token realmente mudou
```

Somando um vetor por posição, a diferença em `t=2` passa a `0.417607`. A degenerescência é a razão de a tabela existir, e ela não é sutil: os quatro números são idênticos bit a bit.

Isso responde também a uma objeção plausível — "a máscara causal já diz quantos tokens vêm antes". Diz **quantos**, não **qual está onde**, e os dois casos comparados têm exatamente o mesmo tamanho de prefixo.

### As duas pendências fechadas

**O `1/√(2·nLayers)`, com o detalhe que faltava.** A Etapa 14 derivou o fator, mas descrevia como se fosse um fator no forward. O GPT-2 aplica na **inicialização**: as duas matrizes que escrevem no fluxo residual nascem com `σ = 0.02/√(2L)`. Isso muda o que é preciso implementar — um argumento a mais em duas camadas, e nenhuma linha no `forward`.

**A inconsistência de inicialização, agora explícita.** O projeto usa `N(0, 0.02²)` na `Embedding` e Kaiming na `Linear` — com `dModel = 128`, `σ = 0.125` contra `0.02`, mais de seis vezes. A §6 apresenta os dois caminhos coerentes e recomenda o de mudança mínima: manter Kaiming e escalar só as projeções residuais. O caminho GPT-2 completo fica registrado como alternativa para a Etapa 18, se o treino mostrar instabilidade.

### O que foi medido

**A contagem, nas três configurações que interessam:**

| | tiny | médio | GPT-2 small |
|---|---|---|---|
| `V` / `d` / `L` | 65 / 128 / 4 | 65 / 384 / 6 | 50.257 / 768 / 12 |
| pilha | 792.064 | 10.642.176 | 85.036.032 |
| total amarrado | **817.024** | **10.766.208** | **124.421.376** |
| total sem amarrar | 825.344 | 10.791.168 | 163.018.752 |

Duas leituras que o capítulo destaca. No tiny, a pilha é **96,9%** do modelo — com 65 caracteres de vocabulário, embeddings e cabeça quase não pesam. No GPT-2, os embeddings são **31,7%**.

E o número famoso finalmente fecha: `163.018.752 − 38.597.376 = 124.421.376`, mais os `18.432` dos vieses `b_K`/`b_V` que este projeto removeu, dá os `124.439.808` oficiais. **O GPT-2 é 124M porque amarra os pesos** — sem amarrar seriam 163M.

O roadmap estimava "1 a 10M" para o tiny; a conta real dá `817 mil`, porque a estimativa presumia vocabulário maior. Com tokenização por caractere, o vocabulário é pequeno por construção.

### Verificação

- **Todos os exemplos numéricos conferidos por script**, incluindo a atenção causal em Python puro usada na prova da permutação.
- **Prosa medida** com o script do §9 do guia: média `14.7` palavras por frase, p90 `23`, `5.3` parênteses por mil palavras. Doze frases longas foram quebradas para o p90 entrar na meta.
- **Os dois `.html` validados no navegador**, com `getBBox` comparado ao `viewBox` das quatro figuras — nenhuma estoura. Console sem erros; o quiz confere 15 questões, 4 alternativas cada, toda `data-answer` apontando para uma alternativa existente.
- **Glossário do `00-overview` atualizado** com seis termos: profundidade, cabeça de linguagem, `ln_f`, equivariância a permutação, pesos amarrados e a entrada de bloco transformer.

### Blocos de armadilha, e de onde vieram

- **§1** — índices de token guardados como `Double` (Etapa 9). Um índice fracionário seria truncado em silêncio, e o modelo leria a linha errada da tabela.
- **§8** — o `Linear` no lugar do `LayerNorm` (Etapa 14, ontem). Entra aqui com a lição refinada: **uma referência montada a partir dos campos do objeto sob teste herda os erros que estão nos campos**. Nesta etapa isso vale em dobro, porque o modelo é composição de composições.

### O que o capítulo deixa encaminhado

- **`GPT(vocabSize, dModel, nHeads, nLayers, contextLength, expansion = 4)`**, com os quatro campos públicos e um `foldLeft` no forward.
- **`parameters` deduplicado.** Com pesos amarrados a tabela aparece em dois lugares do grafo; listar duas vezes faria o otimizador da Etapa 17 aplicar o passo duas vezes ao mesmo tensor. O teste é `toSet.size == size`.
- **Oito frentes de teste**, com "a posição importa" como a que carrega a etapa, e um aviso de custo: o gradient check faz dois forwards completos por elemento, então a configuração precisa ser minúscula.
- **`List.fill(nLayers)(umBlocoSó)`** é a mutação mais traiçoeira da etapa: passa em formato, causalidade e gradient check, porque é uma arquitetura válida — só que com `L` vezes menos capacidade.

Próximo passo: implementar o `GPT` (escrito em `gpt/nn/`, movido para `gpt/model/` em 2026-08-31).

---

## 2026-08-31 — Etapa 15: o modelo completo, e uma demonstração que valia menos do que dizia

O usuário escreveu o modelo. A revisão pegou um erro de camada, e a validação por mutação derrubou uma afirmação do capítulo escrito horas antes. Suíte de **370 testes** (210 `scalagrad` + 160 `gpt`), sendo 16 novos.

O `forward` é um `foldLeft` de quatro linhas. Como na Etapa 14, o que interessa é o que quase passou.

### O erro do código: a cabeça de linguagem era uma atenção

```scala
val head = MultiHeadAttention(dModel, nHeads)   // deveria ser Linear(dModel, vocabSize, useBias = false)
```

O modelo nunca chegava ao vocabulário: a saída era `[B, T, dModel]` em vez de `[B, T, vocabSize]`. Compila porque a MHA também preserva o formato. É a mesma família do `Linear` no lugar do `LayerNorm` de ontem — camada errada com assinatura compatível —, e mais uma vez o que denuncia é o **formato da saída**, não o valor.

O usuário corrigiu junto com o `x` do `forward`, que sombreava dois tensores de ranks diferentes com o mesmo nome. As duas validações que faltavam entraram na sequência: `nLayers >= 1` no construtor, e `rank == 2` mais `T <= contextLength` no `forward`.

### O achado que corrige a teoria: a invariância é de **uma** camada

A mutação "remover o embedding posicional" passou em **16 de 16**. Ou seja: o teste que o capítulo chamava de "o que carrega a etapa" não pegava exatamente o erro para o qual foi desenhado.

A causa não é o teste, é a afirmação. A §2 provou que uma camada de atenção causal sem posição enxerga o *conjunto* de tokens visíveis, não a ordem. Isso é verdade — e para de valer com dois blocos:

| camadas de atenção | diferença em `t=2`, prefixo trocado |
|---|---|
| 1 | `0.000000` |
| 2 | `5.09e-2` |
| 3 | `6.43e-1` |

A máscara causal é assimétrica por construção: a posição 0 vê um token, a 1 vê dois, a 2 vê três. Essa assimetria já aparece na saída da primeira camada — `h₀` depende só de `x₀`, e `x₀` mudou —, e a segunda camada a lê. **Um modelo causal profundo reconstrói a ordem sem tabela posicional nenhuma.**

Isso não invalida a tabela: ela dá a informação de forma direta, desde a primeira camada, em vez de exigir que o modelo a deduza de um efeito colateral. Mas muda o que se pode afirmar, e muda o teste.

Correções: o teste passou a usar `nLayers = 1`, com comentário explicando por que a profundidade importa; a §2 ganhou a tabela dos três números e uma ressalva; e o cartão de referência, o guia visual e duas questões dos exercícios foram ajustados junto.

**A lição de método:** a validação por mutação não confere só a suíte. Ela confere a **afirmação** que a suíte tenta proteger. Aqui, o teste estava correto e a hipótese é que estava larga demais.

### O ajuste do `ε` no gradient check

O check do modelo inteiro falhava por pouco com o `ε = 1e-5` padrão: `1.86e-5` contra o teto de `1e-5`. Antes de mexer na tolerância, medi a curva:

| `ε` | erro máximo entre os 19 parâmetros |
|---|---|
| `1e-3` | `7.4e-2` |
| `1e-4` | `1.7e-3` |
| `1e-5` | `2.4e-4` |
| `1e-6` | `2e-7` a `9e-7`, em quatro modelos independentes |
| `1e-7` | `4e-6` a `6e-6` |

É a curva em U da Etapa 4 §1, com o mínimo deslocado: a composição de embedding, bloco, LayerNorm e cabeça tem curvatura alta o bastante para o erro de truncamento dominar em `1e-5`. O gradiente analítico não é pequeno — o máximo absoluto medido foi `78` —, então não é o caso de gradiente verdadeiro zero da Etapa 11.

A suíte usa `ε = 1e-6` e mantém a tolerância em `1e-5`. Ajustar o `ε` é a correção certa aqui; afrouxar a tolerância teria escondido a informação.

### A suíte: 16 testes

Formato de ponta a ponta em quatro configurações; composição contra os próprios campos; ordem dos blocos contra a inversa; o LayerNorm final contra a versão sem ele; permutação do prefixo com `nLayers = 1`; causalidade com contraprova; `parameters` sem repetição e com todos treináveis; blocos com tensores próprios (`14 · nLayers` distintos); contagem contra a fórmula, generalizada no fator de expansão; a proporção de 96% da pilha no tiny, conferida pela fórmula sem construir o modelo; gradient check nos 19 parâmetros da configuração minúscula; e as três validações.

Duas notas de teste que valem para as próximas etapas:

- **A fórmula da contagem precisou ser generalizada.** `12·d² + 11·d` vale para `expansion = 4`, e a suíte roda com 2. A forma geral é `(4 + 2e)·d² + (e + 7)·d` por bloco, e um teste separado confirma que ela reduz à canônica quando `e = 4`.
- **Não há gradient check em relação à entrada.** Os tokens são índices inteiros; perturbá-los não significa nada.

### Validação por mutação

| mutação | testes que falharam |
|---|---|
| cabeça com dimensões trocadas | 8 de 16 |
| `List.fill(nLayers)(umBlocoSó)` | 4 de 16 |
| esquecer o LayerNorm final | 2 de 16 |
| aplicar os blocos na ordem inversa | 2 de 16 |
| remover o embedding posicional | 1 de 16 |

A última só passou a ser pega depois da correção do `nLayers`.

### O que ficou aberto

- **Weight tying**, opcional, não implementado. Exige expor a tabela de token na `Embedding`.
- **Escala residual `1/√(2·nLayers)`** e a inconsistência entre `0.02` e Kaiming. A §6 do capítulo tem os dois caminhos e a recomendação; a decisão fica para quando houver treino para medir.
- **`GPTConfig`**, adiado para a Etapa 18, que é quem precisa serializar.

### Próximo passo

Etapa 16, a perda: cross-entropy sobre os logits. É a primeira vez que o projeto produz **um único número** medindo o erro — o `L` que a notação `dX ≡ ∂L/∂X` usa desde a Etapa 2. Como sempre, `theory/16-cross-entropy/` antes do código.

---

## 2026-08-31 — `theory/16-cross-entropy/` escrita antes da implementação (Etapa 16)

Segue a convenção da Etapa 4. Três arquivos, no padrão do `theory/STYLE-GUIDE.md`.

- **`16-cross-entropy.md`** — 8 seções numeradas. É a etapa que finalmente produz o `L` que a notação `dX ≡ ∂L/∂X` usa desde a Etapa 2. Até aqui, nos testes, um `.sum` qualquer fazia o papel dele.
- **`cross-entropy.html`** — quatro figuras. FIG 1, o caminho de `[B,T,V]` a um escalar, com a máscara one-hot; FIG 2, a curva de `−log(p)` com os quatro pontos da tabela marcados; FIG 3, os dois modos de falha numérica lado a lado; FIG 4, as barras do gradiente `p − y`, com a soma zero visível.
- **`exercises.html`** — 15 questões (4 fácil, 5 médio, 4 difícil, 2 desafio).

### O que a etapa aproveita do que já existe

Duas peças chegaram prontas, e isso mudou o escopo do capítulo:

- **`logSoftmax` já existe**, estável, desde a Etapa 6. O capítulo não precisa implementá-lo — precisa explicar por que ele existe como operação própria, e é o que a §3 faz com os dois modos de falha medidos.
- **O `BatchSampler` já entrega o alvo deslocado.** `inputs = corpus[i .. i+T)` e `targets = corpus[i+1 .. i+T]`, alinhados posição a posição. A §5 registra isso de forma explícita, porque deslocar de novo dentro da perda é o erro que o próprio roadmap alerta.

### O que foi medido

**Os dois modos de falha, que são independentes:**

```
z = [1000, 1001, 1002]:   exp direto → OverflowError
                          com o máximo subtraído → [0.09003, 0.24473, 0.66524]

z = [0, −800]:            softmax → [1.0, 0.0]  →  log(0) = −infinito
                          logSoftmax → [0.0, −800.0]  exato
```

Vale registrar a distinção, porque ela costuma ser confundida: subtrair o máximo resolve **só** o estouro para cima. O de baixo exige calcular o logaritmo sem nunca materializar a probabilidade — e é isso, e não a subtração do máximo, que justifica o `logSoftmax` existir.

**O gradiente, conferido por diferenças finitas.** Com `z = [2.0, 1.0, 0.1, −0.5]` e alvo 2:

```
analítico (p − y) = [0.62518, 0.22999, -0.90649, 0.05132]      soma = 0.0
numérico          = [0.62518, 0.22999, -0.90649, 0.05132]
```

A derivação em dois passos mostra por que a fórmula é tão simples: o `p_a` do Jacobiano do softmax cancela com o `1/p_a` que vem do `−log`. É o argumento de por que cross-entropy e softmax são sempre implementados fundidos.

**A tabela de sanidade**, que é o que mais deve economizar tempo na Etapa 18:

| `V` | perda esperada no primeiro passo |
|---|---|
| 4 | `1.38629` |
| 65 | `4.17439` |
| 50.257 | `10.82491` |

Com logits uniformes e `V = 65`, o valor medido é exatamente `4.17439` — não aproximadamente. E a perplexidade de um modelo aleatório é exatamente `V`, o que dá uma leitura imediata do progresso.

### A decisão de implementação: máscara one-hot

O `scalagrad` não tem `gather`, e selecionar o logit do alvo exige escolher uma coluna por linha. A §7 recomenda a máscara one-hot multiplicada elemento a elemento, em vez de acrescentar uma operação ao núcleo.

O argumento que decide não é "é mais simples": é que **a máscara tem exatamente o tamanho dos logits, que já existem**. Ela duplica um tensor sem mudar a ordem de memória, e usa só operações já testadas. O `gather` fica para quando houver desempenho medido pedindo — é o que o PyTorch faz, e por esse motivo.

### Blocos de armadilha, e de onde vieram

- **§3** — o softmax no eixo errado (Etapa 12). Aqui o alvo é o eixo do vocabulário, e o eixo errado produz números plausíveis com formato idêntico. O corolário de teste é `T ≠ V`.
- **§5** — índices de token como `Double` (Etapa 9). Os `targets` chegam como `1.0`, `4.0`, `2.0`, e um alvo truncado em silêncio faz o modelo aprender a associação errada sem nenhum sintoma imediato.

### Verificação

- **Todos os exemplos numéricos conferidos por script**, incluindo o gradiente contra diferenças finitas e os dois estouros reproduzidos de verdade.
- **Prosa medida:** média `14.0` palavras por frase, p90 `23`, `12.5` parênteses por mil palavras. Dentro das três metas, sem reescrita.
- **Os dois `.html` validados no navegador**, com `getBBox` conferido contra o `viewBox` das quatro figuras — nenhuma estoura. Console limpo; o quiz confere 15 questões com `data-answer` válida.
- **Glossário do `00-overview` atualizado** com quatro termos: função de perda, cross-entropy, log-sum-exp e perplexidade.

### O que o capítulo deixa encaminhado

- **`gpt/loss/CrossEntropy.scala`**, um `object` com `apply(logits, targets)` devolvendo um `Tensor` escalar — não um `Double`, porque é nele que o `backward()` é chamado.
- **Seis mutações** na §7. A mais perigosa é o `logSoftmax` no eixo das posições: passa em formato, passa no gradient check, e só um valor calculado à mão com `T ≠ V` reprova.
- **O teste que carrega a etapa** é o do gradiente contra `(p − y)/N`, porque ele confere intenção — coisa que o gradient check, por definição, não faz.

Próximo passo: implementar `gpt/loss/CrossEntropy.scala`.

---

## 2026-08-31 — `GPT` movido para `gpt/model/`, e a fronteira com `nn/` documentada

Reorganização de pacote, decidida em conversa e feita antes de a Etapa 17 começar. Suíte de **370 testes** verde depois da mudança, sem nenhum ajuste além dos `package`.

**O que mudou:** `gpt/nn/GPT.scala` → `gpt/model/GPT.scala`, e a spec junto. Duas linhas de `package`, mais um `import gpt.nn.*` no modelo.

**O critério, agora no `CLAUDE.md`:** `nn/` guarda as **peças** — genéricas, reutilizáveis, cada uma com o seu `parameters`. `model/` guarda o **artefato montado**, que consome as peças e é o que treino e geração vão importar. É a mesma fronteira que separou `scalagrad/` de `gpt/`, um nível abaixo.

Pelo mesmo critério, a `CrossEntropy` da Etapa 16 vai para `loss/` e não para `nn/`: ela não tem parâmetro, não entra na lista que o otimizador percorre, e o `backward()` **começa** nela em vez de atravessá-la.

**Por que agora:** nada além da própria spec importava o `GPT`. Depois da Etapa 18 seriam o laço de treino, o gerador e o checkpoint também. O custo de mover só cresce.

**O contra-argumento, registrado:** um pacote com uma classe é organização prematura. O que decide a favor é a Etapa 18 — o `GPTConfig` que o checkpoint precisa serializar mora junto do modelo, não junto do `Linear`, e aí o pacote deixa de ter um arquivo só.

Aproveitei para corrigir duas coisas desatualizadas na mesma seção do `CLAUDE.md`: o `gradcheck/` estava marcado como planejado desde a Etapa 4, e a nota final ainda dizia que `gpt/src/` não existia fisicamente.

---

## 2026-08-31 — Etapa 16 concluída: a perda, quatro bugs e um `oneHot` novo no núcleo

O usuário escreveu a perda em três rodadas de revisão. Suíte de **390 testes** (215 `scalagrad` + 175 `gpt`), sendo 20 novos: 15 do `CrossEntropySpec` e 5 do `Tensor.oneHot`.

O projeto tem, pela primeira vez, um `L` de verdade — o número que a notação `dX ≡ ∂L/∂X` usa desde a Etapa 2.

### Os quatro bugs, todos silenciosos de alguma forma

**1. `Masks.causalMask` no lugar da máscara one-hot.** A primeira versão selecionava o alvo com a máscara da *atenção* — `[T,T]`, com `−inf` acima da diagonal. As duas não têm parentesco: uma esconde o futuro somando antes do softmax, a outra seleciona a resposta certa multiplicando depois. O formato `[B·T, B·T]` não faz broadcast com `[B,T,V]` a não ser por coincidência, e onde fizesse, `−inf × 0` daria `NaN`. O sintoma que resume: `targets` não era usado em lugar nenhum da função.

**2. `oneHot` comparando a linha em vez da coluna.** Escrito como fábrica nova no `scalagrad`, ele nasceu com `if i == indices(i)` onde devia ser `j`. Cada linha saía inteira de zeros ou inteira de uns:

```
esperado, indices = [2,0,3]        obtido
[0, 0, 1, 0]                       [0, 0, 0, 0]
[1, 0, 0, 0]                       [0, 0, 0, 0]
[0, 0, 0, 1]                       [0, 0, 0, 0]
```

Com a máscara toda zero, a **perda seria sempre `0.0`** — gradiente zero, treino que roda e não move um parâmetro.

**3. Falta do `reshape`.** `oneHot` devolve `[B·T, V]` e os logits são `[B,T,V]`. O broadcasting alinha pela direita, então `T` contra `B·T` só bate quando `B = 1`. Funciona no primeiro teste de console e estoura em qualquer lote maior.

**4. Falta do sinal negativo.** `(logSoftmax * oneHot).sum` soma logaritmos de probabilidade, todos ≤ 0. A perda sairia negativa, e o treino da Etapa 18 minimizaria o oposto do que deve — rodando perfeitamente enquanto faz isso.

### As decisões de projeto que a etapa fechou

**Máscara one-hot em vez de `gather`.** O argumento que decidiu não foi simplicidade: é que a máscara tem exatamente o tamanho dos logits, que já existem. Ela duplica um tensor sem mudar a ordem de memória, e usa só operações testadas.

**`oneHot` no `scalagrad`, e não em `gpt`.** Maquinaria genérica de tensor, sem nada de GPT dentro — mora ao lado de `Tensor.zeros` e `Tensor.fill`. A assinatura espelha a do `indexSelect`: recebe `Array[Int]` achatado, devolve rank 2, e o chamador faz o `reshape`. O que é específico da perda — alvo inteiro e dentro de `[0, V)` — ficou no `CrossEntropy`.

**`sum / N` e não `.mean`.** O `.mean` dividiria por `B·T·V`; só `B·T` valores são não nulos depois da máscara. Vale registrar porque a troca parece uma simplificação inofensiva.

### Validação por mutação

| mutação | testes que falharam |
|---|---|
| `logSoftmax` no eixo 1 em vez do 2 | 7 de 15 |
| esquecer o sinal negativo | 6 de 15 |
| somar em vez de mediar | 2 de 15 |
| deslocar os alvos em uma posição | 2 de 15 |
| dividir por `B` em vez de `B·T` | 2 de 15 |
| `softmax` seguido de `log` | **1** de 15 |
| `oneHot` com `i == indices(i)` | 2 no `TensorSpec`, 7 no `CrossEntropySpec` |

**A linha do `softmax`+`log` é a que ensina.** Com logits de tamanho normal, os dois caminhos produzem o **mesmo número**: o valor à mão passa, a média passa, o `log(V)` passa, o gradiente passa. Só o caso com diferença de 800 entre logits reprova. Uma suíte sem teste de estabilidade daria a troca por boa, e o erro apareceria meses depois como um `NaN` no meio do treino. A §7 do capítulo foi atualizada com os números medidos e com essa observação.

### A suíte

Valor à mão (`2.36971`); média do lote (`1.33821`); `log(V)` exato com logits uniformes, para `V ∈ {4, 65, 256}`; independência do alvo escolhido quando os logits são uniformes; invariância a somar `1000` a todos os logits; o caso `[0, −800]` dando `800.0` sem `NaN`; o gradiente conferido contra `(p − y)/N` posição a posição; a soma dos gradientes de cada posição dando zero; gradient check; perplexidade `65.0`; e cinco validações.

`T = 2` e `V = 4` são distintos de propósito — com `T == V`, o `logSoftmax` no eixo errado sobrevive a qualquer teste de valor.

No `TensorSpec`, cinco testes para o `oneHot`: a matriz esperada posição a posição, exatamente um `1.0` por linha, linhas repetidas para índices repetidos, `requiresGradient = false`, e a rejeição de índice fora da faixa.

### Próximo passo

Etapa 17, o AdamW. É a primeira peça do projeto cujo trabalho é **mudar estado** — até aqui tudo era função pura de tensores. O `Gradient` mutável, decidido lá na Etapa 2, existe para este momento. Como sempre, `theory/17-adamw-optimizer/` antes do código.

---

## 2026-09-01 — Etapa 17 começa pela teoria: o capítulo do AdamW, e duas decisões de projeto tomadas antes do código

`theory/17-adamw-optimizer/` escrita: o capítulo (`17-adamw-optimizer.md`, §1 a §9), o guia visual (`adamw.html`, 5 figuras) e o `exercises.html` com 14 questões. Nenhum `.scala` novo — o código da etapa ainda não existe.

Prosa medida pelo script do `theory/STYLE-GUIDE.md`: média de **12,2** palavras por frase, p90 **20**, **5,5** parênteses por mil palavras. As três metas passam. Os `.html` foram validados no navegador: 5 SVGs sem nada fora do `viewBox` (um estouro de 2,8 px no rótulo do eixo da figura 3 foi corrigido aumentando o `viewBox` para 310) e zero erro de console. O `exercises.html` foi exercitado programaticamente — os 14 gabaritos existem como opção, o placar fecha em 14/14 e o resumo final aparece.

### As duas decisões que a etapa precisou fechar antes da teoria

**O estado do otimizador é imutável.** O `step()` não muda campo nenhum: ele devolve um `AdamW` novo, com `t + 1` e com o mapa de `m`/`v` atualizado. O loop de treino da Etapa 18 vira um `foldLeft` carregando o otimizador como acumulador, sem `var`. O custo é honesto e ficou registrado no §7 do capítulo: dois arrays novos por parâmetro a cada passo, mais o array do `updateData`. A conclusão fica pendente de medição, pelo precedente do `matmul` de 2026-08-21, onde a formulação mais funcional foi também a mais rápida.

**Os parâmetros continuam sendo mutados no lugar.** Mesmo argumento do `Gradient` na Etapa 2: outros nós do grafo guardam a referência do tensor, e trocar a instância quebraria essas referências.

### O bloqueio que apareceu ao conferir o código

`data` é `private[scalagrad]`. O `AdamW` mora em `gpt.optim`, fora do módulo, então `p.data(i) = ...` **não compila** — o único lugar que escreve em `data` hoje é o `Gradcheck`, e só porque ele mora dentro do `scalagrad`. Não havia nenhuma API pública de escrita em `Tensor`.

Foram consideradas três saídas: um `updateData` recebendo o array pronto, um `updateData` recebendo uma função `(Int, Double) => Double`, e mover o otimizador para `scalagrad.optim` (onde ele enxergaria `data` sem API nova, já que AdamW não tem nada de GPT dentro). Escolhida a primeira:

```scala
def updateData(values: Array[Double]): Unit
```

com duas pré-condições, ambas `require` pela regra da Etapa 7 — `values.length == size` e `isContiguous`. Os motivos: o chamador monta o array com `Array.tabulate`, no estilo do resto do projeto; a Etapa 18 usa o mesmo método para restaurar checkpoint num modelo já construído; e o `AdamW` fica em `gpt/optim/` como a estrutura de diretórios do `CLAUDE.md` planeja.

O `require(isContiguous)` existe por causa do bug de 2026-08-21: `data` é indexado fisicamente e `gradient`, canonicamente. Todo parâmetro nasce contíguo, então a checagem nunca dispara por uso legítimo — ela guarda o dia em que alguém passar uma view.

**O `zeroGrad` não precisou de API nova.** O campo `gradient` é público e `Gradient.zero()` também, então `parameters.foreach(_.gradient.zero())` já resolve de dentro do `gpt`.

### O que o capítulo ensina, em uma linha por seção

SGD e por que um `lr` só não serve para gradientes de escalas diferentes; o momento como média móvel exponencial; o segundo momento e a invariância de escala (gradientes multiplicados por 1000 dão o mesmo passo, medido igual até `1e-10`); a correção de viés, que faz o primeiro passo valer exatamente `lr`; o weight decay desacoplado, com a tabela onde o decaimento via L2 se espalha por um fator de **333** enquanto o do AdamW fica constante; o estado indexado por identidade de referência, que só funciona porque `Tensor` não é `case class`; e o `updateData`.

Todos os números do capítulo foram calculados por script antes de escrever, e duas tabelas foram corrigidas na revisão: eu havia misturado comprimento de trajeto (SGD) com deslocamento líquido (Adam), e o valor da posição 2 estava em `8.11e-4` quando o correto é `7.99e-4`.

### O glossário

`theory/00-overview/00-overview.md` §5 ganhou onze termos novos na tabela de Treinamento: otimizador, SGD, média móvel exponencial, primeiro e segundo momento, passo adaptativo, correção de viés, weight decay, regularização L2 e hiperparâmetro.

### Próximo passo

O código da Etapa 17, na ordem do checklist: `Tensor.updateData` no `scalagrad` (com os testes de `require`, incluindo o caso da view não contígua), depois SGD como ponto de partida, e então o AdamW completo. O teste que não pode faltar é o do primeiro passo: `|passo| = lr` para gradientes de tamanhos bem diferentes. É ele que pega a falta da correção de viés, que de outra forma é silenciosa.

---

## 2026-09-01 — Etapa 17 concluída: AdamW, SGD e a porta de escrita no `Tensor`

Suíte de **419 testes** (224 `scalagrad` + 195 `gpt`), 29 novos: 9 do `Tensor` (`toArray` e `updateData`), 20 do `AdamWSpec` e 8 do `SGDSpec`. O usuário pediu que a implementação fosse feita por mim a partir daqui, para revisar depois.

### O que o usuário já tinha escrito

`Tensor.updateData` e o esqueleto do `AdamW`. Na revisão apontei três defeitos, todos corrigidos por ele antes de eu seguir:

1. **A mensagem do `require` citava `$rank` no lugar de `$size`.** Num parâmetro `[4, 8]` ela diria "must be equal to 2" quando o esperado eram 32 valores — mensagem que faz duvidar do array em vez do código.
2. **O laço percorria `data.indices`, mas o contrato é sobre `size`.** Coincidem hoje em todo parâmetro real; divergiriam num tensor contíguo sobre buffer maior, e aí o `values(i)` estouraria. Mesmo raciocínio do `Gradient.zeros(shape)`: quem define o invariante é o `shape`.
3. **O retorno era `Tensor`, devolvendo `this` mutado.** `reshape` e `transpose` têm a mesma assinatura e devolvem instância **nova** — `val p2 = p.updateData(v)` parecia cópia e não era. Virou `Unit`, como `Gradient.zero()`.

Ele também renomeou `gpt/optm/` para `gpt/optim/` e escreveu o `Tensor.toArray`, que faltava: o `step` precisa **ler** o valor do parâmetro para o weight decay, e `data` é `private[scalagrad]` na leitura também. O `updateData` tinha resolvido só a escrita.

### O `AdamW`

`step()` em três fases, e a separação não é estética: enquanto a fase 1 é pura, dá para testar a aritmética do passo sem tocar em parâmetro nenhum.

1. Calcular, sem escrever — `m`, `v` e os valores novos, lendo `p.toArray` e `p.gradient`.
2. Escrever, via `updateData`.
3. Devolver o `AdamW` novo, com `t + 1` e o mapa reconstruído.

O contador avança **antes** de ser usado (`val nextT = t + 1`): com `t = 0` a correção de viés dividiria por `1 − β⁰ = 0`. E o decaimento usa os valores de **antes** do passo, junto com o termo adaptativo — decair primeiro e calcular o passo sobre o `p` já decaído dá outro resultado, e um que ninguém notaria.

`step(stepLr)` sobrescreve a taxa de um passo sem alterar a do otimizador. É por aí que o schedule da Etapa 18 injeta o `lr` de cada passo, sem precisar reconstruir o otimizador. O `AdamW` devolvido preserva o `lr` configurado.

O construtor recusa lista vazia, parâmetro com `requiresGradient = false` (que viraria um otimizador que roda sem otimizar), `β` fora de `[0, 1)`, `eps` não positivo e `weightDecay` negativo.

Ficou como `final class` e não `case class`. O `copy` seria uma defesa boa contra esquecer um hiperparâmetro ao reconstruir — se o `step()` não repassasse o `lr`, quem configurou `1e-3` voltaria em silêncio para o default no segundo passo. Mas há um único ponto de reconstrução, o gotcha de `equals` com campo `Array` continua valendo, e o resto do projeto é `final class`. Em vez do `copy`, um teste cobre o risco: `keep the hyperparameters on the returned instance`.

### O `SGD`

Escrito como o ponto de partida que o checklist pede, e como linha de base. Por não ter estado, o `step()` dele devolve `Unit`, ao contrário do `AdamW` — não há instância nova a propagar, e fingir que há seria pior. Um dos testes fixa justamente a propriedade que o AdamW abandona: com SGD, gradiente vinte vezes menor anda vinte vezes menos.

### Os testes que importam

**`|passo| = lr` no primeiro passo**, para gradientes de `0.30`, `0.03` e `−0.60`. É o teste que pega a falta da correção de viés, e um segundo teste fixa o número que apareceria sem ela: `1.34164e-4`, ou 44,7% do pretendido. Sem esse par, a ausência da correção é silenciosa — o treino roda e a perda cai.

Também: os três passos do capítulo conferidos posição a posição; invariância a gradientes ×1000; o momento absorvendo a troca de sinal no passo 3 (deslocamento de `6.8776e-5`, ainda descendo, contra os `3e-4` do primeiro passo); decaimento puro com gradiente zero dando `(1 − lr·λ)·p`; parâmetro negativo sendo puxado **para cima**, na direção do zero; `t` e momentos propagando entre instâncias; e o `step` **não** zerando os gradientes, que é responsabilidade do loop.

No `TensorSpec`, o teste que mais vale é o `toArray` sobre tensor transposto: `data` continua na ordem física original, e ler o buffer direto devolveria `[1,2,3,4,5,6]` em vez de `[1,4,2,5,3,6]`. E o `updateData` recusando view transposta e view de broadcast.

### Uma correção minha no caminho

O teste dos três passos falhou na primeira execução por erro meu de asserção, não de código: `p.toArray shouldBe Array(...).map(_ +- tolerance)` compara array contra array de tolerâncias, sem cair no caminho elemento a elemento. Os valores estavam certos por `1e-11`. Trocado por três asserções escalares.

### Próximo passo

Etapa 18, o loop de treino. Ela precisa de duas coisas que ainda não existem: `Gradient.scale`, para o gradient clipping reescalar o que já foi acumulado, e um formato de checkpoint. O `updateData` desta etapa é o caminho para restaurar os parâmetros num modelo já construído.

---

## 2026-09-01 — Etapa 18 concluída: o modelo treina

Suíte de **477 testes** (227 `scalagrad` + 250 `gpt`). O marco desta etapa não é uma peça nova: é a primeira vez que as dezoito anteriores rodam juntas e a perda cai.

### Uma inversão deliberada de ordem, para registro

O projeto escreve `theory/` antes do código desde a Etapa 4. Nas Etapas 18 e 19 eu inverti: implementei primeiro e escrevi o capítulo depois, contra o código pronto. O motivo é que o usuário está fora enquanto isso roda, então a ordem "teoria primeiro" não cumpre o papel didático dela agora — e a queixa recorrente deste `HISTORY.md` é divergência entre `theory/` e implementação. Escrevendo depois, toda assinatura e todo número do capítulo saiu do código real. Se o usuário preferir a ordem original, é só dizer: o resultado final é o mesmo par de arquivos.

### O que foi construído

**`Gradient.scale`**, no `scalagrad`. O clipping precisa multiplicar o gradiente acumulado por um fator, e a API do `Gradient` expunha só `accumulate`, `zero` e `seed`. Dava para fazer sem método novo — `accumulate(i, (c−1)·g(i))` reescala — mas isso lê como truque. O `scale` não é a sobrescrita que a API se recusa a expor: o conteúdo acumulado é preservado, só reescalado.

**`LRSchedule.cosine`.** O roadmap chamava de `getLR`; renomeei para dizer qual decaimento é. Rampa linear no aquecimento, meia volta de cosseno depois, e o progresso travado em 1 no fim. O travamento não é detalhe: sem ele o cosseno volta a **subir** depois de `totalSteps`, e um treino retomado de checkpoint passa desse ponto com facilidade.

**`GradientClipping`.** Norma global, com `globalNorm` exposto à parte para o log. O `clipByGlobalNorm` devolve a norma de **antes** do corte, porque a de depois não informa nada — vale `maxNorm` quando houve corte e é igual à de antes quando não houve.

**`Checkpoint`.** Formato binário: magic, versão, `t`, número de parâmetros, marcador de momentos, e por parâmetro o tamanho, os valores, `m` e `v`. O tamanho de cada parâmetro é conferido na leitura; sem isso, um `dModel` diferente leria bytes do parâmetro seguinte e produziria um modelo embaralhado que ainda roda.

**`Trainer`**, com `TrainingConfig`, `TrainingStep`, `TrainingResult` e `evaluate`. O laço é um `foldLeft` carregando o otimizador como acumulador — sem `var`, como o resto do projeto. O histórico é construído com `::` e invertido uma vez no fim.

**`BatchSampler.split`**, para o item de validação do checklist. Corta **em ordem**, não sorteia: com corte aleatório, trechos vizinhos cairiam dos dois lados e a validação mediria texto que o treino praticamente já viu.

### O teste que fecha o projeto até aqui

Um corpus perfeitamente cíclico — `0, 1, 2, 0, 1, 2, ...` — com `dModel = 16` e uma camada, em 60 passos:

```
perda do passo 1:            ≈ 1.0986 = log(3)
média dos 10 últimos passos: < 0.3
```

A cadeia inteira funciona: tokenização, embedding, atenção com máscara causal, MLP, bloco, modelo, perda, autograd e otimizador. Se **essa** regra não fosse aprendida, o bug não estaria nos hiperparâmetros.

### Um defeito real encontrado pelos testes

`TrainingConfig` aceitava `warmupSteps >= steps` sem reclamar. O erro só aparecia dentro do `LRSchedule`, no primeiro passo — depois de o modelo já ter sido construído e o primeiro lote amostrado. A checagem foi duplicada no `TrainingConfig`, e junto vieram `lrMax >= lrMin` e `maxGradNorm > 0`. Vale como padrão: validação de configuração pertence ao ponto onde a configuração é criada, não ao ponto onde ela é usada.

Também corrigi um teste meu que assumia que o modelo aprenderia devagar. Com `lr = 1e-2` num corpus cíclico, a média das dez primeiras perdas já era `0.64` — o modelo aprende antes do décimo passo. A asserção passou a olhar a perda do passo 1 contra `log(3)`, que é o valor previsto pela Etapa 16.

### As três armadilhas que o capítulo registra

A ordem das seis operações vira contrato nesta etapa, e duas das três inversões possíveis são silenciosas. Zerar depois do `backward` apaga o que se acabou de calcular, e a perda fica parada num valor plausível. Nunca zerar faz o gradiente do passo 10 ser a soma dos dez lotes — e com Adam isso **não** muda muito o tamanho do passo, porque a divisão por `√v̂` absorve a escala. Só a direção sai errada.

### `theory/18-training-loop/`

Capítulo (§1 a §8), guia visual com 5 figuras e `exercises.html` com 14 questões. Prosa medida: média **15,0** palavras por frase, p90 **24**, **4,9** parênteses por mil. Os `.html` validados no navegador — nenhum elemento fora do `viewBox`, zero erro de console, e os 14 gabaritos exercitados até o placar fechar. Nove termos novos no glossário do `00-overview`.

O exemplo numérico que eu destacaria é o da §2: gradientes `a = [3, −4]` e `b = [0.1]`, cortados por norma global e por norma separada. A razão `a₀/b₀` vale `30.0` antes, continua `30.0` com norma global, e cai para `6.0` com norma por parâmetro. É a demonstração de que fatores distintos giram a direção do passo.

### Próximo passo

Etapa 19, a geração. Ela usa duas coisas construídas aqui: o `noGrad` da avaliação, que é o mesmo modo em que a geração roda, e o checkpoint, que permite gerar texto de um modelo treinado em outra sessão.

---

## 2026-09-01 — Etapa 19 concluída: o modelo gera. **O projeto está completo.**

Suíte de **510 testes** (227 `scalagrad` + 283 `gpt`), 33 novos: 20 do `SamplerSpec`, 15 do `GeneratorSpec` e 2 do `GPTSpec`. As 19 etapas do `PLAN.md` estão implementadas, testadas e documentadas.

### O teste que fecha tudo

```
corpus:   0, 1, 2, 0, 1, 2, ...   (600 tokens)
treino:   80 passos, dModel = 16, 1 camada
prompt:   [0, 1, 2]
gerado:   [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
```

O ciclo continua exatamente. Um bug em qualquer uma das 19 etapas derruba este teste — tokenização, embedding, atenção causal, MLP, bloco, modelo, perda, autograd, otimizador, schedule, clipping e amostragem, todos de pé ao mesmo tempo.

### O que foi construído

**`Sampler`**, com três estratégias num `sealed trait`: `Greedy`, `Temperature(t)` e `TopK(k, t)`. O `keepTopK` põe `-inf` nos descartados em vez de zerar probabilidades depois — `exp(-inf)` é exatamente 0, então eles somem no próprio softmax, sem uma segunda normalização.

**`Generator`**, com o laço autoregressivo, a janela deslizante e o callback `onToken` para impressão progressiva. Tudo dentro de `Tensor.noGrad`.

Duas decisões que valem registro. O `TopK` **não** tem `k = 40` por default, ao contrário do que o checklist pedia: um default numérico escondido numa estratégia de amostragem é o tipo de coisa que ninguém revisa. O 40 ficou documentado como valor usual no cartão de referência. E empates na fronteira do corte mantêm mais de `k` candidatos, porque desempatar por índice escolheria pela ordem alfabética do vocabulário — e essa não é uma razão.

### O bug de ponto flutuante que o teste força

A amostragem pela inversa da CDF procura o primeiro índice cuja acumulada passa de `u`. A soma acumulada pode parar um epsilon **abaixo** de 1, e aí nenhum índice satisfaz a condição. O retorno seria um índice inválido, que só estouraria mais tarde no `decode`.

Acontece raramente e depende do sorteio, então o teste passaria quase sempre. O `SamplerSpec` força a situação com um `Random` que devolve `0.9999999999999999`.

### A intermitência que apareceu, e a correção de fundo

Rodando a suíte inteira duas vezes seguidas, dois testes do `TrainerSpec` passaram numa rodada e falharam na outra. A causa: o `GPT` não aceitava `rng`, então cada execução inicializava pesos diferentes, e a perda inicial variava de `1.10` a `1.84`.

O `CLAUDE.md` já apontava esse risco no `Tensor.randn` — *"sem isso nenhuma inicialização de modelo é reproduzível, e a partir da Etapa 18 'rodei de novo e deu diferente' fica indistinguível de 'mudei alguma coisa'"*. Era exatamente isso acontecendo.

O `rng` foi threaded por `GPT → Embedding`, `TransformerBlock → MultiHeadAttention → Linear` e `MLP → Linear`, todos com default `new Random()` para não quebrar chamador nenhum. Dois testes novos no `GPTSpec` fixam a propriedade: mesma semente dá modelos idênticos parâmetro a parâmetro, sementes diferentes dão modelos diferentes.

Com isso, **uma intermitência pré-existente também sumiu**: o gradient check dos catorze parâmetros do `TransformerBlock` falhava esporadicamente (uma vez em cerca de cinco execuções). Os specs de `nn/` já tinham `rng` semeado para as **entradas**, mas os pesos dos módulos nasciam de `Random` próprio. Passando o `rng` do spec para os construtores em `TransformerBlockSpec`, `MLPSpec`, `MultiHeadAttentionSpec` e `GPTSpec`, a suíte virou determinística. Três execuções seguidas, 510 testes, zero falhas.

### Uma afirmação minha que a medição derrubou

Eu havia escrito no capítulo 18 que a perda inicial "tem que estar perto de `log(V)`", e que **acima** disso indicaria bug. A medição mostrou o contrário: ela fica em `log(V)` **ou acima**, e um pouco acima é o normal.

O piso tem prova. Um modelo recém-inicializado não chuta uniformemente — os logits são aleatórios, com dispersão. Como o alvo não tem relação com eles, a perda é a média de `−log(pᵢ)`, e Jensen dá `média(−log pᵢ) ≥ −log(média pᵢ) = log(V)`, com igualdade só na uniforme. Medido com `V = 3`: entre `1.10` e `1.84`, contra `log(3) = 1.0986`.

A §7 e a §5 do capítulo 18 foram corrigidas, e o teste passou a asserir `>= log(V) − 0.05` em vez de `± 0.4`. Continua valendo o outro lado, que é o mais importante: perda inicial **abaixo** do piso significa que o modelo está vendo a resposta, e o suspeito é a máscara causal.

### `theory/19-inference-generation/`

Capítulo (§1 a §8), guia visual com 5 figuras e `exercises.html` com 14 questões. Prosa medida: média **13,4** palavras por frase, p90 **23**, **2,5** parênteses por mil. Os `.html` validados no navegador — dois estouros de `viewBox` corrigidos, zero erro de console, 14 gabaritos exercitados. Sete termos novos no glossário.

### O estado final do projeto

| | |
|---|---|
| Etapas | 19 de 19 |
| Testes | 510 (227 `scalagrad` + 283 `gpt`) |
| Capítulos de teoria | 19 + overview, todos com guia visual e exercícios |
| Dependências de ML/matemática | nenhuma |

O que ficou deliberadamente de fora, e que seria o caminho natural para continuar: KV cache (a geração recalcula o contexto inteiro a cada token), tokenização por sub-palavra em vez de caractere, `top-p`/nucleus sampling, e um backend de GPU — que é justamente o motivo de o `scalagrad` ter sido separado do `gpt` na reestruturação de 2026-07-22.

### Adendo do mesmo dia — `GPTConfig`, e o checkpoint que carrega a própria forma

Ao conferir os checklists no fim, sobrou um item da Etapa 15 que era desta etapa: *"`case class GPTConfig` — adiado para a Etapa 18, que é quem precisa serializar"*. O `Checkpoint` tinha sido escrito sem ele, e portanto guardava pesos sem guardar a forma do modelo — carregar exigia lembrar de cabeça as dimensões, e errar dava um erro de tamanho de parâmetro em vez de uma mensagem útil.

Fechado: `GPTConfig` criada, `GPT` ganhou construtores secundários que a recebem e um `val config`, e o formato do checkpoint subiu para a **versão 2**, com a configuração logo após o cabeçalho. Duas consequências práticas:

- `Checkpoint.load` confere a configuração **antes** dos tamanhos, e a mensagem nomeia as duas: `saved from GPTConfig(6,8,2,1,4,4), but the model given is GPTConfig(6,16,2,1,4,4)`.
- `Checkpoint.loadModel(file)` reconstrói o modelo sozinho a partir do arquivo. É o caminho para gerar texto de um treino antigo sem saber com que dimensões ele rodou.

Suíte final: **513 testes** (227 `scalagrad` + 286 `gpt`), três execuções seguidas sem intermitência.

Fica pendente, de propósito, o outro item aberto da Etapa 15: a escala residual `1/√(2·nLayers)` na inicialização. Não é da Etapa 18 nem da 19, e mexer nela mudaria a inicialização de todo modelo — melhor ser uma decisão consciente do que um efeito colateral de arrumação de checklist.

---

## 2026-09-01 — Escala residual `1/√(2·nLayers)`: a última pendência da Etapa 15, fechada

Autorizada pelo usuário depois de eu tê-la deixado aberta de propósito no fecho do projeto. Suíte de **523 testes** (227 `scalagrad` + 296 `gpt`), 10 novos, duas execuções seguidas sem intermitência.

### O que foi implementado

O caminho 1 da §6 do capítulo 15, que era a recomendação registrada lá: manter Kaiming e escalar só as duas projeções que escrevem no fluxo residual.

- `Linear` ganhou `initScale: Double = 1.0`, que multiplica o desvio padrão de Kaiming.
- `MultiHeadAttention` e `MLP` ganharam `residualScale`, repassado só para `outProj` e `down`.
- `TransformerBlock` repassa, e o `GPT` calcula `1.0 / Math.sqrt(2.0 * nLayers)` e expõe como `val residualScale`.

Todos com default que preserva o comportamento anterior, então nenhum chamador quebrou. Um teste fixa isso: `Linear` com `initScale = 1.0` explícito produz os mesmos pesos que sem o argumento, dada a mesma semente.

### A medição, e o que ela desmentiu

Desvio padrão do fluxo residual na saída de uma pilha de blocos, entrada `N(0,1)`, `dModel = 32`:

| `L` | sem escala | com `1/√(2L)` |
|---|---|---|
| 1 | `2.14` | `1.67` |
| 2 | `3.64` | `2.05` |
| 4 | `8.11` | `2.95` |
| 8 | `18.39` | `4.53` |
| 16 | `40.00` | `6.94` |

A escala funciona — sem ela o crescimento é quase linear na profundidade, com ela é bem mais lento, e em 16 camadas a diferença chega a quase seis vezes.

**Mas ela não deixa o desvio constante, que era o que a derivação da Etapa 14 §5 previa.** A conta diz que a soma telescopa para variância `2`, ou seja, desvio `1.41` em qualquer profundidade. O medido vai de `1.67` a `6.94`.

A causa estava escrita na própria §6, dois parágrafos antes da recomendação: a derivação supõe subcamada de variância 1, e aqui ela devolve mais, porque o fator 2 do Kaiming está em **todas** as `Linear` e não só na que vem depois da GELU. Cada bloco amplifica um pouco antes de a escala dividir, e o excedente se acumula.

Ou seja, a medição confirmou empiricamente o argumento que o capítulo já registrava a favor do caminho 2 (adotar `0.02` em tudo, à moda do GPT-2). A §6 foi reescrita: de "recomendação" para "decidido e medido", com a tabela e com essa ressalva explícita.

### O que fica aberto, agora como escolha e não como esquecimento

A inconsistência entre `0.02` na `Embedding` e Kaiming na `Linear` — o caminho 2. Corrigi-la de vez troca a derivação da Etapa 8 §5 por outra, e é mudança de porte diferente. O checklist 15 registra isso no lugar da antiga pendência.

O **weight tying**, também da Etapa 15, foi marcado como decisão de não implementar em vez de item em aberto: exigiria expor a tabela de token na `Embedding`, e é otimização de tamanho de modelo, não de correção.

Com isso, os 19 checklists estão fechados.

---

## 2026-09-01 — O projeto vira executável: corpus, CLI e o primeiro treino de verdade

Última lacuna fechada. Até aqui as 19 etapas estavam implementadas e testadas, mas não havia como **rodar** nada: nenhum `@main`, nenhum corpus, e o único treino que existia era o sintético de três tokens dentro dos testes. Suíte de **537 testes** (227 `scalagrad` + 310 `gpt`).

### O corpus

**Dom Casmurro**, de Machado de Assis, domínio público, baixado do Project Gutenberg. Cabeçalho e rodapé removidos: **380.928 caracteres, 96 símbolos** — letras, pontuação, acentos do português e as aspas em guilhemet. Nenhum caractere corrompido; o que parecia lixo na primeira inspeção era o console do Windows renderizando acento.

### O `gpt/cli/`

Dois pontos de entrada, `train` e `chat`, com um parser de flags de três linhas. O `train` lê o corpus, tokeniza, faz o split de validação, treina com log e checkpoint periódico, e no fim imprime uma amostra do que o modelo aprendeu. O `chat` carrega o checkpoint com `Checkpoint.loadModel` — que existe justamente porque o `GPTConfig` entrou no arquivo — e abre um laço de conversa.

O chat **acumula o contexto**: cada linha do usuário é anexada ao histórico, e o modelo continua de onde parou, inclusive do que ele mesmo gerou. `:temp`, `:topk`, `:tokens`, `:limpar`, `:ajuda` e `:sair` funcionam no meio da sessão.

`Tokenizer` ganhou `val alphabet: Set[Char]`, para a CLI filtrar o que o usuário digita. Sem ele, um emoji numa linha faria o `encode` lançar e derrubar a sessão inteira.

### Três bugs reais, todos achados testando de verdade

Nenhum apareceria só lendo o código, e o primeiro deixava o programa **inutilizável**:

1. **O sbt não repassa stdin para um `run` não forkado.** O `chat` recebia EOF na primeira leitura e imprimia "até mais." imediatamente. Corrigido com `Compile / run / fork := true` e `connectInput := true`. Foi encontrado canalizando `printf 'Bentinho\n:sair\n' | sbt ...` — se eu tivesse me contentado com "compila e os testes passam", a entrega estaria quebrada.
2. **O fork roda a partir do diretório do submódulo.** `gpt/data/corpus.txt` virava `gpt/gpt/data/corpus.txt`, e o `chat` dizia que não achava o checkpoint que existia. Corrigido com `Compile / run / baseDirectory := (ThisBuild / baseDirectory).value`.
3. **A JVM forkada escrevia os acentos em latin-1.** Confirmado decodificando os bytes de saída: `ç` saía como `0xe7` solto, e num terminal UTF-8 todo acento viraria lixo — o que num corpus em português é constante. Corrigido com `-Dstdout.encoding=UTF-8` nas `javaOptions`, e verificado de novo pelos bytes: agora decodifica como UTF-8, com `ã` em U+00E3 e `ç` em U+00E7.

### Uma correção de estilo que o usuário pegou

Escrevi o `Main.scala` inteiro com identificadores em português — `Estado`, `conversa`, `aplicar`, `responder` — e ainda com acento em alguns (`começo`, `duração`, `estratégia`). O usuário perguntou por quê, e a resposta é que não havia motivo: a regra do `theory/STYLE-GUIDE.md` §8 é prosa em português e **identificadores em inglês**, e todo o resto de `src/main` segue.

Aconteceu por contágio de escrever prosa e helpers de teste em português a sessão toda. Renomeado: `ChatState`, `chatLoop`, `applyCommand`, `respond`, `startedAt`, `elapsed`, `history`, `strategy`, `summary`, `known`, `dropped`. As strings visíveis continuam em português, que é interface e não identificador. Nos testes ficou o padrão que o próprio usuário estabeleceu no `CrossEntropySpec`, com locais em português.

Vale a checagem que sobrou: um script varre `src/main` procurando identificador acentuado ou em português. Deu zero nos dois.

### O primeiro treino de verdade

```
modelo      GPTConfig(96, 64, 4, 2, 32, 4)   -> 114.176 parâmetros
corpus      389.182 tokens, vocabulário 96
duração     27min46s, 552 ms por passo, 3000 passos
perda       5,2073 no primeiro passo -> 1,8637 no último
validação   2,46 (perplexidade ~11,7)
```

A perda inicial de `5,21` contra o piso de `log(96) = 4,56` bate com o que a Etapa 18 §7 prevê: começa **acima** do chute uniforme, nunca abaixo.

Amostra gerada com `top-k 20` e `T = 0,8`, a partir de "Capitu":

```
Capituras de persas outro ais suinha no
creplijo, timo bem viam sai a istro subirmade
liculade aindo mai que vinsem le que mas me nomaneda, tamos; o que foi e esta
inha e mão nostos. Visto semina do posca, e entranço, da ando mal cressa
```

Palavras reais do português aparecem sozinhas — *que, mas, mais, era, foi, esta, uma, bem, mal, outro, viam, sai*. A morfologia é plausível, a pontuação está no lugar, as quebras de linha imitam o diálogo do romance. Não é texto coerente, e não seria: são 114 mil parâmetros com 32 caracteres de contexto, contra os 124 milhões do menor GPT-2.

**E o motivo do tamanho é medido, não estimado:** 552 ms por passo em Scala puro sem BLAS. Dobrar o `dModel` custa perto de 4x, porque o termo dominante é `dModel²`. Um modelo de qualidade perceptível levaria dias nesta implementação — que é exatamente por que o `scalagrad` foi separado do `gpt` na reestruturação de 2026-07-22, para permitir um backend de GPU sem tocar no modelo.

### Estado final

O projeto está completo e **demonstrável**: `sbt "gpt/runMain gpt.cli.train"` treina, `sbt "gpt/runMain gpt.cli.chat"` conversa.

---

## 2026-09-25 — Ferramental alinhado ao `project-templates`: Scala 3.9.0, sbt 2.0.9 e o formatador saindo do editor

Não muda nada do modelo nem do `scalagrad` — é o ferramental em volta deles. Os projetos Scala pessoais passaram a partir de uma configuração única, no repositório `project-templates` (camada `scala-sbt`), e este foi o que mais estava fora dela.

- **Versões:** Scala 3.8.4 → 3.9.0, sbt 2.0.2 → 2.0.9, scalafmt 3.9.4 → 3.11.5. A suíte passou inteira antes e depois de cada troca: 227 testes no `scalagrad`, 310 no `gpt`.
- **O formatador agora é checado pelo build, não só pelo editor.** Faltava o plugin `sbt-scalafmt` em `project/plugins.sbt`, então a formatação só acontecia no *format on save* do Metals — nada impedia um arquivo de ficar fora do padrão se fosse editado em outro lugar. Com o plugin, `sbt scalafmtCheckAll` passa a valer como verificação.
- **Sintaxe sem chaves (Scala 3 braceless), decidida conscientemente.** O `.scalafmt.conf` antigo deixava a remoção de chaves de fora de propósito, por causa de chaves com intenção no código. Adotar o padrão do template reformatou 62 arquivos (+542 −692), e o diff é só remoção de chaves de métodos, classes e blocos de controle. As chaves que motivaram a exceção — o `{ for ... yield ... }.toArray` em `Masks`, `Tensor` e `IndexOps` — ficaram intactas: o scalafmt não remove chaves de um bloco usado como expressão antes de uma chamada de método.
- **Editor e git:** entram `.vscode/settings.json` e `extensions.json` (Metals, Error Lens, Docs View, Better Comments), `.editorconfig` e `.gitattributes` (tudo LF, scripts `.ps1` em CRLF). O `.gitignore` genérico de Java/Maven/Spring foi trocado pelo do template, mantendo a única linha que era deste projeto: o checkpoint `gpt/data/model.bin`.
- **O que continua igual de propósito:** o ScalaTest, que é o framework deste projeto (o template usa munit por padrão, com o ScalaTest documentado como alternativa), e as `javaOptions` de encoding do `run` forkado.
