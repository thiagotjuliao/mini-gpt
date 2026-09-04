## Etapa 2 — Motor de Autodiferenciação (Autograd)

### Por que existe

Treinar uma rede neural requer calcular a derivada da função de perda em relação a cada parâmetro do modelo. Num transformer com milhões de parâmetros, fazer isso à mão é impossível. O autograd automatiza isso através de uma técnica chamada **diferenciação automática em modo reverso** — mais conhecida como backpropagation.

A ideia central: cada operação que aplicamos a tensores registra, no momento em que acontece, como calcular o gradiente em relação às suas entradas, dado o gradiente em relação à sua saída. Depois, percorremos esse registro de trás para frente, e os gradientes se propagam automaticamente pela cadeia de operações.

### O que implementar

**Extensão do Tensor com campos de autograd**

O Tensor precisa de três campos adicionais:
- `grad: Array[Double]`: acumula o gradiente. Mesma shape que `data`. Inicialmente zeros.
- `requiresGrad: Boolean`: marca se este tensor é um parâmetro que precisa de gradiente. Tensores de entrada de dados não precisam, pesos sim.
- `_backward: () => Unit`: uma closure que, quando chamada, propaga o gradiente deste tensor para seus inputs. É o coração do sistema.
- `_prev: Set[Tensor]`: os tensores que foram inputs da operação que gerou este tensor. Forma as arestas do grafo.

**Grafo de computação (DAG)**

Cada vez que aplicamos uma operação (ex: `A matmul B = C`), além de computar `C.data`, registramos em `C._prev = Set(A, B)` e definimos `C._backward` como a closure que, dado `C.grad`, vai acumular o gradiente correto em `A.grad` e `B.grad`.

O grafo é uma DAG (Directed Acyclic Graph): as arestas apontam de outputs para inputs, e não existem ciclos (uma operação não pode depender de si mesma).

**Ordenação topológica**

Para fazer o backward pass corretamente, precisamos processar os nós do grafo numa ordem que garanta: quando calcularmos o gradiente de um nó, seu próprio gradiente (`self.grad`) já está completamente acumulado. Isso requer ordenação topológica do grafo — um DFS clássico que visita cada nó recursivamente e o adiciona a uma lista *depois* de visitar todos os seus filhos.

Implementar `topologicalSort(root: Tensor): List[Tensor]` usando o algoritmo padrão com um `Set` de visitados para evitar duplicatas.

**O método `backward()`**

Chamado na raiz do grafo (normalmente a perda). Inicializa `self.grad` com `1.0` (derivada de si mesmo em relação a si mesmo). Depois percorre a lista topológica em ordem reversa, chamando `._backward()` em cada nó.

**Por que `grad` é acumulado com `+=` e não `=`**

Um mesmo tensor pode ser usado como input de múltiplas operações diferentes. Por exemplo, um peso W pode ser usado em dois matmuls. O gradiente de W é a soma das contribuições de todos esses caminhos — o que é exatamente o que `+=` garante. Se usarmos `=`, o segundo caminho sobrescreve o primeiro e perdemos a metade do gradiente. É um bug silencioso devastador.

**Zeragem de gradientes**

Antes de cada passo de treinamento, todos os gradientes devem ser zerados. Implementar `zeroGrad()` em todos os tensores que têm `requiresGrad = true`. Se não zeramos, os gradientes do passo anterior se acumulam no próximo, corrompendo completamente o treinamento.

**Modo de inferência (`noGrad`)**

Durante inferência, não precisamos construir o grafo nem acumular gradientes — isso economiza memória e tempo. Implementar um mecanismo para desativar o registro do grafo globalmente (uma variável de estado global `gradEnabled: Boolean` é suficiente).

### Por que essa etapa importa

Autograd é o motor. Sem ele, toda a matemática do transformer existe mas não pode ser treinada. Com ele implementado corretamente, cada camada que construirmos nas próximas etapas ganhará treinabilidade automaticamente, apenas definindo como computar o backward de cada operação.
