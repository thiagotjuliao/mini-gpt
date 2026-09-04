# Etapa 2 — Motor de Autodiferenciação (Autograd)

Status: [ ] Não iniciado · [ ] Em andamento · [x] Concluído

## Extensão do Tensor
- [x] Campo `grad: Array[Double]` (mesma shape que `data`, inicia zerado) — chamado `gradient`
- [x] Campo `requiresGrad: Boolean` — chamado `requiresGradient`
- [x] Campo `_backward: () => Unit`
- [x] Campo `_prev: Set[Tensor]` — chamado `previous`

## Grafo de computação (DAG)
- [x] Cada operação registra `_prev` com seus inputs (`add`/`mul` em `ops/TensorOps.scala`)
- [x] Cada operação define `_backward` corretamente

## Ordenação topológica
- [x] `topologicalSort(root: Tensor): List[Tensor]` via DFS com set de visitados

## backward()
- [x] Inicializa `grad = 1.0` na raiz
- [x] Percorre a lista topológica em ordem reversa chamando `_backward()`

## Acúmulo de gradiente
- [x] Todo acúmulo usa `+=`, nunca `=` (checar isso em toda operação futura)

## Zeragem de gradientes
- [x] `zeroGrad()` para todos os tensores com `requiresGrad = true`

## Modo de inferência
- [x] Variável global `gradEnabled: Boolean`
- [x] Mecanismo `noGrad { ... }` que desativa registro do grafo

## Validação manual
- [x] Grafo simples `a + b`: conferido grad na mão
- [x] Grafo `c * d` com reuso do mesmo tensor duas vezes: conferido que grad soma as duas contribuições
