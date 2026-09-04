# Etapa 1 — Tensor: A Estrutura de Dados Fundamental

Status: [ ] Não iniciado · [ ] Em andamento · [x] Concluído

## Representação interna
- [x] Classe `Tensor` com `data: Array[Double]`, `shape: Array[Int]`, `strides: Array[Int]`
- [x] Função que calcula `strides` a partir de `shape` (row-major)

## Indexação multi-dimensional
- [x] `index(indices: Int*): Int` convertendo `(i, j, k, ...)` em índice linear
- [x] Validação de número de índices vs `rank`

## Propriedades derivadas
- [x] `rank`
- [x] `size`

## Operações de shape (sem copiar dados)
- [x] `reshape(newShape)` com validação de `size`
- [x] `transpose()` para 2D (inverte shape e strides)
- [x] `contiguous()` (detecta layout não-canônico e copia)

## Inicialização
- [x] `zeros(shape)`
- [x] `ones(shape)`
- [x] `fill(shape, value)`
- [x] `arange(n)`
- [x] `randn(shape)` via Box-Muller

## Display
- [x] `toString` com shape e indentação por rank

## Validação manual
- [x] Conferir strides de um tensor `[2,3]`
- [x] Reshape `[2,3]` → `[3,2]` e checar valores via `index`
- [x] Transpose `[2,3]` e checar `t.index(i,j) == original.index(j,i)`
- [x] `randn` com N grande: média ≈0, variância ≈1
