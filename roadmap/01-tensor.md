## Etapa 1 — Tensor: A Estrutura de Dados Fundamental

### Por que existe

Toda a matemática de redes neurais opera sobre arrays multidimensionais de números reais. Um tensor é simplesmente isso: um array N-dimensional com metadados que descrevem sua forma. Sem uma abstração própria para tensor, seríamos forçados a trabalhar com `Array[Array[Array[Double]]]` — o que torna impossível escrever operações genéricas, debugar shapes errados, ou construir o grafo de computação que precisaremos depois.

### O que implementar

**Representação interna em memória (layout row-major)**

O tensor internamente vai ser um único `Array[Double]` contíguo em memória, acompanhado de dois arrays auxiliares: `shape` (as dimensões) e `strides` (o quanto avançar no array linear para andar uma posição em cada dimensão).

Por que flat? Porque é assim que a memória funciona. Uma matriz 3×4 é armazenada como 12 doubles sequenciais. Para acessar o elemento `[i][j]`, o índice linear é `i * stride[0] + j * stride[1]`. Entender strides é fundamental para entender reshape, transpose, e broadcasting sem copiar dados.

Os strides para um tensor row-major de shape `[d0, d1, d2]` são `[d1*d2, d2, 1]`. O stride da última dimensão é sempre 1.

**Indexação multi-dimensional**

Implementar uma função que converte um índice multi-dimensional `(i, j, k, ...)` em um índice linear usando os strides. Essa função será usada em toda operação que precisa acessar elementos individuais.

**Rank e size**

`rank` é o número de dimensões. `size` é o produto de todas as dimensões (número total de elementos). São propriedades derivadas de `shape`.

**Operações de shape**

- `reshape(newShape)`: retorna uma nova view do mesmo array de dados com shape diferente, recomputando os strides. Só é válido se o `size` total não mudar. Não copia dados.
- `transpose()`: para matrizes 2D, inverte shape e strides. Para tensores ND, troca duas dimensões específicas. Também não copia dados — apenas reinterpreta os strides.
- `contiguous()`: às vezes, após transpose ou slice, o tensor não está mais em layout contíguo. Essa função força a cópia dos dados em ordem canônica. Necessário antes de certas operações.

**Inicialização**

Métodos estáticos para criar tensores comuns:
- `zeros(shape)`: todos os elementos em 0.0
- `ones(shape)`: todos em 1.0
- `fill(shape, value)`: todos com o mesmo valor.
- `randn(shape)`: valores amostrados de uma distribuição normal padrão N(0,1). Necessário para inicialização de pesos. Para isso, implementar Box-Muller transform ou Ziggurat — dois números uniformes `u1`, `u2` e a fórmula `sqrt(-2 * ln(u1)) * cos(2π * u2)` geram um par de amostras normais.
- `arange(n)`: sequência 0, 1, 2, ..., n-1.

**Display**

Um método `toString` decente que imprime o tensor com indentação proporcional ao rank, mostrando shape e alguns valores. Vai ser usado exaustivamente durante debug.

### Por que essa etapa importa

Tudo o que vem depois — gradientes, atenção, normalização — são operações sobre tensores. Um Tensor bem implementado aqui evita uma classe inteira de bugs silenciosos onde você passa uma matriz 4×3 onde deveria passar 3×4 e o código continua rodando com números errados.
