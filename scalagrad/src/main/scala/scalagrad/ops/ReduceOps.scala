package scalagrad.ops

import scalagrad.core.*

private[ops] trait ReduceOps:

  /** Fábrica compartilhada por `sum`/`mean`/`max`: reduz `t1` a um tensor
    * escalar `value`; `localGrad(i)` devolve quanto a posição `i` de `t1`
    * contribuiu, multiplicado pelo gradiente escalar recebido.
    */
  private def reduce(t1: Tensor)(value: Double)(localGrad: Int => Double): Tensor =
    val data = Array(value)
    val grad = Gradient.zeros(1)
    val reqGrad = t1.requiresGradient && Tensor.gradEnabled
    val prev = if Tensor.gradEnabled then Set(t1) else Set()

    Tensor(data, Shape(1), Strides(1), grad, reqGrad, prev) { () =>
      if Tensor.gradEnabled then
        (0 until t1.size).foreach { i =>
          t1.gradient.accumulate(i, localGrad(i) * grad(0))
        }
    }

  /** Fábrica compartilhada por `sum(dim)`/`mean(dim)`: reduz `t1` ao longo de
    * um único eixo `dim` (mantendo-o como tamanho 1 se `keepDim`, removendo-o
    * caso contrário). `scale` é o único ponto de variação entre as duas: soma
    * bruta (`1.0`) para `sum`, ou dividida por `t1.shape(dim)` para `mean` —
    * como `mean = sum/n`, o mesmo fator escala tanto o forward (soma × scale)
    * quanto o backward (gradiente local × scale), pela regra da cadeia.
    * `t1.shape.groupIndex(_, dim)` mapeia um multi-índice de `t1` pro índice
    * linear da saída (dimensão `dim` colapsada) e é reusado por forward e
    * backward, já que os dois precisam do mesmo mapeamento posição-a-posição.
    * Exemplo numérico (valores distintos por fatia, forward e backward):
    * theory/03-elementary-operations/03-elementary-operations.md §3.
    */
  private def reduceDim(t1: Tensor, dim: Int, keepDim: Boolean)(scale: Double): Tensor =
    val shape = if keepDim then t1.shape.updated(dim, 1) else t1.shape.crop(dim)

    val raw = Array.fill(shape.size)(0.0)
    (0 until t1.size).foreach { i =>
      val inMultiIdx = t1.unravelIndex(i)
      raw(t1.shape.groupIndex(inMultiIdx, dim)) += t1.get(inMultiIdx*)
    }
    val data = raw.map(_ * scale)

    val grad = Gradient.zeros(shape)
    val reqGrad = t1.requiresGradient && Tensor.gradEnabled
    val prev = if Tensor.gradEnabled then Set(t1) else Set()

    Tensor(data, shape, shape.canonicalStrides, grad, reqGrad, prev) { () =>
      if Tensor.gradEnabled then
        (0 until t1.size).foreach { i =>
          t1.gradient.accumulate(i, scale * grad(t1.shape.groupIndex(t1.unravelIndex(i), dim)))
        }
    }

  /** Valores de `t` em ordem canônica. `reduce` acumula gradiente por índice
    * canônico, então ler `t.data` direto -- que é indexado pelas strides reais
    * -- parearia posições diferentes num tensor não contíguo (ver `CLAUDE.md`,
    * o invariante das duas indexações). `contiguous` devolve o próprio tensor
    * quando as strides já são canônicas, então o caso comum não copia nada.
    */
  private def canonicalValues(t: Tensor): Array[Double] = t.contiguous.data

  extension (t1: Tensor)
    def sum: Tensor = reduce(t1)(canonicalValues(t1).sum)(_ => 1.0)

    def mean: Tensor =
      val n = t1.size
      reduce(t1)(canonicalValues(t1).sum / n)(_ => 1.0 / n)

    def max: Tensor =
      val values = canonicalValues(t1)
      val (maxValue, maxIdx) = values.zipWithIndex.foldLeft((Double.NegativeInfinity, -1)) {
        case ((bestV, bestI), (v, i)) => if v > bestV then (v, i) else (bestV, bestI)
      }
      reduce(t1)(maxValue)(i => if i == maxIdx then 1.0 else 0.0)

    def sum(dim: Int, keepDim: Boolean = false): Tensor = reduceDim(t1, dim, keepDim)(1.0)

    def mean(dim: Int, keepDim: Boolean = false): Tensor =
      reduceDim(t1, dim, keepDim)(1.0 / t1.shape(dim))
end ReduceOps
