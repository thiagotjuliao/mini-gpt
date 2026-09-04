package scalagrad.ops

import scalagrad.core.Tensor
import scalagrad.core.Shape
import scalagrad.core.Gradient

private[ops] trait IndexOps {
  extension (t: Tensor) {
    def indexSelect(indices: Array[Int]): Tensor = {
      require(
        t.rank == 2,
        s"indexSelect expects a rank-2 table [numRows, rowDim], but got rank ${t.rank} instead."
      )

      val numRows = t.shape(0)
      val rowDim = t.shape(1)

      (0 until indices.length).foreach { i =>
        require(
          indices(i) >= 0 && indices(i) < numRows,
          s"Index ${indices(i)} at position $i is out of bounds for a table with $numRows rows."
        )
      }

      val shape = Shape(Array(indices.length, rowDim))
      val grad = Gradient.zeros(shape)
      val reqGrad = t.requiresGradient && Tensor.gradEnabled
      val prev = if Tensor.gradEnabled then Set(t) else Set()

      val data = {
        for {
          i <- 0 until indices.length
          j <- 0 until rowDim
        } yield t.get(indices(i), j)
      }.toArray

      Tensor(data, shape, shape.canonicalStrides, grad, reqGrad, prev) { () =>
        if Tensor.gradEnabled then
          (0 until indices.length).foreach { i =>
            val row = indices(i)

            (0 until rowDim).foreach { j =>
              val inIdx = t.shape.index(row, j)
              val outIdx = shape.index(i, j)

              t.gradient.accumulate(inIdx, grad(outIdx))
            }
          }
      }
    }
  }
}
