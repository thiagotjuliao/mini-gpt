package scalagrad.ops

import scalagrad.core.*

private[ops] trait SoftmaxOps {

  /** Duas passadas do truque log-sum-exp (theory/06-softmax/06-softmax.md
    * §1/§5), compartilhadas por `softmax` e `logSoftmax`: acha o máximo de
    * cada grupo (estabilidade numérica, subtraído antes de exponenciar) e
    * depois a soma de exp(x-max) por grupo. `expData` (exp(x-max) por
    * posição) é usado como numerador por `softmax`; `logSoftmax` só precisa
    * de `maxes`/`sums` (via log-sum-exp), ignorando `expData`.
    */
  private def logSumExpStats(t: Tensor, dim: Int): (Array[Double], Array[Double], Array[Double]) = {
    val sliceSize = t.size / t.shape(dim)
    val maxes = Array.fill(sliceSize)(Double.NegativeInfinity)
    val sums = Array.fill(sliceSize)(0.0)
    val expData = Array.fill(t.size)(0.0)

    (0 until t.size).foreach { i =>
      val multiIdx = t.unravelIndex(i)
      val groupIdx = t.shape.groupIndex(multiIdx, dim)
      maxes(groupIdx) = Math.max(maxes(groupIdx), t.get(multiIdx*))
    }

    (0 until t.size).foreach { i =>
      val multiIdx = t.unravelIndex(i)
      val groupIdx = t.shape.groupIndex(multiIdx, dim)
      val e = Math.exp(t.get(multiIdx*) - maxes(groupIdx))

      expData(i) = e
      sums(groupIdx) += e
    }

    (maxes, sums, expData)
  }

  extension (t: Tensor) {
    def softmax(dim: Int): Tensor = {
      val grad = Gradient.zeros(t.shape)
      val reqGrad = t.requiresGradient && Tensor.gradEnabled
      val prev = if Tensor.gradEnabled then Set(t) else Set()

      val (_, sums, expData) = logSumExpStats(t, dim)
      val data = Array.tabulate(t.size) { i =>
        expData(i) / sums(t.shape.groupIndex(t.unravelIndex(i), dim))
      }

      Tensor(data, t.shape, t.shape.canonicalStrides, grad, reqGrad, prev) { () =>
        if Tensor.gradEnabled then
          val sliceSize = t.size / t.shape(dim)
          val groupSums = Array.fill(sliceSize)(0.0)

          (0 until t.size).foreach { i =>
            val groupIdx = t.shape.groupIndex(t.unravelIndex(i), dim)
            groupSums(groupIdx) += data(i) * grad(i)
          }

          (0 until t.size).foreach { i =>
            val groupIdx = t.shape.groupIndex(t.unravelIndex(i), dim)
            val dx = data(i) * (grad(i) - groupSums(groupIdx))

            t.gradient.accumulate(i, dx)
          }
      }
    }

    def logSoftmax(dim: Int): Tensor = {
      val grad = Gradient.zeros(t.shape)
      val reqGrad = t.requiresGradient && Tensor.gradEnabled
      val prev = if Tensor.gradEnabled then Set(t) else Set()

      val (maxes, sums, _) = logSumExpStats(t, dim)
      val data = Array.tabulate(t.size) { i =>
        val multiIdx = t.unravelIndex(i)
        val groupIdx = t.shape.groupIndex(multiIdx, dim)

        (t.get(multiIdx*) - maxes(groupIdx)) - Math.log(sums(groupIdx))
      }

      Tensor(data, t.shape, t.shape.canonicalStrides, grad, reqGrad, prev) { () =>
        if Tensor.gradEnabled then
          val sliceSize = t.size / t.shape(dim)
          val groupSums = Array.fill(sliceSize)(0.0)

          (0 until t.size).foreach { i =>
            val groupIdx = t.shape.groupIndex(t.unravelIndex(i), dim)
            groupSums(groupIdx) += grad(i)
          }

          (0 until t.size).foreach { i =>
            val groupIdx = t.shape.groupIndex(t.unravelIndex(i), dim)
            val dx = grad(i) - Math.exp(data(i)) * groupSums(groupIdx)

            t.gradient.accumulate(i, dx)
          }
      }
    }
  }
}
