package scalagrad.ops

import scalagrad.core.*

private[ops] trait MatmulOps:
  extension (t1: Tensor)

    /** Produto matricial com um número qualquer de dimensões de lote.
      *
      * As duas últimas dimensões são a matriz; todas as anteriores são lote.
      * O segundo operando pode ter o mesmo lote do primeiro, ou ser rank 2 —
      * neste caso a mesma matriz vale pra todo o lote, e o gradiente dela soma
      * sobre ele (ver theory/03-elementary-operations §6, operando
      * compartilhado). Lotes de tamanhos diferentes são recusados, não
      * broadcastados.
      *
      * Não há caminho especial por rank: o laço percorre as posições da saída
      * e lê os operandos pelas strides, então entrada não contígua funciona
      * sem cópia (ver theory/12-multi-head-attention §4).
      */
    def matmul(t2: Tensor): Tensor =
      require(
        t1.rank >= 2 && t2.rank >= 2,
        s"matmul needs at least a matrix on each side, " +
          s"but got ranks ${t1.rank} and ${t2.rank}."
      )

      val M = t1.shape(t1.rank - 2)
      val K = t1.shape(t1.rank - 1)
      val N = t2.shape(t2.rank - 1)

      require(
        t2.shape(t2.rank - 2) == K,
        s"Inner dimensions must match: the first tensor ends in $K, " +
          s"but the second starts its matrix in ${t2.shape(t2.rank - 2)}."
      )

      val batch1 = t1.shape.toArray.dropRight(2)
      val batch2 = t2.shape.toArray.dropRight(2)
      val sharedRhs = batch2.isEmpty

      require(
        sharedRhs || batch1.sameElements(batch2),
        s"Batch dimensions must match, or the second operand must be rank 2 to be shared. " +
          s"Got ${t1.shape.mkString("x")} and ${t2.shape.mkString("x")}."
      )

      val shape = Shape(batch1 :+ M :+ N)
      val grad = Gradient.zeros(shape)
      val reqGrad = (t1.requiresGradient || t2.requiresGradient) && Tensor.gradEnabled
      val prev = if Tensor.gradEnabled then Set(t1, t2) else Set()

      // Dentro da soma em `k` só uma dimensão de cada operando muda, então o
      // deslocamento em `data` avança por um passo constante. Calcular a base
      // uma vez por posição da saída troca o índice multidimensional por
      // aritmética inteira. As strides canônicas de quem varia são conhecidas:
      // 1 para a última dimensão do operando esquerdo, N para a penúltima do
      // direito.
      val leftStep = t1.strides(t1.rank - 1)
      val rightStep = t2.strides(t2.rank - 2)

      def basesFor(out: Int): (Int, Int, Int, Int) =
        val lhs = shape.unravelIndex(out)
        val rhs = if sharedRhs then Array(0, lhs(lhs.length - 1)) else lhs
        val left = lhs.updated(lhs.length - 1, 0)
        val right = rhs.updated(rhs.length - 2, 0)

        (t1.index(left*), t1.shape.index(left*), t2.index(right*), t2.shape.index(right*))

      val data = Array.tabulate(shape.size) { out =>
        val (leftBase, _, rightBase, _) = basesFor(out)

        (0 until K).foldLeft(0.0) { (acc, k) =>
          acc + t1.data(leftBase + k * leftStep) * t2.data(rightBase + k * rightStep)
        }
      }

      Tensor(data, shape, shape.canonicalStrides, grad, reqGrad, prev) { () =>
        if Tensor.gradEnabled then
          (0 until shape.size).foreach { out =>
            val g = grad(out)
            val (leftBase, leftCanon, rightBase, rightCanon) = basesFor(out)

            (0 until K).foreach { k =>
              t1.gradient.accumulate(leftCanon + k, g * t2.data(rightBase + k * rightStep))
              t2.gradient.accumulate(rightCanon + k * N, g * t1.data(leftBase + k * leftStep))
            }
          }
      }
  end extension
end MatmulOps
