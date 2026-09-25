package scalagrad.ops

import scalagrad.core.*

private[ops] trait BinaryOps:

  /** Fábrica compartilhada por `+`/`-`/`*`/`/`: calcula o shape de saída
    * broadcastado, o forward lendo os dois operandos através das views
    * broadcastadas (`t1B`/`t2B`), e o backward aplicando `localGrad1`/
    * `localGrad2` — cada uma recebendo `(a, b, upstream)`, os valores dos
    * dois operandos e o gradiente vindo de cima, na posição correspondente
    * do shape de saída — e só então desfazendo o broadcast via
    * `Broadcast.unbroadcast` antes de acumular em cada operando.
    * Derivadas locais de cada operação: theory/03-elementary-operations/03-elementary-operations.md §1.
    * Mecanismo de broadcasting no forward: idem, §4.
    */
  private def binary(t1: Tensor, t2: Tensor)(op: (Double, Double) => Double)(
      localGrad1: (Double, Double, Double) => Double,
      localGrad2: (Double, Double, Double) => Double
  ): Tensor =
    val shape = Shape.broadcast(t1.shape, t2.shape)
    val t1B = t1.broadcastTo(shape)
    val t2B = t2.broadcastTo(shape)

    val data = Array.tabulate(shape.size) { i =>
      val multiIdx = shape.unravelIndex(i)
      op(t1B.get(multiIdx*), t2B.get(multiIdx*))
    }

    val grad = Gradient.zeros(shape)
    val reqGrad = (t1.requiresGradient || t2.requiresGradient) && Tensor.gradEnabled
    val prev = if Tensor.gradEnabled then Set(t1, t2) else Set()

    Tensor(data, shape, shape.canonicalStrides, grad, reqGrad, prev) { () =>
      if Tensor.gradEnabled then
        val g1B = Gradient.zeros(shape)
        val g2B = Gradient.zeros(shape)

        (0 until shape.size).foreach { i =>
          val multiIdx = shape.unravelIndex(i)
          val a = t1B.get(multiIdx*); val b = t2B.get(multiIdx*)

          g1B.accumulate(i, localGrad1(a, b, grad(i)))
          g2B.accumulate(i, localGrad2(a, b, grad(i)))
        }

        t1.gradient.accumulateAll(Broadcast.unbroadcast(g1B, shape, t1.shape))
        t2.gradient.accumulateAll(Broadcast.unbroadcast(g2B, shape, t2.shape))
    }
  end binary

  extension (t1: Tensor)
    def +(t2: Tensor): Tensor = binary(t1, t2)(_ + _)((_, _, g) => g, (_, _, g) => g)
    def -(t2: Tensor): Tensor = binary(t1, t2)(_ - _)((_, _, g) => g, (_, _, g) => -g)
    def *(t2: Tensor): Tensor = binary(t1, t2)(_ * _)((_, b, g) => b * g, (a, _, g) => a * g)
    def /(t2: Tensor): Tensor =
      binary(t1, t2)(_ / _)((_, b, g) => g / b, (a, b, g) => g * (-a / (b * b)))
end BinaryOps
