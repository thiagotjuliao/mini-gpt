package scalagrad.ops

import scalagrad.core.*

private[ops] trait UnaryOps {

  /** Fábrica compartilhada por `neg`/`pow`/`exp`/`log`/`clamp`: `op` é o forward
    * elemento a elemento; `localGrad(x, fx)` recebe o valor de entrada e o
    * de saída daquela posição (algumas derivadas precisam de um, outras do
    * outro — ex.: `exp` reusa `fx`, `log` usa `x`).
    */
  private def unary(
      t1: Tensor
  )(op: Double => Double)(localGrad: (Double, Double) => Double): Tensor = {
    val shape = t1.shape

    // Posição em `data` de cada índice canônico. As duas indexações do projeto
    // não coincidem em tensor não contíguo: `data` é lido pelas strides reais,
    // enquanto `gradient` é sempre canônico. Misturar as duas fazia o gradiente
    // parear a derivada local com o valor de outra posição -- forward certo,
    // gradiente errado. O mapa é calculado uma vez e reusado no backward.
    val physical = Array.tabulate(shape.size)(i => t1.index(shape.unravelIndex(i)*))
    val data = physical.map(i => op(t1.data(i)))
    val grad = Gradient.zeros(shape)
    val reqGrad = t1.requiresGradient && Tensor.gradEnabled
    val prev = if Tensor.gradEnabled then Set(t1) else Set()

    Tensor(data, shape, shape.canonicalStrides, grad, reqGrad, prev) { () =>
      if Tensor.gradEnabled then
        (0 until shape.size).foreach { i =>
          t1.gradient.accumulate(i, grad(i) * localGrad(t1.data(physical(i)), data(i)))
        }
    }
  }

  extension (t1: Tensor) {
    def neg: Tensor = unary(t1)(x => -x)((_, _) => -1.0)
    def pow(n: Double): Tensor = unary(t1)(Math.pow(_, n))((x, _) => n * Math.pow(x, n - 1))
    def exp: Tensor = unary(t1)(Math.exp)((_, fx) => fx)
    def log: Tensor = unary(t1)(Math.log)((x, _) => 1 / x)

    def clamp(min: Double, max: Double): Tensor =
      unary(t1)(x => if x < min then min else if x > max then max else x) { (x, _) =>
        if min < x && x < max then 1.0 else 0.0
      }

    def relu: Tensor = unary(t1)(x => Math.max(0, x)) { (x, _) =>
      if x > 0 then 1.0 else 0.0
    }

    def sigmoid: Tensor = {
      val sigma = (x: Double) => 1 / (1 + Math.exp(-x))
      unary(t1)(sigma)((_, fx) => fx * (1 - fx))
    }

    def tanh: Tensor = unary(t1)(Math.tanh)((_, fx) => 1 - Math.pow(fx, 2))

    def gelu: Tensor = {
      val c = Math.sqrt(2 / Math.PI)
      val u = (x: Double) => c * (x + 0.044715 * Math.pow(x, 3))
      val gelux = (x: Double) => 0.5 * x * (1 + Math.tanh(u(x)))

      unary(t1)(gelux) { (x, _) =>
        val t = Math.tanh(u(x))
        val uPrime = c * (1 + 0.134145 * Math.pow(x, 2))

        0.5 * (1 + t) + 0.5 * x * (1 - Math.pow(t, 2)) * uPrime
      }
    }
  }
}
