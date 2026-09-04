package scalagrad.gradcheck

import scalagrad.core.Tensor

object Gradcheck {
  def run(input: Tensor, eps: Double = 1e-5)(f: Tensor => Tensor): Double = {
    // Posição em `data` de cada índice canônico: `gradient` é canônico, e a
    // perturbação precisa mexer na MESMA posição lógica que o gradiente
    // analítico reporta. Num tensor não contíguo os dois índices divergem.
    val physical = Array.tabulate(input.size)(i => input.index(input.shape.unravelIndex(i)*))

    // O gradiente do próprio `input` é zerado antes: `Gradient` só acumula, e
    // sem isto um segundo `run` sobre o mesmo tensor leria a soma do anterior.
    input.gradient.zero()

    val output = f(input)
    require(
      output.size == 1,
      s"The loss must be a scalar, but got shape ${output.shape.mkString("x")}. " +
        "Multiply the output by distinct weights and sum it -- see theory/04-gradient-check §3."
    )
    output.backward()
    val gradA = input.gradient.toArray

    val gradN = Array.tabulate(input.size) { i =>
      val pos = physical(i)
      val orig = input.data(pos)

      input.data(pos) = orig + eps
      val fPlus = Tensor.noGrad { f(input) }.data(0)

      input.data(pos) = orig - eps
      val fMinus = Tensor.noGrad { f(input) }.data(0)

      input.data(pos) = orig

      (fPlus - fMinus) / (2 * eps)
    }

    // Erro Máximo entre os gradientes Analítico e Numérico
    gradA
      .zip(gradN)
      .map { (a, n) =>
        Math.abs(a - n) / Math.max(Math.max(Math.abs(a), Math.abs(n)), 1e-8)
      }
      .max
  }
}
