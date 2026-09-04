package gpt.nn

import scalagrad.core.Tensor
import scalagrad.ops.tensor.*

final class LayerNorm(dim: Int, eps: Double = 1e-5) {
  private val gamma = Tensor.ones(Array(dim), requiresGradient = true)
  private val beta = Tensor.zeros(Array(dim), requiresGradient = true)

  // Constante do bloco, não do forward: recriar a cada chamada alocava um
  // tensor por passada. Mesmo padrão do `scale` da `Attention`.
  private val epsT = Tensor.fill(Array(1), eps)
  val parameters: List[Tensor] = List(gamma, beta)

  def forward(x: Tensor): Tensor = {
    require(
      x.rank >= 1,
      s"The input tensor for this layer must have at least 1 dimension, but got rank ${x.rank}."
    )

    require(
      x.shape.last == dim,
      s"Last dimension for the input tensor must be equal to $dim, but got ${x.shape.last}."
    )

    val axis = x.rank - 1
    val xc = x - x.mean(axis, keepDim = true)
    val variance = xc.pow(2).mean(axis, keepDim = true)
    val xhat = xc / (variance + epsT).pow(0.5)
    xhat * gamma + beta
  }
}
