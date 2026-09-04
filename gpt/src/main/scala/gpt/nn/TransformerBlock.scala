package gpt.nn

import scalagrad.core.Tensor
import scalagrad.ops.tensor.*
import scala.util.Random

final class TransformerBlock(
    dModel: Int,
    nHeads: Int,
    expansion: Int = 4,
    rng: Random = new Random(),
    residualScale: Double = 1.0
) {
  val ln1: LayerNorm = LayerNorm(dModel)
  val attention: MultiHeadAttention = MultiHeadAttention(dModel, nHeads, rng, residualScale)
  val ln2: LayerNorm = LayerNorm(dModel)
  val mlp: MLP = MLP(dModel, expansion, rng, residualScale)

  val parameters: List[Tensor] =
    ln1.parameters ++
      attention.parameters ++
      ln2.parameters ++
      mlp.parameters

  private def validate(x: Tensor): Unit = {
    require(
      x.rank == 3,
      s"The input tensor for this layer must have 3 dimensions, but got rank ${x.rank}."
    )

    require(
      x.shape.last == dModel,
      s"Last dimension for the input tensor must be equal to dModel = $dModel, but got ${x.shape.last}."
    )
  }

  def forward(x: Tensor): Tensor = {
    validate(x)

    val h = x + attention.forward(ln1.forward(x))
    h + mlp.forward(ln2.forward(h))
  }
}
