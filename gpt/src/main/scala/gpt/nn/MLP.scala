package gpt.nn

import scalagrad.core.Tensor
import scalagrad.ops.tensor.*
import scala.util.Random

final class MLP(
    dModel: Int,
    expansion: Int = 4,
    rng: Random = new Random(),
    residualScale: Double = 1.0
) {
  require(
    expansion >= 1,
    s"The expansion factor must be at least 1, but got $expansion."
  )

  val dFF = dModel * expansion
  private val up = Linear(dModel, dFF, rng = rng)
  private val down = Linear(dFF, dModel, initScale = residualScale, rng = rng)
  val parameters: List[Tensor] = up.parameters ++ down.parameters

  private def validate(x: Tensor): Unit = {
    require(
      x.rank >= 2,
      s"The input tensor for this layer must have at least 2 dimensions, but got rank ${x.rank}."
    )

    require(
      x.shape.last == dModel,
      s"Last dimension for the input tensor must be equal to dModel = $dModel, but got ${x.shape.last}."
    )
  }

  def forward(x: Tensor): Tensor = {
    validate(x)
    down.forward(up.forward(x).gelu)
  }
}
