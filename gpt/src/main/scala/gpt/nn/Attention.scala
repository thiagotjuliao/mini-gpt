package gpt.nn

import scalagrad.core.Tensor
import scalagrad.ops.tensor.*

final class Attention(dModel: Int, dHead: Int) {
  private val query = Linear(dModel, dHead)
  private val key = Linear(dModel, dHead, useBias = false)
  private val value = Linear(dModel, dHead)
  private val scale = Tensor.fill(Array(1), Math.sqrt(dHead))
  val parameters: List[Tensor] = query.parameters ++ key.parameters ++ value.parameters

  def forward(x: Tensor): Tensor = {
    require(
      x.rank == 3,
      s"The input tensor for this layer must have 3 dimensions (B, T, dModel), but got rank ${x.rank}."
    )

    require(
      x.shape.last == dModel,
      s"Last dimension for the input tensor must be equal to dModel = $dModel, but got ${x.shape.last}."
    )

    val seqLen = x.shape(1)
    val q = query.forward(x)
    val k = key.forward(x)
    val v = value.forward(x)

    // A escala vem antes da mascara: o backward de `/` le o valor do operando,
    // e um -inf ali viraria NaN. O de `+` so repassa o gradiente, entao a
    // mascara e segura depois da divisao.
    val scores = q.matmul(k.transpose()) / scale
    val mask = Masks.causalMask(seqLen)
    val weights = (scores + mask).softmax(scores.rank - 1)
    weights.matmul(v)
  }
}
