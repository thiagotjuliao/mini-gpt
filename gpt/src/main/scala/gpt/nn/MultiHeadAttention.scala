package gpt.nn

import scalagrad.core.Tensor
import scalagrad.ops.tensor.*
import scala.util.Random

final class MultiHeadAttention(
    dModel: Int,
    nHeads: Int,
    rng: Random = new Random(),
    residualScale: Double = 1.0
):
  require(
    dModel % nHeads == 0,
    s"""
            |Number of heads(nHeads) must divide the input token vector (dModel) in equal parts.
            |Actual: $dModel % $nHeads = ${dModel % nHeads}.
        """.stripMargin
  )

  val dHead: Int = dModel / nHeads
  private val query = Linear(dModel, dModel, rng = rng)
  private val key = Linear(dModel, dModel, useBias = false, rng = rng)
  private val value = Linear(dModel, dModel, useBias = false, rng = rng)
  private val outProj = Linear(dModel, dModel, initScale = residualScale, rng = rng)
  private val scale = Tensor.fill(Array(1), Math.sqrt(dHead))

  val parameters: List[Tensor] =
    query.parameters ++ key.parameters ++ value.parameters ++ outProj.parameters

  private def split(t: Tensor): Tensor =
    t.reshape(Array(t.shape(0), t.shape(1), nHeads, dHead)).transpose(1, 2)

  private def validate(x: Tensor): Unit =
    require(
      x.rank == 3,
      s"The input tensor for this layer must have 3 dimensions (B, T, dModel), but got rank ${x.rank}."
    )

    require(
      x.shape.last == dModel,
      s"Last dimension for the input tensor must be equal to dModel = $dModel, but got ${x.shape.last}."
    )

  /** Pesos de atencao por cabeca, `[B, nHeads, T, T]`. Janela de diagnostico:
    * o `forward` consome o resultado, e os testes de propriedade conferem aqui
    * o que o gradient check nao ve -- soma das linhas e causalidade
    * (ver theory/11-attention/11-attention.md secao 8).
    */
  def attentionWeights(x: Tensor): Tensor =
    validate(x)

    val qh = split(query.forward(x))
    val kh = split(key.forward(x))

    // A escala vem antes da mascara: o backward de `/` le o valor do operando,
    // e um -inf ali viraria NaN. O de `+` so repassa o gradiente, entao a
    // mascara e segura depois da divisao.
    val scores = qh.matmul(kh.transpose()) / scale

    (scores + Masks.causalMask(x.shape(1))).softmax(scores.rank - 1)

  def forward(x: Tensor): Tensor =
    validate(x)

    val weights = attentionWeights(x)
    val vh = split(value.forward(x))
    val context = weights.matmul(vh)

    val merged = context.transpose(1, 2).reshape(Array(x.shape(0), x.shape(1), dModel))
    outProj.forward(merged)
end MultiHeadAttention
