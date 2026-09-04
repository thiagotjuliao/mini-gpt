package gpt.nn

import scalagrad.core.Tensor
import scalagrad.ops.tensor.*
import scala.util.Random

final class Embedding(
    vocabSize: Int,
    dModel: Int,
    contextLength: Int,
    rng: Random = new Random()
) {
  val tokenTable =
    Tensor.randn(Array(vocabSize, dModel), std = 0.02, requiresGradient = true, rng = rng)

  private val positionTable =
    Tensor.randn(Array(contextLength, dModel), std = 0.02, requiresGradient = true, rng = rng)

  val parameters: List[Tensor] = List(tokenTable, positionTable)

  def forward(tokens: Tensor): Tensor = {
    require(
      tokens.rank == 2,
      s"forward expects a rank-2 tensor of token indices [batchSize, seqLen], but got rank ${tokens.rank} instead."
    )

    val batchSize = tokens.shape(0)
    val seqLen = tokens.shape(1)

    require(
      seqLen <= contextLength,
      s"Cannot embed a sequence of length $seqLen: the position table only covers $contextLength positions."
    )

    val shape = Array(batchSize, seqLen, dModel)

    val flatIdx = Array.tabulate(batchSize * seqLen) { n =>
      val value = tokens.get(n / seqLen, n % seqLen)

      require(
        value == Math.floor(value),
        s"Token indices must be whole numbers, but position $n holds $value."
      )

      value.toInt
    }

    tokenTable.indexSelect(flatIdx).reshape(shape) +
      positionTable.indexSelect(Array.range(0, seqLen))
  }
}
