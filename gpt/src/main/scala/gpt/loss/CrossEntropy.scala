package gpt.loss

import scalagrad.core.Tensor
import scalagrad.ops.tensor.*

object CrossEntropy {
  def apply(logits: Tensor, targets: Tensor): Tensor = {
    require(
      logits.rank == 3,
      s"logits must have 3 dimensions (batchSize, seqLen, vocabSize), but got rank ${logits.rank}."
    )

    require(
      targets.rank == 2,
      s"targets must have 2 dimensions (batchSize, seqLen), but got rank ${targets.rank}."
    )

    require(
      logits.shape(0) == targets.shape(0) && logits.shape(1) == targets.shape(1),
      s"logits and targets must agree on batch and sequence, but got " +
        s"${logits.shape.mkString("x")} and ${targets.shape.mkString("x")}."
    )

    val batchSize = targets.shape(0)
    val seqLen = targets.shape(1)
    val vocabSize = logits.shape(2)

    val flatIdx = Array.tabulate(batchSize * seqLen) { n =>
      val value = targets.get(n / seqLen, n % seqLen)

      // O `Tensor` so guarda `Double`, entao um alvo 2.9999 seria truncado em
      // silencio e ensinaria a associacao errada
      // (ver theory/16-cross-entropy/16-cross-entropy.md secao 5).
      require(
        value == Math.floor(value),
        s"Token indices must be whole numbers, but position $n holds $value."
      )

      require(
        value >= 0 && value < vocabSize,
        s"Target $value at position $n is out of bounds for a vocabulary of $vocabSize tokens."
      )

      value.toInt
    }

    val onehotT = Tensor.oneHot(flatIdx, logits.shape(2)).reshape(logits.shape)

    val N = Tensor.make(Array(targets.size), Array(1))
    (logits.logSoftmax(2) * onehotT).sum.neg / N
  }

  def perplexity(loss: Tensor): Double = Math.exp(loss.get(0))
}
