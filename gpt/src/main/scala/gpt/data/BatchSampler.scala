package gpt.data

import scala.util.Random
import scalagrad.core.Tensor

final class BatchSampler(
    corpus: Array[Int],
    contextLength: Int,
    randomizer: Random = new Random()
) {
  require(
    contextLength > 0 && contextLength < corpus.length,
    s"Context length must be positive and less than the corpus size ${corpus.length}, " +
      s"but got $contextLength."
  )

  def sample(batchSize: Int): (Tensor, Tensor) = {
    val shape = Array(batchSize, contextLength)

    val starts = (0 until batchSize)
      .map(_ => randomizer.nextInt(corpus.length - contextLength))

    val inputs = starts
      .flatMap(i => corpus.slice(i, i + contextLength))
      .map(_.toDouble)
      .toArray

    val targets = starts
      .flatMap(i => corpus.slice(i + 1, i + contextLength + 1))
      .map(_.toDouble)
      .toArray

    Tensor.make(inputs, shape) -> Tensor.make(targets, shape)
  }
}

object BatchSampler {

  /** Divide o corpus em treino e validação, cortando **em ordem** em vez de
    * sortear: com um corte aleatório, trechos vizinhos cairiam dos dois lados, e
    * a validação mediria texto que o treino praticamente já viu
    * (ver theory/18-training-loop/18-training-loop.md §5).
    */
  def split(
      corpus: Array[Int],
      contextLength: Int,
      validationFraction: Double = 0.1,
      randomizer: Random = new Random()
  ): (BatchSampler, BatchSampler) = {
    require(
      validationFraction > 0 && validationFraction < 1,
      s"validationFraction must be in (0, 1), but got $validationFraction."
    )

    val cut = (corpus.length * (1 - validationFraction)).toInt

    require(
      cut > contextLength && corpus.length - cut > contextLength,
      s"A corpus of ${corpus.length} tokens split at $cut leaves a side shorter than the " +
        s"context length $contextLength."
    )

    val train = new BatchSampler(corpus.take(cut), contextLength, randomizer)
    val validation = new BatchSampler(corpus.drop(cut), contextLength, randomizer)

    train -> validation
  }
}
