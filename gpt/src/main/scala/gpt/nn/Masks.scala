package gpt.nn

import scalagrad.core.Tensor

object Masks:
  def causalMask(seqLen: Int): Tensor =
    val shape = Array(seqLen, seqLen)
    val data = {
      for
        i <- 0 until seqLen
        j <- 0 until seqLen
      yield if j <= i then 0.0 else Double.NegativeInfinity
    }.toArray

    Tensor.make(data, shape)
