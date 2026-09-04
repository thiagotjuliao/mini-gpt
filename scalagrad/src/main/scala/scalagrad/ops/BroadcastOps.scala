package scalagrad.ops

import scalagrad.core.{Shape, Gradient}

object Broadcast {

  /** Colapsa `grad` (no shape "esticado" `gradShape`, resultado de uma
    * operação broadcastada) de volta pro shape original menor `targetShape`,
    * somando as contribuições de toda posição que foi broadcastada.
    * Os dois casos (dimensão nova por padding vs. dimensão que já existia com
    * tamanho 1) com exemplo numérico: ver
    * theory/03-elementary-operations/03-elementary-operations.md §4.
    */
  def unbroadcast(grad: Gradient, gradShape: Shape, targetShape: Shape): Gradient = {
    val result = Gradient.zeros(targetShape)

    val padLen = Math.max(0, gradShape.rank - targetShape.rank)
    val paddedTarget = targetShape.padTo(gradShape.rank, 1)

    (0 until gradShape.size).foreach { i =>
      val gradIdx = gradShape.unravelIndex(i)

      val targetCoords = Array.tabulate(targetShape.rank) { d =>
        val paddedD = d + padLen
        if paddedTarget(paddedD) == 1 then 0 else gradIdx(paddedD)
      }

      result.accumulate(targetShape.index(targetCoords*), grad(i))
    }

    result
  }
}
