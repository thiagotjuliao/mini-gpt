package gpt.train

import scalagrad.core.Tensor

object GradientClipping {

  /** Norma global: uma raiz sobre a soma dos quadrados de **todos** os
    * gradientes, e não uma norma por parâmetro. A distinção importa -- normas
    * separadas reescalariam cada parâmetro por um fator diferente, o que muda a
    * direção do passo conjunto (theory/18-training-loop §2).
    */
  def globalNorm(parameters: List[Tensor]): Double = {
    val sumOfSquares = parameters.foldLeft(0.0) { (acc, p) =>
      val gradient = p.gradient
      acc + (0 until p.size).foldLeft(0.0)((sum, i) => sum + gradient(i) * gradient(i))
    }

    Math.sqrt(sumOfSquares)
  }

  /** Reescala os gradientes se a norma global passar de `maxNorm`, e devolve a
    * norma **antes** do corte -- é ela que serve de diagnóstico no log.
    */
  def clipByGlobalNorm(parameters: List[Tensor], maxNorm: Double): Double = {
    require(maxNorm > 0, s"maxNorm must be positive, but got $maxNorm.")

    val norm = globalNorm(parameters)

    if norm > maxNorm then
      val factor = maxNorm / norm
      parameters.foreach(_.gradient.scale(factor))

    norm
  }
}
