package gpt.nn

import scalagrad.core.Tensor
import scalagrad.ops.tensor.*
import scala.util.Random

final class Linear(
    inputDim: Int,
    outputDim: Int,
    useBias: Boolean = true,
    initScale: Double = 1.0,
    rng: Random = Random()
) {
  // Dimensao nao positiva produz tensores vazios sem estourar nada: o forward
  // roda, devolve zero elemento, e o erro so aparece muito depois, longe da
  // causa. A checagem mora aqui, uma vez, e nao em cada camada que usa
  // `Linear` -- ver theory/13-mlp/13-mlp.md secao 8.
  require(
    inputDim >= 1,
    s"Input dimension must be at least 1, but got $inputDim."
  )

  require(
    outputDim >= 1,
    s"Output dimension must be at least 1, but got $outputDim."
  )

  private val W = initializeWeights()
  private val b = Option.when(useBias)(initializeBias())

  val weights: Tensor = W
  val bias: Option[Tensor] = b
  val parameters: List[Tensor] = W :: b.toList

  private def initializeBias(): Tensor = {
    val data = Array.fill(outputDim)(0.0)
    val shape = Array(outputDim)

    Tensor.make(data, shape, requiresGradient = true)
  }

  private def initializeWeights(): Tensor = {
    // `initScale` e a escala residual da Etapa 15 §6: as duas projecoes que
    // escrevem no fluxo residual nascem menores por 1/sqrt(2*nLayers).
    val std = Math.sqrt(2.0 / inputDim) * initScale
    val shape = Array(inputDim, outputDim)

    Tensor.randn(shape, std, requiresGradient = true, rng = rng)
  }

  def forward(x: Tensor): Tensor = {
    val z = x.matmul(W)
    b.fold(z)(z + _)
  }
}
