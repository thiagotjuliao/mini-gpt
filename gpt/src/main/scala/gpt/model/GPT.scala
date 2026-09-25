package gpt.model

import gpt.nn.*
import scalagrad.core.Tensor
import scalagrad.ops.tensor.*
import scala.util.Random

final class GPT(
    val vocabSize: Int,
    val dModel: Int,
    val nHeads: Int,
    val nLayers: Int,
    val contextLength: Int,
    val expansion: Int = 4,
    rng: Random = new Random()
):
  def this(config: GPTConfig, rng: Random) = this(
    config.vocabSize,
    config.dModel,
    config.nHeads,
    config.nLayers,
    config.contextLength,
    config.expansion,
    rng
  )

  def this(config: GPTConfig) = this(config, new Random())

  require(
    nLayers >= 1,
    s"The model must have at least 1 transformer block, but got nLayers = $nLayers."
  )

  val config: GPTConfig =
    GPTConfig(vocabSize, dModel, nHeads, nLayers, contextLength, expansion)

  val embedding: Embedding = Embedding(vocabSize, dModel, contextLength, rng)

  /** Escala residual do GPT-2 (Etapa 15 §6): as duas projecoes que escrevem no
    * fluxo residual nascem com desvio padrao dividido por sqrt(2*nLayers), o que
    * mantem a variancia da pilha em ~2 em vez de crescer com 1 + 2L.
    */
  val residualScale: Double = 1.0 / Math.sqrt(2.0 * nLayers)

  val blocks: List[TransformerBlock] =
    List.fill(nLayers)(TransformerBlock(dModel, nHeads, expansion, rng, residualScale))

  val lnFinal: LayerNorm = LayerNorm(dModel)
  val head: Linear = Linear(dModel, vocabSize, useBias = false, rng = rng)

  val parameters: List[Tensor] = {
    embedding.parameters ++
      blocks.flatMap(_.parameters) ++
      lnFinal.parameters ++
      head.parameters
  }.distinct

  // A `Embedding` ja checa as duas condicoes, e a duplicacao se paga na
  // mensagem: quem chamou `GPT.forward` precisa ler o nome do que chamou.
  // Mesma decisao da `TransformerBlock` (ver theory/14-transformer-block §7).
  private def validate(tokens: Tensor): Unit =
    require(
      tokens.rank == 2,
      s"The input tensor for this model must have 2 dimensions (batchSize, seqLen) of token indices, " +
        s"but got rank ${tokens.rank}."
    )

    require(
      tokens.shape(1) <= contextLength,
      s"Cannot run a sequence of length ${tokens.shape(1)}: " +
        s"the model context length is $contextLength."
    )

  def forward(tokens: Tensor): Tensor =
    validate(tokens)

    head.forward(
      lnFinal.forward(
        blocks.foldLeft(embedding.forward(tokens))((x, b) => b.forward(x))
      )
    )
end GPT
