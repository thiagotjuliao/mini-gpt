package gpt.generate

import gpt.data.Tokenizer
import gpt.model.GPT
import scalagrad.core.Tensor

import scala.util.Random

/** Geração autoregressiva: a saída de um passo vira a entrada do seguinte
  * (ver theory/19-inference-generation/19-inference-generation.md §1).
  */
final class Generator(model: GPT, randomizer: Random = new Random()) {

  /** Gera `maxNewTokens` tokens a partir de `prompt`, devolvendo o prompt e a
    * continuação juntos.
    *
    * `onToken` é chamado a cada token novo, para impressão progressiva.
    */
  def generate(
      prompt: Array[Int],
      maxNewTokens: Int,
      strategy: SamplingStrategy = Greedy,
      onToken: Int => Unit = _ => ()
  ): Array[Int] = {
    require(prompt.nonEmpty, "The prompt needs at least 1 token to condition on.")
    require(maxNewTokens >= 0, s"maxNewTokens cannot be negative, but got $maxNewTokens.")
    require(
      prompt.forall(t => t >= 0 && t < model.vocabSize),
      s"Every prompt token must be in [0, ${model.vocabSize})."
    )

    // Nenhum grafo, nenhum gradiente: a geração só lê o modelo.
    Tensor.noGrad {
      (1 to maxNewTokens)
        .foldLeft(prompt.toVector) { (tokens, _) =>
          val token = Sampler.next(lastLogits(tokens), strategy, randomizer)
          onToken(token)
          tokens :+ token
        }
        .toArray
    }
  }

  def generate(prompt: String, maxNewTokens: Int, tokenizer: Tokenizer): String =
    generate(prompt, maxNewTokens, tokenizer, Greedy)

  def generate(
      prompt: String,
      maxNewTokens: Int,
      tokenizer: Tokenizer,
      strategy: SamplingStrategy
  ): String =
    tokenizer.decode(generate(tokenizer.encode(prompt), maxNewTokens, strategy))

  /** Os logits da **última** posição, que é a única que prevê um token novo.
    *
    * A janela deslizante corta pela esquerda quando a sequência passa do
    * contexto: o modelo recusa entradas mais longas, e os tokens mais antigos
    * são os que menos importam para a próxima previsão.
    */
  private def lastLogits(tokens: Vector[Int]): Array[Double] = {
    val context = tokens.takeRight(model.contextLength)
    val input = Tensor.make(context.map(_.toDouble).toArray, Array(1, context.length))
    val logits = model.forward(input)
    val last = context.length - 1

    Array.tabulate(model.vocabSize)(v => logits.get(0, last, v))
  }
}
