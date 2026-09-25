package gpt.generate

import scala.util.Random

/** Como escolher um token a partir dos logits da última posição. */
sealed trait SamplingStrategy

/** O token mais provável, sempre. Determinístico e repetitivo. */
case object Greedy extends SamplingStrategy

/** Amostra da distribuição inteira, com os logits divididos por `value`.
  * Abaixo de 1 concentra a distribuição, acima de 1 espalha.
  */
final case class Temperature(value: Double) extends SamplingStrategy:
  require(value > 0, s"Temperature must be positive, but got $value.")

/** Amostra só entre os `k` tokens mais prováveis, renormalizando entre eles. */
final case class TopK(k: Int, temperature: Double = 1.0) extends SamplingStrategy:
  require(k >= 1, s"k must be at least 1, but got $k.")
  require(temperature > 0, s"Temperature must be positive, but got $temperature.")

object Sampler:

  def next(logits: Array[Double], strategy: SamplingStrategy, rng: Random): Int =
    require(logits.nonEmpty, "Cannot sample from an empty logit vector.")

    strategy match
      case Greedy => argmax(logits)
      case Temperature(t) => sample(softmax(logits.map(_ / t)), rng)
      case TopK(k, t) => sample(softmax(keepTopK(logits, k).map(_ / t)), rng)

  def argmax(logits: Array[Double]): Int =
    logits.indices.foldLeft(0)((best, i) => if logits(i) > logits(best) then i else best)

  /** Softmax estável: subtrair o máximo não muda o resultado e evita que
    * `exp` estoure (ver theory/06-softmax §2).
    */
  def softmax(logits: Array[Double]): Array[Double] =
    val max = logits.max
    val exps = logits.map(l => Math.exp(l - max))
    val total = exps.sum

    exps.map(_ / total)

  /** Zera a probabilidade fora dos `k` maiores, pondo `-inf` nos logits — o
    * `exp` disso é exatamente 0, então eles somem do softmax seguinte.
    *
    * Empates na fronteira mantêm mais de `k` candidatos. É o comportamento
    * desejável: desempatar por índice escolheria por ordem alfabética do
    * vocabulário, o que não é uma razão.
    */
  def keepTopK(logits: Array[Double], k: Int): Array[Double] =
    if k >= logits.length then logits
    else
      val threshold = logits.sorted.reverse.apply(k - 1)
      logits.map(l => if l >= threshold then l else Double.NegativeInfinity)

  /** Amostragem pela inversa da CDF: sorteia `u` uniforme e devolve o primeiro
    * índice onde a soma acumulada passa de `u`.
    */
  private def sample(probabilities: Array[Double], rng: Random): Int =
    val cumulative = probabilities.scanLeft(0.0)(_ + _).tail
    val u = rng.nextDouble()
    val index = cumulative.indexWhere(_ > u)

    // A soma acumulada pode parar um epsilon abaixo de 1 por arredondamento, e
    // aí nenhum índice passa de `u`. A última posição é a resposta certa.
    if index >= 0 then index else probabilities.length - 1
end Sampler
