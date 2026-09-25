package gpt.generate

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.util.Random

class SamplerSpec extends AnyFlatSpec with Matchers:

  private val logits = Array(2.0, 1.0, 0.0, -1.0)

  "argmax" should "find the index of the largest logit" in {
    Sampler.argmax(Array(0.1, 5.0, -3.0, 4.9)) shouldBe 1
  }

  it should "keep the first index on a tie" in {
    Sampler.argmax(Array(1.0, 1.0)) shouldBe 0
  }

  "Greedy" should "be deterministic" in {
    val escolhas = (1 to 20).map(_ => Sampler.next(logits, Greedy, new Random()))

    escolhas.distinct shouldBe Seq(0)
  }

  "softmax" should "produce probabilities that sum to 1" in {
    Sampler.softmax(logits).sum shouldBe 1.0 +- 1e-12
  }

  it should "survive logits large enough to overflow exp" in {
    // sem subtrair o maximo, e^1000 estoura e tudo vira NaN
    val p = Sampler.softmax(Array(1000.0, 999.0))

    p.forall(_.isNaN) shouldBe false
    p.sum shouldBe 1.0 +- 1e-12
    p(0) shouldBe 0.7310585786 +- 1e-9
  }

  "a temperature below 1" should "concentrate the distribution" in {
    val quente = Sampler.softmax(logits.map(_ / 1.0))
    val fria = Sampler.softmax(logits.map(_ / 0.1))

    fria(0) should be > quente(0)
    fria(0) should be > 0.99
  }

  "a temperature above 1" should "flatten the distribution" in {
    val quente = Sampler.softmax(logits.map(_ / 10.0))

    // com T alto todos se aproximam de 1/4
    quente.foreach(p => p shouldBe 0.25 +- 0.1)
    quente(0) should be < Sampler.softmax(logits)(0)
  }

  "temperature 1" should "be the plain softmax" in {
    val comTemperatura = Sampler.softmax(logits.map(_ / 1.0))

    comTemperatura shouldBe Sampler.softmax(logits)
  }

  "keepTopK" should "send everything outside the top k to negative infinity" in {
    val filtrado = Sampler.keepTopK(logits, 2)

    filtrado(0) shouldBe 2.0
    filtrado(1) shouldBe 1.0
    filtrado(2) shouldBe Double.NegativeInfinity
    filtrado(3) shouldBe Double.NegativeInfinity
  }

  it should "zero the probability of the discarded tokens" in {
    val p = Sampler.softmax(Sampler.keepTopK(logits, 2))

    p(2) shouldBe 0.0
    p(3) shouldBe 0.0
    p.sum shouldBe 1.0 +- 1e-12
  }

  it should "leave the logits untouched when k covers everything" in {
    Sampler.keepTopK(logits, 4) shouldBe logits
    Sampler.keepTopK(logits, 99) shouldBe logits
  }

  it should "keep more than k on a tie at the boundary" in {
    // desempatar por indice escolheria pela ordem do vocabulario, o que nao e
    // uma razao -- melhor manter os empatados
    val comEmpate = Array(3.0, 1.0, 1.0, 0.0)
    val filtrado = Sampler.keepTopK(comEmpate, 2)

    filtrado(1) shouldBe 1.0
    filtrado(2) shouldBe 1.0
    filtrado(3) shouldBe Double.NegativeInfinity
  }

  "TopK sampling" should "never return a token outside the top k" in {
    val rng = new Random(11)
    val escolhas = (1 to 500).map(_ => Sampler.next(logits, TopK(2), rng))

    escolhas.toSet shouldBe Set(0, 1)
  }

  "sampling" should "follow the distribution it was given" in {
    val rng = new Random(3)
    // softmax([ln 7, ln 2, ln 1]) = [0.7, 0.2, 0.1]
    val alvo = Array(Math.log(7.0), Math.log(2.0), Math.log(1.0))
    val amostras = 20000

    val contagem = (1 to amostras)
      .map(_ => Sampler.next(alvo, Temperature(1.0), rng))
      .groupBy(identity)
      .view
      .mapValues(_.size.toDouble / amostras)
      .toMap

    contagem(0) shouldBe 0.7 +- 0.02
    contagem(1) shouldBe 0.2 +- 0.02
    contagem(2) shouldBe 0.1 +- 0.02
  }

  it should "stay in bounds when the uniform draw lands past the accumulated sum" in {
    // a soma acumulada pode parar um epsilon abaixo de 1 por arredondamento
    val quaseUm = new Random:
      override def nextDouble(): Double = 0.9999999999999999

    val escolha = Sampler.next(logits, Temperature(1.0), quaseUm)

    escolha should be >= 0
    escolha should be < logits.length
  }

  "the strategies" should "reject invalid parameters" in {
    an[IllegalArgumentException] should be thrownBy Temperature(0.0)
    an[IllegalArgumentException] should be thrownBy Temperature(-1.0)
    an[IllegalArgumentException] should be thrownBy TopK(0)
    an[IllegalArgumentException] should be thrownBy TopK(5, temperature = 0.0)
  }

  it should "reject an empty logit vector" in {
    an[IllegalArgumentException] should be thrownBy
      Sampler.next(Array.empty, Greedy, new Random())
  }
end SamplerSpec
