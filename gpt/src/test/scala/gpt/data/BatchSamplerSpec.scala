package gpt.data

import scalagrad.core.Tensor
import scala.util.Random
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class BatchSamplerSpec extends AnyFlatSpec with Matchers:

  // Devolve, em ordem, os valores dados a cada chamada de nextInt -- permite
  // controlar exatamente quais `start` o BatchSampler vai sortear, sem
  // depender do algoritmo interno do java.util.Random.
  private class FixedRandom(values: Int*) extends Random:
    private val it = values.iterator
    override def nextInt(n: Int): Int = it.next()

  private def row(t: Tensor, r: Int, len: Int): List[Double] =
    (0 until len).map(j => t.get(r, j)).toList

  "BatchSampler.sample" should "match the verified numeric example from theory/07-tokenization §5" in {
    // encode("abacate") = [0,1,0,2,0,4,3] (theory/07-tokenization/07-tokenization.md §2-3)
    val corpus = Array(0, 1, 0, 2, 0, 4, 3)
    val sampler = new BatchSampler(corpus, contextLength = 4, new FixedRandom(0, 2))

    val (input, target) = sampler.sample(batchSize = 2)

    input.rank shouldBe 2
    input.size shouldBe 8
    target.rank shouldBe 2
    target.size shouldBe 8

    // start=0: corpus[0:5]=[0,1,0,2,0] -> input=[0,1,0,2], target=[1,0,2,0]
    row(input, 0, 4) shouldBe List(0.0, 1.0, 0.0, 2.0)
    row(target, 0, 4) shouldBe List(1.0, 0.0, 2.0, 0.0)

    // start=2: corpus[2:7]=[0,2,0,4,3] -> input=[0,2,0,4], target=[2,0,4,3]
    row(input, 1, 4) shouldBe List(0.0, 2.0, 0.0, 4.0)
    row(target, 1, 4) shouldBe List(2.0, 0.0, 4.0, 3.0)
  }

  it should "produce tensors of shape (batchSize, contextLength)" in {
    val corpus = (0 until 50).toArray
    val sampler = new BatchSampler(corpus, contextLength = 6)

    val (input, target) = sampler.sample(batchSize = 5)

    input.rank shouldBe 2
    input.size shouldBe 30
    target.rank shouldBe 2
    target.size shouldBe 30
  }

  it should "always satisfy target = input shifted one position within the same window" in {
    val corpus = (0 until 50).toArray
    val contextLength = 6
    val sampler = new BatchSampler(corpus, contextLength, new Random(42))

    val (input, target) = sampler.sample(batchSize = 20)

    for r <- 0 until 20 do
      // target[i] == input[i+1] para toda posicao i dentro da mesma janela
      // (mesma janela deslocada de 1, nao uma janela nova -- theory §4)
      row(target, r, contextLength - 1) shouldBe row(input, r, contextLength).drop(1)
  }

  it should "draw the only valid window when the corpus has exactly contextLength + 1 tokens" in {
    // corpusLength=5, contextLength=4 -> start valido: 0 ate 5-4-1=0, ou seja, so start=0.
    val corpus = Array(10, 20, 30, 40, 50)
    val sampler = new BatchSampler(corpus, contextLength = 4, new Random())

    val (input, target) = sampler.sample(batchSize = 10)

    for r <- 0 until 10 do
      row(input, r, 4) shouldBe List(10.0, 20.0, 30.0, 40.0)
      row(target, r, 4) shouldBe List(20.0, 30.0, 40.0, 50.0)
  }

  it should "sample independent windows across the batch, not the same one repeated" in {
    // seed fixa escolhida por inspecao: produz starts diferentes entre si num
    // corpus longo, confirmando que cada linha do batch e amostrada de forma
    // independente (nao e sempre a mesma janela, nem uma varredura sequencial).
    val corpus = (0 until 100).toArray
    val sampler = new BatchSampler(corpus, contextLength = 4, new Random(1))

    val (input, _) = sampler.sample(batchSize = 8)

    val firstElementPerRow = (0 until 8).map(r => input.get(r, 0)).toSet
    firstElementPerRow.size should be > 1
  }

  "BatchSampler.split" should "cut the corpus in order, not at random" in {
    val corpus = Array.tabulate(100)(identity)
    val (treino, validacao) = BatchSampler.split(corpus, contextLength = 4, 0.2)

    // 80 primeiros para treino, 20 ultimos para validacao
    val (entradas, _) = treino.sample(20)
    val (entradasVal, _) = validacao.sample(20)

    (0 until entradas.size).foreach(i => entradas.toArray(i) should be < 80.0)
    (0 until entradasVal.size).foreach(i => entradasVal.toArray(i) should be >= 80.0)
  }

  it should "reject a fraction outside (0, 1)" in {
    val corpus = Array.tabulate(100)(identity)

    an[IllegalArgumentException] should be thrownBy BatchSampler.split(corpus, 4, 0.0)
    an[IllegalArgumentException] should be thrownBy BatchSampler.split(corpus, 4, 1.0)
  }

  it should "reject a split that leaves a side shorter than the context" in {
    val corpus = Array.tabulate(20)(identity)

    an[IllegalArgumentException] should be thrownBy BatchSampler.split(corpus, 16, 0.1)
  }
end BatchSamplerSpec
