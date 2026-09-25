package gpt.generate

import gpt.data.{BatchSampler, Tokenizer}
import gpt.model.GPT
import gpt.optim.AdamW
import gpt.train.{Trainer, TrainingConfig}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.util.Random

class GeneratorSpec extends AnyFlatSpec with Matchers:

  private val vocabSize = 3
  private val contextLength = 8

  private def modelo(ctx: Int = contextLength, seed: Int = 42): GPT =
    GPT(
      vocabSize,
      dModel = 16,
      nHeads = 2,
      nLayers = 1,
      contextLength = ctx,
      rng = new Random(seed)
    )

  private def gerador(m: GPT, seed: Int = 5): Generator = new Generator(m, new Random(seed))

  private val prompt = Array(0, 1, 2)

  "generate" should "return the prompt plus the requested number of tokens" in {
    val saida = gerador(modelo()).generate(prompt, maxNewTokens = 5)

    saida.length shouldBe prompt.length + 5
    saida.take(3) shouldBe prompt
  }

  it should "return the prompt untouched when asked for zero tokens" in {
    gerador(modelo()).generate(prompt, maxNewTokens = 0) shouldBe prompt
  }

  it should "only produce tokens inside the vocabulary" in {
    val saida = gerador(modelo()).generate(prompt, 20, Temperature(1.5))

    saida.foreach { t =>
      t should be >= 0
      t should be < vocabSize
    }
  }

  it should "be deterministic under Greedy" in {
    val m = modelo()

    gerador(m).generate(prompt, 10) shouldBe gerador(m, seed = 999).generate(prompt, 10)
  }

  it should "vary under temperature sampling with different seeds" in {
    val m = modelo()

    val a = gerador(m, seed = 1).generate(prompt, 30, Temperature(2.0))
    val b = gerador(m, seed = 2).generate(prompt, 30, Temperature(2.0))

    a should not be b
  }

  "the sliding window" should "let generation run past the context length" in {
    // sem a janela, o modelo recusaria a entrada assim que a sequencia
    // passasse de contextLength -- e o teste estouraria
    val m = modelo(ctx = 4)
    val saida = gerador(m).generate(prompt, maxNewTokens = 20)

    saida.length shouldBe 23
  }

  it should "keep the most recent tokens, dropping the oldest" in {
    val m = modelo(ctx = 4)

    noException should be thrownBy gerador(m).generate(Array(0, 1, 2, 0, 1, 2, 0), 5)
  }

  "onToken" should "fire once per generated token, in order" in {
    val vistos = scala.collection.mutable.ListBuffer.empty[Int]
    val saida = gerador(modelo()).generate(prompt, 6, Greedy, vistos.append(_))

    vistos.length shouldBe 6
    vistos.toArray shouldBe saida.drop(prompt.length)
  }

  "generation" should "not accumulate gradients on the parameters" in {
    val m = modelo()
    m.parameters.foreach(_.gradient.zero())

    gerador(m).generate(prompt, 10, Temperature(1.0))

    m.parameters.foreach(_.gradient.toArray.forall(_ == 0.0) shouldBe true)
  }

  "generate" should "reject an empty prompt" in {
    an[IllegalArgumentException] should be thrownBy
      gerador(modelo()).generate(Array.empty[Int], 5)
  }

  it should "reject a prompt token outside the vocabulary" in {
    an[IllegalArgumentException] should be thrownBy
      gerador(modelo()).generate(Array(0, 7), 5)
  }

  it should "reject a negative number of new tokens" in {
    an[IllegalArgumentException] should be thrownBy gerador(modelo()).generate(prompt, -1)
  }

  "the text overload" should "encode, generate and decode in one call" in {
    val tokenizer = Tokenizer.charLevel("abc")
    val m = modelo()

    val saida = gerador(m).generate("abc", 5, tokenizer)

    saida.length shouldBe 8
    saida.startsWith("abc") shouldBe true
    saida.forall("abc".contains(_)) shouldBe true
  }

  "a trained model" should "continue the pattern it learned" in {
    // o teste de fumaca definitivo: treinar num corpus ciclico e conferir que a
    // geração continua o ciclo. Se isto passa, a cadeia inteira funciona --
    // tokenizacao, embedding, atencao causal, perda, autograd, otimizador e
    // amostragem.
    val corpus = Array.tabulate(600)(_ % vocabSize)
    val m = modelo()

    Trainer.train(
      m,
      new AdamW(m.parameters),
      new BatchSampler(corpus, contextLength, new Random(7)),
      TrainingConfig(
        steps = 80,
        batchSize = 4,
        lrMax = 1e-2,
        lrMin = 1e-3,
        warmupSteps = 5,
        logInterval = 0
      ),
      log = _ => ()
    )

    val saida = gerador(m).generate(Array(0, 1, 2), maxNewTokens = 12)
    val esperado = Array.tabulate(15)(_ % vocabSize)

    saida shouldBe esperado
  }
end GeneratorSpec
