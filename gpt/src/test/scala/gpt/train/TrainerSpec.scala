package gpt.train

import gpt.data.BatchSampler
import gpt.model.GPT
import gpt.optim.AdamW
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import java.io.File
import scala.util.Random

class TrainerSpec extends AnyFlatSpec with Matchers:

  private val vocabSize = 3
  private val contextLength = 8

  /** Um padrao ciclico perfeito: dado o token atual, o proximo e' determinado.
    * A perda tem que cair de log(3) para perto de zero -- se nao cair, o bug
    * esta na cadeia inteira, nao no corpus.
    */
  private val corpus = Array.tabulate(600)(_ % vocabSize)

  /** Semeado: sem `rng` injetado, a perda inicial varia de 1.10 a 1.84 entre
    * execucoes, e qualquer asserção sobre ela vira teste intermitente.
    */
  private def modelo(seed: Int = 42): GPT =
    GPT(
      vocabSize,
      dModel = 16,
      nHeads = 2,
      nLayers = 1,
      contextLength = contextLength,
      rng = new Random(seed)
    )

  private def amostrador(seed: Int = 7): BatchSampler =
    new BatchSampler(corpus, contextLength, new Random(seed))

  private def config(steps: Int, logInterval: Int = 0): TrainingConfig =
    TrainingConfig(
      steps = steps,
      batchSize = 4,
      lrMax = 1e-2,
      lrMin = 1e-3,
      warmupSteps = Math.min(5, steps - 1),
      logInterval = logInterval
    )

  private def semLog(s: String): Unit = ()

  "the training loop" should "drive the loss down on a learnable corpus" in {
    val m = modelo()
    val resultado =
      Trainer.train(m, new AdamW(m.parameters), amostrador(), config(60), log = semLog)

    val ultimas = resultado.losses.takeRight(10).sum / 10

    // Um modelo recem-inicializado nao chuta uniforme: os logits sao aleatorios,
    // e qualquer desvio da uniforme SOBE a cross-entropy quando o alvo nao tem
    // relacao com eles (desigualdade de Jensen). Entao a perda inicial fica em
    // log(3) = 1.0986 ou acima, nunca bem abaixo.
    resultado.losses.head should be >= Math.log(vocabSize) - 0.05
    ultimas should be < 0.3
    ultimas should be < resultado.losses.head / 3
  }

  it should "record one entry per step, numbered from 1" in {
    val m = modelo()
    val resultado =
      Trainer.train(m, new AdamW(m.parameters), amostrador(), config(5), log = semLog)

    resultado.history.length shouldBe 5
    resultado.history.map(_.step) shouldBe List(1, 2, 3, 4, 5)
  }

  it should "keep the history in chronological order" in {
    val m = modelo()
    val resultado =
      Trainer.train(m, new AdamW(m.parameters), amostrador(), config(4), log = semLog)

    resultado.history.head.step shouldBe 1
    resultado.history.last.step shouldBe 4
  }

  it should "follow the schedule, step by step" in {
    val m = modelo()
    val cfg = config(20)
    val resultado = Trainer.train(m, new AdamW(m.parameters), amostrador(), cfg, log = semLog)

    resultado.history.foreach { passo =>
      val esperado =
        LRSchedule.cosine(passo.step, cfg.steps, cfg.lrMax, cfg.lrMin, cfg.warmupSteps)

      passo.lr shouldBe esperado +- 1e-15
    }
  }

  it should "advance the optimizer t to the number of steps" in {
    val m = modelo()
    val resultado =
      Trainer.train(m, new AdamW(m.parameters), amostrador(), config(6), log = semLog)

    resultado.optimizer.t shouldBe 6
  }

  it should "report the gradient norm of every step" in {
    val m = modelo()
    val resultado =
      Trainer.train(m, new AdamW(m.parameters), amostrador(), config(4), log = semLog)

    resultado.history.foreach(_.gradientNorm should be > 0.0)
  }

  it should "log only on the logging interval" in {
    val m = modelo()
    val linhas = scala.collection.mutable.ListBuffer.empty[String]

    Trainer.train(
      m,
      new AdamW(m.parameters),
      amostrador(),
      config(10, logInterval = 5),
      log = linhas.append(_)
    )

    linhas.length shouldBe 2
    linhas.head should include("step")
    linhas.head should include("loss")
  }

  "validation" should "only be measured on the eval interval" in {
    val m = modelo()
    val cfg = config(6).copy(evalInterval = 3, evalBatches = 2)

    val resultado = Trainer.train(
      m,
      new AdamW(m.parameters),
      amostrador(),
      cfg,
      validationSampler = Some(amostrador(seed = 99)),
      log = semLog
    )

    resultado.history.filter(_.validationLoss.isDefined).map(_.step) shouldBe List(3, 6)
  }

  it should "stay empty when no validation sampler is given" in {
    val m = modelo()
    val cfg = config(4).copy(evalInterval = 2)

    val resultado =
      Trainer.train(m, new AdamW(m.parameters), amostrador(), cfg, log = semLog)

    resultado.history.forall(_.validationLoss.isEmpty) shouldBe true
  }

  "evaluate" should "leave the gradients untouched" in {
    val m = modelo()
    val opt = new AdamW(m.parameters)
    opt.zeroGrad()

    Trainer.evaluate(m, amostrador(), batchSize = 2, batches = 3)

    // sem `noGrad`, cada avaliacao penduraria um grafo e acumularia gradiente
    // nos parametros, poluindo o proximo passo de treino
    m.parameters.foreach(_.gradient.toArray.forall(_ == 0.0) shouldBe true)
  }

  it should "return a finite loss near log(vocabSize) for an untrained model" in {
    val m = modelo()
    val perda = Trainer.evaluate(m, amostrador(), batchSize = 4, batches = 3)

    perda.isNaN shouldBe false
    perda should be >= Math.log(vocabSize) - 0.05
    perda should be < Math.log(vocabSize) + 1.2
  }

  "checkpointing" should "write the file on the configured interval" in {
    val arquivo = File.createTempFile("mini-gpt-trainer", ".bin")
    arquivo.delete()
    arquivo.deleteOnExit()

    val m = modelo()
    val cfg = config(4).copy(checkpointInterval = 4, checkpointFile = Some(arquivo))

    Trainer.train(m, new AdamW(m.parameters), amostrador(), cfg, log = semLog)

    arquivo.exists() shouldBe true
    arquivo.length() should be > 0L
  }

  it should "not write anything when no file is configured" in {
    val m = modelo()
    val cfg = config(4).copy(checkpointInterval = 2)

    noException should be thrownBy
      Trainer.train(m, new AdamW(m.parameters), amostrador(), cfg, log = semLog)
  }

  "the config" should "reject a non-positive number of steps" in {
    an[IllegalArgumentException] should be thrownBy TrainingConfig(steps = 0, batchSize = 4)
  }

  it should "reject a non-positive batch size" in {
    an[IllegalArgumentException] should be thrownBy TrainingConfig(steps = 4, batchSize = 0)
  }

  it should "reject a warmup that covers the whole run, at construction time" in {
    // a checagem mora no config, e nao so no schedule: senao a incompatibilidade
    // so apareceria no primeiro passo, depois do modelo ja construido
    an[IllegalArgumentException] should be thrownBy
      TrainingConfig(steps = 10, batchSize = 4, warmupSteps = 10)
  }

  it should "reject a non-positive maxGradNorm" in {
    an[IllegalArgumentException] should be thrownBy
      TrainingConfig(steps = 10, batchSize = 4, maxGradNorm = 0.0)
  }
end TrainerSpec
