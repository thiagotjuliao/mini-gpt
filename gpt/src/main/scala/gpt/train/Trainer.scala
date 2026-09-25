package gpt.train

import gpt.data.BatchSampler
import gpt.loss.CrossEntropy
import gpt.model.GPT
import gpt.optim.AdamW
import scalagrad.core.Tensor

import java.io.File

final case class TrainingConfig(
    steps: Int,
    batchSize: Int,
    lrMax: Double = 3e-4,
    lrMin: Double = 3e-5,
    warmupSteps: Int = 0,
    maxGradNorm: Double = 1.0,
    logInterval: Int = 10,
    evalInterval: Int = 0,
    evalBatches: Int = 5,
    checkpointInterval: Int = 0,
    checkpointFile: Option[File] = None
):
  require(steps >= 1, s"steps must be at least 1, but got $steps.")
  require(batchSize >= 1, s"batchSize must be at least 1, but got $batchSize.")
  require(evalBatches >= 1, s"evalBatches must be at least 1, but got $evalBatches.")

  // Checado aqui, e não só dentro do schedule: sem isto a incompatibilidade só
  // apareceria no primeiro passo, depois de o modelo já ter sido construído.
  require(
    warmupSteps >= 0 && warmupSteps < steps,
    s"warmupSteps must be in [0, steps), but got $warmupSteps for $steps steps."
  )

  require(lrMax >= lrMin, s"lrMax ($lrMax) cannot be smaller than lrMin ($lrMin).")
  require(maxGradNorm > 0, s"maxGradNorm must be positive, but got $maxGradNorm.")
end TrainingConfig

final case class TrainingStep(
    step: Int,
    lr: Double,
    gradientNorm: Double,
    loss: Double,
    validationLoss: Option[Double]
):
  def perplexity: Double = Math.exp(loss)

final case class TrainingResult(optimizer: AdamW, history: List[TrainingStep]):
  def losses: List[Double] = history.map(_.loss)

object Trainer:

  /** Perda média em `batches` lotes, sem construir grafo.
    *
    * O `noGrad` não é só economia: sem ele, cada avaliação penduraria um grafo
    * inteiro na memória e acumularia gradiente nos parâmetros, poluindo o
    * próximo passo de treino (theory/18-training-loop §5).
    */
  def evaluate(model: GPT, sampler: BatchSampler, batchSize: Int, batches: Int): Double =
    Tensor.noGrad {
      val total = (1 to batches).foldLeft(0.0) { (acc, _) =>
        val (inputs, targets) = sampler.sample(batchSize)
        acc + CrossEntropy(model.forward(inputs), targets).get(0)
      }

      total / batches
    }

  def train(
      model: GPT,
      optimizer: AdamW,
      trainSampler: BatchSampler,
      config: TrainingConfig,
      validationSampler: Option[BatchSampler] = None,
      log: String => Unit = println
  ): TrainingResult =
    val initial = TrainingResult(optimizer, List.empty)

    val result = (1 to config.steps).foldLeft(initial) { (acc, step) =>
      val (inputs, targets) = trainSampler.sample(config.batchSize)
      val loss = CrossEntropy(model.forward(inputs), targets)

      // A ordem é contrato: zerar DEPOIS de ler a perda e ANTES do backward.
      // Zerar depois do backward apagaria justamente o que se acabou de calcular.
      acc.optimizer.zeroGrad()
      loss.backward()

      val gradientNorm = GradientClipping.clipByGlobalNorm(model.parameters, config.maxGradNorm)
      val lr = LRSchedule.cosine(
        step,
        config.steps,
        config.lrMax,
        config.lrMin,
        config.warmupSteps
      )

      val nextOptimizer = acc.optimizer.step(lr)

      val validationLoss = validationSampler.filter(_ => shouldRun(step, config.evalInterval)).map {
        sampler => evaluate(model, sampler, config.batchSize, config.evalBatches)
      }

      val metrics = TrainingStep(step, lr, gradientNorm, loss.get(0), validationLoss)

      if shouldRun(step, config.logInterval) then log(format(metrics))

      config.checkpointFile
        .filter(_ => shouldRun(step, config.checkpointInterval))
        .foreach(Checkpoint.save(_, model, nextOptimizer))

      TrainingResult(nextOptimizer, metrics :: acc.history)
    }

    result.copy(history = result.history.reverse)
  end train

  private def shouldRun(step: Int, interval: Int): Boolean = interval > 0 && step % interval == 0

  private def format(m: TrainingStep): String =
    val base = f"step ${m.step}%5d | lr ${m.lr}%.3e | |grad| ${m.gradientNorm}%7.4f | " +
      f"loss ${m.loss}%7.4f | ppl ${m.perplexity}%8.2f"

    m.validationLoss.fold(base)(v => base + f" | val ${v}%7.4f")
end Trainer
