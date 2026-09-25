package gpt.optim

import scalagrad.core.Tensor

/** Estado imutável: `step()` devolve uma instância nova, com `t + 1` e o mapa de
  * momentos atualizado. O mapa é indexado por identidade de referência, o que só
  * funciona porque `Tensor` não é `case class`
  * (ver theory/17-adamw-optimizer/17-adamw-optimizer.md §7).
  */
final class AdamW(
    val parameters: List[Tensor],
    val lr: Double = 3e-4,
    val beta1: Double = 0.9,
    val beta2: Double = 0.95,
    val eps: Double = 1e-8,
    val weightDecay: Double = 0.1,
    val t: Int = 0,
    // `private[gpt]` e não `private`: o `Checkpoint` da Etapa 18 precisa ler os
    // momentos para salvá-los, e escrevê-los de volta ao retomar um treino.
    private[gpt] val state: Map[Tensor, (Array[Double], Array[Double])] = Map.empty
):
  require(
    parameters.nonEmpty,
    "The optimizer needs at least 1 parameter to update, but got an empty list."
  )

  require(
    parameters.forall(_.requiresGradient),
    "Every parameter given to the optimizer must have `requiresGradient = true`, " +
      "otherwise `backward()` never fills its gradient and the optimizer runs without optimizing."
  )

  require(beta1 >= 0 && beta1 < 1, s"beta1 must be in [0, 1), but got $beta1.")
  require(beta2 >= 0 && beta2 < 1, s"beta2 must be in [0, 1), but got $beta2.")
  require(eps > 0, s"eps must be positive, but got $eps.")
  require(weightDecay >= 0, s"weightDecay cannot be negative, but got $weightDecay.")

  def zeroGrad(): Unit = parameters.foreach(_.gradient.zero())

  def step(): AdamW = step(lr)

  /** `stepLr` sobrescreve a taxa deste passo, sem alterar a do otimizador --
    * é como o schedule da Etapa 18 injeta o valor de cada passo.
    */
  def step(stepLr: Double): AdamW =
    // O contador avança antes de ser usado: com `t = 0` a correção de viés
    // dividiria por `1 - beta^0 = 0`.
    val nextT = t + 1
    val biasCorrection1 = 1.0 - Math.pow(beta1, nextT)
    val biasCorrection2 = 1.0 - Math.pow(beta2, nextT)

    // Fase 1: puramente calcular. Nada é escrito enquanto os valores antigos
    // ainda estão sendo lidos -- o decaimento usa o `p` de antes do passo.
    val updates = parameters.map { p =>
      val (m, v) = state.getOrElse(p, (Array.fill(p.size)(0.0), Array.fill(p.size)(0.0)))
      val gradient = p.gradient
      val values = p.toArray

      val nextM = Array.tabulate(p.size)(i => beta1 * m(i) + (1 - beta1) * gradient(i))
      val nextV =
        Array.tabulate(p.size)(i => beta2 * v(i) + (1 - beta2) * gradient(i) * gradient(i))

      val nextValues = Array.tabulate(p.size) { i =>
        val mHat = nextM(i) / biasCorrection1
        val vHat = nextV(i) / biasCorrection2
        val decay = stepLr * weightDecay * values(i)

        values(i) - decay - stepLr * mHat / (Math.sqrt(vHat) + eps)
      }

      (p, nextM, nextV, nextValues)
    }

    // Fase 2: escrever.
    updates.foreach((p, _, _, nextValues) => p.updateData(nextValues))

    // Fase 3: o estado do próximo passo.
    new AdamW(
      parameters,
      lr,
      beta1,
      beta2,
      eps,
      weightDecay,
      nextT,
      updates.map((p, nextM, nextV, _) => p -> (nextM, nextV)).toMap
    )
  end step
end AdamW
