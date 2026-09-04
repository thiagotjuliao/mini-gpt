package gpt.train

/** A taxa de aprendizado passo a passo: rampa linear no aquecimento, cosseno
  * depois (ver theory/18-training-loop/18-training-loop.md §3).
  */
object LRSchedule {

  /** `step` é 1-based, igual ao `t` do `AdamW`. Depois de `totalSteps` a taxa
    * fica em `lrMin` em vez de continuar caindo -- um treino que passa do
    * planejado não deve começar a andar para trás.
    */
  def cosine(
      step: Int,
      totalSteps: Int,
      lrMax: Double = 3e-4,
      lrMin: Double = 3e-5,
      warmupSteps: Int = 0
  ): Double = {
    require(step >= 1, s"Step is 1-based, but got $step.")
    require(totalSteps >= 1, s"totalSteps must be at least 1, but got $totalSteps.")
    require(
      warmupSteps >= 0 && warmupSteps < totalSteps,
      s"warmupSteps must be in [0, totalSteps), but got $warmupSteps for $totalSteps steps."
    )
    require(lrMax >= lrMin, s"lrMax ($lrMax) cannot be smaller than lrMin ($lrMin).")
    require(lrMin >= 0, s"lrMin cannot be negative, but got $lrMin.")

    if step <= warmupSteps then lrMax * step / warmupSteps
    else
      val progress = (step - warmupSteps).toDouble / (totalSteps - warmupSteps)
      val clamped = Math.min(Math.max(progress, 0.0), 1.0)

      lrMin + 0.5 * (lrMax - lrMin) * (1 + Math.cos(Math.PI * clamped))
  }
}
