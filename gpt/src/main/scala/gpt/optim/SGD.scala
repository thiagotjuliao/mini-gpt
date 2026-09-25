package gpt.optim

import scalagrad.core.Tensor

/** O ponto de partida da Etapa 17: `p ← p − lr · g`, sem estado nenhum.
  *
  * Por não guardar momento nem contador, `step()` devolve `Unit` -- não há
  * instância nova a propagar, ao contrário do [[AdamW]]. Serve de linha de base
  * para comparar: o passo aqui é proporcional ao gradiente, e é exatamente essa
  * proporcionalidade que o AdamW abandona
  * (ver theory/17-adamw-optimizer/17-adamw-optimizer.md §2).
  */
final class SGD(val parameters: List[Tensor], val lr: Double = 3e-4):
  require(
    parameters.nonEmpty,
    "The optimizer needs at least 1 parameter to update, but got an empty list."
  )

  require(
    parameters.forall(_.requiresGradient),
    "Every parameter given to the optimizer must have `requiresGradient = true`, " +
      "otherwise `backward()` never fills its gradient and the optimizer runs without optimizing."
  )

  def zeroGrad(): Unit = parameters.foreach(_.gradient.zero())

  def step(): Unit = step(lr)

  def step(stepLr: Double): Unit = parameters.foreach { p =>
    val values = p.toArray
    val gradient = p.gradient

    p.updateData(Array.tabulate(p.size)(i => values(i) - stepLr * gradient(i)))
  }
