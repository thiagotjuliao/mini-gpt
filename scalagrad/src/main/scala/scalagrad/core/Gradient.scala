package scalagrad.core

/** Gradiente acumulado de um `Tensor`. Wrapper sobre `Array[Double]`, mesmo
  * padrão de construção restrita de `Shape`/`Strides` — mas, ao contrário
  * delas, é deliberadamente mutável (ver `CLAUDE.md`): múltiplos caminhos do
  * grafo de autograd precisam acumular no mesmo `Gradient` ao longo de uma
  * única passada de `backward()`.
  *
  * A API só expõe as três operações que essa mutação legitimamente precisa
  * (`accumulate`, `zero`, `seed`) — não expõe um `update`/`:=` genérico, pra
  * tornar impossível sobrescrever um gradiente por engano em vez de acumular
  * (`+=`), que é o bug silencioso que quebraria grafos com tensores reusados.
  */
final class Gradient private[scalagrad] (private val values: Array[Double]):
  def apply(i: Int): Double = values(i)
  def toArray: Array[Double] = values.clone()
  def length: Int = values.length
  def toList: List[Double] = values.toList

  /** A única forma de escrever num gradiente existente: soma `delta` na
    * posição `i`. Nunca sobrescreve — preserva contribuições já acumuladas
    * de outros caminhos do grafo.
    */
  def accumulate(i: Int, delta: Double): Unit = values(i) += delta

  /** Acumula posição a posição. Valida o comprimento antes de escrever: sem
    * isso, um array curto acumularia até a metade e só então lançaria, deixando
    * o gradiente num estado parcial.
    */
  def accumulateAll(deltas: Array[Double]): Unit =
    require(
      deltas.length == values.length,
      s"Expected ${values.length} deltas to accumulate, but got ${deltas.length}."
    )

    values.indices.foreach(i => accumulate(i, deltas(i)))

  /** Mesma coisa, sem materializar um array intermediário -- evita o `.clone()`
    * que `toArray` faria só pra ser lido posição a posição.
    */
  def accumulateAll(other: Gradient): Unit =
    require(
      other.length == values.length,
      s"Expected a gradient of length ${values.length}, but got ${other.length}."
    )

    values.indices.foreach(i => accumulate(i, other(i)))

  /** Multiplica todas as posições por `factor`.
    *
    * Não é a sobrescrita que esta API se recusa a expor: o conteúdo acumulado é
    * preservado, só reescalado. É exatamente o que o gradient clipping da Etapa
    * 18 faz depois que o `backward()` terminou — todos os gradientes encolhem
    * pelo mesmo fator, o que muda o tamanho do passo sem mudar a direção.
    */
  def scale(factor: Double): Unit = values.mapInPlace(_ * factor)

  /** Zera todas as posições (usado por `Tensor.zeroGrad`). */
  def zero(): Unit = values.mapInPlace(_ => 0.0)

  /** Semeia todas as posições com `1.0` (usado por `Tensor.backward` só na
    * raiz do grafo: `dLoss/dLoss = 1`).
    */
  def seed(): Unit = values.mapInPlace(_ => 1.0)

  override def toString: String = values.mkString("Gradient(", ", ", ")")
end Gradient

object Gradient:

  /** Um gradiente tem exatamente uma posição por elemento do tensor, indexada
    * canonicamente. Preferir esta sobrecarga: passar `data.length` em vez de
    * `shape.size` já causou inconsistência real (ver `HISTORY.md`, 2026-08-21).
    */
  def zeros(shape: Shape): Gradient = zeros(shape.size)

  def zeros(n: Int): Gradient = new Gradient(Array.fill(n)(0.0))

  private[scalagrad] def apply(values: Array[Double]): Gradient = new Gradient(values)
