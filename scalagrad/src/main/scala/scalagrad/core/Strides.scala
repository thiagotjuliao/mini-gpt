package scalagrad.core

/** Strides de um `Tensor`: passo em `data` (número de elementos, não bytes)
  * pra avançar cada dimensão em 1. Wrapper sobre `Array[Int]`, mesmo padrão de
  * construção restrita de `Shape`.
  */
final class Strides private[scalagrad] (private val values: Array[Int]) {
  def apply(i: Int): Int = values(i)
  def length: Int = values.length
  def mkString(sep: String): String = values.mkString(sep)
  /** O array interno, **sem cópia** -- ao contrário de `Shape.toArray`. É
    * deliberado: `Shape.linearIndex` lê isto a cada acesso a elemento, e clonar
    * ali dominaria o custo. O nome diz `unsafe` para que quem chamar saiba que
    * está segurando o estado interno, e nunca deve mutá-lo.
    */
  def unsafeValues: Array[Int] = values

  /** Cópia defensiva, mesma semântica de `Shape.toArray`. */
  def toArray: Array[Int] = values.clone()

  def updated(i: Int, value: Int): Strides = new Strides(values.updated(i, value))
  def leftPad(n: Int, value: Int): Strides = new Strides(Array.fill(n)(value) ++ values)

  /** `this` alinhado ao comprimento `targetLength`, preenchendo `value` à
    * esquerda se `this` for mais curto. Mesma ideia de `Shape.padTo`.
    */
  def padTo(targetLength: Int, value: Int): Strides =
    leftPad(Math.max(0, targetLength - length), value)

  override def toString: String = values.mkString("Strides(", ", ", ")")
}

object Strides {
  def apply(dims: Int*): Strides = new Strides(dims.toArray)
  def apply(dims: Array[Int]): Strides = new Strides(dims)
}
