package scalagrad.core

/** Shape de um `Tensor`: quantidade de elementos em cada dimensão.
  * Wrapper sobre `Array[Int]` — construtor privado ao projeto, só `Shape.apply`
  * (ou métodos que já devolvem `Shape`, como `updated`/`leftPad`) criam instâncias.
  */
final class Shape private[scalagrad] (private val values: Array[Int]):
  val rank: Int = values.length
  val size: Int = values.product

  def apply(i: Int): Int = values(i)
  def indices: Range = values.indices
  def isDefinedAt(i: Int): Boolean = values.isDefinedAt(i)
  def mkString(sep: String): String = values.mkString(sep)
  def toArray: Array[Int] = values.clone()
  def toList: List[Int] = values.toList

  def head: Int = values(0)
  def last: Int = values(rank - 1)
  def updated(i: Int, value: Int): Shape = Shape(values.updated(i, value))
  def leftPad(n: Int, value: Int): Shape = Shape(Array.fill(n)(value) ++ values)
  def zip(other: Shape): Array[(Int, Int)] = values.zip(other.values)
  def crop(dim: Int): Shape = Shape(values.take(dim) ++ values.drop(dim + 1))

  /** Índice linear do "grupo" ao qual `multiIdx` pertence quando a dimensão
    * `dim` é colapsada -- usado por reduções ao longo de um eixo (`sum(dim)`/
    * `mean(dim)`, Etapa 3) e por operações que precisam mapear cada posição
    * de volta pra fatia que a originou (`softmax`, Etapa 6). Equivalente a
    * indexar em `crop(dim)` (dimensão removida) ou em `updated(dim, 1)`
    * (dimensão zerada, mantendo o rank) -- os dois dão o mesmo valor, já que
    * reduzir uma dimensão pra tamanho 1 não muda as strides relativas das
    * demais (a stride daquela dimensão fica irrelevante, sempre multiplicada
    * por índice 0).
    *
    * Calculado por Horner sobre as dimensões mantidas, sem construir o shape
    * colapsado nem o multi-índice reduzido: este método roda uma vez por
    * elemento dentro de `softmax` e `reduceDim`, e antes alocava um `Shape`
    * descartável em cada chamada.
    */
  def groupIndex(multiIdx: Array[Int], dim: Int): Int =
    indices.foldLeft(0)((acc, i) => if i == dim then acc else acc * values(i) + multiIdx(i))

  /** `this` alinhado ao rank `targetRank`, preenchendo `value` à esquerda se
    * `this` for mais curto. Não faz nada se `this` já tiver rank igual ou maior.
    */
  def padTo(targetRank: Int, value: Int): Shape = leftPad(Math.max(0, targetRank - rank), value)

  /** Strides canônicas (row-major, contíguas) para este shape. Calculada uma
    * única vez por instância — `Shape` é imutável, cachear é sempre seguro.
    */
  lazy val canonicalStrides: Strides =
    if values.isEmpty then Strides(Array.empty[Int]) else Strides(values.tail.scanRight(1)(_ * _))

  /** Índice linear (row-major) de um multi-índice contra `strides` — que não
    * precisam ser as canônicas: `Tensor.index` passa as strides *reais*
    * daquele tensor (podem não ser canônicas após `transpose`/`broadcastTo`),
    * enquanto `Shape.index` sempre passa `canonicalStrides`. A validação
    * (contagem de dimensões, limites de cada eixo) é a mesma nos dois casos.
    */
  def linearIndex(dims: Seq[Int], strides: Strides): Int =
    require(
      dims.length == rank,
      s"All dimensions must be provided. Got ${dims.length} out of $rank."
    )

    val offending = dims.indices.find(i => dims(i) < 0 || dims(i) >= values(i))
    require(
      offending.isEmpty,
      offending.fold("") { i =>
        s"Index ${dims(i)} is out of bounds for dimension $i of shape ${mkString("x")}, " +
          s"which has size ${values(i)}."
      }
    )

    dims.indices.foldLeft(0)((acc, i) => acc + dims(i) * strides(i))

  /** Índice linear assumindo strides canônicas (ver `linearIndex`). */
  def index(dims: Int*): Int = linearIndex(dims, canonicalStrides)

  /** Inverso de um índice linear (row-major) num multi-índice, um valor por
    * dimensão. Não depende de nenhuma instância de `Tensor` — é só função do
    * shape em si (via `canonicalStrides`), por isso mora aqui e não lá.
    */
  def unravelIndex(idx: Int): Array[Int] =
    require(
      idx >= 0 && idx <= size - 1,
      s"Index must be between 0 and ${size - 1}."
    )

    val strides = canonicalStrides

    Array.tabulate(rank)(i => (idx / strides(i)) % values(i))

  override def toString: String = values.mkString("Shape(", ", ", ")")
end Shape

object Shape:
  def apply(dims: Int*): Shape = Shape(dims.toArray)
  def apply(dims: Array[Int]): Shape = new Shape(dims)

  /** Alinha dois shapes pela direita (padding de `1`s à esquerda) e valida a
    * regra de broadcasting: cada par de dimensões deve ser igual, ou um dos
    * dois deve ser `1`. Retorna o shape resultante (rank = max dos dois ranks).
    */
  def broadcast(shape1: Shape, shape2: Shape): Shape =
    val shape1B = shape1.padTo(shape2.rank, 1)
    val shape2B = shape2.padTo(shape1.rank, 1)
    val zipped = shape1B.zip(shape2B)

    require(
      zipped.forall((a, b) => a == b || a == 1 || b == 1),
      s"Shapes $shape1 and $shape2 cannot be broadcasted."
    )

    Shape(zipped.map((a, b) => Math.max(a, b)))
