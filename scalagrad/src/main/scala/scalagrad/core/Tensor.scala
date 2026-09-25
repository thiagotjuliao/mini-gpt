package scalagrad.core

import scala.util.Random
import scala.util.DynamicVariable
import scalagrad.ops.tensor
import scala.compiletime.ops.boolean

final class Tensor private[scalagrad] (
    private[scalagrad] val data: Array[Double],
    val shape: Shape,
    private[scalagrad] val strides: Strides,
    val gradient: Gradient,
    val requiresGradient: Boolean = false,
    private[core] val previous: Set[Tensor] = Set()
)(private[core] val backwardStep: () => Unit = () => ()):
  private val maxReachableIndex: Int =
    shape.indices.foldLeft(0)((acc, i) => acc + (shape(i) - 1) * strides(i))

  assert(
    maxReachableIndex < data.length,
    s"Shape ${shape.mkString("x")} with strides ${strides.mkString("x")} reaches index $maxReachableIndex, " +
      s"but data has only ${data.length} element(s)."
  )

  val rank: Int = shape.rank
  val size: Int = shape.size

  /** Comparação por valor, não por referência: um tensor transposto duas vezes
    * tem strides canônicas de fato, e precisa ser reconhecido como contíguo.
    */
  lazy val isContiguous: Boolean =
    shape.indices.forall(i => strides(i) == shape.canonicalStrides(i))

  def get(dims: Int*): Double = data(index(dims*))

  /** Índice linear pras strides *reais* deste tensor (podem não ser
    * canônicas, ex.: depois de `transpose`/`broadcastTo`).
    */
  def index(dims: Int*): Int = shape.linearIndex(dims, strides)

  def unravelIndex(idx: Int): Array[Int] = shape.unravelIndex(idx)

  def reshape(newShape: Array[Int]): Tensor =
    require(
      newShape.product == size,
      "New shape must be of the same size as the current shape."
    )

    val t = this.contiguous
    val shapeT = Shape(newShape)
    val grad = Gradient.zeros(shapeT)
    val reqGrad = requiresGradient && Tensor.gradEnabled
    val prev = if Tensor.gradEnabled then Set(this) else Set()

    Tensor(t.data, shapeT, shapeT.canonicalStrides, grad, reqGrad, prev) { () =>
      if Tensor.gradEnabled then
        (0 until size).foreach { i =>
          gradient.accumulate(i, grad(i))
        }
    }

  def reshape(newShape: Shape): Tensor = reshape(newShape.toArray)

  def transpose(dim0: Int = rank - 2, dim1: Int = rank - 1): Tensor =
    require(
      shape.isDefinedAt(dim0) && shape.isDefinedAt(dim1),
      s"Provided dimensions must be between 0 and ${rank - 1}."
    )

    val d0 = shape(dim0); val d1 = shape(dim1)
    val newShape = shape.updated(dim0, d1).updated(dim1, d0)

    val s0 = strides(dim0); val s1 = strides(dim1)
    val newStrides = strides.updated(dim0, s1).updated(dim1, s0)

    val grad = Gradient.zeros(newShape)
    val reqGrad = requiresGradient && Tensor.gradEnabled
    val prev = if Tensor.gradEnabled then Set(this) else Set()

    Tensor(data, newShape, newStrides, grad, reqGrad, prev) { () =>
      if Tensor.gradEnabled then
        (0 until size).foreach { i =>
          val outIdx = newShape.unravelIndex(i)
          val inIdx = outIdx.updated(dim0, outIdx(dim1)).updated(dim1, outIdx(dim0))
          gradient.accumulate(shape.index(inIdx*), grad(i))
        }
    }

  def contiguous: Tensor =
    if isContiguous then this
    else
      // `mapping` traduz índice canônico -> posição física, e só vale pra
      // `data`. O `gradient` já é canônico (todas as ops acumulam por
      // `shape.index`), então ele é copiado sem permutar -- aplicar o mapping
      // aqui embaralharia o gradiente de um tensor não contíguo.
      val mapping = Array.tabulate(size)(n => index(unravelIndex(n)*))
      val contData = mapping.map(i => data(i))
      val contGrad = gradient.toArray

      Tensor(
        contData,
        shape,
        shape.canonicalStrides,
        Gradient(contGrad),
        requiresGradient,
        previous
      )(
        backwardStep
      )

  def backward(): Unit =
    require(size == 1, "`backward` should be called on a Scalar root, like Loss.")

    gradient.seed()
    val nodes = Tensor.topologicalSort(this).reverse
    nodes.foreach(_.backwardStep())

  def zeroGrad(): Unit =
    val nodes = Tensor.topologicalSort(this)
    nodes.filter(_.requiresGradient).foreach(_.gradient.zero())

  /** View broadcastada de `this` para `newShape`: mesmo `data` (sem copiar),
    * com dimensões de tamanho 1 "esticadas" via stride 0. `newShape` deve ser
    * compatível com `shape` pela regra de broadcasting (ver `Shape.broadcast`).
    * Mecanismo (view de stride 0) e os dois casos de padding de rank: ver
    * theory/03-elementary-operations/03-elementary-operations.md §4.
    *
    * **Só para leitura, e por isso não é público.** A view compartilha o objeto
    * `gradient` do tensor original, que tem menos posições que o `shape`
    * anunciado. Acumular nela por índice canônico do shape novo corrompe ou
    * estoura -- foi o motivo de esta rota ser descartada no `matmul` da Etapa 11.
    */
  private[scalagrad] def broadcastTo(newShape: Shape): Tensor =
    val shapeB = Shape.broadcast(shape, newShape)
    val paddedShape = shape.padTo(shapeB.rank, 1)
    val paddedStrides = strides.padTo(shapeB.rank, 0)

    val finalShape = Shape(
      shapeB.indices
        .map(i => if shapeB(i) > paddedShape(i) then shapeB(i) else paddedShape(i))
        .toArray
    )
    val finalStrides = Strides(
      shapeB.indices.map(i => if shapeB(i) > paddedShape(i) then 0 else paddedStrides(i)).toArray
    )

    Tensor(data, finalShape, finalStrides, gradient, requiresGradient, previous)(backwardStep)

  def updateData(values: Array[Double]): Unit =
    require(
      values.length == size,
      s"The input values array size must be equal to $size, but got ${values.length}."
    )

    require(
      isContiguous,
      "This tensor must be made contiguous before updating its data."
    )

    values.indices.foreach { i =>
      data(i) = values(i)
    }

  def toArray: Array[Double] =
    if isContiguous then data.clone()
    else this.contiguous.toArray

  override def toString: String =
    def fmt(dim: Int, prefix: Array[Int]): String =
      val cells = Tensor.truncatedRange(shape(dim))

      if dim == rank - 1 then
        cells
          .map {
            case Some(i) => String.format(java.util.Locale.US, "%.4f", get(prefix :+ i*))
            case None => "..."
          }
          .mkString("[", ", ", "]")
      else
        val indent = "  " * (dim + 1)
        cells
          .map {
            case Some(i) => fmt(dim + 1, prefix :+ i)
            case None => "..."
          }
          .mkString(s"[\n$indent", s",\n$indent", s"\n${"  " * dim}]")

    s"Tensor(shape=${shape.mkString("x")})\n${fmt(0, Array.empty)}"
end Tensor

object Tensor:
  private val edgeItems = 3
  private val summarizeThreshold = 2 * edgeItems + 1

  /** `DynamicVariable` em vez de `var` simples: cada thread enxerga seu
    * próprio valor (herdado pela thread que a criou), então `noGrad` rodando
    * numa thread nunca some com o `gradEnabled` de outra rodando em paralelo
    * -- diferente de um `var` compartilhado, onde duas threads alternando
    * `noGrad` concorrentemente podem fazer uma "roubar" o toggle da outra.
    */
  private val gradEnabledVar: DynamicVariable[Boolean] = new DynamicVariable(true)
  private[scalagrad] def gradEnabled: Boolean = gradEnabledVar.value

  private def truncatedRange(n: Int): Seq[Option[Int]] =
    if n <= summarizeThreshold then (0 until n).map(Some(_))
    else (0 until edgeItems).map(Some(_)) ++ Seq(None) ++ (n - edgeItems until n).map(Some(_))

  /** Box-Muller: transforma pares de amostras uniformes em pares de amostras
    * N(0,1). Derivação (coordenadas polares, por que R²/θ têm essas
    * distribuições): ver theory/01-tensor/01-tensor.md §4.
    */
  private def getGaussianSamples(rng: Random): LazyList[Double] =
    val u1 = 1.0 - rng.nextDouble()
    val u2 = rng.nextDouble()

    val r = Math.sqrt(-2 * Math.log(u1))
    val t = 2 * Math.PI * u2

    val z1 = r * Math.cos(t)
    val z2 = r * Math.sin(t)

    z1 #:: z2 #:: getGaussianSamples(rng)

  /** Ordem topológica reversa a partir de `root`, via DFS com pilha explícita
    * (stack-safe pra grafos profundos). Por que essa ordem é necessária pro
    * backward, e o caso do grafo diamante com nó compartilhado não-folha: ver
    * theory/02-autograd/02-autograd.md §4.
    */
  private def topologicalSort(root: Tensor): List[Tensor] =
    @scala.annotation.tailrec
    def visit(
        tensors: List[Tensor],
        visited: Set[Tensor],
        added: Set[Tensor],
        sorted: Vector[Tensor]
    ): List[Tensor] =
      if tensors.isEmpty then sorted.toList
      else
        val tensor = tensors.head
        val parents = tensor.previous.toList

        if parents.forall(visited.contains(_)) then
          if !added.contains(tensor) then
            visit(tensors.tail, visited + tensor, added + tensor, sorted :+ tensor)
          else visit(tensors.tail, visited + tensor, added, sorted)
        else visit(parents ::: tensors, visited + tensor, added, sorted)

    visit(List(root), Set(), Set(), Vector())

  def make(data: Array[Double], shape: Array[Int], requiresGradient: Boolean = false): Tensor =
    val shapeT = Shape(shape)
    val grad = Gradient.zeros(shapeT)

    Tensor(data, shapeT, shapeT.canonicalStrides, grad, requiresGradient)()

  def zeros(shape: Array[Int], requiresGradient: Boolean = false): Tensor =
    fill(shape, 0.0, requiresGradient)

  def ones(shape: Array[Int], requiresGradient: Boolean = false): Tensor =
    fill(shape, 1.0, requiresGradient)

  def fill(shape: Array[Int], value: Double, requiresGradient: Boolean = false): Tensor =
    val data = Array.fill(shape.product)(value)
    Tensor.make(data, shape, requiresGradient)

  /** `rng` injetável pelo mesmo motivo do `BatchSampler`: sem isso nenhuma
    * inicialização de modelo é reproduzível, e a partir da Etapa 18 "rodei de
    * novo e deu diferente" fica indistinguível de "mudei alguma coisa".
    */
  def randn(
      shape: Array[Int],
      std: Double = 1.0,
      requiresGradient: Boolean = false,
      rng: Random = Random
  ): Tensor =
    val data = getGaussianSamples(rng).take(shape.product).map(_ * std).toArray
    Tensor.make(data, shape, requiresGradient)

  def arange(n: Int): Tensor =
    val data = Array.tabulate(n)(_.toDouble)
    Tensor.make(data, Array(n))

  def noGrad[T](block: => T): T = gradEnabledVar.withValue(false)(block)

  def oneHot(indices: Array[Int], numClasses: Int): Tensor =
    require(
      numClasses >= 1,
      s"A one-hot row needs at least 1 class, but got $numClasses."
    )

    // Sem esta checagem, um indice fora da faixa nao acende coluna nenhuma e
    // devolve uma linha toda zero -- em silencio. Mesmo cuidado do
    // `indexSelect`, que valida do mesmo jeito.
    (0 until indices.length).foreach { i =>
      require(
        indices(i) >= 0 && indices(i) < numClasses,
        s"Index ${indices(i)} at position $i is out of bounds for $numClasses classes."
      )
    }

    val shape = Array(indices.length, numClasses)

    val data = {
      for
        i <- 0 until indices.length
        j <- 0 until numClasses
      yield if j == indices(i) then 1.0 else 0.0
    }.toArray

    Tensor.make(data, shape)
  end oneHot
end Tensor
