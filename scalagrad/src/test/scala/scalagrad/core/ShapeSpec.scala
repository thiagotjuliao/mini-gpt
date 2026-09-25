package scalagrad.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ShapeSpec extends AnyFlatSpec with Matchers:

  "Shape" should "expose size, rank and per-dimension access via apply" in {
    val s = Shape(2, 3, 4)

    s.rank shouldBe 3
    s.size shouldBe 24
    s(0) shouldBe 2
    s(1) shouldBe 3
    s(2) shouldBe 4
  }

  it should "report valid indices via isDefinedAt" in {
    val s = Shape(2, 3)

    s.isDefinedAt(0) shouldBe true
    s.isDefinedAt(1) shouldBe true
    s.isDefinedAt(2) shouldBe false
    s.isDefinedAt(-1) shouldBe false
  }

  it should "expose its dimensions as toArray/toList" in {
    val s = Shape(2, 3)

    s.toArray.toList shouldBe List(2, 3)
    s.toList shouldBe List(2, 3)
  }

  it should "render a readable toString" in {
    Shape(2, 3).toString shouldBe "Shape(2, 3)"
  }

  "updated" should "return a new Shape with a single dimension changed, without mutating the original" in {
    val s = Shape(2, 3)
    val u = s.updated(1, 5)

    u.toList shouldBe List(2, 5)
    s.toList shouldBe List(2, 3) // original inalterado
  }

  "leftPad" should "prepend the given value n times" in {
    Shape(3).leftPad(1, 1).toList shouldBe List(1, 3)
    Shape(2, 3).leftPad(2, 1).toList shouldBe List(1, 1, 2, 3)
    Shape(2, 3).leftPad(0, 1).toList shouldBe List(2, 3)
  }

  "zip" should "pair dimensions position by position" in {
    Shape(2, 3).zip(Shape(10, 20)).toList shouldBe List((2, 10), (3, 20))
  }

  "canonicalStrides" should "compute row-major strides for a multi-dimensional shape" in {
    Shape(2, 3, 4).canonicalStrides.toArray.toList shouldBe List(12, 4, 1)
  }

  it should "be 1 for a rank-1 shape" in {
    Shape(5).canonicalStrides.toArray.toList shouldBe List(1)
  }

  it should "be empty for a rank-0 shape, instead of throwing on an already-empty array" in {
    // regressao: values.tail estourava UnsupportedOperationException quando values
    // ja vinha vazio (rank 0) -- caso real, produzido por Shape.crop(dim) ao colapsar
    // a unica dimensao de um shape rank 1 (ex.: softmax(dim=0) sobre um vetor)
    Shape(Array.empty[Int]).canonicalStrides.toArray.toList shouldBe List.empty[Int]
  }

  "unravelIndex" should "be the inverse of index for every linear position" in {
    val s = Shape(2, 3, 4)

    for n <- 0 until s.size do
      val multiIdx = s.unravelIndex(n)
      s.index(multiIdx*) shouldBe n
  }

  "index" should "compute the correct linear index for a 2D row-major shape" in {
    val s = Shape(2, 3)

    s.index(0, 0) shouldBe 0
    s.index(0, 2) shouldBe 2
    s.index(1, 0) shouldBe 3
    s.index(1, 2) shouldBe 5
  }

  it should "reject a dimension value out of bounds for its axis" in {
    val s = Shape(2, 3)

    an[IllegalArgumentException] should be thrownBy s.index(2, 0)
  }

  it should "reject the wrong number of dimensions" in {
    val s = Shape(2, 3)

    an[IllegalArgumentException] should be thrownBy s.index(0)
  }

  "groupIndex" should "map every position in the same row to the same group when collapsing the last dimension" in {
    val s = Shape(2, 3)

    s.groupIndex(Array(0, 0), dim = 1) shouldBe 0
    s.groupIndex(Array(0, 1), dim = 1) shouldBe 0
    s.groupIndex(Array(0, 2), dim = 1) shouldBe 0
    s.groupIndex(Array(1, 0), dim = 1) shouldBe 1
    s.groupIndex(Array(1, 1), dim = 1) shouldBe 1
    s.groupIndex(Array(1, 2), dim = 1) shouldBe 1
  }

  it should "map every position in the same column to the same group when collapsing the first dimension" in {
    val s = Shape(2, 3)

    s.groupIndex(Array(0, 0), dim = 0) shouldBe 0
    s.groupIndex(Array(1, 0), dim = 0) shouldBe 0
    s.groupIndex(Array(0, 2), dim = 0) shouldBe 2
    s.groupIndex(Array(1, 2), dim = 0) shouldBe 2
  }

  it should "collapse every position of a rank-1 shape into the single group 0" in {
    // regressao: colapsar a unica dimensao de um shape rank 1 produz um shape rank 0
    // internamente (crop(0)), o caso que expos o bug de canonicalStrides acima
    val s = Shape(3)

    s.groupIndex(Array(0), dim = 0) shouldBe 0
    s.groupIndex(Array(1), dim = 0) shouldBe 0
    s.groupIndex(Array(2), dim = 0) shouldBe 0
  }

  it should "agree with indexing crop(dim) directly" in {
    val s = Shape(2, 3, 4)

    for n <- 0 until s.size do
      val multiIdx = s.unravelIndex(n)
      val collapsed = multiIdx.take(1) ++ multiIdx.drop(2) // drop dim=1
      s.groupIndex(multiIdx, dim = 1) shouldBe s.crop(1).index(collapsed*)
  }

  it should "agree with indexing the dim-zeroed, rank-preserving shape (keepDim=true equivalent)" in {
    val s = Shape(2, 3, 4)
    val keepDimShape = s.updated(1, 1)

    for n <- 0 until s.size do
      val multiIdx = s.unravelIndex(n)
      s.groupIndex(multiIdx, dim = 1) shouldBe keepDimShape.index(multiIdx.updated(1, 0)*)
  }

  "Shape.broadcast" should "align shapes of the same rank, taking the max per dimension" in {
    Shape.broadcast(Shape(2, 1), Shape(1, 3)).toList shouldBe List(2, 3)
  }

  it should "left-pad the shorter shape before aligning" in {
    Shape.broadcast(Shape(3), Shape(2, 3)).toList shouldBe List(2, 3)
  }

  it should "reject incompatible shapes" in {
    an[IllegalArgumentException] should be thrownBy Shape.broadcast(Shape(3), Shape(4))
    an[IllegalArgumentException] should be thrownBy Shape.broadcast(Shape(2, 3), Shape(4, 3))
  }
end ShapeSpec
