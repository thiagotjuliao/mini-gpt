package scalagrad.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class GradientSpec extends AnyFlatSpec with Matchers:

  "Gradient.zeros" should "create a gradient with every position at 0.0" in {
    val g = Gradient.zeros(3)

    g.length shouldBe 3
    g.toList shouldBe List(0.0, 0.0, 0.0)
  }

  "accumulate" should "add to the existing value instead of overwriting it" in {
    val g = Gradient.zeros(2)

    g.accumulate(0, 5.0)
    g.accumulate(0, 3.0) // segunda contribuição no mesmo índice: deve somar, não sobrescrever
    g.accumulate(1, 1.0)

    g(0) shouldBe 8.0
    g(1) shouldBe 1.0
  }

  it should "accept negative deltas (used to express subtraction as accumulate(-delta))" in {
    val g = Gradient.zeros(1)

    g.accumulate(0, 5.0)
    g.accumulate(0, -2.0)

    g(0) shouldBe 3.0
  }

  "zero" should "reset every position back to 0.0" in {
    val g = Gradient.zeros(2)
    g.accumulate(0, 4.0)
    g.accumulate(1, -1.0)

    g.zero()

    g.toList shouldBe List(0.0, 0.0)
  }

  "seed" should "set every position to 1.0" in {
    val g = Gradient.zeros(3)
    g.accumulate(0, 4.0)

    g.seed()

    g.toList shouldBe List(1.0, 1.0, 1.0)
  }

  "toString" should "render a readable representation" in {
    val g = Gradient.zeros(2)
    g.accumulate(0, 1.5)

    g.toString shouldBe "Gradient(1.5, 0.0)"
  }

  "scale" should "multiply every position by the factor" in {
    val g = Gradient.zeros(3)
    g.accumulate(0, 2.0)
    g.accumulate(1, -6.0)
    g.accumulate(2, 0.5)

    g.scale(0.5)

    g.toList shouldBe List(1.0, -3.0, 0.25)
  }

  it should "preserve what was accumulated, unlike an overwrite" in {
    // reescalar nao e sobrescrever: as contribuicoes de varios caminhos do
    // grafo continuam la, so menores. E o que o gradient clipping precisa.
    val g = Gradient.zeros(1)
    g.accumulate(0, 3.0)
    g.accumulate(0, 1.0)

    g.scale(2.0)

    g(0) shouldBe 8.0
  }

  it should "zero everything when the factor is zero" in {
    val g = Gradient.zeros(2)
    g.accumulate(0, 5.0)

    g.scale(0.0)

    g.toList shouldBe List(0.0, 0.0)
  }
end GradientSpec
