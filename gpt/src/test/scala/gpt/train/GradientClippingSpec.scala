package gpt.train

import scalagrad.core.Tensor
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class GradientClippingSpec extends AnyFlatSpec with Matchers:

  private def parametro(gradiente: Double*): Tensor =
    val t = Tensor.make(Array.fill(gradiente.length)(0.0), Array(gradiente.length), true)
    gradiente.indices.foreach(i => t.gradient.accumulate(i, gradiente(i)))
    t

  "globalNorm" should "span every parameter, not one at a time" in {
    val a = parametro(3.0, -4.0)
    val b = parametro(0.0, 2.0, -1.0)

    // sqrt(9 + 16 + 0 + 4 + 1) = sqrt(30)
    GradientClipping.globalNorm(List(a, b)) shouldBe Math.sqrt(30.0) +- 1e-12
  }

  it should "be zero when every gradient is zero" in {
    GradientClipping.globalNorm(List(parametro(0.0, 0.0))) shouldBe 0.0
  }

  "clipByGlobalNorm" should "bring the norm down to maxNorm when it is exceeded" in {
    val a = parametro(3.0, -4.0)
    val b = parametro(0.0, 2.0, -1.0)

    GradientClipping.clipByGlobalNorm(List(a, b), 1.0)

    GradientClipping.globalNorm(List(a, b)) shouldBe 1.0 +- 1e-12
  }

  it should "return the norm from before the clip, which is the diagnostic" in {
    val p = parametro(3.0, -4.0)

    GradientClipping.clipByGlobalNorm(List(p), 1.0) shouldBe 5.0 +- 1e-12
  }

  it should "rescale every gradient by the same factor, preserving the direction" in {
    val a = parametro(3.0, -4.0)
    val b = parametro(0.0, 2.0, -1.0)

    GradientClipping.clipByGlobalNorm(List(a, b), 1.0)

    val fator = 1.0 / Math.sqrt(30.0)
    a.gradient.toArray shouldBe Array(3.0 * fator, -4.0 * fator)
    b.gradient.toArray shouldBe Array(0.0, 2.0 * fator, -1.0 * fator)
  }

  it should "leave the gradients untouched when the norm is within maxNorm" in {
    val p = parametro(0.3, -0.4)

    val norma = GradientClipping.clipByGlobalNorm(List(p), 1.0)

    norma shouldBe 0.5 +- 1e-12
    p.gradient.toArray shouldBe Array(0.3, -0.4)
  }

  it should "not divide by zero when every gradient is zero" in {
    val p = parametro(0.0, 0.0)

    GradientClipping.clipByGlobalNorm(List(p), 1.0) shouldBe 0.0
    p.gradient.toArray shouldBe Array(0.0, 0.0)
  }

  it should "reject a maxNorm that is not positive" in {
    val p = parametro(1.0)

    an[IllegalArgumentException] should be thrownBy
      GradientClipping.clipByGlobalNorm(List(p), 0.0)
  }

  "Gradient.scale" should "rescale without losing what was accumulated" in {
    val p = parametro(2.0, -6.0)

    p.gradient.scale(0.5)

    p.gradient.toArray shouldBe Array(1.0, -3.0)
  }
end GradientClippingSpec
