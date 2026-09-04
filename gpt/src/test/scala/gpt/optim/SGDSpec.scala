package gpt.optim

import scalagrad.core.Tensor
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SGDSpec extends AnyFlatSpec with Matchers {

  private val tolerance = 1e-12
  private val lr = 3e-4

  private def parametro(valores: Double*): Tensor =
    Tensor.make(valores.toArray, Array(valores.length), true)

  private def semear(p: Tensor, gradiente: Double*): Unit = {
    p.gradient.zero()
    gradiente.indices.foreach(i => p.gradient.accumulate(i, gradiente(i)))
  }

  "SGD" should "move each parameter by lr times its own gradient" in {
    val p = parametro(0.60, -0.20, 0.05)
    semear(p, 0.30, 0.03, -0.60)

    new SGD(List(p), lr).step()

    p.toArray(0) shouldBe 0.60 - 9.0e-5 +- tolerance
    p.toArray(1) shouldBe -0.20 - 9.0e-6 +- tolerance
    p.toArray(2) shouldBe 0.05 + 1.8e-4 +- tolerance
  }

  it should "keep the step proportional to the gradient" in {
    // e exatamente essa proporcionalidade que o AdamW abandona: aqui o
    // gradiente vinte vezes menor anda vinte vezes menos.
    val p = parametro(0.0, 0.0)
    semear(p, 0.60, 0.03)

    new SGD(List(p), lr).step()

    val razao = Math.abs(p.toArray(0)) / Math.abs(p.toArray(1))
    razao shouldBe 20.0 +- 1e-9
  }

  it should "descend, not ascend, the loss" in {
    val p = parametro(1.0)
    semear(p, 0.5)

    new SGD(List(p), lr).step()

    p.toArray(0) should be < 1.0
  }

  it should "use the overridden lr when given one" in {
    val p = parametro(1.0)
    semear(p, 1.0)

    new SGD(List(p), lr).step(0.1)

    p.toArray(0) shouldBe 0.9 +- tolerance
  }

  it should "have no state to carry between steps" in {
    val p = parametro(1.0)
    val otimizador = new SGD(List(p), lr = 0.1)

    semear(p, 1.0)
    otimizador.step()
    semear(p, 1.0)
    otimizador.step()

    // dois passos identicos, porque nao ha momento acumulado
    p.toArray(0) shouldBe 0.8 +- tolerance
  }

  "zeroGrad" should "zero the gradient of every parameter" in {
    val a = parametro(1.0, 2.0)
    val b = parametro(3.0)
    semear(a, 0.5, -0.5)
    semear(b, 0.25)

    new SGD(List(a, b)).zeroGrad()

    a.gradient.toArray shouldBe Array(0.0, 0.0)
    b.gradient.toArray shouldBe Array(0.0)
  }

  "the constructor" should "reject an empty parameter list" in {
    an[IllegalArgumentException] should be thrownBy new SGD(List.empty)
  }

  it should "reject a parameter that does not require gradient" in {
    an[IllegalArgumentException] should be thrownBy
      new SGD(List(Tensor.make(Array(1.0), Array(1))))
  }
}
