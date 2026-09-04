package scalagrad.ops

import scalagrad.core.{Shape, Gradient}
import scalagrad.ops.Broadcast.unbroadcast
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class BroadcastOpsSpec extends AnyFlatSpec with Matchers {

  private def onesGradient(n: Int): Gradient = {
    val g = Gradient.zeros(n)
    (0 until n).foreach(i => g.accumulate(i, 1.0))
    g
  }

  "unbroadcast" should "sum a broadcasted dimension away entirely when it did not exist in the original shape" in {
    // M(3,4) + v(4,): v foi broadcastado ao longo da dimensão nova (linhas).
    // dC tudo 1 -> v.grad[j] = soma da coluna j = 3, pra cada uma das 4 colunas.
    val dC = onesGradient(12)

    val dv = unbroadcast(dC, Shape(3, 4), Shape(4))

    dv.toList shouldBe List(3.0, 3.0, 3.0, 3.0)
  }

  it should "sum along a dimension that already existed with size 1, keeping the dimension" in {
    // M(3,4) + b(3,1): b foi broadcastado ao longo das colunas.
    // dC tudo 1 -> b.grad[i] = soma da linha i = 4, pra cada uma das 3 linhas.
    val dC = onesGradient(12)

    val db = unbroadcast(dC, Shape(3, 4), Shape(3, 1))

    db.toList shouldBe List(4.0, 4.0, 4.0)
  }

  it should "act as identity when there was no broadcasting at all" in {
    val dC = Gradient.zeros(6)
    (0 until 6).foreach(i => dC.accumulate(i, (i + 1).toDouble))

    val result = unbroadcast(dC, Shape(2, 3), Shape(2, 3))

    result.toList shouldBe List(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
  }

  it should "sum a fully broadcasted scalar back down to a single value" in {
    // shape () conceitual representado aqui como Shape(1): todo o gradiente cai numa posição só.
    val dC = onesGradient(6)

    val result = unbroadcast(dC, Shape(2, 3), Shape(1))

    result.toList shouldBe List(6.0)
  }
}
