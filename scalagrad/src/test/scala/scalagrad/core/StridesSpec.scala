package scalagrad.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class StridesSpec extends AnyFlatSpec with Matchers:

  "Strides" should "expose length and per-position access via apply" in {
    val st = Strides(3, 1)

    st.length shouldBe 2
    st(0) shouldBe 3
    st(1) shouldBe 1
  }

  it should "expose its values via toArray" in {
    Strides(3, 1).toArray.toList shouldBe List(3, 1)
  }

  it should "render a readable toString" in {
    Strides(3, 1).toString shouldBe "Strides(3, 1)"
  }

  "updated" should "return a new Strides with a single position changed, without mutating the original" in {
    val st = Strides(3, 1)
    val u = st.updated(0, 0)

    u.toArray.toList shouldBe List(0, 1)
    st.toArray.toList shouldBe List(3, 1) // original inalterado
  }

  "leftPad" should "prepend the given value n times" in {
    Strides(1).leftPad(1, 0).toArray.toList shouldBe List(0, 1)
    Strides(3, 1).leftPad(0, 0).toArray.toList shouldBe List(3, 1)
  }
end StridesSpec
