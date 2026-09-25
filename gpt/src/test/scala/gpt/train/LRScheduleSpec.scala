package gpt.train

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class LRScheduleSpec extends AnyFlatSpec with Matchers:

  private val lrMax = 3e-4
  private val lrMin = 3e-5
  private val total = 1000
  private val warmup = 100

  private def lr(step: Int): Double = LRSchedule.cosine(step, total, lrMax, lrMin, warmup)

  "the warmup" should "ramp linearly from lrMax/warmupSteps up to lrMax" in {
    lr(1) shouldBe 3.0e-6 +- 1e-12
    lr(25) shouldBe 7.5e-5 +- 1e-12
    lr(50) shouldBe 1.5e-4 +- 1e-12
    lr(100) shouldBe lrMax +- 1e-12
  }

  it should "never start at zero, which would make the first step do nothing" in {
    lr(1) should be > 0.0
  }

  it should "be skipped entirely when warmupSteps is 0" in {
    LRSchedule.cosine(1, total, lrMax, lrMin, warmupSteps = 0) shouldBe lrMax +- 1e-8
  }

  "the cosine decay" should "pass through the midpoint halfway between warmup and the end" in {
    // no meio do cosseno o valor e a media aritmetica dos extremos
    lr(550) shouldBe (lrMax + lrMin) / 2 +- 1e-12
  }

  it should "land exactly on lrMin at the last step" in {
    lr(total) shouldBe lrMin +- 1e-12
  }

  it should "stay at lrMin past totalSteps instead of turning around" in {
    // sem o clamp, o cosseno voltaria a subir e o treino andaria para tras
    lr(total + 200) shouldBe lrMin +- 1e-12
    lr(total * 3) shouldBe lrMin +- 1e-12
  }

  it should "decrease monotonically after the warmup" in {
    val valores = (warmup to total).map(lr)

    valores.zip(valores.tail).foreach { (atual, proximo) =>
      proximo should be <= atual
    }
  }

  it should "match the values computed by hand" in {
    lr(200) shouldBe 2.918585e-4 +- 1e-10
    lr(775) shouldBe 6.954058e-5 +- 1e-10
  }

  "the schedule" should "reject a step below 1, since t is 1-based" in {
    an[IllegalArgumentException] should be thrownBy LRSchedule.cosine(0, total)
  }

  it should "reject a warmup that covers the whole run" in {
    an[IllegalArgumentException] should be thrownBy
      LRSchedule.cosine(1, total, warmupSteps = total)
  }

  it should "reject lrMax below lrMin" in {
    an[IllegalArgumentException] should be thrownBy
      LRSchedule.cosine(1, total, lrMax = 1e-5, lrMin = 1e-4)
  }
end LRScheduleSpec
