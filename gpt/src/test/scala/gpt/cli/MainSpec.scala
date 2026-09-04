package gpt.cli

import gpt.generate.{Greedy, Temperature, TopK}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MainSpec extends AnyFlatSpec with Matchers {

  "the argument parser" should "fall back to the default when the flag is absent" in {
    intArg(Seq("--steps", "10"), "--batch", 8) shouldBe 8
    doubleArg(Seq.empty, "--lr", 1e-3) shouldBe 1e-3
    stringArg(Seq("--outro", "x"), "--corpus", "padrao.txt") shouldBe "padrao.txt"
  }

  it should "read the value that follows the flag" in {
    intArg(Seq("--steps", "500", "--batch", "4"), "--steps", 3000) shouldBe 500
    intArg(Seq("--steps", "500", "--batch", "4"), "--batch", 8) shouldBe 4
    doubleArg(Seq("--lr", "0.01"), "--lr", 1e-3) shouldBe 0.01
    stringArg(Seq("--corpus", "livro.txt"), "--corpus", "padrao.txt") shouldBe "livro.txt"
  }

  it should "find a flag in any position" in {
    val args = Seq("--corpus", "a.txt", "--steps", "7", "--checkpoint", "b.bin")

    intArg(args, "--steps", 0) shouldBe 7
    stringArg(args, "--checkpoint", "") shouldBe "b.bin"
  }

  "the sampling state" should "become Greedy when the temperature is effectively zero" in {
    ChatState(temperature = 0.0).strategy shouldBe Greedy
    ChatState(temperature = 0.005).strategy shouldBe Greedy
  }

  it should "use TopK while k is positive" in {
    ChatState(temperature = 0.8, topK = 20).strategy shouldBe TopK(20, 0.8)
  }

  it should "fall back to plain temperature when the cut is disabled" in {
    ChatState(temperature = 1.1, topK = 0).strategy shouldBe Temperature(1.1)
  }

  it should "describe itself for the prompt line" in {
    ChatState(0.8, 20, 200).summary should include("top-k 20")
    ChatState(0.0, 20, 200).summary should include("greedy")
    ChatState(1.5, 0, 50).summary should (include("T = 1,50").or(include("T = 1.50")))
  }

  "the commands" should "change one setting at a time" in {
    val initial = ChatState()

    applyCommand(":temp 1.2", initial).temperature shouldBe 1.2
    applyCommand(":topk 5", initial).topK shouldBe 5
    applyCommand(":tokens 40", initial).maxNewTokens shouldBe 40

    // os outros campos ficam intactos
    applyCommand(":temp 1.2", initial).topK shouldBe initial.topK
  }

  it should "keep the state when the value is not a number" in {
    val initial = ChatState()

    applyCommand(":temp muito", initial) shouldBe initial
    applyCommand(":topk todos", initial) shouldBe initial
  }

  it should "keep the state on an unknown command" in {
    val initial = ChatState()

    applyCommand(":voar", initial) shouldBe initial
    applyCommand(":temp", initial) shouldBe initial
  }

  it should "forget the conversation on :limpar, keeping the settings" in {
    val withHistory = ChatState(temperature = 1.1, topK = 5, history = Vector(1, 2, 3))
    val cleared = applyCommand(":limpar", withHistory)

    cleared.history shouldBe empty
    cleared.temperature shouldBe 1.1
    cleared.topK shouldBe 5
  }

  it should "leave the history alone when changing a sampling setting" in {
    val withHistory = ChatState(history = Vector(4, 5))

    applyCommand(":temp 0.5", withHistory).history shouldBe Vector(4, 5)
  }
}
