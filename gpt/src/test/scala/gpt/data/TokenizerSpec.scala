package gpt.data

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TokenizerSpec extends AnyFlatSpec with Matchers {

  // Exemplo verificado em theory/07-tokenization/07-tokenization.md §2-3.
  private val corpus = "abacate"
  private val tokenizer = Tokenizer.charLevel(corpus)

  "Tokenizer.charLevel" should "assign vocabSize equal to the number of distinct characters" in {
    tokenizer.vocabSize shouldBe 5 // a, b, c, e, t
  }

  it should "collapse repeated characters to a single vocabulary entry" in {
    Tokenizer.charLevel("aaaa").vocabSize shouldBe 1
  }

  it should "assign indices in alphabetical order, not order of appearance" in {
    // "abacate" apresenta 't' antes de 'e' na ordem de aparição, mas 'e' < 't' alfabeticamente.
    tokenizer.encode("a").toList shouldBe List(0)
    tokenizer.encode("b").toList shouldBe List(1)
    tokenizer.encode("c").toList shouldBe List(2)
    tokenizer.encode("e").toList shouldBe List(3)
    tokenizer.encode("t").toList shouldBe List(4)
  }

  "encode" should "match the verified numeric example from theory/07-tokenization" in {
    tokenizer.encode("abacate").toList shouldBe List(0, 1, 0, 2, 0, 4, 3)
  }

  it should "map an empty string to an empty array" in {
    tokenizer.encode("").toList shouldBe List.empty
  }

  it should "throw NoSuchElementException for a character outside the vocabulary" in {
    val exception = intercept[NoSuchElementException] {
      tokenizer.encode("z")
    }
    exception.getMessage should include("z")
  }

  it should "report the correct position when the unknown character is not the first one" in {
    val exception = intercept[NoSuchElementException] {
      tokenizer.encode("cat!")
    }
    exception.getMessage should include("3")
  }

  "decode" should "match the verified numeric example from theory/07-tokenization" in {
    tokenizer.decode(Array(0, 1, 0, 2, 0, 4, 3)) shouldBe "abacate"
  }

  it should "map an empty array to an empty string" in {
    tokenizer.decode(Array.empty[Int]) shouldBe ""
  }

  it should "throw NoSuchElementException for an index at or beyond vocabSize" in {
    val exception = intercept[NoSuchElementException] {
      tokenizer.decode(Array(tokenizer.vocabSize))
    }
    exception.getMessage should include(tokenizer.vocabSize.toString)
  }

  it should "throw NoSuchElementException for a negative index" in {
    intercept[NoSuchElementException] {
      tokenizer.decode(Array(-1))
    }
  }

  "decode(encode(text))" should "round-trip for samples drawn from the corpus" in {
    for (sample <- Seq("abacate", "cat", "tea", "a", "bacatea")) {
      tokenizer.decode(tokenizer.encode(sample)) shouldBe sample
    }
  }

  "alphabet" should "hold exactly the characters the tokenizer knows" in {
    val t = Tokenizer.charLevel("abacate")

    t.alphabet shouldBe Set('a', 'b', 'c', 'e', 't')
    t.alphabet.size shouldBe t.vocabSize
  }

  it should "be the filter that keeps `encode` from throwing on outside text" in {
    // a CLI recebe texto digitado por gente, e `encode` lanca em desconhecido
    val t = Tokenizer.charLevel("abacate")
    val digitado = "abaca-te!"

    an[NoSuchElementException] should be thrownBy t.encode(digitado)
    noException should be thrownBy t.encode(digitado.filter(t.alphabet.contains))
  }
}
