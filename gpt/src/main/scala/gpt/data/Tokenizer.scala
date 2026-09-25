package gpt.data

import java.io.File
import scala.io.Source
import scala.util.Using

private[gpt] final class Tokenizer private (charToIdx: Map[Char, Int]):
  private val idxToChar = charToIdx.map((k, v) => (v, k))
  val vocabSize: Int = charToIdx.size

  /** Os caracteres que este tokenizador conhece. Quem recebe texto de fora
    * (a CLI) filtra por aqui antes de chamar `encode`, que lanca em desconhecido.
    */
  val alphabet: Set[Char] = charToIdx.keySet

  def encode(input: String): Array[Int] =
    input.toArray.zipWithIndex.map { case (c, i) =>
      charToIdx.getOrElse(
        c,
        throw new NoSuchElementException(
          s"Char `$c` @ index $i cannot be found in the current vocabulary."
        )
      )
    }

  def decode(input: Array[Int]): String =
    input.map { i =>
      idxToChar.getOrElse(
        i,
        throw new NoSuchElementException(
          s"Trying to get index $i on a vocabulary size of $vocabSize."
        )
      )
    }.mkString
end Tokenizer

object Tokenizer:
  def charLevel(corpus: String): Tokenizer =
    new Tokenizer(corpus.distinct.sorted.zipWithIndex.toMap)

  def charLevel(file: File): Tokenizer =
    charLevel(Using.resource(Source.fromFile(file))(_.mkString))
