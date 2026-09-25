package gpt.loss

import scalagrad.core.Tensor
import scalagrad.ops.tensor.*
import scalagrad.gradcheck.Gradcheck
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class CrossEntropySpec extends AnyFlatSpec with Matchers:

  // T = 2 e V = 4 sao distintos de proposito: com T == V, o `logSoftmax` no
  // eixo das posicoes em vez do vocabulario produz numeros plausiveis e
  // formato identico (theory/16-cross-entropy §3).
  private val batchSize = 2
  private val seqLen = 2
  private val vocabSize = 4

  /** O lote do §5 do capitulo. Perdas por posicao:
    * 2.36971, 1.38629, 0.13993, 1.45693 -- media 1.33821.
    */
  private val loteValores = Array(
    2.0, 1.0, 0.1, -0.5, 0.3, 0.3, 0.3, 0.3, -1.0, 3.0, 0.0, 0.5, 1.5, -2.0, 0.7, 0.7
  )

  private val loteLogits =
    Tensor.make(loteValores.clone(), Array(batchSize, seqLen, vocabSize))

  /** Uma instancia nova por teste de gradiente: `Gradient` so acumula, entao
    * reusar o mesmo tensor entre dois `backward` leria a soma dos dois.
    */
  private def logitsComGradiente: Tensor =
    Tensor.make(loteValores.clone(), Array(batchSize, seqLen, vocabSize), true)

  private val loteAlvos = Tensor.make(Array(2.0, 0.0, 1.0, 3.0), Array(batchSize, seqLen))

  private def logits(values: Double*): Tensor =
    Tensor.make(values.toArray, Array(1, 1, values.length))

  private def alvo(index: Int): Tensor = Tensor.make(Array(index.toDouble), Array(1, 1))

  private def softmax(values: Seq[Double]): Seq[Double] =
    val m = values.max
    val e = values.map(v => Math.exp(v - m))
    e.map(_ / e.sum)

  // ---- valores conhecidos ----

  "CrossEntropy" should "match a loss computed by hand for a single position" in {
    // p = [0.62518, 0.22999, 0.09351, 0.05132], alvo 2 -> -log(0.09351)
    val loss = CrossEntropy(logits(2.0, 1.0, 0.1, -0.5), alvo(2))

    loss.size shouldBe 1
    loss.get(0) shouldBe 2.36971 +- 1e-5
  }

  it should "average over the B·T positions of a batch" in {
    // pega tanto somar em vez de mediar quanto dividir pelo numero errado
    CrossEntropy(loteLogits, loteAlvos).get(0) shouldBe 1.33821 +- 1e-5
  }

  it should "give exactly log(V) when every logit is equal" in {
    // a checagem de sanidade do §6: um modelo que nao sabe nada paga log(V)
    for v <- Seq(4, 65, 256) do
      withClue(s"V=$v: ") {
        val uniformes = Tensor.make(Array.fill(v)(0.0), Array(1, 1, v))

        CrossEntropy(uniformes, alvo(v / 2)).get(0) shouldBe Math.log(v) +- 1e-9
      }
  }

  it should "not depend on which target is chosen, when the logits are uniform" in {
    val uniformes = Tensor.make(Array.fill(vocabSize)(0.0), Array(1, 1, vocabSize))
    val perdas = (0 until vocabSize).map(i => CrossEntropy(uniformes, alvo(i)).get(0))

    perdas.distinct.size shouldBe 1
  }

  // ---- estabilidade numerica ----

  it should "be invariant to adding a constant to every logit" in {
    // o softmax nao muda com deslocamento; se a perda mudar, o log-sum-exp
    // nao esta sendo usado (§3)
    val base = CrossEntropy(logits(2.0, 1.0, 0.1, -0.5), alvo(2)).get(0)
    val deslocado = CrossEntropy(logits(1002.0, 1001.0, 1000.1, 999.5), alvo(2)).get(0)

    deslocado shouldBe base +- 1e-9
  }

  it should "survive a logit gap that underflows the probability" in {
    // softmax daria [1.0, 0.0] e log(0) = -infinito. O logSoftmax devolve
    // -800 exato, sem nunca materializar a probabilidade.
    val loss = CrossEntropy(logits(0.0, -800.0), alvo(1)).get(0)

    loss.isNaN shouldBe false
    loss.isInfinite shouldBe false
    loss shouldBe 800.0 +- 1e-6
  }

  // ---- gradiente ----

  "CrossEntropy backward" should "produce (p − y)/N on the logits" in {
    // O teste que carrega a etapa: ele confere a INTENCAO. O gradient check
    // abaixo so confere coerencia entre forward e backward, e aprovaria uma
    // formula errada desde que derivada corretamente (§4 e §7).
    val z = logitsComGradiente
    val n = batchSize * seqLen

    CrossEntropy(z, loteAlvos).backward()

    val gradiente = z.gradient.toArray

    for b <- 0 until batchSize; t <- 0 until seqLen do
      val posicao = (0 until vocabSize).map(v => loteLogits.get(b, t, v))
      val p = softmax(posicao)
      val alvoIdx = loteAlvos.get(b, t).toInt

      for v <- 0 until vocabSize do
        withClue(s"posicao (b=$b, t=$t, v=$v): ") {
          val esperado = (p(v) - (if v == alvoIdx then 1.0 else 0.0)) / n

          gradiente(b * seqLen * vocabSize + t * vocabSize + v) shouldBe esperado +- 1e-12
        }
  }

  it should "have gradients that sum to zero within each position" in {
    // somar uma constante a todos os logits nao muda a perda, entao o
    // gradiente de uma posicao nao pode ter soma diferente de zero
    val z = logitsComGradiente

    CrossEntropy(z, loteAlvos).backward()

    val gradiente = z.gradient.toArray

    for b <- 0 until batchSize; t <- 0 until seqLen do
      withClue(s"posicao (b=$b, t=$t): ") {
        val soma =
          (0 until vocabSize).map(v => gradiente(b * seqLen * vocabSize + t * vocabSize + v)).sum

        soma shouldBe 0.0 +- 1e-15
      }
  }

  it should "pass gradient check on the logits" in {
    val z = logitsComGradiente

    Gradcheck.run(z)(x => CrossEntropy(x, loteAlvos)) should be < 1e-5
  }

  // ---- perplexidade ----

  "CrossEntropy.perplexity" should "be exp(loss), and equal V for a model that knows nothing" in {
    val uniformes = Tensor.make(Array.fill(65)(0.0), Array(1, 1, 65))
    val loss = CrossEntropy(uniformes, alvo(7))

    loss.get(0) shouldBe 4.17439 +- 1e-5
    CrossEntropy.perplexity(loss) shouldBe 65.0 +- 1e-9
  }

  // ---- validacao de entrada ----

  "CrossEntropy" should "reject logits that are not rank 3" in {
    val rank2 = Tensor.make(Array(1.0, 2.0, 3.0, 4.0), Array(2, 2))

    an[IllegalArgumentException] should be thrownBy CrossEntropy(rank2, loteAlvos)
  }

  it should "reject targets that are not rank 2" in {
    val rank1 = Tensor.make(Array(1.0, 0.0, 1.0, 3.0), Array(4))

    an[IllegalArgumentException] should be thrownBy CrossEntropy(loteLogits, rank1)
  }

  it should "reject a batch or sequence mismatch between logits and targets" in {
    val outroLote = Tensor.make(Array(1.0, 0.0, 1.0, 3.0, 2.0, 2.0), Array(3, 2))

    an[IllegalArgumentException] should be thrownBy CrossEntropy(loteLogits, outroLote)
  }

  it should "reject a target outside [0, V)" in {
    val foraDaFaixa = Tensor.make(Array(2.0, 0.0, 1.0, 9.0), Array(batchSize, seqLen))

    an[IllegalArgumentException] should be thrownBy CrossEntropy(loteLogits, foraDaFaixa)
  }

  it should "reject a fractional target" in {
    // o Tensor so guarda Double, entao um alvo 2.9999 seria truncado em
    // silencio e ensinaria a associacao errada (§5)
    val fracionario = Tensor.make(Array(2.0, 0.0, 1.5, 3.0), Array(batchSize, seqLen))

    an[IllegalArgumentException] should be thrownBy CrossEntropy(loteLogits, fracionario)
  }
end CrossEntropySpec
