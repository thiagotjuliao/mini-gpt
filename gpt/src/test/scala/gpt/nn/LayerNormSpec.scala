package gpt.nn

import scalagrad.core.Tensor
import scalagrad.ops.tensor.*
import scalagrad.gradcheck.Gradcheck
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class LayerNormSpec extends AnyFlatSpec with Matchers {

  private val eps = 1e-5

  private def tensorOf(rows: Array[Double]*): Tensor =
    Tensor.make(rows.flatten.toArray, Array(rows.length, rows.head.length))

  /** Referencia em Scala puro do forward de uma linha, com gamma=1 e beta=0.
    * Independente da implementacao: usada pra conferir o forward e pra montar
    * o xhat esperado nas identidades do backward.
    */
  private def normalize(row: Array[Double]): Array[Double] = {
    val h = row.length
    val mu = row.sum / h
    val variance = row.map(v => (v - mu) * (v - mu)).sum / h
    val s = Math.sqrt(variance + eps)

    row.map(v => (v - mu) / s)
  }

  /** Pesos distintos e de sinais alternados. Uma perda uniforme nao serve
    * aqui: `forward(x).sum` e identicamente zero em x (ver o teste "should sum
    * to zero across every row"), entao o gradiente verdadeiro dela e zero e o
    * gradient check degenera -- o analitico da exatamente 0.0 e o numerico e
    * so ruido de arredondamento. Medido neste projeto: 2.2e-3 com `.sum`
    * (acima do limite de 1e-5, ou seja, reprova codigo correto) contra 3.1e-10
    * com a perda ponderada.
    */
  private def weightsFor(t: Tensor): Tensor = {
    val data = Array.tabulate(t.size)(i => if i % 2 == 0 then (i + 1).toDouble else -(i + 1).toDouble)

    Tensor.make(data, t.shape.toArray)
  }

  private val sampleRows = Seq(
    Array(2.0, 3.0, 6.0, 9.0),
    Array(4.0, 4.0, 6.0, 6.0),
    Array(-1.0, 0.5, 2.0, -3.0)
  )

  "LayerNorm.forward" should "match a plain Scala implementation, row by row" in {
    val layerNorm = new LayerNorm(dim = 4)
    val y = layerNorm.forward(tensorOf(sampleRows*))

    for (i <- sampleRows.indices) {
      val expected = normalize(sampleRows(i))

      for (j <- expected.indices) y.get(i, j) shouldBe expected(j) +- 1e-12
    }
  }

  it should "match the verified numeric example from theory/10-layer-norm Secao 2" in {
    val layerNorm = new LayerNorm(dim = 4)
    val y = layerNorm.forward(tensorOf(Array(2.0, 3.0, 6.0, 9.0)))
    val expected = Array(-1.0954, -0.7303, 0.3651, 1.4606)

    for (j <- expected.indices) y.get(0, j) shouldBe expected(j) +- 1e-4
  }

  it should "produce rows with mean 0 and variance 1 while gamma=1 and beta=0" in {
    val layerNorm = new LayerNorm(dim = 4)
    val y = layerNorm.forward(tensorOf(sampleRows*))

    for (i <- sampleRows.indices) {
      val row = (0 until 4).map(j => y.get(i, j))
      val mean = row.sum / 4
      val variance = row.map(v => (v - mean) * (v - mean)).sum / 4

      mean shouldBe 0.0 +- 1e-12
      variance shouldBe 1.0 +- 1e-4 // nao e 1 exato por causa do eps
    }
  }

  it should "be invariant to scaling and shifting of the input" in {
    // theory/10-layer-norm Secao 1: a normalizacao descarta escala e
    // deslocamento, e e justamente isso que gamma e beta existem pra devolver.
    val layerNorm = new LayerNorm(dim = 4)
    val original = Array(2.0, 3.0, 6.0, 9.0)

    val y = layerNorm.forward(tensorOf(original))
    val yScaled = layerNorm.forward(tensorOf(original.map(_ * 100)))
    val yShifted = layerNorm.forward(tensorOf(original.map(_ + 1000)))

    for (j <- original.indices) {
      yScaled.get(0, j) shouldBe y.get(0, j) +- 1e-4
      yShifted.get(0, j) shouldBe y.get(0, j) +- 1e-4
    }
  }

  it should "return zeros instead of NaN for a constant row" in {
    // variancia exatamente zero: sem o eps seria 0/0 (theory Secao 3).
    val layerNorm = new LayerNorm(dim = 4)
    val y = layerNorm.forward(tensorOf(Array(7.0, 7.0, 7.0, 7.0)))

    for (j <- 0 until 4) {
      y.get(0, j).isNaN shouldBe false
      y.get(0, j) shouldBe 0.0 +- 1e-12
    }
  }

  it should "sum to zero across every row" in {
    // Nao e curiosidade: e a razao de todo gradient check desta suite usar uma
    // perda ponderada. Como forward(x).sum e constante em x, o gradiente dela e
    // identicamente zero, e o gradient check passa a comparar ruido numerico
    // contra zero -- inutil como teste. Mesma degeneracao do softmax (Etapa 6),
    // cuja saida tambem soma um valor fixo.
    val layerNorm = new LayerNorm(dim = 4)
    val y = layerNorm.forward(tensorOf(sampleRows*))

    for (i <- sampleRows.indices) {
      (0 until 4).map(j => y.get(i, j)).sum shouldBe 0.0 +- 1e-12
    }
  }

  it should "preserve the input shape for rank 1, 2 and 3" in {
    // a camada usa apenas rank-1 como eixo, entao nao ha razao pra exigir
    // rank 3 -- o [B,T,H] do modelo e so o caso mais comum, nao o unico.
    val layerNorm = new LayerNorm(dim = 4)

    val rank1 = layerNorm.forward(Tensor.make(Array(1.0, 2.0, 3.0, 4.0), Array(4)))
    rank1.rank shouldBe 1
    rank1.shape(0) shouldBe 4

    val rank2 = layerNorm.forward(tensorOf(sampleRows*))
    rank2.rank shouldBe 2
    rank2.shape(0) shouldBe 3
    rank2.shape(1) shouldBe 4

    val rank3 = layerNorm.forward(Tensor.make(Array.tabulate(24)(_.toDouble), Array(2, 3, 4)))
    rank3.rank shouldBe 3
    rank3.shape(0) shouldBe 2
    rank3.shape(1) shouldBe 3
    rank3.shape(2) shouldBe 4
  }

  it should "normalize each row of a rank-3 batch independently" in {
    val layerNorm = new LayerNorm(dim = 4)
    val rows = Array(2.0, 3.0, 6.0, 9.0) ++ Array(200.0, 300.0, 600.0, 900.0)
    val y = layerNorm.forward(Tensor.make(rows, Array(2, 1, 4)))
    val expected = normalize(Array(2.0, 3.0, 6.0, 9.0))

    // as duas sequencias estao em escalas muito diferentes e mesmo assim saem
    // identicas: cada uma usou so as proprias estatisticas (theory Secao 7).
    for (j <- expected.indices) {
      y.get(0, 0, j) shouldBe expected(j) +- 1e-4
      y.get(1, 0, j) shouldBe expected(j) +- 1e-4
    }
  }

  it should "reject a rank-0 tensor" in {
    // regressao: o require de rank tem que rodar antes do acesso a shape.last,
    // senao sai um erro cru de indice sem contexto (mesmo bug do indexSelect).
    val layerNorm = new LayerNorm(dim = 4)

    an[IllegalArgumentException] should be thrownBy layerNorm.forward(Tensor.make(Array(1.0), Array()))
  }

  it should "reject an input whose last dimension is not dim" in {
    val layerNorm = new LayerNorm(dim = 4)

    an[IllegalArgumentException] should be thrownBy layerNorm.forward(tensorOf(Array(1.0, 2.0, 3.0)))
  }

  "LayerNorm.parameters" should "return exactly [gamma, beta], as ones and zeros, both trainable" in {
    val layerNorm = new LayerNorm(dim = 6)

    layerNorm.parameters.size shouldBe 2

    val gamma = layerNorm.parameters(0)
    gamma.shape(0) shouldBe 6
    for (i <- 0 until 6) gamma.get(i) shouldBe 1.0

    val beta = layerNorm.parameters(1)
    beta.shape(0) shouldBe 6
    for (i <- 0 until 6) beta.get(i) shouldBe 0.0

    layerNorm.parameters.foreach(p => p.requiresGradient shouldBe true)
  }

  "LayerNorm" should "pass gradient check w.r.t. x" in {
    val layerNorm = new LayerNorm(dim = 4)
    val x = tensorOf(sampleRows*)
    val w = weightsFor(x)

    Gradcheck.run(x)(x => (layerNorm.forward(x) * w).sum) should be < 1e-5
  }

  it should "pass gradient check w.r.t. gamma" in {
    // instancia nova por check: Gradient so acumula, nunca sobrescreve.
    val layerNorm = new LayerNorm(dim = 4)
    val x = tensorOf(sampleRows*)
    val w = weightsFor(x)

    Gradcheck.run(layerNorm.parameters(0))(_ => (layerNorm.forward(x) * w).sum) should be < 1e-5
  }

  it should "pass gradient check w.r.t. beta" in {
    val layerNorm = new LayerNorm(dim = 4)
    val x = tensorOf(sampleRows*)
    val w = weightsFor(x)

    Gradcheck.run(layerNorm.parameters(1))(_ => (layerNorm.forward(x) * w).sum) should be < 1e-5
  }

  "LayerNorm backward" should "produce a gradient that sums to zero within each row" in {
    // theory/10-layer-norm Secao 5: o termo m1 remove o empurrao que apenas
    // deslocaria o vetor, e o forward ignora deslocamento. Vale pra qualquer dy.
    val layerNorm = new LayerNorm(dim = 4)
    val x = Tensor.make(sampleRows.flatten.toArray, Array(3, 4), requiresGradient = true)
    val w = weightsFor(x)

    ((layerNorm.forward(x) * w).sum).backward()

    for (i <- sampleRows.indices) {
      val rowSum = (0 until 4).map(j => x.gradient(x.index(i, j))).sum

      rowSum shouldBe 0.0 +- 1e-12
    }
  }

  it should "produce a gradient orthogonal to xhat within each row" in {
    // a segunda identidade: m2 remove o empurrao que apenas reescalaria o
    // vetor. O residuo e da ordem do eps, nao zero exato (theory Secao 5).
    val layerNorm = new LayerNorm(dim = 4)
    val x = Tensor.make(sampleRows.flatten.toArray, Array(3, 4), requiresGradient = true)
    val w = weightsFor(x)

    ((layerNorm.forward(x) * w).sum).backward()

    for (i <- sampleRows.indices) {
      val xhat = normalize(sampleRows(i))
      val projection = (0 until 4).map(j => x.gradient(x.index(i, j)) * xhat(j)).sum

      projection shouldBe 0.0 +- 1e-4
    }
  }
}
