package gpt.nn

import scala.util.Random
import scalagrad.core.Tensor
import scalagrad.ops.tensor.*
import scalagrad.gradcheck.Gradcheck
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TransformerBlockSpec extends AnyFlatSpec with Matchers:

  // Dimensoes todas distintas entre si -- B, T, dModel, nHeads, expansion e
  // dHead. Um eixo trocado sobrevive a qualquer coincidencia entre elas.
  private val batchSize = 3
  private val seqLen = 5
  private val dModel = 8
  private val nHeads = 4
  private val expansion = 2

  // Semente fixa: os testes comparam numeros, e sem semente uma falha nao e
  // reproduzivel.
  private val rng = new Random(20260831)

  private def randomSequence(
      batchSize: Int,
      seqLen: Int,
      dModel: Int,
      requiresGradient: Boolean = false
  ): Tensor =
    val data = Array.fill(batchSize * seqLen * dModel)(rng.nextDouble() * 2 - 1)
    Tensor.make(data, Array(batchSize, seqLen, dModel), requiresGradient)

  private def lossWeights(shape: Array[Int]): Tensor =
    val size = shape.product
    Tensor.make(Array.tabulate(size)(i => 0.3 + 0.7 * Math.sin(i * 1.7)), shape)

  private def maxDifference(a: Tensor, b: Tensor): Double =
    a.shape.toArray shouldBe b.shape.toArray

    (0 until a.size)
      .map(i => Math.abs(a.get(a.unravelIndex(i)*) - b.get(b.unravelIndex(i)*)))
      .max

  /** A referencia da etapa: o bloco montado a partir das subcamadas expostas,
    * que ja tem suite propria. Nao ha Scala puro aqui de proposito -- a
    * atencao e o MLP foram conferidos contra referencias em laco nas Etapas 12
    * e 13, e reimplementa-los aqui duplicaria o risco em vez de reduzi-lo.
    */
  private def reference(b: TransformerBlock, x: Tensor): Tensor =
    val h = x + b.attention.forward(b.ln1.forward(x))
    h + b.mlp.forward(b.ln2.forward(h))

  private def withoutFirstResidual(b: TransformerBlock, x: Tensor): Tensor =
    val h = b.attention.forward(b.ln1.forward(x))
    h + b.mlp.forward(b.ln2.forward(h))

  private def withoutSecondResidual(b: TransformerBlock, x: Tensor): Tensor =
    val h = x + b.attention.forward(b.ln1.forward(x))
    b.mlp.forward(b.ln2.forward(h))

  private def postLN(b: TransformerBlock, x: Tensor): Tensor =
    val h = b.ln1.forward(x + b.attention.forward(x))
    b.ln2.forward(h + b.mlp.forward(h))

  private def swappedSublayers(b: TransformerBlock, x: Tensor): Tensor =
    val h = x + b.mlp.forward(b.ln1.forward(x))
    h + b.attention.forward(b.ln2.forward(h))

  // ---- forward ----

  "TransformerBlock.forward" should "match the composition of its own sublayers" in {
    // O teste que carrega a etapa. Todos os erros possiveis aqui -- soma
    // faltando, LN no lugar errado, subcamadas trocadas -- produzem forwards
    // perfeitamente derivaveis, que o gradient check aprova
    // (theory/14-transformer-block/14-transformer-block.md secao 7).
    val block = TransformerBlock(dModel, nHeads, expansion, rng)
    val x = randomSequence(batchSize, seqLen, dModel)

    maxDifference(block.forward(x), reference(block, x)) should be < 1e-12
  }

  it should "preserve the input shape, across varied dimensions" in {
    val cases = Seq((1, 1, 2, 1, 1), (2, 3, 4, 2, 4), (3, 5, 8, 4, 2), (2, 7, 6, 3, 3))

    for (b, t, d, h, e) <- cases do
      withClue(s"B=$b T=$t dModel=$d nHeads=$h expansion=$e: ") {
        val y = TransformerBlock(d, h, e, rng).forward(randomSequence(b, t, d))

        y.shape.toArray shouldBe Array(b, t, d)
      }
  }

  it should "keep both residual sums" in {
    // Duas referencias erradas, uma sem cada soma. As duas tem que diferir.
    val block = TransformerBlock(dModel, nHeads, expansion, rng)
    val x = randomSequence(batchSize, seqLen, dModel)
    val y = block.forward(x)

    maxDifference(y, withoutFirstResidual(block, x)) should be > 1e-3
    maxDifference(y, withoutSecondResidual(block, x)) should be > 1e-3
  }

  it should "normalize before each sublayer, not after the sum" in {
    // pre-LN contra post-LN. Sem este teste a troca passa: o formato e o
    // mesmo, a causalidade continua valendo e o gradient check aprova.
    val block = TransformerBlock(dModel, nHeads, expansion, rng)
    val x = randomSequence(batchSize, seqLen, dModel)

    maxDifference(block.forward(x), postLN(block, x)) should be > 1e-3
  }

  it should "run the attention before the MLP" in {
    val block = TransformerBlock(dModel, nHeads, expansion, rng)
    val x = randomSequence(batchSize, seqLen, dModel)

    maxDifference(block.forward(x), swappedSublayers(block, x)) should be > 1e-3
  }

  it should "accept a non-contiguous input" in {
    val block = TransformerBlock(dModel, nHeads, expansion, rng)
    val raw = randomSequence(batchSize, dModel, seqLen)
    val x = raw.transpose(1, 2)

    val same = Tensor.make(
      Array.tabulate(batchSize * seqLen * dModel) { i =>
        val b = i / (seqLen * dModel)
        val t = (i / dModel) % seqLen
        val d = i % dModel
        raw.get(b, d, t)
      },
      Array(batchSize, seqLen, dModel)
    )

    maxDifference(block.forward(x), block.forward(same)) should be < 1e-12
  }

  // ---- propriedades herdadas e propriedades novas ----

  "TransformerBlock" should "stay causal from end to end" in {
    // A mascara vem da Etapa 12, mas heranca se testa: um reshape errado no
    // meio do bloco faria a propriedade sumir sem mudar formato nenhum.
    val block = TransformerBlock(dModel, nHeads, expansion, rng)
    val x = randomSequence(1, seqLen, dModel)

    val changed = Tensor.make(
      Array.tabulate(seqLen * dModel) { i =>
        val t = i / dModel
        val d = i % dModel
        if t == seqLen - 1 then x.get(0, t, d) + 3.0 else x.get(0, t, d)
      },
      Array(1, seqLen, dModel)
    )

    val y = block.forward(x)
    val yChanged = block.forward(changed)

    for t <- 0 until seqLen - 1; d <- 0 until dModel do
      withClue(s"posicao (t=$t, d=$d): ") {
        yChanged.get(0, t, d) shouldBe y.get(0, t, d) +- 1e-12
      }

    // contraprova: a ultima posicao TEM que mudar, senao o teste acima passaria
    // num bloco que ignora a entrada
    val lastChanged = (0 until dModel)
      .map(d => Math.abs(yChanged.get(0, seqLen - 1, d) - y.get(0, seqLen - 1, d)))
      .max
    lastChanged should be > 1e-3
  }

  it should "be stackable: applying it to its own output changes nothing structural" in {
    // a propriedade da qual a Etapa 15 depende
    val block = TransformerBlock(dModel, nHeads, expansion, rng)
    val x = randomSequence(batchSize, seqLen, dModel)

    val twice = block.forward(block.forward(x))

    twice.shape.toArray shouldBe Array(batchSize, seqLen, dModel)
    (0 until twice.size).foreach { i =>
      withClue(s"posicao $i: ") {
        twice.get(twice.unravelIndex(i)*).isNaN shouldBe false
      }
    }
  }

  // ---- parametros ----

  "TransformerBlock.parameters" should "list 14 distinct tensors, in the order ln1, attention, ln2, mlp" in {
    val block = TransformerBlock(dModel, nHeads, expansion, rng)
    val ps = block.parameters

    ps.size shouldBe 14
    // toSet pega uma subcamada compartilhada por engano -- por exemplo, o
    // mesmo LayerNorm nas duas posicoes, que nao muda formato nenhum
    ps.toSet.size shouldBe 14
    ps.foreach(p => p.requiresGradient shouldBe true)

    ps.take(2) shouldBe block.ln1.parameters
    ps.slice(2, 8) shouldBe block.attention.parameters
    ps.slice(8, 10) shouldBe block.ln2.parameters
    ps.drop(10) shouldBe block.mlp.parameters
  }

  it should "give each LayerNorm exactly gamma and beta, both of shape [dModel]" in {
    // regressao: `Linear(dModel, dModel, rng = rng)` no lugar de `LayerNorm(dModel)` tem
    // o mesmo numero de tensores e produz o mesmo formato de saida. O que
    // muda e o formato dos parametros -- e a contagem do teste seguinte.
    val block = TransformerBlock(dModel, nHeads, expansion, rng)

    for (norm, name) <- Seq((block.ln1, "ln1"), (block.ln2, "ln2")) do
      withClue(s"$name: ") {
        norm.parameters.size shouldBe 2
        norm.parameters.foreach(p => p.shape.toArray shouldBe Array(dModel))
      }
  }

  it should "total 12·dModel² + 11·dModel" in {
    for (d, h) <- Seq((2, 1), (8, 4), (16, 4), (64, 8)) do
      withClue(s"dModel=$d nHeads=$h: ") {
        TransformerBlock(d, h, rng = rng).parameters.map(_.size).sum shouldBe 12 * d * d + 11 * d
      }
  }

  // ---- gradientes ----

  "TransformerBlock" should "pass gradient check w.r.t. x" in {
    val block = TransformerBlock(dModel, nHeads, expansion, rng)
    val x = randomSequence(batchSize, seqLen, dModel, requiresGradient = true)
    val w = lossWeights(Array(batchSize, seqLen, dModel))

    Gradcheck.run(x)(x => (block.forward(x) * w).sum) should be < 1e-5
  }

  it should "pass gradient check w.r.t. every one of the fourteen parameters" in {
    // uma instancia nova por parametro: `Gradient` so acumula, entao um
    // segundo run sobre o mesmo bloco leria o gradiente somado do anterior.
    val w = lossWeights(Array(batchSize, seqLen, dModel))

    for i <- 0 until 14 do
      val block = TransformerBlock(dModel, nHeads, expansion, rng)
      val x = randomSequence(batchSize, seqLen, dModel)

      withClue(s"parametro $i: ") {
        Gradcheck.run(block.parameters(i))(_ => (block.forward(x) * w).sum) should be < 1e-5
      }
  }

  it should "pass gradient check with a non-contiguous input" in {
    val block = TransformerBlock(dModel, nHeads, expansion, rng)
    val raw = randomSequence(batchSize, dModel, seqLen, requiresGradient = true)
    val w = lossWeights(Array(batchSize, seqLen, dModel))

    Gradcheck.run(raw)(r => (block.forward(r.transpose(1, 2)) * w).sum) should be < 1e-5
  }

  // ---- validacao de entrada ----

  "TransformerBlock.forward" should "reject an input that is not rank 3" in {
    val block = TransformerBlock(dModel, nHeads, expansion, rng)
    val matrix = Tensor.make(Array.fill(seqLen * dModel)(0.0), Array(seqLen, dModel))

    val error = intercept[IllegalArgumentException](block.forward(matrix))

    error.getMessage should include("2")
  }

  it should "reject an input whose last dimension is not dModel" in {
    val block = TransformerBlock(dModel, nHeads, expansion, rng)
    val wrong = randomSequence(batchSize, seqLen, dModel + 1)

    val error = intercept[IllegalArgumentException](block.forward(wrong))

    error.getMessage should include(dModel.toString)
    error.getMessage should include((dModel + 1).toString)
  }
end TransformerBlockSpec
