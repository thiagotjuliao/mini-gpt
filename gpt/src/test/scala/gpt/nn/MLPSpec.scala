package gpt.nn

import scala.util.Random
import scalagrad.core.Tensor
import scalagrad.ops.tensor.*
import scalagrad.gradcheck.Gradcheck
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MLPSpec extends AnyFlatSpec with Matchers:

  // Dimensoes todas distintas entre si -- B, T, dModel, expansion e dFF. Com
  // dFF == dModel, trocar `W2` por `W2ᵀ` continuaria produzindo shapes
  // compativeis, e o erro viraria resultado silenciosamente errado.
  private val batchSize = 3
  private val seqLen = 5
  private val dModel = 4
  private val expansion = 2
  private val dFF = dModel * expansion

  // Indices em `parameters`, que e [W1, b1, W2, b2]. Nomeados porque lista
  // posicional ja renumerou em tres etapas seguidas.
  private val upWeight = 0
  private val upBias = 1
  private val downWeight = 2
  private val downBias = 3

  // Semente fixa: os testes de valor comparam numeros, e sem semente uma
  // falha nao e reproduzivel.
  private val rng = new Random(20260830)

  private def randomSequence(
      batchSize: Int,
      seqLen: Int,
      dModel: Int,
      requiresGradient: Boolean = false
  ): Tensor =
    val data = Array.fill(batchSize * seqLen * dModel)(rng.nextDouble() * 2 - 1)
    Tensor.make(data, Array(batchSize, seqLen, dModel), requiresGradient)

  // Pesos distintos entre si na perda: `sum` puro daria gradiente uniforme, e
  // uniforme esconde troca de posicao (theory/04-gradient-check §3).
  private def lossWeights(shape: Array[Int]): Tensor =
    val size = shape.product
    Tensor.make(Array.tabulate(size)(i => 0.3 + 0.7 * Math.sin(i * 1.7)), shape)

  // A definicao da GELU, escrita de novo aqui de proposito: a referencia so
  // vale como caminho independente se nao chamar o codigo sob teste.
  private def gelu(x: Double): Double =
    val c = Math.sqrt(2 / Math.PI)
    0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * Math.pow(x, 3))))

  /** Onde a ativacao entra: no meio (o certo), antes da primeira camada, ou
    * depois da segunda. As duas ultimas existem como contraprova -- as tres
    * produzem o mesmo formato, entao so o valor separa uma da outra.
    */
  private enum GeluAt:
    case Middle, Start, End

  /** `Linear -> GELU -> Linear` em Scala puro, sem tensores. */
  private def reference(
      mlp: MLP,
      x: Tensor,
      placement: GeluAt = GeluAt.Middle
  ): Array[Array[Array[Double]]] =
    val ps = mlp.parameters
    val (w1, b1, w2, b2) = (ps(upWeight), ps(upBias), ps(downWeight), ps(downBias))
    val hidden = w1.shape(1)
    val inputDim = x.shape(2)
    val outputDim = w2.shape(1)

    Array.tabulate(x.shape(0), x.shape(1), outputDim) { (b, t, d) =>
      val input = (0 until inputDim).map { k =>
        if placement == GeluAt.Start then gelu(x.get(b, t, k)) else x.get(b, t, k)
      }

      val activations = (0 until hidden).map { i =>
        val z = (0 until inputDim).map(k => input(k) * w1.get(k, i)).sum + b1.get(i)
        if placement == GeluAt.Middle then gelu(z) else z
      }

      val y = (0 until hidden).map(i => activations(i) * w2.get(i, d)).sum + b2.get(d)
      if placement == GeluAt.End then gelu(y) else y
    }

  // A derivada da GELU, tambem reescrita a partir da formula (theory/05 §5).
  private def geluPrime(x: Double): Double =
    val c = Math.sqrt(2 / Math.PI)
    val t = Math.tanh(c * (x + 0.044715 * Math.pow(x, 3)))

    0.5 * (1 + t) + 0.5 * x * (1 - t * t) * c * (1 + 0.134145 * x * x)

  private def scaled(t: Tensor, factor: Double): Tensor =
    Tensor.make(
      Array.tabulate(t.size)(i => t.get(t.unravelIndex(i)*) * factor),
      t.shape.toArray
    )

  // ---- forward ----

  "MLP.forward" should "match a plain Scala implementation of Linear -> GELU -> Linear" in {
    val mlp = MLP(dModel, expansion, rng)
    val x = randomSequence(batchSize, seqLen, dModel)

    val y = mlp.forward(x)
    val expected = reference(mlp, x)

    y.shape.toArray shouldBe Array(batchSize, seqLen, dModel)

    for b <- 0 until batchSize; t <- 0 until seqLen; d <- 0 until dModel do
      withClue(s"posicao (b=$b, t=$t, d=$d): ") {
        y.get(b, t, d) shouldBe expected(b)(t)(d) +- 1e-12
      }
  }

  it should "preserve the input shape, across varied dimensions" in {
    // inclui as pontas: dModel = 1, seqLen = 1, expansion = 1 (sem expansao
    // nenhuma) e o expansion = 4 de producao.
    val cases = Seq((1, 1, 1, 1), (2, 1, 3, 4), (1, 6, 2, 1), (3, 5, 4, 2), (2, 3, 5, 4))

    for (b, t, d, e) <- cases do
      withClue(s"B=$b T=$t dModel=$d expansion=$e: ") {
        val y = MLP(d, e, rng).forward(randomSequence(b, t, d))

        y.shape.toArray shouldBe Array(b, t, d)
      }
  }

  it should "accept a rank-2 input [T, dModel]" in {
    // diferenca deliberada em relacao a MultiHeadAttention, que exige rank 3:
    // o MLP nao interpreta eixo de posicao nenhum, entao rank 2 e valido
    // (theory/13-mlp/13-mlp.md §8).
    val mlp = MLP(dModel, expansion, rng)
    val data = Array.fill(seqLen * dModel)(rng.nextDouble() * 2 - 1)
    val x = Tensor.make(data, Array(seqLen, dModel))

    mlp.forward(x).shape.toArray shouldBe Array(seqLen, dModel)
  }

  it should "accept a non-contiguous input" in {
    // x montado como [B, dModel, T] e transposto: as strides deixam de ser
    // canonicas. Corolario do CLAUDE.md -- toda op nova precisa de um caso
    // com entrada transposta.
    val mlp = MLP(dModel, expansion, rng)
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

    val fromView = mlp.forward(x)
    val fromCopy = mlp.forward(same)

    for b <- 0 until batchSize; t <- 0 until seqLen; d <- 0 until dModel do
      withClue(s"posicao (b=$b, t=$t, d=$d): ") {
        fromView.get(b, t, d) shouldBe fromCopy.get(b, t, d) +- 1e-12
      }
  }

  // ---- posicao a posicao ----

  "MLP.forward" should "give identical outputs to identical tokens, wherever they sit" in {
    val mlp = MLP(dModel, expansion, rng)
    val tokens = Array.fill(seqLen)(Array.fill(dModel)(rng.nextDouble() * 2 - 1))
    val repeated = tokens.updated(3, tokens(0))

    val x = Tensor.make(
      Array.tabulate(seqLen * dModel)(i => repeated(i / dModel)(i % dModel)),
      Array(1, seqLen, dModel)
    )

    val y = mlp.forward(x)

    // igualdade exata, sem tolerancia: e a mesma conta, com os mesmos bits
    for d <- 0 until dModel do
      withClue(s"coordenada $d: ") {
        y.get(0, 3, d) shouldBe y.get(0, 0, d)
      }
  }

  it should "commute with a permutation of the positions" in {
    val mlp = MLP(dModel, expansion, rng)
    val x = randomSequence(1, seqLen, dModel)
    val permutation = Array(2, 0, 4, 1, 3)

    val permuted = Tensor.make(
      Array.tabulate(seqLen * dModel)(i => x.get(0, permutation(i / dModel), i % dModel)),
      Array(1, seqLen, dModel)
    )

    val y = mlp.forward(x)
    val yPermuted = mlp.forward(permuted)

    for t <- 0 until seqLen; d <- 0 until dModel do
      withClue(s"posicao (t=$t, d=$d): ") {
        yPermuted.get(0, t, d) shouldBe y.get(0, permutation(t), d)
      }
  }

  // ---- a nao-linearidade ----

  "MLP" should "not be an affine map" in {
    // O teste que carrega a etapa: sem a `gelu`, a camada vira
    // `Linear -> Linear`, e formato, independencia entre posicoes e gradient
    // check continuam TODOS passando (theory/13-mlp/13-mlp.md §8).
    //
    // Para qualquer f(x) = xA + b vale f(2x) - 2f(x) + f(0) = 0, exato e
    // independente do valor do vies. A forma com f(0) importa: comparar
    // f(2x) com 2f(x) direto so funcionaria enquanto b1 e b2 nascem zerados.
    val mlp = MLP(dModel, expansion, rng)
    val x = randomSequence(1, seqLen, dModel)
    val zeros = Tensor.make(Array.fill(seqLen * dModel)(0.0), Array(1, seqLen, dModel))

    val fx = mlp.forward(x)
    val f2x = mlp.forward(scaled(x, 2.0))
    val f0 = mlp.forward(zeros)

    val gaps = for t <- 0 until seqLen; d <- 0 until dModel
    yield Math.abs(f2x.get(0, t, d) - 2 * fx.get(0, t, d) + f0.get(0, t, d))

    gaps.max should be > 1e-3
  }

  it should "apply the GELU between the two Linear layers, and not before or after them" in {
    val mlp = MLP(dModel, expansion, rng)
    val x = randomSequence(1, seqLen, dModel)
    val y = mlp.forward(x)

    for wrong <- Seq(GeluAt.Start, GeluAt.End) do
      withClue(s"gelu em $wrong: ") {
        val expected = reference(mlp, x, wrong)

        val differences = for t <- 0 until seqLen; d <- 0 until dModel
        yield Math.abs(y.get(0, t, d) - expected(0)(t)(d))

        differences.max should be > 1e-3
      }
  }

  it should "add b1 before the GELU, not after it" in {
    // Nenhum teste de FORWARD consegue separar as duas ordens: `b1` nasce
    // zerado, e somar zero antes ou depois da ativacao da exatamente o mesmo
    // numero. `data` e private[scalagrad], entao a suite tambem nao tem como
    // preencher o vies por fora.
    //
    // O backward separa, e sem preparacao nenhuma: com o vies no lugar certo,
    // `db1 = Σ dZ = Σ dA ⊙ gelu'(Z)`; com ele depois da ativacao, `db1 = Σ dA`.
    // Os dois diferem mesmo com b1 = 0, porque a diferenca esta na regra da
    // cadeia, nao no valor do vies. Confirmado por mutacao: a troca passa em
    // todos os outros 17 testes e falha neste.
    val mlp = MLP(dModel, expansion, rng)
    val x = randomSequence(batchSize, seqLen, dModel)
    val w = lossWeights(Array(batchSize, seqLen, dModel))
    val ps = mlp.parameters
    val (w1, b1, w2) = (ps(upWeight), ps(upBias), ps(downWeight))

    (mlp.forward(x) * w).sum.backward()

    // dW1 e dW2 somam sobre B e T, e db1 tambem -- uma unica soma por neuronio
    val expected = (0 until dFF).map { i =>
      val perPosition = for b <- 0 until batchSize; t <- 0 until seqLen yield
        val z = (0 until dModel).map(k => x.get(b, t, k) * w1.get(k, i)).sum + b1.get(i)
        val dA = (0 until dModel).map(d => w.get(b, t, d) * w2.get(i, d)).sum

        dA * geluPrime(z)

      perPosition.sum
    }

    val actual = b1.gradient.toArray

    for i <- 0 until dFF do
      withClue(s"db1($i): ") {
        actual(i) shouldBe expected(i) +- 1e-9
      }
  }

  // ---- parametros ----

  "MLP.parameters" should "be [W1, b1, W2, b2], with the shapes of an expansion and a projection back" in {
    val mlp = MLP(dModel, expansion, rng)

    mlp.parameters.size shouldBe 4
    mlp.dFF shouldBe dFF

    mlp.parameters(upWeight).shape.toArray shouldBe Array(dModel, dFF)
    mlp.parameters(upBias).shape.toArray shouldBe Array(dFF)
    mlp.parameters(downWeight).shape.toArray shouldBe Array(dFF, dModel)
    mlp.parameters(downBias).shape.toArray shouldBe Array(dModel)

    mlp.parameters.foreach(p => p.requiresGradient shouldBe true)
  }

  it should "total 8·dModel² + 5·dModel with the default expansion" in {
    for d <- Seq(1, 4, 16, 64) do
      withClue(s"dModel=$d: ") {
        MLP(d, rng = rng).parameters.map(_.size).sum shouldBe 8 * d * d + 5 * d
      }
  }

  it should "not share a tensor between the two layers" in {
    // regressao: reusar uma unica `Linear` daria W1 == W2, o que so e sequer
    // representavel quando dFF == dModel -- e nesse caso passa despercebido.
    val mlp = MLP(dModel, expansion, rng)

    mlp.parameters.toSet.size shouldBe 4
  }

  // ---- gradientes ----

  "MLP" should "pass gradient check w.r.t. x" in {
    val mlp = MLP(dModel, expansion, rng)
    val x = randomSequence(batchSize, seqLen, dModel, requiresGradient = true)
    val w = lossWeights(Array(batchSize, seqLen, dModel))

    Gradcheck.run(x)(x => (mlp.forward(x) * w).sum) should be < 1e-5
  }

  it should "pass gradient check w.r.t. every one of the four parameters" in {
    // uma instancia nova por parametro: `Gradient` so acumula (nunca
    // sobrescreve), entao um segundo run sobre a mesma camada leria o
    // gradiente somado do anterior.
    val w = lossWeights(Array(batchSize, seqLen, dModel))

    for i <- 0 until 4 do
      val mlp = MLP(dModel, expansion, rng)
      val x = randomSequence(batchSize, seqLen, dModel)

      withClue(s"parametro $i: ") {
        Gradcheck.run(mlp.parameters(i))(_ => (mlp.forward(x) * w).sum) should be < 1e-5
      }
  }

  it should "pass gradient check with a non-contiguous input" in {
    val mlp = MLP(dModel, expansion, rng)
    val raw = randomSequence(batchSize, dModel, seqLen, requiresGradient = true)
    val w = lossWeights(Array(batchSize, seqLen, dModel))

    Gradcheck.run(raw)(r => (mlp.forward(r.transpose(1, 2)) * w).sum) should be < 1e-5
  }

  // ---- validacao de entrada ----

  "MLP" should "reject an expansion below 1" in {
    val error = intercept[IllegalArgumentException](MLP(dModel, expansion = 0))

    error.getMessage should include("0")
  }

  it should "reject an input with rank below 2" in {
    val mlp = MLP(dModel, expansion, rng)
    val vector = Tensor.make(Array.fill(dModel)(0.0), Array(dModel))

    val error = intercept[IllegalArgumentException](mlp.forward(vector))

    error.getMessage should include("1")
  }

  it should "reject an input whose last dimension is not dModel" in {
    val mlp = MLP(dModel, expansion, rng)
    val wrong = randomSequence(batchSize, seqLen, dModel + 1)

    val error = intercept[IllegalArgumentException](mlp.forward(wrong))

    // a mensagem tem que citar os dois numeros: quem erra isso precisa ver
    // contra o que a dimensao foi comparada
    error.getMessage should include(dModel.toString)
    error.getMessage should include((dModel + 1).toString)
  }

  private def desvioPadrao(t: scalagrad.core.Tensor): Double =
    val v = t.toArray
    val media = v.sum / v.length
    Math.sqrt(v.map(x => (x - media) * (x - media)).sum / v.length)

  "the residual scale" should "shrink the second Linear and leave the first alone" in {
    // parameters = [W_up, b_up, W_down, b_down]
    val escala = 0.25
    val camada = MLP(64, 4, new Random(9), escala)

    val esperadoUp = Math.sqrt(2.0 / 64)
    val esperadoDown = Math.sqrt(2.0 / 256) * escala

    desvioPadrao(camada.parameters(0)) shouldBe esperadoUp +- (esperadoUp * 0.1)
    desvioPadrao(camada.parameters(2)) shouldBe esperadoDown +- (esperadoDown * 0.1)
  }

  it should "default to 1.0" in {
    val camada = MLP(64, 4, new Random(9))
    val esperado = Math.sqrt(2.0 / 256)

    desvioPadrao(camada.parameters(2)) shouldBe esperado +- (esperado * 0.1)
  }
end MLPSpec
