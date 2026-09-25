package gpt.nn

import scala.util.Random
import scalagrad.core.Tensor
import scalagrad.ops.tensor.*
import scalagrad.gradcheck.Gradcheck
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class LinearSpec extends AnyFlatSpec with Matchers:

  private def randomInput(
      batchSize: Int,
      inputDim: Int,
      requiresGradient: Boolean = false
  ): Tensor =
    val data = Array.fill(batchSize * inputDim)(Random.nextDouble() * 2 - 1)
    Tensor.make(data, Array(batchSize, inputDim), requiresGradient)

  private def flatten(t: Tensor): IndexedSeq[Double] =
    (0 until t.size).map(i => t.get(t.unravelIndex(i)*))

  "Linear.forward" should "match a plain Scala implementation of y = xW + b" in {
    val inputDim = 3; val outputDim = 2; val batchSize = 2
    val linear = new Linear(inputDim, outputDim)
    val x = randomInput(batchSize, inputDim)

    val y = linear.forward(x)
    val w = linear.parameters(0)
    val b = linear.parameters(1)

    for i <- 0 until batchSize; j <- 0 until outputDim do
      val expected = (0 until inputDim).map(k => x.get(i, k) * w.get(k, j)).sum + b.get(j)
      y.get(i, j) shouldBe expected +- 1e-9
  }

  it should "produce output of shape (batchSize, outputDim), for varied dimensions" in {
    for (inputDim, outputDim, batchSize) <- Seq((1, 1, 1), (4, 3, 1), (5, 8, 6)) do
      val linear = new Linear(inputDim, outputDim)
      val y = linear.forward(randomInput(batchSize, inputDim))

      y.rank shouldBe 2
      y.size shouldBe batchSize * outputDim
  }

  "Linear.parameters" should "return exactly [W, b], with the shapes [inputDim,outputDim] and [outputDim]" in {
    val linear = new Linear(inputDim = 4, outputDim = 6)

    linear.parameters.size shouldBe 2

    val w = linear.parameters(0)
    w.rank shouldBe 2
    w.size shouldBe 4 * 6

    val b = linear.parameters(1)
    b.rank shouldBe 1
    b.size shouldBe 6
  }

  it should "mark both W and b as requiring gradient" in {
    // regressao: numa versao anterior, b nascia sem requiresGradient=true --
    // o otimizador (Etapa 17) nunca teria como atualiza-lo.
    val linear = new Linear(inputDim = 3, outputDim = 2)
    linear.parameters.foreach(p => p.requiresGradient shouldBe true)
  }

  "Linear weight initialization" should "not fill W with a single repeated constant" in {
    // regressao: uma versao anterior usava Array.fill(n)(std), preenchendo
    // todo peso com o MESMO valor -- exatamente o problema da simetria
    // descrito em theory/08-linear/08-linear.md Secao 4 (neuronios "gemeos").
    val linear = new Linear(inputDim = 10, outputDim = 10)
    val values = flatten(linear.parameters(0)).toSet

    values.size should be > 1
  }

  it should "have variance close to the Kaiming target Var(w) = 2/inputDim" in {
    val inputDim = 50; val outputDim = 200
    val linear = new Linear(inputDim, outputDim)
    val values = flatten(linear.parameters(0))

    val mean = values.sum / values.size
    val variance = values.map(v => (v - mean) * (v - mean)).sum / values.size
    val expectedVariance = 2.0 / inputDim

    // amostra grande (10000 pesos) deixa bastante folga (35%) sem risco de
    // teste instavel -- desvio-padrao amostral esperado da variancia e
    // pequeno pra esse N (ver theory/08-linear/08-linear.md Secao 5).
    variance shouldBe expectedVariance +- (expectedVariance * 0.35)
  }

  "Linear" should "pass gradient check w.r.t. x" in {
    val linear = new Linear(inputDim = 4, outputDim = 3)
    val x = randomInput(batchSize = 5, inputDim = 4, requiresGradient = true)

    Gradcheck.run(x)(x => linear.forward(x).sum) should be < 1e-5
  }

  it should "pass gradient check w.r.t. W" in {
    // linear/x novos, so pra este teste: Gradient so acumula (nunca
    // sobrescreve), entao reusar o mesmo W entre varios Gradcheck.run
    // corromperia o gradiente analitico lido pelo teste seguinte.
    val linear = new Linear(inputDim = 4, outputDim = 3)
    val x = randomInput(batchSize = 5, inputDim = 4)

    Gradcheck.run(linear.parameters(0))(_ => linear.forward(x).sum) should be < 1e-5
  }

  it should "pass gradient check w.r.t. b" in {
    val linear = new Linear(inputDim = 4, outputDim = 3)
    val x = randomInput(batchSize = 5, inputDim = 4)

    Gradcheck.run(linear.parameters(1))(_ => linear.forward(x).sum) should be < 1e-5
  }

  // ---- entrada rank 3 [B, T, inputDim] -- o formato que a atencao usa (Etapa 11) ----
  // Ate a Etapa 11 esta camada so tinha sido exercitada com rank 2, e por isso
  // ninguem notou que matmul nao dispatchava o par (3, 2).

  private def randomSequenceInput(
      batchSize: Int,
      seqLen: Int,
      inputDim: Int,
      requiresGradient: Boolean = false
  ): Tensor =
    val data = Array.fill(batchSize * seqLen * inputDim)(Random.nextDouble() * 2 - 1)
    Tensor.make(data, Array(batchSize, seqLen, inputDim), requiresGradient)

  "Linear.forward with a rank-3 input" should "apply the same W and b to every position" in {
    val inputDim = 3; val outputDim = 2; val batchSize = 2; val seqLen = 4
    val linear = new Linear(inputDim, outputDim)
    val x = randomSequenceInput(batchSize, seqLen, inputDim)

    val y = linear.forward(x)
    val w = linear.parameters(0)
    val b = linear.parameters(1)

    y.shape.toList shouldBe List(batchSize, seqLen, outputDim)

    for i <- 0 until batchSize; t <- 0 until seqLen; j <- 0 until outputDim do
      val expected = (0 until inputDim).map(k => x.get(i, t, k) * w.get(k, j)).sum + b.get(j)
      y.get(i, t, j) shouldBe expected +- 1e-9
  }

  it should "give each position the same result it would get on its own" in {
    // nenhuma posicao pode enxergar as outras: a mistura entre tokens e trabalho
    // da atencao (Etapa 11), nunca do Linear
    val inputDim = 3; val outputDim = 2
    val linear = new Linear(inputDim, outputDim)
    val x = randomSequenceInput(batchSize = 2, seqLen = 4, inputDim = inputDim)

    val y = linear.forward(x)

    for i <- 0 until 2; t <- 0 until 4 do
      val row = Tensor.make((0 until inputDim).map(k => x.get(i, t, k)).toArray, Array(1, inputDim))
      val alone = linear.forward(row)

      (0 until outputDim).foreach(j => y.get(i, t, j) shouldBe alone.get(0, j) +- 1e-12)
  }

  it should "pass gradient check w.r.t. x, W and b" in {
    // W e b sao compartilhados por todas as B*T posicoes, entao o gradiente deles
    // soma sobre lote e sequencia (theory/11-attention/11-attention.md §8)
    // o Linear fica FORA do closure: Gradcheck.run chama f uma vez por perturbacao,
    // e um `new Linear` la dentro sortearia pesos novos a cada chamada
    val forX = new Linear(4, 3)
    val xCheck = randomSequenceInput(2, 4, 4, requiresGradient = true)
    Gradcheck.run(xCheck)(x => forX.forward(x).sum) should be < 1e-5

    val forW = new Linear(4, 3)
    val xW = randomSequenceInput(2, 4, 4)
    Gradcheck.run(forW.parameters(0))(_ => forW.forward(xW).sum) should be < 1e-5

    val forB = new Linear(4, 3)
    val xB = randomSequenceInput(2, 4, 4)
    Gradcheck.run(forB.parameters(1))(_ => forB.forward(xB).sum) should be < 1e-5
  }

  // ---- useBias = false (Etapa 12) ----
  // Usado pela projecao `key` da atencao, cujo vies teria gradiente
  // exatamente zero para sempre (ver AttentionSpec).
  //
  // AVISO sobre o que estes testes NAO conseguem provar: `initializeBias`
  // preenche o vies com zeros, entao no instante da construcao uma camada
  // com vies e outra sem produzem o MESMO forward, bit a bit. Nenhum teste
  // de saida distingue as duas aqui. A diferenca observavel e `parameters` --
  // e e ela que importa, porque e a lista que o otimizador da Etapa 17 vai
  // percorrer. Os testes de forward abaixo cobrem outra coisa: que o ramo
  // `b.fold(z)(z + _)` devolve `z` intacto em vez de quebrar.

  "Linear.parameters with useBias = false" should "return exactly [W], with no bias entry" in {
    val linear = new Linear(inputDim = 4, outputDim = 6, useBias = false)

    linear.parameters.size shouldBe 1

    val w = linear.parameters(0)
    w.rank shouldBe 2
    w.size shouldBe 4 * 6
    w.requiresGradient shouldBe true
  }

  "Linear.forward with useBias = false" should "compute y = xW, with no constant term" in {
    val inputDim = 3; val outputDim = 2; val batchSize = 2
    val linear = new Linear(inputDim, outputDim, useBias = false)
    val x = randomInput(batchSize, inputDim)

    val y = linear.forward(x)
    val w = linear.parameters(0)

    for i <- 0 until batchSize; j <- 0 until outputDim do
      val expected = (0 until inputDim).map(k => x.get(i, k) * w.get(k, j)).sum
      y.get(i, j) shouldBe expected +- 1e-12
  }

  it should "map a zero input to exactly zero" in {
    // sem vies, y = 0W = 0 -- e igualdade exata, nao aproximacao. Um termo
    // constante qualquer que fosse APRENDIDO (nao o zero da inicializacao)
    // apareceria aqui.
    val linear = new Linear(inputDim = 4, outputDim = 3, useBias = false)
    val zeros = Tensor.make(Array.fill(2 * 4)(0.0), Array(2, 4))

    val y = linear.forward(zeros)

    for i <- 0 until y.size do y.get(y.unravelIndex(i)*) shouldBe 0.0
  }

  it should "apply the same W to every position of a rank-3 input" in {
    // o formato que a atencao usa: [B, T, dModel] -- e o caminho de producao
    // do useBias = false hoje.
    val inputDim = 3; val outputDim = 2; val batchSize = 2; val seqLen = 4
    val linear = new Linear(inputDim, outputDim, useBias = false)
    val x = randomSequenceInput(batchSize, seqLen, inputDim)

    val y = linear.forward(x)
    val w = linear.parameters(0)

    y.shape.toList shouldBe List(batchSize, seqLen, outputDim)

    for i <- 0 until batchSize; t <- 0 until seqLen; j <- 0 until outputDim do
      val expected = (0 until inputDim).map(k => x.get(i, t, k) * w.get(k, j)).sum
      y.get(i, t, j) shouldBe expected +- 1e-12
  }

  "Linear with useBias = false" should "pass gradient check w.r.t. x and W" in {
    // garante que o ramo sem vies nao quebrou o grafo de autograd
    val forX = new Linear(4, 3, useBias = false)
    val xCheck = randomSequenceInput(2, 4, 4, requiresGradient = true)
    Gradcheck.run(xCheck)(x => forX.forward(x).sum) should be < 1e-5

    val forW = new Linear(4, 3, useBias = false)
    val xW = randomSequenceInput(2, 4, 4)
    Gradcheck.run(forW.parameters(0))(_ => forW.forward(xW).sum) should be < 1e-5
  }

  // ---- validacao das dimensoes (Etapa 13) ----
  // Dimensao nao positiva criava tensores vazios em silencio. O caso real que
  // motivou a checagem: `MLP` construia `Linear(dModel, dFF)` com `dFF` ainda
  // nao inicializado, valendo 0 -- e nada estourava.

  "Linear" should "reject an inputDim below 1" in {
    val error = intercept[IllegalArgumentException](new Linear(inputDim = 0, outputDim = 3))

    error.getMessage should include("0")
  }

  it should "reject an outputDim below 1" in {
    val error = intercept[IllegalArgumentException](new Linear(inputDim = 3, outputDim = 0))

    error.getMessage should include("0")
  }

  // ---- weights/bias e o Random injetavel (Etapa 13) ----

  "Linear.weights and Linear.bias" should "expose the same tensors that parameters lists" in {
    // acesso por nome em vez de por indice posicional; tem que ser a MESMA
    // instancia, nao uma copia -- o otimizador da Etapa 17 atualiza uma so.
    val linear = new Linear(inputDim = 3, outputDim = 2)

    linear.weights should be theSameInstanceAs linear.parameters(0)
    linear.bias shouldBe defined
    linear.bias.get should be theSameInstanceAs linear.parameters(1)

    new Linear(3, 2, useBias = false).bias shouldBe empty
  }

  "Linear with an injected Random" should "be reproducible across instances" in {
    val a = new Linear(4, 3, rng = new Random(20260830))
    val b = new Linear(4, 3, rng = new Random(20260830))
    val c = new Linear(4, 3, rng = new Random(20260831))

    flatten(a.weights) shouldBe flatten(b.weights)
    flatten(a.weights) should not be flatten(c.weights)
  }

  private def desvioPadrao(t: scalagrad.core.Tensor): Double =
    val v = t.toArray
    val media = v.sum / v.length
    Math.sqrt(v.map(x => (x - media) * (x - media)).sum / v.length)

  "initScale" should "multiply the Kaiming standard deviation" in {
    val semEscala = Linear(64, 64, rng = new Random(3))
    val comEscala = Linear(64, 64, initScale = 0.25, rng = new Random(3))

    val esperado = Math.sqrt(2.0 / 64)
    desvioPadrao(semEscala.weights) shouldBe esperado +- (esperado * 0.1)
    desvioPadrao(comEscala.weights) shouldBe (esperado * 0.25) +- (esperado * 0.25 * 0.1)
  }

  it should "default to 1.0, leaving the initialization untouched" in {
    val padrao = Linear(64, 64, rng = new Random(11))
    val explicito = Linear(64, 64, initScale = 1.0, rng = new Random(11))

    padrao.weights.toArray shouldBe explicito.weights.toArray
  }

  it should "not touch the bias, which starts at zero anyway" in {
    val camada = Linear(8, 8, initScale = 0.1, rng = new Random(5))

    camada.bias.get.toArray.forall(_ == 0.0) shouldBe true
  }
end LinearSpec
