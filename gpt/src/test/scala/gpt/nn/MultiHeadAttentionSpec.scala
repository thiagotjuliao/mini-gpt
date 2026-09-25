package gpt.nn

import scala.util.Random
import scalagrad.core.Tensor
import scalagrad.ops.tensor.*
import scalagrad.gradcheck.Gradcheck
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MultiHeadAttentionSpec extends AnyFlatSpec with Matchers:

  // Dimensoes deliberadamente todas distintas entre si -- B, T, nHeads, dHead
  // e dModel. Duas que coincidam escondem um eixo trocado: durante a escrita
  // desta camada, `Masks.causalMask(x.shape(1))` sobre os scores construiu a
  // mascara com nHeads no lugar de T, e o unico caso que passava era
  // justamente T == nHeads.
  private val batchSize = 3
  private val seqLen = 5
  private val dModel = 8
  private val nHeads = 4
  private val dHead = dModel / nHeads

  // Indices em `parameters`, que e [Wq, bq, Wk, Wv, Wo, bo] -- `key` e `value`
  // nascem com useBias = false (ver o teste de parametros abaixo). Nomeados
  // porque a lista da Etapa 11 ja renumerou duas vezes.
  private val qWeight = 0
  private val qBias = 1
  private val kWeight = 2
  private val vWeight = 3
  private val oWeight = 4
  private val oBias = 5

  // Semente fixa: os testes de valor comparam numeros, e sem semente uma
  // falha nao e reproduzivel.
  private val rng = new Random(20260822)

  private def randomSequence(
      batchSize: Int,
      seqLen: Int,
      dModel: Int,
      requiresGradient: Boolean = false
  ): Tensor =
    val data = Array.fill(batchSize * seqLen * dModel)(rng.nextDouble() * 2 - 1)
    Tensor.make(data, Array(batchSize, seqLen, dModel), requiresGradient)

  /** Pesos distintos pra perda escalar do gradient check. Nunca usar `.sum`
    * puro sobre os pesos de atencao: `softmax(x).sum` e funcao constante e
    * aprova qualquer backward (theory/06-softmax Secao 5).
    */
  private def lossWeights(shape: Array[Int]): Tensor =
    val size = shape.product
    Tensor.make(Array.tabulate(size)(i => 0.3 + 0.7 * Math.sin(i * 1.7)), shape)

  private def project(
      x: Tensor,
      w: Tensor,
      b: Option[Tensor] = None
  ): Array[Array[Array[Double]]] =
    val bs = x.shape(0)
    val t = x.shape(1)
    val din = x.shape(2)
    val dout = w.shape(1)

    Array.tabulate(bs, t, dout) { (i, j, o) =>
      (0 until din).map(k => x.get(i, j, k) * w.get(k, o)).sum + b.fold(0.0)(_.get(o))
    }

  private def softmaxRow(row: Array[Double]): Array[Double] =
    val max = row.max
    val exps = row.map(s => Math.exp(s - max))
    val total = exps.sum

    exps.map(_ / total)

  /** Referencia do bloco inteiro em Scala puro -- nao usa nenhuma operacao de
    * `scalagrad`, so le os 6 parametros. Devolve (pesos por cabeca, saida).
    *
    * A cabeca de uma coordenada `c` e obtida por `c / dHead`, que e o que o
    * par `reshape` + `transpose(1,2)` precisa produzir. Trocar a ordem dos
    * dois na camada muda esse fatiamento, e este teste denuncia.
    */
  private def reference(
      mha: MultiHeadAttention,
      x: Tensor,
      heads: Int
  ): (Array[Array[Array[Array[Double]]]], Array[Array[Array[Double]]]) =
    val ps = mha.parameters
    val q = project(x, ps(qWeight), Some(ps(qBias)))
    val k = project(x, ps(kWeight))
    val v = project(x, ps(vWeight))

    val bs = x.shape(0)
    val t = x.shape(1)
    val dm = x.shape(2)
    val dh = dm / heads
    val scale = Math.sqrt(dh)

    val scores = Array.tabulate(bs, heads, t, t) { (b, h, i, j) =>
      if j > i then Double.NegativeInfinity
      else (0 until dh).map(d => q(b)(i)(h * dh + d) * k(b)(j)(h * dh + d)).sum / scale
    }

    val p = scores.map(_.map(_.map(softmaxRow)))

    val context = Array.tabulate(bs, t, dm) { (b, i, c) =>
      (0 until t).map(j => p(b)(c / dh)(i)(j) * v(b)(j)(c)).sum
    }

    val y = Array.tabulate(bs, t, dm) { (b, i, o) =>
      (0 until dm).map(c => context(b)(i)(c) * ps(oWeight).get(c, o)).sum + ps(oBias).get(o)
    }

    (p, y)
  end reference

  /** Referencia sem eixo de cabeca nenhum: e literalmente o bloco da Etapa 11
    * com dHead = dModel, seguido de W_O. Serve so pro caso nHeads = 1, e e
    * independente da `reference` acima -- nao tem aritmetica de fatiamento.
    */
  private def referenceSingleHead(
      mha: MultiHeadAttention,
      x: Tensor
  ): Array[Array[Array[Double]]] =
    val ps = mha.parameters
    val q = project(x, ps(qWeight), Some(ps(qBias)))
    val k = project(x, ps(kWeight))
    val v = project(x, ps(vWeight))

    val bs = x.shape(0)
    val t = x.shape(1)
    val dm = x.shape(2)
    val scale = Math.sqrt(dm)

    val scores = Array.tabulate(bs, t, t) { (b, i, j) =>
      if j > i then Double.NegativeInfinity
      else (0 until dm).map(d => q(b)(i)(d) * k(b)(j)(d)).sum / scale
    }

    val p = scores.map(_.map(softmaxRow))

    val context = Array.tabulate(bs, t, dm) { (b, i, d) =>
      (0 until t).map(j => p(b)(i)(j) * v(b)(j)(d)).sum
    }

    Array.tabulate(bs, t, dm) { (b, i, o) =>
      (0 until dm).map(c => context(b)(i)(c) * ps(oWeight).get(c, o)).sum + ps(oBias).get(o)
    }
  end referenceSingleHead

  private def perturbPosition(x: Tensor, position: Int, delta: Double): Tensor =
    val bs = x.shape(0)
    val t = x.shape(1)
    val dm = x.shape(2)

    val data = Array.tabulate(bs * t * dm) { i =>
      val b = i / (t * dm)
      val j = (i / dm) % t
      val d = i % dm
      x.get(b, j, d) + (if j == position then delta else 0.0)
    }

    Tensor.make(data, Array(bs, t, dm))

  // ---- forward ----

  "MultiHeadAttention.forward" should "match a plain Scala implementation of the whole block" in {
    val mha = MultiHeadAttention(dModel, nHeads, rng)
    val x = randomSequence(batchSize, seqLen, dModel)

    val y = mha.forward(x)
    val (_, expected) = reference(mha, x, nHeads)

    for b <- 0 until batchSize; t <- 0 until seqLen; d <- 0 until dModel do
      withClue(s"posicao (b=$b, t=$t, d=$d): ") {
        y.get(b, t, d) shouldBe expected(b)(t)(d) +- 1e-12
      }
  }

  it should "reproduce the single-head block when nHeads = 1" in {
    // A melhor regressao possivel da divisao em cabecas: com uma cabeca so, a
    // maquinaria de reshape/transpose tem que colapsar de volta na Etapa 11.
    // A referencia daqui nao tem aritmetica de fatiamento nenhuma.
    val mha = MultiHeadAttention(dModel, 1, rng)
    val x = randomSequence(batchSize, seqLen, dModel)

    val y = mha.forward(x)
    val expected = referenceSingleHead(mha, x)

    for b <- 0 until batchSize; t <- 0 until seqLen; d <- 0 until dModel do
      withClue(s"posicao (b=$b, t=$t, d=$d): ") {
        y.get(b, t, d) shouldBe expected(b)(t)(d) +- 1e-12
      }
  }

  it should "preserve the input shape, for varied dimensions" in {
    // O formato de entrada e o de saida coincidirem e o que permite empilhar
    // o bloco N vezes na Etapa 15 (theory/12-multi-head-attention Secao 10).
    // Inclui as pontas: uma cabeca so, um token so, dHead = 1.
    for (bs, t, dm, h) <- Seq((1, 1, 1, 1), (1, 4, 6, 3), (2, 1, 8, 8), (4, 6, 9, 3)) do
      val y = MultiHeadAttention(dm, h).forward(randomSequence(bs, t, dm))

      withClue(s"(batchSize=$bs, seqLen=$t, dModel=$dm, nHeads=$h): ") {
        y.rank shouldBe 3
        y.shape(0) shouldBe bs
        y.shape(1) shouldBe t
        y.shape(2) shouldBe dm
      }
  }

  it should "accept a non-contiguous input" in {
    // x montado como [B, dModel, seqLen] e transposto: as strides deixam de
    // ser canonicas. Corolario do CLAUDE.md -- toda op nova precisa de um
    // caso com entrada transposta.
    val mha = MultiHeadAttention(dModel, nHeads, rng)
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

    val y = mha.forward(x)
    val yContiguous = mha.forward(same)

    for b <- 0 until batchSize; t <- 0 until seqLen; d <- 0 until dModel do
      withClue(s"posicao (b=$b, t=$t, d=$d): ") {
        y.get(b, t, d) shouldBe yContiguous.get(b, t, d) +- 1e-12
      }
  }

  // ---- divisao em cabecas ----

  "Splitting into heads" should "be undone exactly by the reverse pair" in {
    // Ida e volta sem atencao no meio, com igualdade exata. Este teste sozinho
    // pega a troca de ordem entre reshape e transpose
    // (theory/12-multi-head-attention Secao 9).
    val t = randomSequence(batchSize, seqLen, dModel)

    val split = t.reshape(Array(batchSize, seqLen, nHeads, dHead)).transpose(1, 2)
    val merged = split.transpose(1, 2).reshape(Array(batchSize, seqLen, dModel))

    split.shape.toArray shouldBe Array(batchSize, nHeads, seqLen, dHead)

    for b <- 0 until batchSize; i <- 0 until seqLen; d <- 0 until dModel do
      withClue(s"posicao (b=$b, t=$i, d=$d): ") {
        merged.get(b, i, d) shouldBe t.get(b, i, d)
      }
  }

  it should "not commute: transpose before reshape gives a different tensor" in {
    // Contraprova do teste acima. A ordem trocada produz shapes compativeis --
    // a contagem de elementos e a mesma --, entao nada estoura e o erro vira
    // resultado silenciosamente errado.
    val t = randomSequence(batchSize, seqLen, dModel)

    val right = t.reshape(Array(batchSize, seqLen, nHeads, dHead)).transpose(1, 2)
    val wrong = t.transpose(1, 2).reshape(Array(batchSize, nHeads, seqLen, dHead))

    wrong.shape.toArray shouldBe right.shape.toArray

    val differences = for
      b <- 0 until batchSize
      h <- 0 until nHeads
      i <- 0 until seqLen
      d <- 0 until dHead
    yield Math.abs(right.get(b, h, i, d) - wrong.get(b, h, i, d))

    differences.max should be > 1e-6
  }

  it should "give the heads different attention distributions" in {
    // Se duas cabecas produzem a mesma linha de P, provavelmente estao lendo
    // a mesma fatia. E a razao de existir da etapa: H distribuicoes em vez de
    // uma (theory/12-multi-head-attention Secao 1).
    val mha = MultiHeadAttention(dModel, nHeads, rng)
    val x = randomSequence(batchSize, seqLen, dModel)
    val p = mha.attentionWeights(x)

    // a ultima posicao enxerga a sequencia inteira, entao e a linha com mais
    // liberdade pra diferir entre cabecas
    val rows = (0 until nHeads).map { h =>
      (0 until seqLen).map(j => p.get(0, h, seqLen - 1, j)).toList
    }

    rows.toSet.size shouldBe nHeads
  }

  // ---- pesos de atencao ----

  "MultiHeadAttention.attentionWeights" should "have shape (B, nHeads, T, T)" in {
    val p = MultiHeadAttention(dModel, nHeads, rng).attentionWeights(
      randomSequence(batchSize, seqLen, dModel)
    )

    p.shape.toArray shouldBe Array(batchSize, nHeads, seqLen, seqLen)
  }

  it should "produce rows that sum to 1, with zeros in the masked positions" in {
    // Nenhuma das duas propriedades e coberta pelo gradient check, que valida
    // a coerencia entre forward e backward -- nao se o forward e o pretendido.
    // Normalizar o eixo errado (o das queries em vez do das keys) mantem
    // shape, causalidade e gradient check, e quebra so a soma das linhas.
    val mha = MultiHeadAttention(dModel, nHeads, rng)
    val p = mha.attentionWeights(randomSequence(batchSize, seqLen, dModel))

    for b <- 0 until batchSize; h <- 0 until nHeads; i <- 0 until seqLen do
      val row = (0 until seqLen).map(j => p.get(b, h, i, j))

      withClue(s"linha (b=$b, h=$h, i=$i): ") {
        row.sum shouldBe 1.0 +- 1e-12
        row.drop(i + 1).foreach(w => w shouldBe 0.0)
      }
  }

  it should "give the first position a weight of exactly 1 on itself" in {
    // A linha 0 e [1, 0, ..., 0] quaisquer que sejam os scores. Aplicar a
    // mascara DEPOIS do softmax encolheria esse 1 por um fator < 1
    // (theory/11-attention Secao 5). Sem tolerancia: e igualdade exata.
    val mha = MultiHeadAttention(dModel, nHeads, rng)
    val p = mha.attentionWeights(randomSequence(batchSize, seqLen, dModel))

    for b <- 0 until batchSize; h <- 0 until nHeads do
      withClue(s"(b=$b, h=$h): ") {
        p.get(b, h, 0, 0) shouldBe 1.0
      }
  }

  it should "match the reference weights position by position" in {
    val mha = MultiHeadAttention(dModel, nHeads, rng)
    val x = randomSequence(batchSize, seqLen, dModel)

    val p = mha.attentionWeights(x)
    val (expected, _) = reference(mha, x, nHeads)

    for b <- 0 until batchSize; h <- 0 until nHeads; i <- 0 until seqLen; j <- 0 until seqLen do
      withClue(s"posicao (b=$b, h=$h, i=$i, j=$j): ") {
        p.get(b, h, i, j) shouldBe expected(b)(h)(i)(j) +- 1e-12
      }
  }

  // ---- mascara causal ----

  "The causal mask" should "leave every earlier output untouched when the last token changes" in {
    val mha = MultiHeadAttention(dModel, nHeads, rng)
    val x = randomSequence(batchSize, seqLen, dModel)
    val perturbed = perturbPosition(x, seqLen - 1, 3.0)

    val y = mha.forward(x)
    val yPerturbed = mha.forward(perturbed)

    for b <- 0 until batchSize; t <- 0 until seqLen - 1; d <- 0 until dModel do
      withClue(s"posicao (b=$b, t=$t, d=$d): ") {
        y.get(b, t, d) shouldBe yPerturbed.get(b, t, d) +- 1e-15
      }
  }

  it should "still let an earlier token influence the later ones" in {
    // Contraprova: uma mascara que bloqueasse tudo fora da diagonal passaria
    // no teste acima e estaria igualmente errada.
    val mha = MultiHeadAttention(dModel, nHeads, rng)
    val x = randomSequence(batchSize, seqLen, dModel)
    val perturbed = perturbPosition(x, 0, 3.0)

    val y = mha.forward(x)
    val yPerturbed = mha.forward(perturbed)

    val moved = for
      b <- 0 until batchSize
      d <- 0 until dModel
    yield Math.abs(y.get(b, seqLen - 1, d) - yPerturbed.get(b, seqLen - 1, d))

    moved.max should be > 1e-6
  }

  // ---- parametros ----

  "MultiHeadAttention.parameters" should "expose [Wq, bq, Wk, Wv, Wo, bo]" in {
    // key e value nascem sem vies, pelas duas propriedades estruturais da
    // linha do softmax (theory/12-multi-head-attention Secao 8):
    //   - invariancia a deslocamento constante mata b_K, que teria gradiente
    //     exatamente zero para sempre;
    //   - linhas que somam 1 fazem b_V atravessar a atencao intacto, virando
    //     uma constante que b_O absorve por completo -- uma direcao plana da
    //     loss, dois parametros para um grau de liberdade.
    val ps = MultiHeadAttention(dModel, nHeads, rng).parameters

    ps.size shouldBe 6

    for i <- Seq(qWeight, kWeight, vWeight, oWeight) do
      withClue(s"peso $i: ") {
        ps(i).rank shouldBe 2
        ps(i).shape(0) shouldBe dModel
        ps(i).shape(1) shouldBe dModel
      }

    for i <- Seq(qBias, oBias) do
      withClue(s"vies $i: ") {
        ps(i).rank shouldBe 1
        ps(i).shape(0) shouldBe dModel
      }
  }

  it should "total 4 * dModel^2 + 2 * dModel, independently of nHeads" in {
    // O ponto central da etapa: multi-head nao custa nada. Como
    // nHeads * dHead = dModel, H cabecas tem exatamente o mesmo numero de
    // pesos que uma cabeca de tamanho dModel
    // (theory/12-multi-head-attention Secao 2).
    val expected = 4 * dModel * dModel + 2 * dModel

    for h <- Seq(1, 2, 4, 8) do
      withClue(s"nHeads=$h: ") {
        MultiHeadAttention(dModel, h, rng).parameters.map(_.size).sum shouldBe expected
      }

    // confere contra a tabela da teoria Secao 9
    MultiHeadAttention(4, 2, rng).parameters.map(_.size).sum shouldBe 72
  }

  it should "mark every parameter as requiring gradient" in {
    // sem isso o otimizador da Etapa 17 nao teria como atualizar a camada.
    MultiHeadAttention(dModel, nHeads, rng).parameters.foreach(p =>
      p.requiresGradient shouldBe true
    )
  }

  it should "not share one projection between Q, K, V and O" in {
    // regressao: reusar uma unica Linear faria Q = K, e a matriz de scores
    // viraria simetrica. Formatos batem, o forward roda e o gradient check
    // passa -- so a assimetria denuncia.
    val ps = MultiHeadAttention(dModel, nHeads, rng).parameters
    val projections = Seq(qWeight, kWeight, vWeight, oWeight).map { i =>
      (0 until ps(i).size).map(n => ps(i).get(ps(i).unravelIndex(n)*)).toList
    }

    projections.toSet.size shouldBe 4
  }

  // ---- gradientes ----

  "MultiHeadAttention" should "pass gradient check w.r.t. x" in {
    val mha = MultiHeadAttention(dModel, nHeads, rng)
    val x = randomSequence(batchSize, seqLen, dModel, requiresGradient = true)
    val w = lossWeights(Array(batchSize, seqLen, dModel))

    Gradcheck.run(x)(x => (mha.forward(x) * w).sum) should be < 1e-5
  }

  it should "pass gradient check w.r.t. every one of the six parameters" in {
    // uma instancia nova por parametro: Gradient so acumula (nunca
    // sobrescreve), entao um segundo run sobre a mesma camada leria o
    // gradiente somado do anterior.
    //
    // Sem excecoes: como key e value perderam o vies, todo parametro desta
    // camada tem gradiente nao trivial. Um b_K sobrevivente reprovaria aqui
    // com erro na casa de 1e-3 -- gradiente verdadeiro zero contra a metrica
    // relativa do Gradcheck (theory/11-attention Secao 8).
    val w = lossWeights(Array(batchSize, seqLen, dModel))

    for i <- 0 until 6 do
      val mha = MultiHeadAttention(dModel, nHeads, rng)
      val x = randomSequence(batchSize, seqLen, dModel)

      withClue(s"parametro $i: ") {
        Gradcheck.run(mha.parameters(i))(_ => (mha.forward(x) * w).sum) should be < 1e-5
      }
  }

  it should "pass gradient check with a non-contiguous input" in {
    val mha = MultiHeadAttention(dModel, nHeads, rng)
    val raw = randomSequence(batchSize, dModel, seqLen, requiresGradient = true)
    val w = lossWeights(Array(batchSize, seqLen, dModel))

    Gradcheck.run(raw)(r => (mha.forward(r.transpose(1, 2)) * w).sum) should be < 1e-5
  }

  // ---- validacao de entrada ----

  "MultiHeadAttention" should "reject a dModel that nHeads does not divide" in {
    val error = intercept[IllegalArgumentException](MultiHeadAttention(dModel = 10, nHeads = 4))

    // a mensagem tem que citar os dois numeros: quem erra isso erra
    // escolhendo nHeads, e precisa ver contra o que
    error.getMessage should include("10")
    error.getMessage should include("4")
  }

  it should "accept every nHeads that divides dModel" in {
    for h <- Seq(1, 2, 4, 8) do
      withClue(s"nHeads=$h: ") {
        MultiHeadAttention(dModel, h, rng).dHead shouldBe dModel / h
      }
  }

  "MultiHeadAttention.forward" should "reject an input that is not rank 3" in {
    val mha = MultiHeadAttention(dModel, nHeads, rng)

    an[IllegalArgumentException] should be thrownBy mha.forward(
      Tensor.make(Array.fill(seqLen * dModel)(0.0), Array(seqLen, dModel))
    )
  }

  it should "reject an input whose last dimension is not dModel" in {
    val mha = MultiHeadAttention(dModel, nHeads, rng)

    an[IllegalArgumentException] should be thrownBy mha.forward(
      randomSequence(batchSize, seqLen, dModel + 1)
    )
  }

  "MultiHeadAttention.attentionWeights" should "validate its input too" in {
    // e uma porta de entrada publica, entao se defende sozinha em vez de
    // depender de o forward ter validado antes.
    val mha = MultiHeadAttention(dModel, nHeads, rng)

    an[IllegalArgumentException] should be thrownBy mha.attentionWeights(
      randomSequence(batchSize, seqLen, dModel + 1)
    )
  }

  private def desvioPadrao(t: scalagrad.core.Tensor): Double =
    val v = t.toArray
    val media = v.sum / v.length
    Math.sqrt(v.map(x => (x - media) * (x - media)).sum / v.length)

  "the residual scale" should "shrink W_O and leave W_Q alone" in {
    // parameters = [W_q, b_q, W_k, W_v, W_o, b_o]
    val escala = 0.25
    val atencao = MultiHeadAttention(64, 4, new Random(9), escala)

    val desvioQ = desvioPadrao(atencao.parameters(0))
    val desvioO = desvioPadrao(atencao.parameters(4))

    // W_O e a projecao que escreve no fluxo residual; W_Q nao
    desvioO / desvioQ shouldBe escala +- 0.1
  }

  it should "default to 1.0, keeping every projection on the same scale" in {
    val atencao = MultiHeadAttention(64, 4, new Random(9))

    desvioPadrao(atencao.parameters(4)) / desvioPadrao(atencao.parameters(0)) shouldBe 1.0 +- 0.15
  }
end MultiHeadAttentionSpec
