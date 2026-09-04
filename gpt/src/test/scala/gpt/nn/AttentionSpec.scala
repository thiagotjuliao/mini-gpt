package gpt.nn

import scala.util.Random
import scalagrad.core.Tensor
import scalagrad.ops.tensor.*
import scalagrad.gradcheck.Gradcheck
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class AttentionSpec extends AnyFlatSpec with Matchers {

  // Dimensoes deliberadamente todas diferentes entre si: shapes quadradas
  // escondem indices trocados (theory/03-elementary-operations Secao 6).
  private val batchSize = 2
  private val seqLen = 3
  private val dModel = 4
  private val dHead = 2

  // Indices em `parameters`, que e [Wq, bq, Wk, Wv, bv] -- `key` nasce com
  // useBias = false (ver o teste de parametros abaixo). Nomeados porque a
  // lista ja renumerou duas vezes, e `parameters(4)` cru nao denuncia nada
  // quando o layout muda.
  private val qWeight = 0
  private val qBias = 1
  private val kWeight = 2
  private val vWeight = 3
  private val vBias = 4

  private def randomSequence(
      batchSize: Int,
      seqLen: Int,
      dModel: Int,
      requiresGradient: Boolean = false
  ): Tensor = {
    val data = Array.fill(batchSize * seqLen * dModel)(Random.nextDouble() * 2 - 1)
    Tensor.make(data, Array(batchSize, seqLen, dModel), requiresGradient)
  }

  /** Pesos distintos pra perda escalar do gradient check. Nunca usar `.sum`
    * puro sobre os pesos de atencao: `softmax(x).sum` e funcao constante e
    * aprova qualquer backward (theory/06-softmax Secao 5; o mesmo erro
    * reapareceu na Etapa 10).
    */
  private def lossWeights(shape: Array[Int]): Tensor = {
    val size = shape.product
    Tensor.make(Array.tabulate(size)(i => 0.3 + 0.7 * Math.sin(i * 1.7)), shape)
  }

  /** Projecao de uma camada `Linear` em Scala puro, lendo W e b de
    * `parameters`. Serve de base pra referencia independente do bloco todo.
    * O vies e opcional, espelhando o `Option` do proprio `Linear`.
    */
  private def project(
      x: Tensor,
      w: Tensor,
      b: Option[Tensor] = None
  ): Array[Array[Array[Double]]] = {
    val bs = x.shape(0)
    val t = x.shape(1)
    val din = x.shape(2)
    val dout = w.shape(1)

    Array.tabulate(bs, t, dout) { (i, j, o) =>
      (0 until din).map(k => x.get(i, j, k) * w.get(k, o)).sum + b.fold(0.0)(_.get(o))
    }
  }

  /** Referencia do bloco inteiro em Scala puro -- nao usa nenhuma operacao de
    * `scalagrad`, so le os 5 parametros. Devolve (pesos de atencao, saida).
    */
  private def reference(
      att: Attention,
      x: Tensor
  ): (Array[Array[Array[Double]]], Array[Array[Array[Double]]]) = {
    val ps = att.parameters
    val q = project(x, ps(qWeight), Some(ps(qBias)))
    val k = project(x, ps(kWeight))
    val v = project(x, ps(vWeight), Some(ps(vBias)))

    val bs = x.shape(0)
    val t = x.shape(1)
    val dh = ps(qWeight).shape(1)
    val scale = Math.sqrt(dh)

    val weights = Array.tabulate(bs, t, t) { (b, i, j) =>
      if j > i then Double.NegativeInfinity
      else (0 until dh).map(h => q(b)(i)(h) * k(b)(j)(h)).sum / scale
    }

    // softmax por linha, com subtracao do maximo (log-sum-exp)
    for (b <- 0 until bs; i <- 0 until t) {
      val row = weights(b)(i)
      val max = row.max
      val exps = row.map(s => Math.exp(s - max))
      val total = exps.sum
      for (j <- 0 until t) row(j) = exps(j) / total
    }

    val y = Array.tabulate(bs, t, dh) { (b, i, h) =>
      (0 until t).map(j => weights(b)(i)(j) * v(b)(j)(h)).sum
    }

    (weights, y)
  }

  private def perturbPosition(x: Tensor, position: Int, delta: Double): Tensor = {
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
  }

  // ---- forward ----

  "Attention.forward" should "match a plain Scala implementation of the whole block" in {
    val att = Attention(dModel, dHead)
    val x = randomSequence(batchSize, seqLen, dModel)

    val y = att.forward(x)
    val (_, expected) = reference(att, x)

    for (b <- 0 until batchSize; t <- 0 until seqLen; h <- 0 until dHead)
      withClue(s"posicao (b=$b, t=$t, h=$h): ") {
        y.get(b, t, h) shouldBe expected(b)(t)(h) +- 1e-12
      }
  }

  it should "produce output of shape (batchSize, seqLen, dHead), for varied dimensions" in {
    for ((bs, t, dm, dh) <- Seq((1, 1, 1, 1), (1, 5, 3, 2), (4, 2, 6, 3), (2, 7, 3, 5))) {
      val y = Attention(dm, dh).forward(randomSequence(bs, t, dm))

      withClue(s"(batchSize=$bs, seqLen=$t, dModel=$dm, dHead=$dh): ") {
        y.rank shouldBe 3
        y.shape(0) shouldBe bs
        y.shape(1) shouldBe t
        y.shape(2) shouldBe dh
      }
    }
  }

  it should "accept a non-contiguous input" in {
    // x montado como [B, dModel, seqLen] e transposto: as strides deixam de
    // ser canonicas. E o cenario que quebra qualquer codigo que leia
    // `t.data(i)` em vez de `t.get(...)` (theory/11-attention Secao 7).
    val att = Attention(dModel, dHead)
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

    val y = att.forward(x)
    val yContiguous = att.forward(same)

    for (b <- 0 until batchSize; t <- 0 until seqLen; h <- 0 until dHead)
      withClue(s"posicao (b=$b, t=$t, h=$h): ") {
        y.get(b, t, h) shouldBe yContiguous.get(b, t, h) +- 1e-12
      }
  }

  // ---- parametros ----

  "Attention.parameters" should "expose [Wq, bq, Wk, Wv, bv] -- the key projection has no bias" in {
    // Somar uma constante a TODAS as keys desloca a linha inteira de scores
    // pelo mesmo valor (o termo extra e q_i . bk, que nao depende de j), e o
    // softmax e invariante a isso. Um vies em `key` teria gradiente
    // exatamente zero para sempre -- consequencia de que o gradiente que
    // atravessa o softmax soma zero ao longo da dimensao normalizada
    // (theory/06-softmax Secao 4). Por isso `key` nasce com useBias = false:
    // seria um parametro morto carregado pelo otimizador da Etapa 17.
    // Com bq nao acontece -- ver o teste de gradiente nao trivial abaixo.
    val ps = Attention(dModel, dHead).parameters

    ps.size shouldBe 5

    for (i <- Seq(qWeight, kWeight, vWeight)) withClue(s"peso $i: ") {
      ps(i).rank shouldBe 2
      ps(i).shape(0) shouldBe dModel
      ps(i).shape(1) shouldBe dHead
    }

    for (i <- Seq(qBias, vBias)) withClue(s"vies $i: ") {
      ps(i).rank shouldBe 1
      ps(i).shape(0) shouldBe dHead
    }
  }

  it should "mark every parameter as requiring gradient" in {
    // sem isso o otimizador da Etapa 17 nao teria como atualizar a camada.
    Attention(dModel, dHead).parameters.foreach(p => p.requiresGradient shouldBe true)
  }

  it should "not share one projection between Q, K and V" in {
    // regressao: reusar uma unica Linear pras tres projecoes faria Q = K = V,
    // e a matriz de scores viraria simetrica. Formatos batem, o forward roda e
    // o gradient check passa -- so a assimetria denuncia (theory/11-attention
    // Secao 3, FIG. 2).
    val ps = Attention(dModel, dHead).parameters
    val projections = Seq(qWeight, kWeight, vWeight).map { i =>
      (0 until ps(i).size).map(n => ps(i).get(ps(i).unravelIndex(n)*)).toList
    }

    projections.toSet.size shouldBe 3
  }

  // ---- mascara causal ----

  "The causal mask" should "leave every earlier output untouched when the last token changes" in {
    val att = Attention(dModel, dHead)
    val x = randomSequence(batchSize, seqLen, dModel)
    val perturbed = perturbPosition(x, seqLen - 1, 3.0)

    val y = att.forward(x)
    val yPerturbed = att.forward(perturbed)

    for (b <- 0 until batchSize; t <- 0 until seqLen - 1; h <- 0 until dHead)
      withClue(s"posicao (b=$b, t=$t, h=$h): ") {
        y.get(b, t, h) shouldBe yPerturbed.get(b, t, h) +- 1e-15
      }
  }

  it should "still let an earlier token influence the later ones" in {
    // contraprova do teste acima: uma mascara que bloqueasse tudo fora da
    // diagonal passaria na causalidade e estaria igualmente errada.
    val att = Attention(dModel, dHead)
    val x = randomSequence(batchSize, seqLen, dModel)
    val perturbed = perturbPosition(x, 0, 3.0)

    val y = att.forward(x)
    val yPerturbed = att.forward(perturbed)

    val moved = for {
      b <- 0 until batchSize
      h <- 0 until dHead
    } yield Math.abs(y.get(b, seqLen - 1, h) - yPerturbed.get(b, seqLen - 1, h))

    moved.max should be > 1e-6
  }

  it should "give the first position exactly v0" in {
    // a linha 0 dos pesos e [1, 0, ..., 0] quaisquer que sejam os scores,
    // entao a media ponderada devolve o proprio value. Aplicar a mascara
    // DEPOIS do softmax encolheria y0 por um fator < 1 (theory/11-attention
    // Secao 5). Sem tolerancia de aproximacao: e igualdade exata.
    val att = Attention(dModel, dHead)
    val x = randomSequence(batchSize, seqLen, dModel)

    val y = att.forward(x)
    val v = project(x, att.parameters(vWeight), Some(att.parameters(vBias)))

    for (b <- 0 until batchSize; h <- 0 until dHead)
      withClue(s"posicao (b=$b, h=$h): ") {
        y.get(b, 0, h) shouldBe v(b)(0)(h) +- 1e-15
      }
  }

  it should "make every output row identical when every input token is identical" in {
    // com a sequencia constante, todo score e igual e cada linha de P e
    // uniforme sobre o que ela pode ver -- entao toda saida vale v0, mas SO
    // se cada linha somar 1. Com a mascara depois do softmax as linhas somam
    // 0.446, 0.903 e 1.0, e as saidas saem diferentes umas das outras.
    val att = Attention(dModel, dHead)
    val token = Array.fill(dModel)(Random.nextDouble() * 2 - 1)
    val x = Tensor.make(
      Array.tabulate(batchSize * seqLen * dModel)(i => token(i % dModel)),
      Array(batchSize, seqLen, dModel)
    )

    val y = att.forward(x)

    for (b <- 0 until batchSize; t <- 1 until seqLen; h <- 0 until dHead)
      withClue(s"posicao (b=$b, t=$t, h=$h) contra a linha 0: ") {
        y.get(b, t, h) shouldBe y.get(b, 0, h) +- 1e-12
      }
  }

  it should "keep every output inside the convex envelope of the allowed values" in {
    // consequencia de pesos nao-negativos que somam 1: a saida e uma media
    // ponderada, nunca uma extrapolacao.
    val att = Attention(dModel, dHead)
    val x = randomSequence(batchSize, seqLen, dModel)

    val y = att.forward(x)
    val v = project(x, att.parameters(vWeight), Some(att.parameters(vBias)))

    for (b <- 0 until batchSize; t <- 0 until seqLen; h <- 0 until dHead) {
      val allowed = (0 to t).map(j => v(b)(j)(h))

      withClue(s"posicao (b=$b, t=$t, h=$h): ") {
        y.get(b, t, h) should (be >= allowed.min - 1e-12 and be <= allowed.max + 1e-12)
      }
    }
  }

  it should "produce attention rows that sum to 1, with zeros in the masked positions" in {
    val att = Attention(dModel, dHead)
    val x = randomSequence(batchSize, seqLen, dModel)
    val (weights, _) = reference(att, x)

    for (b <- 0 until batchSize; i <- 0 until seqLen)
      withClue(s"linha (b=$b, i=$i): ") {
        weights(b)(i).sum shouldBe 1.0 +- 1e-12
        weights(b)(i).drop(i + 1).foreach(w => w shouldBe 0.0)
      }
  }

  // ---- gradientes ----

  "Attention" should "pass gradient check w.r.t. x" in {
    val att = Attention(dModel, dHead)
    val x = randomSequence(batchSize, seqLen, dModel, requiresGradient = true)
    val w = lossWeights(Array(batchSize, seqLen, dHead))

    Gradcheck.run(x)(x => (att.forward(x) * w).sum) should be < 1e-5
  }

  it should "pass gradient check w.r.t. Wq, bq, Wk, Wv and bv" in {
    // uma instancia nova por parametro: Gradient so acumula (nunca
    // sobrescreve), entao um segundo Gradcheck.run sobre a mesma camada leria
    // o gradiente somado do anterior (mesma nota do LinearSpec).
    //
    // Sem excecoes: desde que `key` perdeu o vies, todo parametro desta
    // camada tem gradiente nao trivial. Antes, bk precisava ficar de fora --
    // gradiente verdadeiro zero reprova na metrica relativa do Gradcheck.
    val w = lossWeights(Array(batchSize, seqLen, dHead))

    for (i <- Seq(qWeight, qBias, kWeight, vWeight, vBias)) {
      val att = Attention(dModel, dHead)
      val x = randomSequence(batchSize, seqLen, dModel)

      withClue(s"parametro $i: ") {
        Gradcheck.run(att.parameters(i))(_ => (att.forward(x) * w).sum) should be < 1e-5
      }
    }
  }

  it should "give bq a non-trivial gradient" in {
    // Contraprova da assimetria documentada no teste de parametros, e a
    // justificativa de bq ter FICADO enquanto bk saiu: deslocar as queries
    // acrescenta bq . k_j ao score, termo que VARIA com j e portanto muda a
    // distribuicao de atencao. bq aprende; um vies em key nunca aprenderia.
    val att = Attention(dModel, dHead)
    val x = randomSequence(batchSize, seqLen, dModel)
    val w = lossWeights(Array(batchSize, seqLen, dHead))

    val out = (att.forward(x) * w).sum
    out.backward()

    val bq = att.parameters(qBias)
    (0 until bq.size).map(n => Math.abs(bq.gradient(n))).max should be > 1e-6
  }

  it should "give Wq a zero gradient when the sequence has a single position" in {
    // com seqLen = 1 a linha de pesos e constante em [1], entao mexer em Wq
    // nao muda saida nenhuma. E correto, nao bug (theory/11-attention Secao 7,
    // onde a primeira linha de dQ da exatamente [0, 0]).
    val att = Attention(dModel, dHead)
    val x = randomSequence(batchSize, 1, dModel)
    val w = lossWeights(Array(batchSize, 1, dHead))

    val out = (att.forward(x) * w).sum
    out.backward()

    val wq = att.parameters(qWeight)
    for (n <- 0 until wq.size)
      withClue(s"Wq[$n]: ") {
        wq.gradient(n) shouldBe 0.0 +- 1e-15
      }
  }

  // ---- validacao de entrada ----

  "Attention.forward" should "reject an input that is not rank 3" in {
    val att = Attention(dModel, dHead)

    an[IllegalArgumentException] should be thrownBy att.forward(
      Tensor.make(Array.fill(seqLen * dModel)(0.0), Array(seqLen, dModel))
    )
  }

  it should "reject an input whose last dimension is not dModel" in {
    val att = Attention(dModel, dHead)

    an[IllegalArgumentException] should be thrownBy att.forward(
      randomSequence(batchSize, seqLen, dModel + 1)
    )
  }
}
