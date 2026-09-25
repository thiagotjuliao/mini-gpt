package gpt.nn

import scalagrad.core.Tensor
import scalagrad.ops.tensor.*
import scalagrad.gradcheck.Gradcheck
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class EmbeddingSpec extends AnyFlatSpec with Matchers:

  private def tokensOf(rows: Array[Int]*): Tensor =
    val batchSize = rows.length
    val seqLen = rows.head.length
    val data = rows.flatten.map(_.toDouble).toArray

    Tensor.make(data, Array(batchSize, seqLen))

  "Embedding.embed" should "produce output of shape (batchSize, seqLen, dModel)" in {
    val embedding = new Embedding(vocabSize = 10, dModel = 4, contextLength = 5)
    val y = embedding.forward(tokensOf(Array(7, 1, 4), Array(0, 7, 2)))

    y.rank shouldBe 3
    y.shape(0) shouldBe 2
    y.shape(1) shouldBe 3
    y.shape(2) shouldBe 4
  }

  it should "place each token's row plus its position's row at every output slot" in {
    val dModel = 4
    val embedding = new Embedding(vocabSize = 10, dModel, contextLength = 5)
    val rows = Seq(Array(7, 1, 4), Array(0, 7, 2))
    val y = embedding.forward(tokensOf(rows*))

    val tokenTable = embedding.parameters(0)
    val positionTable = embedding.parameters(1)

    for b <- rows.indices; t <- rows.head.indices; d <- 0 until dModel do
      val expected = tokenTable.get(rows(b)(t), d) + positionTable.get(t, d)
      y.get(b, t, d) shouldBe expected +- 1e-12
  }

  it should "give the same token different vectors at different positions" in {
    // se o positional embedding for esquecido, todo teste de shape e de
    // lookup ainda passa -- este e o que quebra.
    val dModel = 3
    val embedding = new Embedding(vocabSize = 6, dModel, contextLength = 4)
    val y = embedding.forward(tokensOf(Array(5, 5)))

    val positionTable = embedding.parameters(1)
    val slot0 = (0 until dModel).map(d => y.get(0, 0, d))
    val slot1 = (0 until dModel).map(d => y.get(0, 1, d))

    slot0 should not equal slot1

    for d <- 0 until dModel do
      val positionalGap = positionTable.get(0, d) - positionTable.get(1, d)
      (slot0(d) - slot1(d)) shouldBe positionalGap +- 1e-12
  }

  it should "reuse the same position rows across every sequence in the batch" in {
    // a soma [B,T,D] + [T,D] depende do broadcast com left-pad: a tabela
    // posicional nao tem eixo de batch e precisa se repetir sobre ele.
    val dModel = 3
    val embedding = new Embedding(vocabSize = 8, dModel, contextLength = 4)
    val rows = Seq(Array(1, 2), Array(3, 4))
    val y = embedding.forward(tokensOf(rows*))

    val tokenTable = embedding.parameters(0)

    for t <- rows.head.indices; d <- 0 until dModel do
      val positionalFirst = y.get(0, t, d) - tokenTable.get(rows(0)(t), d)
      val positionalSecond = y.get(1, t, d) - tokenTable.get(rows(1)(t), d)

      positionalFirst shouldBe positionalSecond +- 1e-12
  }

  it should "reject a tokens tensor that is not rank-2" in {
    val embedding = new Embedding(vocabSize = 10, dModel = 4, contextLength = 5)
    val flatTokens = Tensor.make(Array(1.0, 2.0, 3.0), Array(3))

    an[IllegalArgumentException] should be thrownBy embedding.forward(flatTokens)
  }

  it should "reject a sequence longer than the context length" in {
    val embedding = new Embedding(vocabSize = 10, dModel = 4, contextLength = 2)

    an[IllegalArgumentException] should be thrownBy embedding.forward(tokensOf(Array(1, 2, 3)))
  }

  it should "reject a token index outside the vocabulary" in {
    val embedding = new Embedding(vocabSize = 5, dModel = 4, contextLength = 5)

    an[IllegalArgumentException] should be thrownBy embedding.forward(tokensOf(Array(0, 5)))
  }

  "Embedding.parameters" should "return exactly [tokenTable, positionTable], both trainable" in {
    val embedding = new Embedding(vocabSize = 10, dModel = 4, contextLength = 6)

    embedding.parameters.size shouldBe 2

    val tokenTable = embedding.parameters(0)
    tokenTable.shape(0) shouldBe 10
    tokenTable.shape(1) shouldBe 4

    val positionTable = embedding.parameters(1)
    positionTable.shape(0) shouldBe 6
    positionTable.shape(1) shouldBe 4

    embedding.parameters.foreach(p => p.requiresGradient shouldBe true)
  }

  "Embedding table initialization" should "draw from N(0, 0.02^2) rather than a repeated constant" in {
    val embedding = new Embedding(vocabSize = 100, dModel = 50, contextLength = 8)
    val tokenTable = embedding.parameters(0)
    val values = (0 until tokenTable.size).map(i => tokenTable.get(tokenTable.unravelIndex(i)*))

    values.toSet.size should be > 1

    val mean = values.sum / values.size
    val std = Math.sqrt(values.map(v => (v - mean) * (v - mean)).sum / values.size)

    // 5000 amostras: folga de 20% e larga o bastante pra nao dar flakiness.
    std shouldBe 0.02 +- (0.02 * 0.2)
  }

  // Lote compartilhado pelos dois testes de backward direto: o token 1 aparece
  // tres vezes, o 4 e o 5 nunca aparecem.
  private val sparseRows = Seq(Array(1, 3, 1), Array(0, 2, 1))
  private val sparseVocabSize = 6
  private val sparseDim = 3

  /** Roda o backward de `(forward(tokens) * weights).sum` num `Embedding` novo e
    * devolve a tabela de tokens junto dos pesos usados. Os pesos sao todos
    * distintos de proposito: um gradiente uniforme (`.sum` puro) daria o mesmo
    * resultado com as contribuicoes trocadas de posicao.
    */
  private def runWeightedBackward(): (Tensor, Tensor) =
    val embedding = new Embedding(sparseVocabSize, sparseDim, contextLength = 4)
    val size = sparseRows.length * sparseRows.head.length * sparseDim
    val weights =
      Tensor.make(
        Array.tabulate(size)(i => (i + 1).toDouble),
        Array(sparseRows.length, sparseRows.head.length, sparseDim)
      )

    (embedding.forward(tokensOf(sparseRows*)) * weights).sum.backward()

    (embedding.parameters(0), weights)

  "Embedding backward" should "leave rows of tokens absent from the batch at exactly zero" in {
    val (tokenTable, _) = runWeightedBackward()
    val used = sparseRows.flatten.toSet

    for row <- 0 until sparseVocabSize; d <- 0 until sparseDim do
      val gradient = tokenTable.gradient(tokenTable.index(row, d))

      if used.contains(row) then gradient should not be 0.0
      else gradient shouldBe 0.0
  }

  it should "sum every contribution into a row whose token repeats" in {
    val (tokenTable, weights) = runWeightedBackward()

    for row <- 0 until sparseVocabSize; d <- 0 until sparseDim do
      val expected = {
        for
          b <- sparseRows.indices
          t <- sparseRows.head.indices
          if sparseRows(b)(t) == row
        yield weights.get(b, t, d)
      }.sum

      tokenTable.gradient(tokenTable.index(row, d)) shouldBe expected +- 1e-12
  }

  "Embedding" should "pass gradient check w.r.t. the token table, with a repeated token" in {
    // o token 1 aparece tres vezes: as contribuicoes tem que SOMAR na linha 1
    // (scatter-add, theory/09-embedding/09-embedding.md Secao 3). A linha 5
    // nunca e selecionada e deve ficar em zero -- o Gradcheck compara posicao
    // por posicao da tabela inteira, entao um vazamento apareceria aqui.
    val embedding = new Embedding(vocabSize = 6, dModel = 3, contextLength = 4)
    val tokens = tokensOf(Array(1, 3, 1), Array(0, 2, 1))

    Gradcheck.run(embedding.parameters(0))(_ => embedding.forward(tokens).sum) should be < 1e-5
  }

  it should "pass gradient check w.r.t. the position table, summing over the batch" in {
    // embedding novo: Gradient so acumula, entao reusar o do teste anterior
    // corromperia o gradiente analitico lido aqui.
    val embedding = new Embedding(vocabSize = 6, dModel = 3, contextLength = 4)
    val tokens = tokensOf(Array(0, 1, 2), Array(3, 4, 5))

    Gradcheck.run(embedding.parameters(1))(_ => embedding.forward(tokens).sum) should be < 1e-5
  }
end EmbeddingSpec
