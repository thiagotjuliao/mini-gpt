package gpt.model

import scala.util.Random
import gpt.nn.TransformerBlock
import scalagrad.core.Tensor
import scalagrad.ops.tensor.*
import scalagrad.gradcheck.Gradcheck
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class GPTSpec extends AnyFlatSpec with Matchers:

  // Todas distintas entre si: um eixo trocado sobrevive a qualquer
  // coincidencia. contextLength > seqLen de proposito, pra que o teto da
  // tabela posicional nao seja exercitado por acidente nos testes de valor.
  private val batchSize = 2
  private val seqLen = 3
  private val dModel = 8
  private val nHeads = 4
  private val nLayers = 2
  private val vocabSize = 7
  private val contextLength = 5
  private val expansion = 2

  private val rng = new Random(20260831)

  private def model(
      nLayers: Int = nLayers,
      vocabSize: Int = vocabSize,
      dModel: Int = dModel,
      nHeads: Int = nHeads,
      contextLength: Int = contextLength,
      expansion: Int = expansion
  ): GPT = GPT(vocabSize, dModel, nHeads, nLayers, contextLength, expansion, rng)

  private def tokensOf(rows: Array[Int]*): Tensor =
    Tensor.make(rows.flatten.map(_.toDouble).toArray, Array(rows.size, rows.head.length))

  private def randomTokens(batchSize: Int, seqLen: Int, vocabSize: Int): Tensor =
    Tensor.make(
      Array.fill(batchSize * seqLen)(rng.nextInt(vocabSize).toDouble),
      Array(batchSize, seqLen)
    )

  private def lossWeights(shape: Array[Int]): Tensor =
    val size = shape.product
    Tensor.make(Array.tabulate(size)(i => 0.3 + 0.7 * Math.sin(i * 1.7)), shape)

  private def maxDifference(a: Tensor, b: Tensor): Double =
    a.shape.toArray shouldBe b.shape.toArray

    (0 until a.size)
      .map(i => Math.abs(a.get(a.unravelIndex(i)*) - b.get(b.unravelIndex(i)*)))
      .max

  /** O modelo remontado a partir dos proprios campos. Vale a ressalva da
    * Etapa 14: uma referencia feita dos campos do objeto herda os erros que
    * estao NOS campos -- por isso os testes de formato e de contagem, que
    * olham de fora, sao indispensaveis aqui.
    */
  private def reference(m: GPT, tokens: Tensor): Tensor =
    val x = m.blocks.foldLeft(m.embedding.forward(tokens))((acc, b) => b.forward(acc))
    m.head.forward(m.lnFinal.forward(x))

  private def withBlocksReversed(m: GPT, tokens: Tensor): Tensor =
    val x = m.blocks.reverse.foldLeft(m.embedding.forward(tokens))((acc, b) => b.forward(acc))
    m.head.forward(m.lnFinal.forward(x))

  private def withoutFinalNorm(m: GPT, tokens: Tensor): Tensor =
    val x = m.blocks.foldLeft(m.embedding.forward(tokens))((acc, b) => b.forward(acc))
    m.head.forward(x)

  /** A formula da §7 do capitulo, generalizada no fator de expansao: o
    * `12·d² + 11·d` de la vale para `expansion = 4`, e a suite roda com 2.
    * Por bloco: atencao `4d² + 2d`, MLP `2e·d² + (e+1)d`, dois LayerNorm `4d`.
    */
  private def blockCount(dModel: Int, expansion: Int): Int =
    (4 + 2 * expansion) * dModel * dModel + (expansion + 7) * dModel

  private def parameterCount(
      vocabSize: Int,
      dModel: Int,
      nLayers: Int,
      contextLength: Int,
      expansion: Int
  ): Int =
    vocabSize * dModel + // tabela de token
      contextLength * dModel + // tabela posicional
      nLayers * blockCount(dModel, expansion) + // a pilha
      2 * dModel + // LayerNorm final
      dModel * vocabSize // cabeca de linguagem, sem pesos amarrados

  // ---- forward ----

  "GPT.forward" should "turn [B, T] of token indices into [B, T, vocabSize] of logits" in {
    val logits = model().forward(randomTokens(batchSize, seqLen, vocabSize))

    logits.shape.toArray shouldBe Array(batchSize, seqLen, vocabSize)
  }

  it should "preserve that contract across varied dimensions" in {
    val cases = Seq((1, 1, 2, 1, 1, 3, 4), (3, 4, 4, 2, 2, 5, 6), (2, 6, 6, 3, 3, 9, 6))

    for (b, t, d, h, l, v, c) <- cases do
      withClue(s"B=$b T=$t d=$d heads=$h L=$l V=$v ctx=$c: ") {
        val logits = model(l, v, d, h, c).forward(randomTokens(b, t, v))

        logits.shape.toArray shouldBe Array(b, t, v)
      }
  }

  it should "match the composition of its own pieces" in {
    val gpt = model()
    val tokens = randomTokens(batchSize, seqLen, vocabSize)

    maxDifference(gpt.forward(tokens), reference(gpt, tokens)) should be < 1e-12
  }

  it should "apply the blocks in order" in {
    // com nLayers >= 2 e blocos independentes, inverter a ordem tem que mudar
    // o resultado. Se nao mudar, provavelmente todos os blocos sao a mesma
    // instancia -- veja tambem o teste de independencia abaixo.
    val gpt = model()
    val tokens = randomTokens(batchSize, seqLen, vocabSize)

    maxDifference(gpt.forward(tokens), withBlocksReversed(gpt, tokens)) should be > 1e-6
  }

  it should "run the final LayerNorm before the head" in {
    val gpt = model()
    val tokens = randomTokens(batchSize, seqLen, vocabSize)

    maxDifference(gpt.forward(tokens), withoutFinalNorm(gpt, tokens)) should be > 1e-6
  }

  // ---- o que a arquitetura precisa distinguir ----

  "GPT" should "give different logits when the prefix order changes" in {
    // O teste que carrega a etapa (theory/15-gpt-model §2). Sem embedding
    // posicional, uma camada de atencao causal enxerga o CONJUNTO de tokens
    // visiveis e nao a ordem: a saida da ultima posicao seria identica bit a
    // bit.
    //
    // `nLayers = 1` NAO e detalhe. Com duas camadas ou mais, a propria
    // mascara causal vaza posicao -- a posicao 0 ve um token, a 1 ve dois, e
    // essa assimetria propaga. Medido em Python puro, com o mesmo prefixo
    // trocado: diferenca 0.0 com uma camada, 5.1e-2 com duas, 6.4e-1 com
    // tres. Um modelo profundo distingue a ordem mesmo sem a tabela, e o
    // teste passaria de graca.
    val gpt = model(nLayers = 1)
    val original = tokensOf(Array(1, 4, 6))
    val swapped = tokensOf(Array(4, 1, 6))

    val a = gpt.forward(original)
    val b = gpt.forward(swapped)

    val lastPosition = (0 until vocabSize)
      .map(v => Math.abs(a.get(0, seqLen - 1, v) - b.get(0, seqLen - 1, v)))
      .max

    lastPosition should be > 1e-6
  }

  it should "stay causal from end to end" in {
    val gpt = model()
    val original = tokensOf(Array(1, 4, 6))
    val lastChanged = tokensOf(Array(1, 4, 2))

    val a = gpt.forward(original)
    val b = gpt.forward(lastChanged)

    for t <- 0 until seqLen - 1; v <- 0 until vocabSize do
      withClue(s"logit (t=$t, v=$v): ") {
        b.get(0, t, v) shouldBe a.get(0, t, v) +- 1e-12
      }

    // contraprova: a ultima posicao TEM que mudar, senao o teste acima
    // passaria num modelo que ignora a entrada
    val lastPosition = (0 until vocabSize)
      .map(v => Math.abs(a.get(0, seqLen - 1, v) - b.get(0, seqLen - 1, v)))
      .max

    lastPosition should be > 1e-6
  }

  // ---- parametros ----

  "GPT.parameters" should "have no repeated tensor, and mark all of them trainable" in {
    val gpt = model()

    gpt.parameters.toSet.size shouldBe gpt.parameters.size
    gpt.parameters.foreach(p => p.requiresGradient shouldBe true)
  }

  it should "give every block its own tensors" in {
    // `List.fill(n)(TransformerBlock(...))` avalia a expressao n vezes, e esta
    // certo. `val b = TransformerBlock(...); List.fill(n)(b)` compartilharia
    // uma instancia so -- formato, causalidade e gradient check passariam.
    val gpt = model()

    gpt.blocks.size shouldBe nLayers
    gpt.blocks.flatMap(_.parameters).toSet.size shouldBe 14 * nLayers
  }

  it should "total V·d + C·d + L·bloco + 2d + d·V" in {
    val configs = Seq((7, 8, 2, 5, 2), (5, 4, 1, 3, 4), (13, 16, 3, 9, 1), (6, 8, 2, 4, 4))

    for (v, d, l, c, e) <- configs do
      withClue(s"V=$v d=$d L=$l ctx=$c expansion=$e: ") {
        val gpt =
          model(
            nLayers = l,
            vocabSize = v,
            dModel = d,
            nHeads = 2,
            contextLength = c,
            expansion = e
          )

        gpt.parameters.map(_.size).sum shouldBe parameterCount(v, d, l, c, e)
      }
  }

  it should "reduce to 12·dModel² + 11·dModel per block with the default expansion" in {
    // a forma canonica da §7, que so vale para expansion = 4
    for d <- Seq(4, 8, 128) do
      withClue(s"dModel=$d: ") {
        blockCount(d, 4) shouldBe 12 * d * d + 11 * d
      }
  }

  it should "put 96% of the tiny configuration in the stack" in {
    // sanity check da §7: com vocabulario de 65 caracteres, dModel 128 e 4
    // blocos, quase tudo esta na pilha. Conferido pela formula, sem construir
    // o modelo -- 817 mil parametros seriam lentos de alocar num teste.
    val total = parameterCount(65, 128, 4, 128, expansion = 4)
    val stack = 4 * (12 * 128 * 128 + 11 * 128)

    total shouldBe 825344
    stack shouldBe 792064
    (stack.toDouble / total) should be > 0.95
  }

  // ---- gradientes ----

  "GPT" should "pass gradient check on every parameter, in a tiny configuration" in {
    // Configuracao minuscula de proposito: diferencas finitas fazem DOIS
    // forwards completos por elemento do tensor conferido (§8 do capitulo).
    // Nao ha gradient check em relacao a entrada -- ela e um indice inteiro,
    // e perturba-la nao faz sentido.
    //
    // `eps = 1e-6` em vez do 1e-5 padrao. O modelo inteiro tem curvatura alta
    // o bastante para o erro de truncamento dominar em 1e-5 -- medido: erro
    // maximo 2.4e-4 la, contra 2e-7 a 9e-7 em 1e-6, e de volta a 6e-6 em
    // 1e-7, quando o arredondamento assume. E a curva em U da Etapa 4 §1,
    // com o minimo deslocado por causa da profundidade da composicao.
    val tokens = tokensOf(Array(1, 3, 0), Array(2, 2, 4))
    val w = lossWeights(Array(2, 3, 5))
    val size = GPT(5, 4, 2, 1, 4, expansion, rng).parameters.size

    for i <- 0 until size do
      val gpt = GPT(5, 4, 2, 1, 4, expansion, rng)

      withClue(s"parametro $i de $size: ") {
        Gradcheck.run(gpt.parameters(i), eps = 1e-6)(_ =>
          (gpt.forward(tokens) * w).sum
        ) should be < 1e-5
      }
  }

  // ---- validacao de entrada ----

  "GPT" should "reject nLayers below 1" in {
    val error = intercept[IllegalArgumentException](model(nLayers = 0))

    error.getMessage should include("0")
  }

  "GPT.forward" should "reject an input that is not rank 2" in {
    val gpt = model()
    val wrongRank =
      Tensor.make(Array.fill(batchSize * seqLen * dModel)(0.0), Array(batchSize, seqLen, dModel))

    val error = intercept[IllegalArgumentException](gpt.forward(wrongRank))

    error.getMessage should include("3")
  }

  it should "reject a sequence longer than the context length" in {
    val gpt = model()
    val tooLong = randomTokens(batchSize, contextLength + 1, vocabSize)

    val error = intercept[IllegalArgumentException](gpt.forward(tooLong))

    error.getMessage should include(contextLength.toString)
    error.getMessage should include((contextLength + 1).toString)
  }

  "two models built with the same seed" should "be identical, parameter by parameter" in {
    // sem `rng` injetado nao ha como reproduzir um treino: "rodei de novo e deu
    // diferente" fica indistinguivel de "mudei alguma coisa".
    val a = GPT(8, 8, 2, 2, 4, rng = new scala.util.Random(123))
    val b = GPT(8, 8, 2, 2, 4, rng = new scala.util.Random(123))

    a.parameters.length shouldBe b.parameters.length
    a.parameters.zip(b.parameters).foreach((x, y) => x.toArray shouldBe y.toArray)
  }

  it should "differ when the seeds differ" in {
    val a = GPT(8, 8, 2, 2, 4, rng = new scala.util.Random(1))
    val b = GPT(8, 8, 2, 2, 4, rng = new scala.util.Random(2))

    val iguais =
      a.parameters.zip(b.parameters).forall((x, y) => x.toArray.sameElements(y.toArray))

    iguais shouldBe false
  }

  private def desvioPadrao(t: Tensor): Double =
    val v = t.toArray
    val media = v.sum / v.length
    Math.sqrt(v.map(x => (x - media) * (x - media)).sum / v.length)

  "the residual scale" should "be 1/sqrt(2 * nLayers)" in {
    // os tres valores da tabela de theory/15-gpt-model §6
    GPT(8, 8, 2, 4, 4).residualScale shouldBe 0.35355 +- 1e-5
    GPT(8, 8, 2, 6, 4).residualScale shouldBe 0.28868 +- 1e-5
    GPT(8, 8, 2, 12, 4).residualScale shouldBe 0.20412 +- 1e-5
  }

  it should "shrink W_O and the MLP down projection, and nothing else" in {
    val nLayers = 8
    val modelo = GPT(8, 64, 4, nLayers, 4, rng = new Random(21))
    val bloco = modelo.blocks.head

    val desvioQ = desvioPadrao(bloco.attention.parameters(0))
    val desvioO = desvioPadrao(bloco.attention.parameters(4))
    val desvioUp = desvioPadrao(bloco.mlp.parameters(0))
    val desvioDown = desvioPadrao(bloco.mlp.parameters(2))

    val escala = 1.0 / Math.sqrt(2.0 * nLayers)

    desvioO / desvioQ shouldBe escala +- 0.05
    desvioDown / (Math.sqrt(2.0 / (64 * 4))) shouldBe escala +- 0.05
    desvioUp shouldBe Math.sqrt(2.0 / 64) +- (Math.sqrt(2.0 / 64) * 0.1)
  }

  it should "keep the residual stream from growing with depth" in {
    // Sem escala o desvio do fluxo cresce quase linearmente com a profundidade;
    // com escala, bem mais devagar. Medido em 2026-09-01: 18.39 contra 4.53 em
    // 8 camadas (ver theory/15-gpt-model §6).
    val dModel = 32
    val nLayers = 8

    val entrada =
      val r = new Random(1)
      Tensor.make(Array.fill(2 * 6 * dModel)(r.nextGaussian()), Array(2, 6, dModel))

    def pilha(escala: Double): Double = Tensor.noGrad {
      val blocos = List.fill(nLayers)(TransformerBlock(dModel, 4, 4, new Random(7), escala))
      desvioPadrao(blocos.foldLeft(entrada)((x, b) => b.forward(x)))
    }

    val semEscala = pilha(1.0)
    val comEscala = pilha(1.0 / Math.sqrt(2.0 * nLayers))

    comEscala should be < semEscala / 3
  }
end GPTSpec
