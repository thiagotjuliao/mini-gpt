package gpt.optim

import scalagrad.core.Tensor
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class AdamWSpec extends AnyFlatSpec with Matchers:

  private val tolerance = 1e-9

  /** O exemplo que atravessa theory/17-adamw-optimizer: as tres posicoes contam
    * historias diferentes. A 0 troca de sinal no passo 3, a 1 tem gradiente dez
    * vezes menor, e a 2 tem o maior gradiente com sinal constante.
    */
  private val valoresIniciais = Array(0.60, -0.20, 0.05)
  private val g1 = Array(0.30, 0.03, -0.60)
  private val g2 = Array(0.10, 0.05, -0.20)
  private val g3 = Array(-0.20, 0.04, -0.10)

  private val lr = 3e-4

  private def parametro(valores: Array[Double]): Tensor =
    Tensor.make(valores.clone(), Array(valores.length), true)

  private def semDecay(p: Tensor): AdamW =
    new AdamW(List(p), lr = lr, weightDecay = 0.0)

  /** O `Gradient` so acumula, entao cada passo zera antes de semear o proximo --
    * e o que o loop de treino faz entre os lotes.
    */
  private def semear(p: Tensor, gradiente: Array[Double]): Unit =
    p.gradient.zero()
    gradiente.indices.foreach(i => p.gradient.accumulate(i, gradiente(i)))

  private def passo(otimizador: AdamW, p: Tensor, gradiente: Array[Double]): AdamW =
    semear(p, gradiente)
    otimizador.step()

  "AdamW" should "give a first step of exactly lr, whatever the gradient scale" in {
    val p = parametro(valoresIniciais)
    passo(semDecay(p), p, g1)

    // Gradientes de 0.30, 0.03 e -0.60 -- vinte vezes de diferenca entre eles.
    // So o sinal sobrevive. E este o teste que pega a falta da correcao de
    // vies: sem ela o passo sairia com 44.7% deste tamanho.
    val deslocamentos = p.toArray.zip(valoresIniciais).map((novo, velho) => novo - velho)

    deslocamentos(0) shouldBe -lr +- tolerance
    deslocamentos(1) shouldBe -lr +- tolerance
    deslocamentos(2) shouldBe +lr +- tolerance
  }

  it should "not be the 0.4472 factor that omitting bias correction would give" in {
    val p = parametro(valoresIniciais)
    passo(semDecay(p), p, g1)

    val semCorrecao = lr * (0.1 / Math.sqrt(0.05))
    val deslocamento = Math.abs(p.toArray(0) - valoresIniciais(0))

    deslocamento should not be (semCorrecao +- 1e-7)
    semCorrecao shouldBe 1.34164e-4 +- 1e-9
  }

  it should "reproduce the three steps of the chapter" in {
    val p = parametro(valoresIniciais)

    val apos1 = passo(semDecay(p), p, g1)
    p.toArray(0) shouldBe 0.5997 +- tolerance
    p.toArray(1) shouldBe -0.2003 +- tolerance
    p.toArray(2) shouldBe 0.0503 +- tolerance

    val apos2 = passo(apos1, p, g2)
    p.toArray(0) shouldBe 0.59943601 +- 1e-8
    p.toArray(1) shouldBe -0.20059311 +- 1e-8
    p.toArray(2) shouldBe 0.050563988 +- 1e-8

    passo(apos2, p, g3)
    p.toArray(0) shouldBe 0.59936724 +- 1e-8
    p.toArray(1) shouldBe -0.20088846 +- 1e-8
    p.toArray(2) shouldBe 0.050798518 +- 1e-8
  }

  it should "take the same step when every gradient is multiplied by 1000" in {
    val original = parametro(valoresIniciais)
    val escalado = parametro(valoresIniciais)

    passo(semDecay(original), original, g1)
    passo(semDecay(escalado), escalado, g1.map(_ * 1000))

    // A razao m/sqrt(v) e adimensional: o numerador e o denominador crescem
    // juntos. A diferenca que sobra e o eps.
    original.toArray.zip(escalado.toArray).foreach { (a, b) =>
      a shouldBe b +- 1e-9
    }
  }

  it should "let momentum absorb a sign flip instead of reversing the step" in {
    val p = parametro(valoresIniciais)

    val apos1 = passo(semDecay(p), p, g1)
    val apos2 = passo(apos1, p, g2)
    val antesDoTerceiro = p.toArray(0)

    // O gradiente foi +0.30, +0.10 e agora -0.20: trocou de sinal.
    passo(apos2, p, g3)
    val deslocamento = p.toArray(0) - antesDoTerceiro

    // O passo continua descendo (negativo), mas bem menor que os anteriores.
    deslocamento should be < 0.0
    Math.abs(deslocamento) shouldBe 6.8776027e-5 +- 1e-10
    Math.abs(deslocamento) should be < lr
  }

  it should "decay the parameter by (1 - lr * lambda) when the gradient is zero" in {
    val p = parametro(valoresIniciais)
    val otimizador = new AdamW(List(p), lr = lr, weightDecay = 0.1)

    passo(otimizador, p, Array(0.0, 0.0, 0.0))

    // Com gradiente zero, m e v ficam zerados e o passo adaptativo e 0/(0+eps).
    // Sobra so o decaimento, que e puramente multiplicativo.
    val fator = 1.0 - lr * 0.1
    p.toArray(0) shouldBe valoresIniciais(0) * fator +- tolerance
    p.toArray(1) shouldBe valoresIniciais(1) * fator +- tolerance
    p.toArray(2) shouldBe valoresIniciais(2) * fator +- tolerance
  }

  it should "pull a negative parameter towards zero, not downwards" in {
    val p = parametro(Array(-0.20))
    val otimizador = new AdamW(List(p), lr = lr, weightDecay = 0.1)

    passo(otimizador, p, Array(0.0))

    p.toArray(0) should be > -0.20
    p.toArray(0) should be < 0.0
  }

  it should "reproduce the full update of section 8, with decay and gradient together" in {
    val p = parametro(valoresIniciais)
    val otimizador = new AdamW(List(p), lr = lr, weightDecay = 0.1)

    passo(otimizador, p, g1)

    p.toArray(0) shouldBe 0.599682 +- 1e-9
    p.toArray(1) shouldBe -0.200294 +- 1e-9
    p.toArray(2) shouldBe 0.0502985 +- 1e-9
  }

  it should "carry t forward on the returned instance" in {
    val p = parametro(valoresIniciais)
    val inicial = semDecay(p)

    inicial.t shouldBe 0

    val apos1 = passo(inicial, p, g1)
    apos1.t shouldBe 1

    val apos2 = passo(apos1, p, g2)
    apos2.t shouldBe 2
  }

  it should "leave the previous instance untouched when stepping" in {
    val p = parametro(valoresIniciais)
    val inicial = semDecay(p)
    val apos1 = passo(inicial, p, g1)

    inicial.t shouldBe 0
    apos1.t shouldBe 1
  }

  it should "propagate m and v, so a second step differs from a repeated first step" in {
    val comEstado = parametro(valoresIniciais)
    val semEstado = parametro(valoresIniciais)

    val apos1 = passo(semDecay(comEstado), comEstado, g1)
    val antes = comEstado.toArray(0)
    passo(apos1, comEstado, g2)
    val deslocamentoComEstado = comEstado.toArray(0) - antes

    // Um otimizador novo trata g2 como se fosse o primeiro passo.
    passo(semDecay(semEstado), semEstado, g2)
    val deslocamentoSemEstado = semEstado.toArray(0) - valoresIniciais(0)

    deslocamentoComEstado should not be (deslocamentoSemEstado +- 1e-7)
    Math.abs(deslocamentoSemEstado) shouldBe lr +- tolerance
  }

  it should "keep the hyperparameters on the returned instance" in {
    val p = parametro(valoresIniciais)
    val otimizador = new AdamW(List(p), lr = 1e-3, beta1 = 0.8, beta2 = 0.99, weightDecay = 0.2)

    val proximo = passo(otimizador, p, g1)

    proximo.lr shouldBe 1e-3
    proximo.beta1 shouldBe 0.8
    proximo.beta2 shouldBe 0.99
    proximo.weightDecay shouldBe 0.2
  }

  it should "use the overridden lr for the step without changing the configured one" in {
    val p = parametro(valoresIniciais)
    val otimizador = semDecay(p)

    semear(p, g1)
    val proximo = otimizador.step(1e-3)

    Math.abs(p.toArray(0) - valoresIniciais(0)) shouldBe 1e-3 +- tolerance
    proximo.lr shouldBe lr
  }

  it should "update every parameter in the list, of any shape" in {
    val vetor = parametro(Array(0.60, -0.20, 0.05))
    val matriz = Tensor.make(Array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6), Array(2, 3), true)

    semear(vetor, g1)
    matriz.gradient.zero()
    (0 until 6).foreach(i => matriz.gradient.accumulate(i, 0.5))

    new AdamW(List(vetor, matriz), lr = lr, weightDecay = 0.0).step()

    Math.abs(vetor.toArray(0) - 0.60) shouldBe lr +- tolerance
    matriz.toArray.zip(Array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)).foreach { (novo, velho) =>
      novo shouldBe (velho - lr) +- tolerance
    }
  }

  it should "not zero the gradients as a side effect of step" in {
    val p = parametro(valoresIniciais)
    passo(semDecay(p), p, g1)

    // Zerar e responsabilidade do loop de treino, via `zeroGrad`. Se `step`
    // zerasse sozinho, um segundo `backward` acumularia sobre o lixo errado.
    p.gradient.toArray shouldBe g1
  }

  "zeroGrad" should "zero the gradient of every parameter" in {
    val a = parametro(Array(1.0, 2.0))
    val b = parametro(Array(3.0))

    semear(a, Array(0.5, -0.5))
    semear(b, Array(0.25))

    new AdamW(List(a, b)).zeroGrad()

    a.gradient.toArray shouldBe Array(0.0, 0.0)
    b.gradient.toArray shouldBe Array(0.0)
  }

  "the constructor" should "reject an empty parameter list" in {
    an[IllegalArgumentException] should be thrownBy new AdamW(List.empty)
  }

  it should "reject a parameter that does not require gradient" in {
    val congelado = Tensor.make(Array(1.0), Array(1))

    an[IllegalArgumentException] should be thrownBy new AdamW(List(congelado))
  }

  it should "reject betas outside [0, 1)" in {
    val p = parametro(Array(1.0))

    an[IllegalArgumentException] should be thrownBy new AdamW(List(p), beta1 = 1.0)
    an[IllegalArgumentException] should be thrownBy new AdamW(List(p), beta2 = -0.1)
  }

  it should "reject a negative weight decay" in {
    val p = parametro(Array(1.0))

    an[IllegalArgumentException] should be thrownBy new AdamW(List(p), weightDecay = -0.1)
  }
end AdamWSpec
