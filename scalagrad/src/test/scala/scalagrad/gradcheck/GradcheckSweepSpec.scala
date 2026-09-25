package scalagrad.gradcheck

import scalagrad.core.Tensor
import scalagrad.ops.tensor.*
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Varredura do item "Aplicação" de `checklists/04-gradient-check.md`: roda
  * `Gradcheck.run` em todas as ops da Etapa 3, com inputs "aleatórios" de
  * shapes variadas. Usa um `Random` com seed fixa -- aleatório o suficiente
  * pra não depender de valores escolhidos a dedo, mas determinístico entre
  * execuções (um teste que falha só às vezes, por sorte do gerador, é pior
  * que um teste fixo).
  */
class GradcheckSweepSpec extends AnyFlatSpec with Matchers:
  private val rng = new scala.util.Random(42)
  private val shapes = List(Array(4), Array(2, 3))

  private def randomTensor(
      shape: Array[Int],
      lo: Double,
      hi: Double,
      requiresGradient: Boolean = true
  ): Tensor =
    val data = Array.fill(shape.product)(lo + rng.nextDouble() * (hi - lo))
    Tensor.make(data, shape, requiresGradient)

  "Gradcheck" should "pass for add, across varied shapes" in {
    shapes.foreach { shape =>
      val b = randomTensor(shape, -3.0, 3.0, requiresGradient = false)
      Gradcheck.run(randomTensor(shape, -3.0, 3.0))(a => (a + b).sum) should be < 1e-5
    }
  }

  it should "pass for sub, across varied shapes" in {
    shapes.foreach { shape =>
      val b = randomTensor(shape, -3.0, 3.0, requiresGradient = false)
      Gradcheck.run(randomTensor(shape, -3.0, 3.0))(a => (a - b).sum) should be < 1e-5
    }
  }

  it should "pass for mul, across varied shapes" in {
    shapes.foreach { shape =>
      val b = randomTensor(shape, -3.0, 3.0, requiresGradient = false)
      Gradcheck.run(randomTensor(shape, -3.0, 3.0))(a => (a * b).sum) should be < 1e-5
    }
  }

  it should "pass for div, across varied shapes (divisor kept away from zero)" in {
    shapes.foreach { shape =>
      val b = randomTensor(shape, 1.0, 3.0, requiresGradient = false)
      Gradcheck.run(randomTensor(shape, -3.0, 3.0))(a => (a / b).sum) should be < 1e-5
    }
  }

  it should "pass for pow, across varied shapes" in {
    shapes.foreach { shape =>
      Gradcheck.run(randomTensor(shape, -3.0, 3.0))(a => a.pow(2).sum) should be < 1e-5
    }
  }

  it should "pass for neg, across varied shapes" in {
    shapes.foreach { shape =>
      Gradcheck.run(randomTensor(shape, -3.0, 3.0))(a => a.neg.sum) should be < 1e-5
    }
  }

  it should "pass for exp, across varied shapes" in {
    shapes.foreach { shape =>
      Gradcheck.run(randomTensor(shape, -2.0, 2.0))(a => a.exp.sum) should be < 1e-5
    }
  }

  it should "pass for log, across varied shapes (input kept away from zero)" in {
    shapes.foreach { shape =>
      Gradcheck.run(randomTensor(shape, 0.5, 3.0))(a => a.log.sum) should be < 1e-5
    }
  }

  it should "pass for sum, across varied shapes" in {
    shapes.foreach { shape =>
      Gradcheck.run(randomTensor(shape, -3.0, 3.0))(a => a.sum) should be < 1e-5
    }
  }

  it should "pass for mean, across varied shapes" in {
    shapes.foreach { shape =>
      Gradcheck.run(randomTensor(shape, -3.0, 3.0))(a => a.mean) should be < 1e-5
    }
  }

  it should "pass for max, across varied shapes" in {
    shapes.foreach { shape =>
      Gradcheck.run(randomTensor(shape, -3.0, 3.0))(a => a.max) should be < 1e-5
    }
  }

  it should "pass for sum(dim), across varied shapes" in {
    Gradcheck.run(randomTensor(Array(2, 3), -3.0, 3.0))(a => a.sum(1).sum) should be < 1e-5
    Gradcheck.run(randomTensor(Array(2, 3), -3.0, 3.0))(a => a.sum(0).sum) should be < 1e-5
  }

  it should "pass for mean(dim), across varied shapes" in {
    Gradcheck.run(randomTensor(Array(2, 3), -3.0, 3.0))(a => a.mean(1).sum) should be < 1e-5
    Gradcheck.run(randomTensor(Array(2, 3), -3.0, 3.0))(a => a.mean(0).sum) should be < 1e-5
  }

  it should "pass for clamp, across varied shapes (mixing saturated and inside regions)" in {
    shapes.foreach { shape =>
      Gradcheck.run(randomTensor(shape, -8.0, 8.0))(a => a.clamp(-5.0, 5.0).sum) should be < 1e-5
    }
  }

  it should "pass for matmul 2D, w.r.t. both operands" in {
    val a = randomTensor(Array(3, 4), -3.0, 3.0)
    val b = randomTensor(Array(4, 2), -3.0, 3.0, requiresGradient = false)
    val bFixed = randomTensor(Array(3, 4), -3.0, 3.0, requiresGradient = false)

    Gradcheck.run(a)(x => x.matmul(b).sum) should be < 1e-5
    Gradcheck.run(randomTensor(Array(4, 2), -3.0, 3.0))(x => bFixed.matmul(x).sum) should be < 1e-5
  }

  it should "pass for matmul 3D (batch), w.r.t. both operands" in {
    val a = randomTensor(Array(2, 3, 4), -3.0, 3.0)
    val b = randomTensor(Array(2, 4, 5), -3.0, 3.0, requiresGradient = false)
    val bFixed = randomTensor(Array(2, 3, 4), -3.0, 3.0, requiresGradient = false)

    Gradcheck.run(a)(x => x.matmul(b).sum) should be < 1e-5
    Gradcheck.run(randomTensor(Array(2, 4, 5), -3.0, 3.0))(x =>
      bFixed.matmul(x).sum
    ) should be < 1e-5
  }

  it should "pass for matmul (batch x shared matrix), w.r.t. both operands" in {
    val a = randomTensor(Array(2, 3, 4), -3.0, 3.0)
    val w = randomTensor(Array(4, 5), -3.0, 3.0, requiresGradient = false)
    val aFixed = randomTensor(Array(2, 3, 4), -3.0, 3.0, requiresGradient = false)

    Gradcheck.run(a)(x => x.matmul(w).sum) should be < 1e-5
    // o gradiente da matriz compartilhada soma sobre o lote -- a diferenca finita
    // perturba uma posicao de W e sente as duas fatias de uma vez
    Gradcheck.run(randomTensor(Array(4, 5), -3.0, 3.0))(x => aFixed.matmul(x).sum) should be < 1e-5
  }

  it should "pass for matmul on a non-contiguous batched operand" in {
    val w = randomTensor(Array(3, 2), -3.0, 3.0, requiresGradient = false)

    Gradcheck.run(randomTensor(Array(2, 3, 4), -3.0, 3.0))(x =>
      x.transpose(1, 2).matmul(w).sum
    ) should be < 1e-5
  }

  // ---- dois eixos de lote (Etapa 12) ----
  // A perda usa pesos distintos antes do `.sum`: com `.sum` puro sobre um
  // produto matricial o gradiente de cada entrada vira uma constante, e o
  // check aprova backwards que trocam posicoes entre si.

  private def distinctWeights(shape: Array[Int]): Tensor =
    Tensor.make(Array.tabulate(shape.product)(i => 0.3 + 0.7 * Math.sin(i * 1.7)), shape)

  it should "pass for matmul with two batch axes, w.r.t. both operands" in {
    val w = distinctWeights(Array(2, 2, 3, 3))
    val bFixed = randomTensor(Array(2, 2, 4, 3), -3.0, 3.0, requiresGradient = false)
    val aFixed = randomTensor(Array(2, 2, 3, 4), -3.0, 3.0, requiresGradient = false)

    Gradcheck.run(randomTensor(Array(2, 2, 3, 4), -3.0, 3.0))(x =>
      (x.matmul(bFixed) * w).sum
    ) should be < 1e-5

    Gradcheck.run(randomTensor(Array(2, 2, 4, 3), -3.0, 3.0))(x =>
      (aFixed.matmul(x) * w).sum
    ) should be < 1e-5
  }

  it should "pass for a shared matrix under two batch axes" in {
    // o gradiente da matriz compartilhada soma sobre as B*H fatias de uma vez
    val w = distinctWeights(Array(2, 2, 3, 3))
    val aFixed = randomTensor(Array(2, 2, 3, 4), -3.0, 3.0, requiresGradient = false)

    Gradcheck.run(randomTensor(Array(4, 3), -3.0, 3.0))(x =>
      (aFixed.matmul(x) * w).sum
    ) should be < 1e-5
  }

  it should "pass for a non-contiguous rank-4 operand" in {
    // a divisao em cabecas produz exatamente isto: transpose(1, 2) antes do matmul
    val w = distinctWeights(Array(2, 3, 5, 5))
    val bFixed = randomTensor(Array(2, 3, 4, 5), -3.0, 3.0, requiresGradient = false)

    Gradcheck.run(randomTensor(Array(2, 5, 3, 4), -3.0, 3.0))(x =>
      (x.transpose(1, 2).matmul(bFixed) * w).sum
    ) should be < 1e-5
  }

  // ---- ops unarias sobre entrada NAO contigua ----
  // Regressao de um bug real: o backward de `unary` misturava as duas
  // indexacoes do projeto -- `data` e lido pelas strides reais, `gradient` e
  // sempre canonico. Em tensor contiguo as duas coincidem e nada aparece; num
  // transposto a derivada local era pareada com o valor de outra posicao.
  // Forward continuava perfeito. Medido antes da correcao: exp dava erro
  // 0.632 aqui contra 8.5e-11 no mesmo exp sobre entrada contigua.
  //
  // Shape 3x3 entra de proposito: quadrado NAO mascara o defeito, porque o
  // transpose troca as strides mesmo quando as dimensoes sao iguais.

  private val unaryOps: List[(String, Tensor => Tensor)] = List(
    "neg" -> (_.neg),
    "pow" -> (_.pow(3)),
    "exp" -> (_.exp),
    "log" -> (_.log),
    "clamp" -> (_.clamp(-2.0, 2.0)),
    "relu" -> (_.relu),
    "sigmoid" -> (_.sigmoid),
    "tanh" -> (_.tanh),
    "gelu" -> (_.gelu)
  )

  it should "pass for every unary op on a non-contiguous input" in {
    // `log` exige entrada positiva, entao a faixa comeca acima de zero. Os
    // limites evitam tambem os pontos onde clamp e relu nao sao derivaveis.
    unaryOps.foreach { (nome, op) =>
      List(Array(2, 3), Array(3, 3), Array(2, 3, 4)).foreach { shape =>
        val w = Tensor.make(
          Array.tabulate(shape.product)(i => 0.3 + 0.7 * Math.sin(i * 1.7)),
          shape.reverse
        )

        withClue(s"$nome com shape ${shape.toList} transposto: ") {
          Gradcheck.run(randomTensor(shape, 0.5, 1.8)) { x =>
            val t = if x.rank == 2 then x.transpose(0, 1) else x.transpose(0, 2)
            (op(t) * w).sum
          } should be < 1e-5
        }
      }
    }
  }

  it should "give the same gradient whether the input arrives contiguous or transposed" in {
    // contraprova direta: a mesma funcao matematica, montada por dois caminhos
    // que so diferem no layout de memoria, tem que dar o mesmo gradiente.
    val values = Array(0.7, 1.1, 1.5, 0.9, 1.3, 1.7)

    unaryOps.foreach { (nome, op) =>
      val direto = Tensor.make(values, Array(3, 2), requiresGradient = true)
      val viaTranspose = Tensor.make(values, Array(3, 2), requiresGradient = true)
      val w = Tensor.make(Array(1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0), Array(3, 2))
      val wT = Tensor.make(Array(1.0, 100.0, 10000.0, 10.0, 1000.0, 100000.0), Array(2, 3))

      (op(direto) * w).sum.backward()
      (op(viaTranspose.transpose(0, 1)) * wT).sum.backward()

      withClue(s"$nome: ") {
        direto.gradient.toList.zip(viaTranspose.gradient.toList).foreach { (a, b) =>
          a shouldBe b +- 1e-12
        }
      }
    }
  }

  it should "pass for relu, across varied shapes" in {
    shapes.foreach { shape =>
      Gradcheck.run(randomTensor(shape, -3.0, 3.0))(a => a.relu.sum) should be < 1e-5
    }
  }

  it should "pass for sigmoid, across varied shapes" in {
    shapes.foreach { shape =>
      Gradcheck.run(randomTensor(shape, -3.0, 3.0))(a => a.sigmoid.sum) should be < 1e-5
    }
  }

  it should "pass for tanh, across varied shapes" in {
    shapes.foreach { shape =>
      Gradcheck.run(randomTensor(shape, -3.0, 3.0))(a => a.tanh.sum) should be < 1e-5
    }
  }

  it should "pass for gelu, across varied shapes (the densest backward derivation so far)" in {
    shapes.foreach { shape =>
      Gradcheck.run(randomTensor(shape, -3.0, 3.0))(a => a.gelu.sum) should be < 1e-5
    }
  }

  // softmax(dim).sum() sozinho e uma funcao constante (probabilidades sempre somam 1,
  // gradiente analitico E numerico dariam ~0 os dois, "passando" mesmo com um backward
  // quebrado) -- por isso todo teste de softmax aqui multiplica por pesos fixos antes
  // do sum, igual a tecnica ja usada nos testes unitarios de backward em TensorOpsSpec.
  it should "pass for softmax, across varied shapes and dims" in {
    val w1 = randomTensor(Array(4), -3.0, 3.0, requiresGradient = false)
    Gradcheck.run(randomTensor(Array(4), -3.0, 3.0))(a => (a.softmax(0) * w1).sum) should be < 1e-5

    val w2 = randomTensor(Array(2, 3), -3.0, 3.0, requiresGradient = false)
    Gradcheck.run(randomTensor(Array(2, 3), -3.0, 3.0))(a =>
      (a.softmax(1) * w2).sum
    ) should be < 1e-5
    Gradcheck.run(randomTensor(Array(2, 3), -3.0, 3.0))(a =>
      (a.softmax(0) * w2).sum
    ) should be < 1e-5
  }

  it should "pass for softmax with logits of very different magnitudes (numerical stability)" in {
    // sem a subtracao do maximo, exp(900) ja estoura pra Infinity em Double
    val a = Tensor.make(Array(700.0, 800.0, 900.0, 750.0), Array(4), requiresGradient = true)
    val w = Tensor.make(Array(1.0, -2.0, 0.5, 3.0), Array(4))

    Gradcheck.run(a)(x => (x.softmax(0) * w).sum) should be < 1e-5
  }

  it should "pass for logSoftmax, across varied shapes and dims" in {
    val w1 = randomTensor(Array(4), -3.0, 3.0, requiresGradient = false)
    Gradcheck.run(randomTensor(Array(4), -3.0, 3.0))(a =>
      (a.logSoftmax(0) * w1).sum
    ) should be < 1e-5

    val w2 = randomTensor(Array(2, 3), -3.0, 3.0, requiresGradient = false)
    Gradcheck.run(randomTensor(Array(2, 3), -3.0, 3.0))(a =>
      (a.logSoftmax(1) * w2).sum
    ) should be < 1e-5
    Gradcheck.run(randomTensor(Array(2, 3), -3.0, 3.0))(a =>
      (a.logSoftmax(0) * w2).sum
    ) should be < 1e-5
  }

  it should "pass for logSoftmax with logits of very different magnitudes (numerical stability)" in {
    val a = Tensor.make(Array(700.0, 800.0, 900.0, 750.0), Array(4), requiresGradient = true)
    val w = Tensor.make(Array(1.0, -2.0, 0.5, 3.0), Array(4))

    Gradcheck.run(a)(x => (x.logSoftmax(0) * w).sum) should be < 1e-5
  }
end GradcheckSweepSpec
