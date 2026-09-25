package scalagrad.gradcheck

import scalagrad.core.{Tensor, Shape, Strides, Gradient}
import scalagrad.ops.tensor.*
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class GradcheckSpec extends AnyFlatSpec with Matchers:

  // Op deliberadamente quebrada (backward errado: ignora o valor de entrada,
  // devolve o upstream sem multiplicar por `2x`) -- usada só pra confirmar que
  // o Gradcheck de fato denuncia um backward incorreto, e nao so "carimba"
  // qualquer coisa como correta.
  private def brokenSquare(t: Tensor): Tensor =
    val data = Array(t.data(0) * t.data(0))
    val grad = Gradient.zeros(1)

    Tensor(data, Shape(1), Strides(1), grad, true, Set(t)) { () =>
      t.gradient.accumulate(0, grad(0)) // bug: deveria ser 2 * t.data(0) * grad(0)
    }

  "Gradcheck.run" should "report a near-zero error for a correct backward (sum of squares)" in {
    // mesmo exemplo de theory/04-gradient-check/04-gradient-check.md §3
    val x = Tensor.make(Array(3.0, -2.0), Array(2), requiresGradient = true)

    val error = Gradcheck.run(x)(t => (t * t).sum)

    error should be < 1e-5
  }

  it should "report a near-zero error for matmul composed with sum" in {
    val a = Tensor.make(Array(1.0, 2.0, 3.0, 4.0), Array(2, 2), requiresGradient = true)
    val w = Tensor.make(Array(5.0, 6.0, 7.0, 8.0), Array(2, 2))

    val error = Gradcheck.run(a)(t => t.matmul(w).sum)

    error should be < 1e-5
  }

  it should "report a near-zero error for clamp, away from the non-differentiable borders" in {
    // valores estritamente dentro/fora do intervalo, evitando os pontos exatos
    // min/max -- ali o gradiente numerico e o analitico divergem por construcao
    // (a funcao tem um "bico"), o que nao e um bug do backward, e um limite
    // conhecido do proprio metodo de diferenca finita perto de nao-diferenciabilidades
    val a = Tensor.make(Array(-3.0, 2.0, 4.5), Array(3), requiresGradient = true)

    val error = Gradcheck.run(a)(t => t.clamp(0.0, 5.0).sum)

    error should be < 1e-5
  }

  it should "report a large error for a deliberately incorrect backward" in {
    val x = Tensor.make(Array(3.0), Array(1), requiresGradient = true)

    val error = Gradcheck.run(x)(brokenSquare)

    // analitico (com bug) = 1.0; numerico (correto) = 2*3 = 6.0
    error should be > 1e-3
  }

  it should "work when the input tensor itself is non-contiguous" in {
    // regressao: `run` perturbava `input.data(i)` tratando `i` como indice
    // canonico, mas `data` e indexado pelas strides reais. Num transposto os
    // dois divergem, e a diferenca finita media a derivada de OUTRA posicao.
    // Medido antes da correcao: 0.99 contra um limiar de 1e-5.
    val w = Tensor.make(Array(1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0), Array(3, 2))
    val x = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), Array(2, 3), requiresGradient = true)

    Gradcheck.run(x.transpose(0, 1))(t => (t * w).sum) should be < 1e-5
  }

  it should "reject a loss that is not a scalar" in {
    // sem isso, `run` lia o primeiro elemento e devolvia um numero plausivel
    // e sem sentido -- silencio no unico utilitario cujo trabalho e avisar.
    val x = Tensor.make(Array(1.0, 2.0, 3.0, 4.0), Array(2, 2), requiresGradient = true)

    an[IllegalArgumentException] should be thrownBy Gradcheck.run(x)(t => t * t)
  }

  it should "give the same answer when run twice on the same tensor" in {
    // `Gradient` so acumula, entao sem zerar a entrada o segundo `run` leria o
    // gradiente somado do primeiro. Antes era preciso criar um tensor novo por
    // chamada; agora `run` e idempotente.
    val w = Tensor.make(Array(1.0, 10.0, 100.0, 1000.0), Array(2, 2))
    val x = Tensor.make(Array(0.5, 1.5, 2.5, 3.5), Array(2, 2), requiresGradient = true)

    val primeira = Gradcheck.run(x)(t => (t * w).sum)
    val segunda = Gradcheck.run(x)(t => (t * w).sum)

    segunda shouldBe primeira +- 1e-12
  }
end GradcheckSpec
