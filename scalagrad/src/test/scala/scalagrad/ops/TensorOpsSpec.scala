package scalagrad.ops

import scalagrad.core.Tensor
import scalagrad.core.Shape
import scalagrad.ops.tensor.*
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TensorOpsSpec extends AnyFlatSpec with Matchers {

  private def trainable(value: Double): Tensor =
    Tensor.make(Array(value), Array(1), requiresGradient = true)

  "+" should "compute the element-wise sum" in {
    val a = Tensor.make(Array(1.0, 2.0, 3.0), Array(3))
    val b = Tensor.make(Array(10.0, 20.0, 30.0), Array(3))

    val c = a + b

    c.data.toList shouldBe List(11.0, 22.0, 33.0)
  }

  it should "reject tensors with different shapes" in {
    val a = Tensor.make(Array(1.0, 2.0), Array(2))
    val b = Tensor.make(Array(1.0, 2.0, 3.0), Array(3))

    an[IllegalArgumentException] should be thrownBy (a + b)
  }

  it should "distribute gradient equally to both operands (derivative of a sum is 1)" in {
    val a = Tensor.make(Array(2.0), Array(1))
    val b = Tensor.make(Array(3.0), Array(1))

    val loss = a + b
    loss.backward()

    a.gradient(0) shouldBe 1.0
    b.gradient(0) shouldBe 1.0
  }

  it should "broadcast a smaller operand across a larger one in the forward pass" in {
    val a = Tensor.make(Array(1.0, 2.0, 3.0), Array(1, 3))
    val b = Tensor.make(Array(10.0, 20.0, 30.0, 40.0, 50.0, 60.0), Array(2, 3))

    val c = a + b

    c.data.toList shouldBe List(11.0, 22.0, 33.0, 41.0, 52.0, 63.0)
  }

  it should "broadcast an operand with fewer dimensions, left-padding rank before aligning" in {
    val a = Tensor.make(Array(1.0, 2.0, 3.0), Array(3))
    val b = Tensor.make(Array(10.0, 20.0, 30.0, 40.0, 50.0, 60.0), Array(2, 3))

    val c = a + b

    c.data.toList shouldBe List(11.0, 22.0, 33.0, 41.0, 52.0, 63.0)
  }

  it should "sum broadcasted gradient contributions back into the smaller operand's original shape" in {
    val a = Tensor.make(Array(1.0, 2.0, 3.0), Array(1, 3), requiresGradient = true)
    val b =
      Tensor.make(Array(10.0, 20.0, 30.0, 40.0, 50.0, 60.0), Array(2, 3), requiresGradient = true)

    val loss = (a + b).sum
    loss.backward()

    // a foi usado 2x (uma vez por linha) -- cada posição de a.grad acumula as duas linhas
    a.gradient.toList shouldBe List(2.0, 2.0, 2.0)
    b.gradient.toList shouldBe List(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
  }

  "*" should "compute the element-wise product" in {
    val a = Tensor.make(Array(2.0, 3.0), Array(2))
    val b = Tensor.make(Array(4.0, 5.0), Array(2))

    val c = a * b

    c.data.toList shouldBe List(8.0, 15.0)
  }

  it should "reject tensors with different shapes" in {
    val a = Tensor.make(Array(1.0, 2.0), Array(2))
    val b = Tensor.make(Array(1.0, 2.0, 3.0), Array(3))

    an[IllegalArgumentException] should be thrownBy (a * b)
  }

  it should "route gradient through the other operand (derivative of a product)" in {
    val a = Tensor.make(Array(3.0), Array(1))
    val b = Tensor.make(Array(4.0), Array(1))

    val loss = a * b
    loss.backward()

    a.gradient(0) shouldBe 4.0 // dLoss/da = b
    b.gradient(0) shouldBe 3.0 // dLoss/db = a
  }

  it should "broadcast a smaller operand across a larger one in the forward pass" in {
    val a = Tensor.make(Array(2.0, 3.0, 4.0), Array(1, 3))
    val b = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), Array(2, 3))

    val c = a * b

    c.data.toList shouldBe List(2.0, 6.0, 12.0, 8.0, 15.0, 24.0)
  }

  it should "read the other operand through its broadcasted view when accumulating gradient" in {
    val a = Tensor.make(Array(2.0, 3.0, 4.0), Array(1, 3), requiresGradient = true)
    val b = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), Array(2, 3), requiresGradient = true)

    val loss = (a * b).sum
    loss.backward()

    // da[j] = Σᵢ b[i,j] (a foi usado nas duas linhas, com um b diferente por linha)
    a.gradient.toList shouldBe List(1.0 + 4.0, 2.0 + 5.0, 3.0 + 6.0)
    // db[i,j] = a[j] (sem broadcasting em b, cada posição recebe a própria coluna de a)
    b.gradient.toList shouldBe List(2.0, 3.0, 4.0, 2.0, 3.0, 4.0)
  }

  "-" should "compute the element-wise difference" in {
    val a = Tensor.make(Array(5.0, 7.0, 9.0), Array(3))
    val b = Tensor.make(Array(1.0, 2.0, 3.0), Array(3))

    val c = a - b

    c.data.toList shouldBe List(4.0, 5.0, 6.0)
  }

  it should "reject tensors with different shapes" in {
    val a = Tensor.make(Array(1.0, 2.0), Array(2))
    val b = Tensor.make(Array(1.0, 2.0, 3.0), Array(3))

    an[IllegalArgumentException] should be thrownBy (a - b)
  }

  it should "negate the gradient of the subtrahend (derivative of a difference)" in {
    val a = Tensor.make(Array(5.0), Array(1))
    val b = Tensor.make(Array(2.0), Array(1))

    val loss = a - b
    loss.backward()

    a.gradient(0) shouldBe 1.0
    b.gradient(0) shouldBe -1.0
  }

  it should "broadcast a smaller operand across a larger one in the forward pass" in {
    val a = Tensor.make(Array(1.0, 2.0, 3.0), Array(1, 3))
    val b = Tensor.make(Array(10.0, 20.0, 30.0, 40.0, 50.0, 60.0), Array(2, 3))

    val c = a - b

    c.data.toList shouldBe List(-9.0, -18.0, -27.0, -39.0, -48.0, -57.0)
  }

  it should "sum broadcasted gradient for the minuend and negate it for the subtrahend" in {
    val a = Tensor.make(Array(1.0, 2.0, 3.0), Array(1, 3), requiresGradient = true)
    val b =
      Tensor.make(Array(10.0, 20.0, 30.0, 40.0, 50.0, 60.0), Array(2, 3), requiresGradient = true)

    val loss = (a - b).sum
    loss.backward()

    a.gradient.toList shouldBe List(2.0, 2.0, 2.0) // usado 2x, derivada +1 em cada uso
    b.gradient.toList shouldBe List(-1.0, -1.0, -1.0, -1.0, -1.0,
      -1.0) // sem broadcasting, derivada -1
  }

  "/" should "compute the element-wise quotient" in {
    val a = Tensor.make(Array(10.0, 20.0), Array(2))
    val b = Tensor.make(Array(2.0, 5.0), Array(2))

    val c = a / b

    c.data.toList shouldBe List(5.0, 4.0)
  }

  it should "reject tensors with different shapes" in {
    val a = Tensor.make(Array(1.0, 2.0), Array(2))
    val b = Tensor.make(Array(1.0, 2.0, 3.0), Array(3))

    an[IllegalArgumentException] should be thrownBy (a / b)
  }

  it should "apply the quotient rule in the backward pass" in {
    val a = Tensor.make(Array(6.0), Array(1))
    val b = Tensor.make(Array(3.0), Array(1))

    val loss = a / b
    loss.backward()

    a.gradient(0) shouldBe (1.0 / 3.0 +- 1e-9) // dLoss/da = 1/b
    b.gradient(0) shouldBe (-6.0 / 9.0 +- 1e-9) // dLoss/db = -a/b^2
  }

  it should "broadcast a smaller operand across a larger one in the forward pass" in {
    val a = Tensor.make(Array(10.0, 20.0), Array(1, 2))
    val b = Tensor.make(Array(2.0, 4.0, 5.0, 10.0), Array(2, 2))

    val c = a / b

    c.data.toList shouldBe List(5.0, 5.0, 2.0, 2.0)
  }

  it should "read the other operand through its broadcasted view for the quotient rule" in {
    val a = Tensor.make(Array(10.0, 20.0), Array(1, 2), requiresGradient = true)
    val b = Tensor.make(Array(2.0, 4.0, 5.0, 10.0), Array(2, 2), requiresGradient = true)

    val loss = (a / b).sum
    loss.backward()

    // da[j] = Σᵢ 1/b[i,j]
    a.gradient(0) shouldBe (1.0 / 2.0 + 1.0 / 5.0 +- 1e-9)
    a.gradient(1) shouldBe (1.0 / 4.0 + 1.0 / 10.0 +- 1e-9)
    // db[i,j] = -a[j]/b[i,j]^2
    b.gradient(0) shouldBe (-10.0 / 4.0 +- 1e-9)
    b.gradient(1) shouldBe (-20.0 / 16.0 +- 1e-9)
    b.gradient(2) shouldBe (-10.0 / 25.0 +- 1e-9)
    b.gradient(3) shouldBe (-20.0 / 100.0 +- 1e-9)
  }

  it should "propagate IEEE 754 semantics on division by zero instead of guarding against it" in {
    val a = Tensor.make(Array(5.0, 0.0), Array(2))
    val b = Tensor.make(Array(0.0, 0.0), Array(2))

    val c = a / b

    c.data(0) shouldBe Double.PositiveInfinity
    c.data(1).isNaN shouldBe true
  }

  "neg" should "negate every element" in {
    val a = Tensor.make(Array(3.0, -2.0), Array(2))

    val c = a.neg

    c.data.toList shouldBe List(-3.0, 2.0)
  }

  it should "flip the sign of the incoming gradient" in {
    val a = Tensor.make(Array(4.0), Array(1))

    val loss = a.neg
    loss.backward()

    a.gradient(0) shouldBe -1.0
  }

  "pow" should "raise every element to the given exponent" in {
    val a = Tensor.make(Array(2.0, 3.0), Array(2))

    val c = a.pow(2)

    c.data.toList shouldBe List(4.0, 9.0)
  }

  it should "apply the power rule in the backward pass" in {
    val a = Tensor.make(Array(3.0), Array(1))

    val loss = a.pow(2)
    loss.backward()

    a.gradient(0) shouldBe 6.0 // dLoss/da = n * a^(n-1) = 2 * 3^1
  }

  "backward" should "sum contributions when a tensor is reused across the graph" in {
    val a = Tensor.make(Array(2.0), Array(1))
    val b = Tensor.make(Array(3.0), Array(1))

    val c = a + b // c = 5, depende de a e b
    val d = c * a // d = 10, reusa `a` de novo

    d.backward()

    // via mul direto: dD/da = c.data = 5
    // via add -> c: dD/dc = a.data = 2, dc/da = 1, contribui +2
    a.gradient(0) shouldBe 7.0
    b.gradient(0) shouldBe 2.0
    c.gradient(0) shouldBe 2.0
  }

  "zeroGrad" should "zero gradients only for tensors with requiresGradient = true" in {
    val a = trainable(2.0)
    val b = Tensor.make(Array(3.0), Array(1)) // requiresGradient = false

    val loss = a + b
    loss.backward()
    loss.zeroGrad()

    a.gradient(0) shouldBe 0.0
    b.gradient(0) shouldBe 1.0 // não é zerado: requiresGradient = false
  }

  "gradEnabled" should "be true by default" in {
    Tensor.gradEnabled shouldBe true
  }

  "noGrad" should "force requiresGradient = false on results, even if an input requires it" in {
    val a = trainable(2.0)
    val b = Tensor.make(Array(3.0), Array(1))

    val result = Tensor.noGrad { a + b }

    result.requiresGradient shouldBe false
  }

  it should "restore gradEnabled to its previous value after the block" in {
    Tensor.noGrad { () }

    Tensor.gradEnabled shouldBe true
  }

  it should "restore gradEnabled even if the block throws" in {
    val previous = Tensor.gradEnabled

    a[RuntimeException] should be thrownBy {
      Tensor.noGrad { throw new RuntimeException("boom") }
    }

    Tensor.gradEnabled shouldBe previous
  }

  "exp" should "compute the element-wise exponential" in {
    val a = Tensor.make(Array(0.0, 1.0, 2.0), Array(3))

    val c = a.exp

    c.data(0) shouldBe (1.0 +- 1e-9)
    c.data(1) shouldBe (Math.E +- 1e-9)
    c.data(2) shouldBe (Math.exp(2.0) +- 1e-9)
  }

  it should "propagate gradient as the exponential of the input (derivative of exp is itself)" in {
    val a = trainable(2.0)

    val loss = a.exp
    loss.backward()

    a.gradient(0) shouldBe (Math.exp(2.0) +- 1e-9) // dLoss/da = e^a, não `a`
  }

  "log" should "compute the element-wise natural logarithm" in {
    val a = Tensor.make(Array(1.0, Math.E, 10.0), Array(3))

    val c = a.log

    c.data(0) shouldBe (0.0 +- 1e-9)
    c.data(1) shouldBe (1.0 +- 1e-9)
    c.data(2) shouldBe (Math.log(10.0) +- 1e-9)
  }

  it should "apply the reciprocal rule in the backward pass (derivative of log(x) is 1/x)" in {
    val a = trainable(4.0)

    val loss = a.log
    loss.backward()

    a.gradient(0) shouldBe (0.25 +- 1e-9)
  }

  "sum" should "add all elements into a scalar tensor" in {
    val a = Tensor.make(Array(1.0, 2.0, 3.0, 4.0), Array(4))

    val c = a.sum

    c.shape.toList shouldBe List(1)
    c.data.toList shouldBe List(10.0)
  }

  it should "not crash for tensors with rank > 1 (regression: output shape must be Array(1), not the input shape)" in {
    val a = Tensor.make(Array(1.0, 2.0, 3.0, 4.0), Array(2, 2))

    val c = a.sum

    c.data.toList shouldBe List(10.0)
  }

  it should "route the same upstream gradient to every element (derivative of sum is 1)" in {
    val a = Tensor.make(Array(1.0, 2.0, 3.0), Array(3), requiresGradient = true)

    val loss = a.sum
    loss.backward()

    a.gradient.toList shouldBe List(1.0, 1.0, 1.0)
  }

  "mean" should "compute the average of all elements" in {
    val a = Tensor.make(Array(2.0, 4.0, 6.0), Array(3))

    val c = a.mean

    c.data.toList shouldBe List(4.0)
  }

  it should "divide the upstream gradient equally among all elements" in {
    val a = Tensor.make(Array(2.0, 4.0, 6.0), Array(3), requiresGradient = true)

    val loss = a.mean
    loss.backward()

    a.gradient(0) shouldBe (1.0 / 3.0 +- 1e-9)
    a.gradient(1) shouldBe (1.0 / 3.0 +- 1e-9)
    a.gradient(2) shouldBe (1.0 / 3.0 +- 1e-9)
  }

  "max" should "return the maximum value as a scalar tensor" in {
    val a = Tensor.make(Array(3.0, 7.0, 2.0), Array(3))

    val c = a.max

    c.data.toList shouldBe List(7.0)
  }

  it should "route the gradient only to the index of the maximum" in {
    val a = Tensor.make(Array(3.0, 7.0, 2.0), Array(3), requiresGradient = true)

    val loss = a.max
    loss.backward()

    a.gradient.toList shouldBe List(0.0, 1.0, 0.0)
  }

  it should "break ties by routing the gradient to the first occurrence of the maximum" in {
    val a = Tensor.make(Array(5.0, 5.0, 2.0), Array(3), requiresGradient = true)

    val loss = a.max
    loss.backward()

    a.gradient.toList shouldBe List(1.0, 0.0, 0.0)
  }

  "sum(dim)" should "reduce along the given dimension, dropping it by default (keepDim=false)" in {
    val a = Tensor.make(Array(1.0, 4.0, 2.0, 5.0, 0.0, 3.0), Array(2, 3))

    val c = a.sum(1)

    c.shape.toList shouldBe List(2)
    c.data.toList shouldBe List(7.0, 8.0) // linha 0: 1+4+2, linha 1: 5+0+3
  }

  it should "reduce along dimension 0 the same way" in {
    val a = Tensor.make(Array(1.0, 4.0, 2.0, 5.0, 0.0, 3.0), Array(2, 3))

    val c = a.sum(0)

    c.shape.toList shouldBe List(3)
    c.data.toList shouldBe List(6.0, 4.0, 5.0) // coluna 0: 1+5, coluna 1: 4+0, coluna 2: 2+3
  }

  it should "keep the reduced dimension as size 1 when keepDim=true" in {
    val a = Tensor.make(Array(1.0, 4.0, 2.0, 5.0, 0.0, 3.0), Array(2, 3))

    val c = a.sum(1, keepDim = true)

    c.shape.toList shouldBe List(2, 1)
    c.data.toList shouldBe List(7.0, 8.0)
  }

  it should "route each output's upstream gradient to every input position in its own slice" in {
    val a = Tensor.make(Array(1.0, 4.0, 2.0, 5.0, 0.0, 3.0), Array(2, 3), requiresGradient = true)
    // pesos distintos por linha (10 e 100), pra garantir que cada linha de `a` receba
    // o gradiente vindo da SUA própria posição de saída, não uma constante genérica
    val weights = Tensor.make(Array(10.0, 100.0), Array(2))

    val loss = (a.sum(1) * weights).sum
    loss.backward()

    a.gradient.toList shouldBe List(10.0, 10.0, 10.0, 100.0, 100.0, 100.0)
  }

  it should "do the same when keepDim=true" in {
    val a = Tensor.make(Array(1.0, 4.0, 2.0, 5.0, 0.0, 3.0), Array(2, 3), requiresGradient = true)
    val weights = Tensor.make(Array(10.0, 100.0), Array(2, 1))

    val loss = (a.sum(1, keepDim = true) * weights).sum
    loss.backward()

    a.gradient.toList shouldBe List(10.0, 10.0, 10.0, 100.0, 100.0, 100.0)
  }

  "mean(dim)" should "reduce along the given dimension, dropping it by default (keepDim=false)" in {
    val a = Tensor.make(Array(1.0, 4.0, 2.0, 5.0, 0.0, 3.0), Array(2, 3))

    val c = a.mean(1)

    c.shape.toList shouldBe List(2)
    c.data(0) shouldBe (7.0 / 3.0 +- 1e-9) // linha 0: (1+4+2)/3
    c.data(1) shouldBe (8.0 / 3.0 +- 1e-9) // linha 1: (5+0+3)/3
  }

  it should "reduce along dimension 0 the same way" in {
    val a = Tensor.make(Array(1.0, 4.0, 2.0, 5.0, 0.0, 3.0), Array(2, 3))

    val c = a.mean(0)

    c.shape.toList shouldBe List(3)
    c.data.toList shouldBe List(
      3.0,
      2.0,
      2.5
    ) // coluna 0: (1+5)/2, coluna 1: (4+0)/2, coluna 2: (2+3)/2
  }

  it should "keep the reduced dimension as size 1 when keepDim=true" in {
    val a = Tensor.make(Array(1.0, 4.0, 2.0, 5.0, 0.0, 3.0), Array(2, 3))

    val c = a.mean(1, keepDim = true)

    c.shape.toList shouldBe List(2, 1)
    c.data(0) shouldBe (7.0 / 3.0 +- 1e-9)
    c.data(1) shouldBe (8.0 / 3.0 +- 1e-9)
  }

  it should "route each output's upstream gradient, scaled by 1/n, to every input position in its own slice" in {
    val a = Tensor.make(Array(1.0, 4.0, 2.0, 5.0, 0.0, 3.0), Array(2, 3), requiresGradient = true)
    // mesmos pesos distintos por linha do teste de sum(dim), pra confirmar que o
    // fator de escala (1/n = 1/3) se soma corretamente à contribuição de cada linha
    val weights = Tensor.make(Array(10.0, 100.0), Array(2))

    val loss = (a.mean(1) * weights).sum
    loss.backward()

    a.gradient(0) shouldBe (10.0 / 3.0 +- 1e-9)
    a.gradient(1) shouldBe (10.0 / 3.0 +- 1e-9)
    a.gradient(2) shouldBe (10.0 / 3.0 +- 1e-9)
    a.gradient(3) shouldBe (100.0 / 3.0 +- 1e-9)
    a.gradient(4) shouldBe (100.0 / 3.0 +- 1e-9)
    a.gradient(5) shouldBe (100.0 / 3.0 +- 1e-9)
  }

  it should "do the same when keepDim=true" in {
    val a = Tensor.make(Array(1.0, 4.0, 2.0, 5.0, 0.0, 3.0), Array(2, 3), requiresGradient = true)
    val weights = Tensor.make(Array(10.0, 100.0), Array(2, 1))

    val loss = (a.mean(1, keepDim = true) * weights).sum
    loss.backward()

    a.gradient(0) shouldBe (10.0 / 3.0 +- 1e-9)
    a.gradient(3) shouldBe (100.0 / 3.0 +- 1e-9)
  }

  "matmul" should "compute the 2D matrix product for non-square shapes" in {
    val a = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), Array(2, 3))
    val b = Tensor.make(Array(7.0, 8.0, 9.0, 10.0, 11.0, 12.0), Array(3, 2))

    val c = a.matmul(b)

    c.shape.toList shouldBe List(2, 2)
    c.data.toList shouldBe List(58.0, 64.0, 139.0, 154.0)
  }

  it should "reject 2D tensors whose inner dimensions do not match" in {
    val a = Tensor.make(Array(1.0, 2.0), Array(1, 2))
    val b = Tensor.make(Array(1.0, 2.0, 3.0), Array(1, 3))

    an[IllegalArgumentException] should be thrownBy a.matmul(b)
  }

  it should "apply dA = dC @ Bt and dB = At @ dC in the 2D backward pass" in {
    val a = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), Array(2, 3), requiresGradient = true)
    val b =
      Tensor.make(Array(7.0, 8.0, 9.0, 10.0, 11.0, 12.0), Array(3, 2), requiresGradient = true)
    // pesos distintos por posicao de C, pra que dC chegue ao matmul com um valor
    // diferente por elemento (dC = weights, ja que d(C*weights)/dC = weights) em vez
    // de uma constante uniforme, que esconderia erro de indexacao no backward
    val weights = Tensor.make(Array(1.0, 2.0, 3.0, 4.0), Array(2, 2))

    val loss = (a.matmul(b) * weights).sum
    loss.backward()

    // dC = [[1,2],[3,4]]; dA = dC @ Bt, dB = At @ dC (theory/03-elementary-operations/03-elementary-operations.md §6)
    a.gradient.toList shouldBe List(23.0, 29.0, 35.0, 53.0, 67.0, 81.0)
    b.gradient.toList shouldBe List(13.0, 18.0, 17.0, 24.0, 21.0, 30.0)
  }

  it should "compute the batched 3D matrix product independently per slice" in {
    val a = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0), Array(2, 2, 2))
    val b = Tensor.make(Array(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0), Array(2, 2, 2))

    val c = a.matmul(b)

    c.shape.toList shouldBe List(2, 2, 2)
    // batch 0: [[1,2],[3,4]] @ [[9,10],[11,12]]; batch 1: [[5,6],[7,8]] @ [[13,14],[15,16]]
    c.data.toList shouldBe List(31.0, 34.0, 71.0, 78.0, 155.0, 166.0, 211.0, 226.0)
  }

  it should "reject 3D tensors with mismatched batch size instead of broadcasting it" in {
    val a = Tensor.make(Array.fill(8)(1.0), Array(2, 2, 2))
    val b = Tensor.make(Array.fill(4)(1.0), Array(1, 2, 2))

    an[IllegalArgumentException] should be thrownBy a.matmul(b)
  }

  it should "apply dA and dB independently per batch slice in the 3D backward pass" in {
    val a = Tensor.make(
      Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
      Array(2, 2, 2),
      requiresGradient = true
    )
    val b = Tensor.make(
      Array(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0),
      Array(2, 2, 2),
      requiresGradient = true
    )
    // pesos distintos entre as DUAS fatias de batch (nao so entre posicoes dentro da
    // mesma fatia), pra pegar um eventual bug de indice vazando de uma fatia pra outra
    val weights = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0), Array(2, 2, 2))

    val loss = (a.matmul(b) * weights).sum
    loss.backward()

    // dC0 = [[1,2],[3,4]], dC1 = [[5,6],[7,8]]; dA/dB por fatia, mesma formula do caso 2D
    a.gradient.toList shouldBe List(29.0, 35.0, 67.0, 81.0, 149.0, 171.0, 203.0, 233.0)
    b.gradient.toList shouldBe List(10.0, 14.0, 14.0, 20.0, 74.0, 86.0, 86.0, 100.0)
  }

  it should "reject rank combinations it does not know how to dispatch" in {
    val a = Tensor.make(Array(1.0, 2.0, 3.0, 4.0), Array(2, 2))
    val b = Tensor.make(Array(1.0, 2.0, 3.0, 4.0), Array(1, 2, 2))

    an[IllegalArgumentException] should be thrownBy a.matmul(b)
  }

  // ---- (3, 2): matriz compartilhada por todo o lote (Etapa 11, projecoes Q/K/V) ----
  // B, M, K e N todos diferentes: shapes quadradas escondem troca de eixo
  // (theory/03-elementary-operations/03-elementary-operations.md §6, bloco Armadilha)
  private val sharedLhs = Tensor.make(
    Array(1.0, 2.0, 0.0, -1.0, 3.0, -2.0, 1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 0.0, -3.0, 1.0, 1.0,
      1.0, 1.0, 1.0, -1.0, 2.0, 0.0, 3.0),
    Array(2, 3, 4)
  )
  private val sharedRhs = Tensor.make(Array(1.0, -2.0, 0.0, 3.0, 2.0, 1.0, -1.0, 0.0), Array(4, 2))

  it should "multiply a batched tensor by a single shared matrix" in {
    val c = sharedLhs.matmul(sharedRhs)

    c.shape.toList shouldBe List(2, 3, 2)
    c.data.toList shouldBe List(2.0, 4.0, 5.0, -11.0, 2.0, 5.0, -5.0, -7.0, 2.0, 2.0, -4.0, 8.0)
  }

  it should "give the same result as multiplying each batch slice separately in 2D" in {
    val c = sharedLhs.matmul(sharedRhs)

    (0 until 2).foreach { b =>
      val slice = Tensor.make(sharedLhs.data.slice(b * 12, (b + 1) * 12), Array(3, 4))
      val expected = slice.matmul(sharedRhs)

      c.data.slice(b * 6, (b + 1) * 6).toList shouldBe expected.data.toList
    }
  }

  it should "sum the shared matrix's gradient over the batch, and keep dA per slice" in {
    val a = Tensor.make(sharedLhs.data.clone(), Array(2, 3, 4), requiresGradient = true)
    val w = Tensor.make(sharedRhs.data.clone(), Array(4, 2), requiresGradient = true)
    // pesos distintos em toda posicao E entre as duas fatias, pra que um indice
    // vazando de uma fatia pra outra apareca
    val weights = Tensor.make(
      Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0),
      Array(2, 3, 2)
    )

    val loss = (a.matmul(w) * weights).sum
    loss.backward()

    // dA[b,m,k] = sum_n dC[b,m,n]*W[k,n] -- uma fatia nao enxerga a outra
    a.gradient.toList shouldBe List(-3.0, 6.0, 4.0, -1.0, -5.0, 12.0, 10.0, -3.0, -7.0, 18.0, 16.0,
      -5.0, -9.0, 24.0, 22.0, -7.0, -11.0, 30.0, 28.0, -9.0, -13.0, 36.0, 34.0, -11.0)
    // dW[k,n] = sum_b sum_m dC[b,m,n]*A[b,m,k] -- soma TAMBEM sobre o lote,
    // porque W foi compartilhada (theory/11-attention/11-attention.md §8)
    w.gradient.toList shouldBe List(22.0, 28.0, 32.0, 36.0, 1.0, 2.0, 58.0, 64.0)
  }

  it should "double the shared matrix's gradient when the batch repeats the same slice" in {
    // duas fatias identicas de A e de dC: dW tem que dar exatamente o dobro do dB
    // do teste 2D acima ([13,18,17,24,21,30]). Um laco de batch faltando no
    // backward da metade disso, sem errar shape nenhum.
    val a = Tensor.make(
      Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
      Array(2, 2, 3),
      requiresGradient = true
    )
    val w =
      Tensor.make(Array(7.0, 8.0, 9.0, 10.0, 11.0, 12.0), Array(3, 2), requiresGradient = true)
    val weights = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0), Array(2, 2, 2))

    val loss = (a.matmul(w) * weights).sum
    loss.backward()

    w.gradient.toList shouldBe List(26.0, 36.0, 34.0, 48.0, 42.0, 60.0)
    // dA repete, porque as duas fatias sao identicas
    a.gradient.toList shouldBe List(23.0, 29.0, 35.0, 53.0, 67.0, 81.0, 23.0, 29.0, 35.0, 53.0,
      67.0, 81.0)
  }

  it should "read a non-contiguous batched operand through its strides" in {
    // primeiro tensor nao-contiguo a chegar no matmul: K.transpose() da Etapa 11
    val x = Tensor.make(
      Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0),
      Array(2, 3, 2)
    )
    val w = Tensor.make(Array(1.0, 0.0, 0.0, 1.0, 1.0, 1.0), Array(3, 2))

    val c = x.transpose(1, 2).matmul(w)

    c.shape.toList shouldBe List(2, 2, 2)
    c.data.toList shouldBe List(6.0, 8.0, 8.0, 10.0, 18.0, 20.0, 20.0, 22.0)
  }

  it should "reject a shared matrix whose first dimension does not match the inner one" in {
    val a = Tensor.make(Array.fill(24)(1.0), Array(2, 3, 4))
    val w = Tensor.make(Array.fill(10)(1.0), Array(5, 2))

    an[IllegalArgumentException] should be thrownBy a.matmul(w)
  }

  // ---- lote de rank qualquer (Etapa 12): [B, H, T, dHead] tem DOIS eixos de lote ----
  // O matmul deixou de ter caminho por rank: as duas ultimas dimensoes sao a
  // matriz, todas as anteriores sao lote. Estes testes cobrem o caminho novo,
  // que nasceu sem cobertura nenhuma (mesma licao do (3,2) na Etapa 11).

  private def ramp(shape: Array[Int], requiresGradient: Boolean = false): Tensor =
    Tensor.make(
      Array.tabulate(shape.product)(i => (i % 7) * 0.5 - 1.0),
      shape,
      requiresGradient
    )

  it should "multiply with two batch axes, matching a manual product position by position" in {
    // B, H, M, K e N todos diferentes: shapes quadradas escondem eixo trocado
    val B = 2; val H = 3; val M = 4; val K = 5; val N = 6
    val a = ramp(Array(B, H, M, K))
    val b = ramp(Array(B, H, K, N))

    val c = a.matmul(b)

    c.shape.toList shouldBe List(B, H, M, N)

    for (i <- 0 until B; h <- 0 until H; m <- 0 until M; n <- 0 until N)
      withClue(s"posicao (b=$i, h=$h, m=$m, n=$n): ") {
        val expected = (0 until K).map(k => a.get(i, h, m, k) * b.get(i, h, k, n)).sum
        c.get(i, h, m, n) shouldBe expected +- 1e-12
      }
  }

  it should "share a rank-2 matrix across two batch axes" in {
    // o caminho das projecoes Q/K/V depois da divisao em cabecas
    val B = 2; val H = 3; val M = 4; val K = 5; val N = 6
    val a = ramp(Array(B, H, M, K))
    val w = ramp(Array(K, N))

    val c = a.matmul(w)

    c.shape.toList shouldBe List(B, H, M, N)

    for (i <- 0 until B; h <- 0 until H; m <- 0 until M; n <- 0 until N)
      withClue(s"posicao (b=$i, h=$h, m=$m, n=$n): ") {
        val expected = (0 until K).map(k => a.get(i, h, m, k) * w.get(k, n)).sum
        c.get(i, h, m, n) shouldBe expected +- 1e-12
      }
  }

  it should "not impose a rank ceiling" in {
    // nada no codigo conta dimensoes de lote, entao rank 6 tem que funcionar
    // igual. Guarda contra uma regressao pra dispatcher por rank.
    val a = ramp(Array(2, 2, 2, 3, 4, 5))
    val b = ramp(Array(2, 2, 2, 3, 5, 6))

    a.matmul(b).shape.toList shouldBe List(2, 2, 2, 3, 4, 6)
  }

  it should "read a non-contiguous rank-4 operand through its strides, without copying it" in {
    // exatamente a divisao em cabecas: [B, T, H, dHead] -> transpose(1, 2)
    val B = 2; val T = 5; val H = 3; val dHead = 4
    val x = ramp(Array(B, T, H, dHead))
    val q = x.transpose(1, 2)
    val k = x.transpose(1, 2).transpose(2, 3)

    val scores = q.matmul(k)

    scores.shape.toList shouldBe List(B, H, T, T)

    for (b <- 0 until B; h <- 0 until H; i <- 0 until T; j <- 0 until T)
      withClue(s"posicao (b=$b, h=$h, i=$i, j=$j): ") {
        val expected = (0 until dHead).map(d => q.get(b, h, i, d) * k.get(b, h, d, j)).sum
        scores.get(b, h, i, j) shouldBe expected +- 1e-12
      }
  }

  it should "reject rank-4 operands whose batch dimensions differ" in {
    val a = ramp(Array(2, 3, 4, 5))
    val b = ramp(Array(2, 5, 5, 6))

    val thrown = the[IllegalArgumentException] thrownBy a.matmul(b)
    // a mensagem cita os shapes ORIGINAIS, nao os achatados: quem le o erro
    // precisa reconhecer os numeros que escreveu
    thrown.getMessage should include("2x3x4x5")
    thrown.getMessage should include("2x5x5x6")
  }

  it should "sum the shared matrix gradient over every batch position" in {
    // dW = soma de A_i^T . dC_i sobre as B*H fatias. Com dC distinto por
    // posicao, uma soma que esquecesse uma fatia mudaria o resultado.
    val a = Tensor.make(Array.tabulate(2 * 2 * 2 * 2)(i => (i + 1).toDouble), Array(2, 2, 2, 2))
    val w = Tensor.make(Array(1.0, 0.0, 0.0, 1.0), Array(2, 2), requiresGradient = true)
    val weights = Tensor.make(
      Array.tabulate(2 * 2 * 2 * 2)(i => (i + 1).toDouble),
      Array(2, 2, 2, 2)
    )

    val out = (a.matmul(w) * weights).sum
    out.backward()

    // dW[k][n] = soma sobre as 8 linhas de a[.., m, k] * weights[.., m, n]
    val expected = for (k <- 0 until 2; n <- 0 until 2) yield {
      (for (i <- 0 until 2; h <- 0 until 2; m <- 0 until 2)
        yield a.get(i, h, m, k) * weights.get(i, h, m, n)).sum
    }

    w.gradient.toList shouldBe expected.toList
  }

  // ---- reducoes sobre entrada NAO contigua ----
  // Regressao: `sum`/`mean`/`max` liam `t1.data`, que e indexado pelas strides
  // reais, enquanto o backward de `reduce` acumula por indice canonico. Em
  // `max` isso mandava o gradiente pra posicao errada; em views broadcastadas
  // (onde data.length < size) `sum` somava o array fisico e devolvia menos.

  "max sobre um tensor transposto" should "creditar o gradiente na posicao logica do maximo" in {
    // o maximo (9.0) esta na posicao fisica 3, que no transposto e a logica (1,0)
    val t = Tensor.make(Array(1.0, 2.0, 3.0, 9.0, 5.0, 6.0), Array(2, 3), requiresGradient = true)

    val out = t.transpose(0, 1).max
    out.backward()

    out.data(0) shouldBe 9.0
    t.gradient.toList shouldBe List(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
  }

  "sum e mean sobre uma view broadcastada" should "usar o tamanho logico" in {
    // a view estica [1,3] pra [2,3] com stride 0: data.length = 3, size = 6
    val linha = Tensor.make(Array(1.0, 2.0, 3.0), Array(1, 3))
    val esticada = linha.broadcastTo(Shape(2, 3))

    esticada.size shouldBe 6
    esticada.sum.data(0) shouldBe 12.0
    esticada.mean.data(0) shouldBe 2.0
  }

  "sum e mean sobre um transposto" should "continuar dando o mesmo total" in {
    // controle: a soma independe da ordem, entao o transposto ja passava antes.
    // O teste existe pra que uma correcao futura nao quebre este caso.
    val t = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), Array(2, 3))

    t.transpose(0, 1).sum.data(0) shouldBe 21.0
    t.transpose(0, 1).mean.data(0) shouldBe 3.5
  }

  "clamp" should "saturate values outside [min, max] and pass through values inside it" in {
    // cobre os tres regimes: abaixo do min, dentro, acima do max, e exatamente nas duas bordas
    val a = Tensor.make(Array(-3.0, 2.0, 7.0, 5.0, 0.0, 4.5), Array(6))

    val c = a.clamp(0.0, 5.0)

    c.data.toList shouldBe List(0.0, 2.0, 5.0, 5.0, 0.0, 4.5)
  }

  it should "route gradient only through positions strictly inside (min, max), zeroing saturated and border positions" in {
    val a = Tensor.make(Array(-3.0, 2.0, 7.0, 5.0, 0.0, 4.5), Array(6), requiresGradient = true)
    // pesos distintos por posicao, pra que dC chegue variado (dC = weights) em vez de
    // uma constante uniforme, que esconderia erro de indexacao no backward
    val weights = Tensor.make(Array(10.0, 20.0, 30.0, 40.0, 50.0, 60.0), Array(6))

    val loss = (a.clamp(0.0, 5.0) * weights).sum
    loss.backward()

    // so os indices 1 (2.0) e 5 (4.5) estao estritamente dentro de (0,5); o resto
    // (saturado ou exatamente na borda) corta o gradiente (theory §7)
    a.gradient.toList shouldBe List(0.0, 20.0, 0.0, 0.0, 0.0, 60.0)
  }

  "relu" should "zero out negative values and pass through non-negative ones" in {
    val a = Tensor.make(Array(-3.0, -0.5, 0.0, 2.0, 5.0), Array(5))

    val c = a.relu

    c.data.toList shouldBe List(0.0, 0.0, 0.0, 2.0, 5.0)
  }

  it should "route gradient only through strictly positive positions, zeroing negative and exactly-zero ones" in {
    val a = Tensor.make(Array(-3.0, -0.5, 0.0, 2.0, 5.0), Array(5), requiresGradient = true)
    // pesos distintos por posicao, pra que dC chegue variado (dC = weights) em vez de
    // uma constante uniforme, que esconderia erro de indexacao no backward
    val weights = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 5.0), Array(5))

    val loss = (a.relu * weights).sum
    loss.backward()

    // so os indices 3 (2.0) e 4 (5.0) tem x>0; x=0 exato (indice 2) tambem corta o
    // gradiente, mesma convencao do clamp (theory/05-activations/05-activations.md §2)
    a.gradient.toList shouldBe List(0.0, 0.0, 0.0, 4.0, 5.0)
  }

  "sigmoid" should "compute the logistic function" in {
    val a = Tensor.make(Array(0.0, 1.0), Array(2))

    val c = a.sigmoid

    c.data(0) shouldBe (0.5 +- 1e-9)
    c.data(1) shouldBe (1.0 / (1.0 + Math.exp(-1.0)) +- 1e-9)
  }

  it should "apply the sigma*(1-sigma) rule in the backward pass" in {
    val a = Tensor.make(Array(0.0, 1.0), Array(2), requiresGradient = true)
    val weights = Tensor.make(Array(1.0, 2.0), Array(2))

    val loss = (a.sigmoid * weights).sum
    loss.backward()

    val sigma1 = 1.0 / (1.0 + Math.exp(-1.0))
    a.gradient(0) shouldBe (0.25 +- 1e-9) // sigma(0)=0.5 -> 0.5*0.5*1
    a.gradient(1) shouldBe (sigma1 * (1 - sigma1) * 2.0 +- 1e-9)
  }

  "tanh" should "compute the hyperbolic tangent" in {
    val a = Tensor.make(Array(0.0, 1.0), Array(2))

    val c = a.tanh

    c.data(0) shouldBe (0.0 +- 1e-9)
    c.data(1) shouldBe (Math.tanh(1.0) +- 1e-9)
  }

  it should "apply the 1-tanh^2 rule in the backward pass" in {
    val a = Tensor.make(Array(0.0, 1.0), Array(2), requiresGradient = true)
    val weights = Tensor.make(Array(1.0, 2.0), Array(2))

    val loss = (a.tanh * weights).sum
    loss.backward()

    val t1 = Math.tanh(1.0)
    a.gradient(0) shouldBe (1.0 +- 1e-9)
    a.gradient(1) shouldBe ((1 - t1 * t1) * 2.0 +- 1e-9)
  }

  "gelu" should "compute the tanh approximation of GELU" in {
    val a = Tensor.make(Array(1.0, -1.0), Array(2))

    val c = a.gelu

    val cConst = Math.sqrt(2.0 / Math.PI)
    def geluRef(x: Double): Double =
      0.5 * x * (1 + Math.tanh(cConst * (x + 0.044715 * Math.pow(x, 3))))

    c.data(0) shouldBe (geluRef(1.0) +- 1e-9)
    c.data(1) shouldBe (geluRef(-1.0) +- 1e-9)
    // valores de referencia (theory/05-activations/05-activations.md §5)
    c.data(0) shouldBe (0.8411 +- 1e-3)
    c.data(1) shouldBe (-0.1589 +- 1e-3)
  }

  it should "apply the product rule combined with tanh's chain rule in the backward pass" in {
    val a = Tensor.make(Array(1.0), Array(1), requiresGradient = true)

    val loss = a.gelu.sum
    loss.backward()

    // formula derivada independentemente do backward implementado, pra servir de
    // oraculo real (nao so reusar a mesma expressao do codigo sob teste)
    val cConst = Math.sqrt(2.0 / Math.PI)
    val u = cConst * (1.0 + 0.044715 * 1.0)
    val t = Math.tanh(u)
    val uPrime = cConst * (1 + 0.134145 * 1.0)
    val expected = 0.5 * (1 + t) + 0.5 * 1.0 * (1 - t * t) * uPrime

    a.gradient(0) shouldBe (expected +- 1e-9)
    // valor de referencia (theory §5)
    a.gradient(0) shouldBe (1.08301 +- 1e-3)
  }

  "softmax" should "compute a numerically stable probability distribution along a dimension" in {
    val a = Tensor.make(Array(1.0, 2.0, 3.0), Array(3))

    val c = a.softmax(0)

    val exps = Array(1.0, 2.0, 3.0).map(Math.exp)
    val expected = exps.map(_ / exps.sum)
    c.data.toList.zip(expected).foreach { case (actual, exp) => actual shouldBe (exp +- 1e-9) }
    c.data.sum shouldBe (1.0 +- 1e-9)
  }

  it should "normalize each row independently when dim=1" in {
    // linha 0 tem a mesma distribuicao relativa do teste anterior; linha 1 e uniforme
    val a = Tensor.make(Array(1.0, 2.0, 3.0, 1.0, 1.0, 1.0), Array(2, 3))

    val c = a.softmax(1)

    val exps = Array(1.0, 2.0, 3.0).map(Math.exp)
    val expectedRow0 = exps.map(_ / exps.sum)
    c.data.toList.take(3).zip(expectedRow0).foreach { case (actual, exp) =>
      actual shouldBe (exp +- 1e-9)
    }
    c.data.toList.drop(3).foreach(_ shouldBe (1.0 / 3.0 +- 1e-9))
  }

  it should "remain finite and correctly normalized for logits that would overflow naive exponentiation" in {
    // sem subtrair o maximo, exp(900) ja estoura pra Infinity em Double (theory §1)
    val a = Tensor.make(Array(700.0, 800.0, 900.0, 750.0), Array(4))

    val c = a.softmax(0)

    c.data.toList.foreach(_.isNaN shouldBe false)
    c.data.sum shouldBe (1.0 +- 1e-9)
    c.data(2) shouldBe (1.0 +- 1e-9) // logit dominante (900, o maximo) carrega quase toda a massa
  }

  it should "apply dx = s*(dOut - (dOut*s).sum()) in the backward pass" in {
    val a = Tensor.make(Array(1.0, 2.0, 3.0), Array(3), requiresGradient = true)
    // usar os proprios pesos como dOut (d(sum(s*w))/ds = w), pra reusar o mesmo
    // exemplo numerico de theory/06-softmax/06-softmax.md (dOut=[0.1,0.2,0.3])
    val dOut = Tensor.make(Array(0.1, 0.2, 0.3), Array(3))

    val loss = (a.softmax(0) * dOut).sum
    loss.backward()

    // oraculo independente: recalcula s e dx pela formula, sem reusar o codigo sob teste
    val s = Array(1.0, 2.0, 3.0).map(Math.exp)
    val sNorm = s.map(_ / s.sum)
    val dOutArr = Array(0.1, 0.2, 0.3)
    val dot = sNorm.zip(dOutArr).map(_ * _).sum
    val expected = sNorm.zip(dOutArr).map { case (si, di) => si * (di - dot) }

    a.gradient.toList.zip(expected).foreach { case (actual, exp) => actual shouldBe (exp +- 1e-9) }
  }

  it should "keep each row's backward independent of the other rows when dim=1" in {
    val a = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), Array(2, 3), requiresGradient = true)
    // pesos distintos por linha, pra pegar vazamento de gradiente entre grupos
    // (o bug real encontrado na revisao: somar sobre o tensor inteiro em vez do grupo)
    val dOut = Tensor.make(Array(0.1, 0.2, 0.3, 0.5, 0.3, 0.2), Array(2, 3))

    val loss = (a.softmax(1) * dOut).sum
    loss.backward()

    def softmaxDxRow(xs: Array[Double], d: Array[Double]): Array[Double] = {
      val exps = xs.map(Math.exp)
      val s = exps.map(_ / exps.sum)
      val dot = s.zip(d).map(_ * _).sum
      s.zip(d).map { case (si, di) => si * (di - dot) }
    }
    val expected = softmaxDxRow(Array(1.0, 2.0, 3.0), Array(0.1, 0.2, 0.3)) ++
      softmaxDxRow(Array(4.0, 5.0, 6.0), Array(0.5, 0.3, 0.2))

    a.gradient.toList.zip(expected).foreach { case (actual, exp) => actual shouldBe (exp +- 1e-9) }
  }

  "logSoftmax" should "compute log(softmax(x)) via log-sum-exp, without dividing by a probability" in {
    val a = Tensor.make(Array(1.0, 2.0, 3.0), Array(3))

    val c = a.logSoftmax(0)

    val logSumExp = Math.log(Array(1.0, 2.0, 3.0).map(Math.exp).sum)
    c.data.toList.zip(Array(1.0, 2.0, 3.0)).foreach { case (actual, x) =>
      actual shouldBe (x - logSumExp +- 1e-9)
    }
  }

  it should "equal the log of softmax's output" in {
    val a = Tensor.make(Array(1.0, 2.0, 3.0, -1.0), Array(4))

    val softmaxOut = a.softmax(0)
    val logSoftmaxOut = a.logSoftmax(0)

    softmaxOut.data.toList.zip(logSoftmaxOut.data.toList).foreach { case (s, ls) =>
      Math.log(s) shouldBe (ls +- 1e-9)
    }
  }

  it should "remain finite for logits that would overflow naive exponentiation" in {
    val a = Tensor.make(Array(700.0, 800.0, 900.0, 750.0), Array(4))

    val c = a.logSoftmax(0)

    c.data.toList.foreach { x =>
      x.isNaN shouldBe false; x.isInfinite shouldBe false
    }
    c.data(2) shouldBe (0.0 +- 1e-9) // logit dominante: log(prob~1) ~ 0
  }

  it should "apply dx = dOut - s*dOut.sum() in the backward pass" in {
    val a = Tensor.make(Array(1.0, 2.0, 3.0), Array(3), requiresGradient = true)
    val dOut = Tensor.make(Array(0.1, 0.2, 0.3), Array(3))

    val loss = (a.logSoftmax(0) * dOut).sum
    loss.backward()

    val s = Array(1.0, 2.0, 3.0).map(Math.exp)
    val sNorm = s.map(_ / s.sum)
    val dOutArr = Array(0.1, 0.2, 0.3)
    val sumDOut = dOutArr.sum
    val expected = dOutArr.zip(sNorm).map { case (di, si) => di - si * sumDOut }

    a.gradient.toList.zip(expected).foreach { case (actual, exp) => actual shouldBe (exp +- 1e-9) }
  }

  it should "keep each row's backward independent of the other rows when dim=1" in {
    val a = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), Array(2, 3), requiresGradient = true)
    val dOut = Tensor.make(Array(0.1, 0.2, 0.3, 0.5, 0.3, 0.2), Array(2, 3))

    val loss = (a.logSoftmax(1) * dOut).sum
    loss.backward()

    def logSoftmaxDxRow(xs: Array[Double], d: Array[Double]): Array[Double] = {
      val exps = xs.map(Math.exp)
      val s = exps.map(_ / exps.sum)
      val sumD = d.sum
      d.zip(s).map { case (di, si) => di - si * sumD }
    }
    val expected = logSoftmaxDxRow(Array(1.0, 2.0, 3.0), Array(0.1, 0.2, 0.3)) ++
      logSoftmaxDxRow(Array(4.0, 5.0, 6.0), Array(0.5, 0.3, 0.2))

    a.gradient.toList.zip(expected).foreach { case (actual, exp) => actual shouldBe (exp +- 1e-9) }
  }

  // Mesma tabela e mesmo exemplo numerico de theory/09-embedding/09-embedding.md SS1/SS3
  // (5 linhas, embeddingDim=2), reusado nos testes de indexSelect abaixo.
  private def sampleTable(requiresGradient: Boolean = true): Tensor =
    Tensor.make(
      Array(
        0.10, -0.20, 0.30, 0.40, -0.10, 0.50, 0.05, -0.05, 0.20, 0.20
      ),
      Array(5, 2),
      requiresGradient
    )

  "indexSelect" should "gather rows by index, repeating a row when its index repeats" in {
    val table = sampleTable()

    val out = table.indexSelect(Array(1, 0, 1))

    out.shape.toList shouldBe List(3, 2)
    out.data.toList shouldBe List(0.30, 0.40, 0.10, -0.20, 0.30, 0.40)
  }

  it should "scatter-add gradient into only the rows that were selected (theory SS3)" in {
    val table = sampleTable()
    // pesos distintos por posicao de saida, pra que o gradiente que chega em
    // cada linha selecionada nao seja uma constante uniforme (esconderia erro
    // de indexacao) -- mesma tecnica ja usada nos testes de matmul/softmax.
    val weights = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 0.5, -1.0), Array(3, 2))

    val loss = (table.indexSelect(Array(1, 0, 1)) * weights).sum
    loss.backward()

    // linha 0: so a posicao 1 (dY=[3,4]); linha 1: posicoes 0 e 2 somadas ([1,2]+[0.5,-1]=[1.5,1])
    table.gradient.toList shouldBe List(3.0, 4.0, 1.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  }

  it should "reject a table that is not rank-2 (regression: rank check must run before shape(1) is read)" in {
    val rank1Table = Tensor.make(Array(1.0, 2.0, 3.0), Array(3))

    an[IllegalArgumentException] should be thrownBy rank1Table.indexSelect(Array(0))
  }

  it should "reject an index at or beyond the table's row count" in {
    val table = sampleTable()

    an[IllegalArgumentException] should be thrownBy table.indexSelect(Array(0, 5))
  }

  it should "reject a negative index" in {
    val table = sampleTable()

    an[IllegalArgumentException] should be thrownBy table.indexSelect(Array(-1))
  }

  it should "mark the output as not requiring gradient when the table doesn't" in {
    val table = sampleTable(requiresGradient = false)

    table.indexSelect(Array(0, 1)).requiresGradient shouldBe false
  }
}
