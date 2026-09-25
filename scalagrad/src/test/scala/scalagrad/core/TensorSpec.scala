package scalagrad.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TensorSpec extends AnyFlatSpec with Matchers:

  "A Tensor" should "compute rank and size from its shape" in {
    val t = Tensor.make(Array.fill(24)(0.0), Array(2, 3, 4))
    t.rank shouldBe 3
    t.size shouldBe 24
  }

  it should "throw when data length does not match the product of shape" in {
    an[AssertionError] should be thrownBy Tensor.make(Array(1.0, 2.0, 3.0), Array(2, 2))
  }

  "index" should "compute the correct linear index for a 2D row-major tensor" in {
    val t = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), Array(2, 3))
    t.index(0, 0) shouldBe 0
    t.index(0, 2) shouldBe 2
    t.index(1, 0) shouldBe 3
    t.index(1, 2) shouldBe 5
  }

  it should "compute the correct linear index for a 3D row-major tensor" in {
    val t = Tensor.make(Array.tabulate(2 * 3 * 4)(_.toDouble), Array(2, 3, 4))
    t.index(0, 0, 0) shouldBe 0
    t.index(0, 1, 0) shouldBe 4
    t.index(1, 0, 0) shouldBe 12
    t.index(1, 2, 3) shouldBe 23
  }

  it should "reject a dimension value that is out of bounds for its axis" in {
    val t = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), Array(2, 3))
    an[IllegalArgumentException] should be thrownBy t.index(2, 0)
  }

  "get" should "read back the values used to construct the tensor" in {
    val t = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), Array(2, 3))
    t.get(0, 0) shouldBe 1.0
    t.get(0, 2) shouldBe 3.0
    t.get(1, 0) shouldBe 4.0
    t.get(1, 2) shouldBe 6.0
  }

  "unravelIndex" should "be the inverse of index for every linear position" in {
    val shape = Array(2, 3, 4)
    val t = Tensor.make(Array.fill(shape.product)(0.0), shape)
    for n <- 0 until t.size do
      val multiIdx = t.unravelIndex(n)
      t.index(multiIdx*) shouldBe n
  }

  it should "reject an index outside the valid range" in {
    val t = Tensor.make(Array(1.0, 2.0, 3.0, 4.0), Array(2, 2))
    an[IllegalArgumentException] should be thrownBy t.unravelIndex(-1)
    an[IllegalArgumentException] should be thrownBy t.unravelIndex(4)
  }

  "reshape" should "preserve the flat data in row-major order" in {
    val t = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), Array(2, 3))
    val r = t.reshape(Array(3, 2))
    r.rank shouldBe 2
    r.size shouldBe 6
    r.get(0, 0) shouldBe 1.0
    r.get(0, 1) shouldBe 2.0
    r.get(1, 0) shouldBe 3.0
    r.get(1, 1) shouldBe 4.0
    r.get(2, 0) shouldBe 5.0
    r.get(2, 1) shouldBe 6.0
  }

  it should "reject a new shape whose size does not match the current size" in {
    val t = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), Array(2, 3))
    an[IllegalArgumentException] should be thrownBy t.reshape(Array(4, 2))
  }

  it should "reinterpret the logical (not physical) order when reshaping a non-contiguous tensor" in {
    // A=[[1,2,3],[4,5,6]] transposto e' logicamente [[1,4],[2,5],[3,6]] (fisicamente
    // ainda [1,2,3,4,5,6]); reshape precisa ler pela ordem logica, nao pelo array cru
    val a = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), Array(2, 3))
    val r = a.transpose().reshape(Array(6))

    r.data.toList shouldBe List(1.0, 4.0, 2.0, 5.0, 3.0, 6.0)
  }

  it should "route the upstream gradient back to the original tensor via its own backward node" in {
    val a = Tensor.make(Array(1.0, 4.0, 2.0, 5.0, 0.0, 3.0), Array(2, 3), requiresGradient = true)
    val r = a.reshape(Array(3, 2))

    // reshape nao reordena: a posicao canonica i do gradiente de saida
    // corresponde a mesma posicao i no gradiente de entrada
    List(10.0, 20.0, 30.0, 40.0, 50.0, 60.0).zipWithIndex.foreach { case (v, i) =>
      r.gradient.accumulate(i, v)
    }
    r.backwardStep()

    a.gradient.toList shouldBe List(10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
  }

  it should "force requiresGradient = false and disconnect from the graph inside noGrad" in {
    val a = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), Array(2, 3), requiresGradient = true)

    val r = Tensor.noGrad { a.reshape(Array(3, 2)) }

    r.requiresGradient shouldBe false
    r.previous shouldBe empty
  }

  "transpose" should "not mutate the original tensor" in {
    val t = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), Array(2, 3))
    val transposed = t.transpose()
    t.rank shouldBe 2
    t.get(0, 0) shouldBe 1.0
    t.get(1, 2) shouldBe 6.0
    transposed should not be theSameInstanceAs(t)
  }

  it should "swap the last two dimensions by default" in {
    val t = Tensor.make(Array.fill(24)(0.0), Array(2, 3, 4))
    val transposed = t.transpose()
    transposed.rank shouldBe 3
    transposed.get(0, 0, 0) // just checking it does not throw for a valid index
  }

  it should "produce values equivalent to the mathematical transpose of a 2D matrix" in {
    // matrix [[1,2,3],[4,5,6]] transposed is [[1,4],[2,5],[3,6]]
    val t = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), Array(2, 3))
    val transposed = t.transpose(0, 1)

    transposed.rank shouldBe 2
    transposed.size shouldBe 6

    transposed.get(0, 0) shouldBe t.get(0, 0)
    transposed.get(0, 1) shouldBe t.get(1, 0)
    transposed.get(1, 0) shouldBe t.get(0, 1)
    transposed.get(1, 1) shouldBe t.get(1, 1)
    transposed.get(2, 0) shouldBe t.get(0, 2)
    transposed.get(2, 1) shouldBe t.get(1, 2)
  }

  it should "route the upstream gradient back to the original shape, undoing the same axis swap" in {
    // mesmo A=[[1,4,2],[5,0,3]] da teoria; At=[[1,5],[4,0],[2,3]] shape (3,2)
    val a = Tensor.make(Array(1.0, 4.0, 2.0, 5.0, 0.0, 3.0), Array(2, 3), requiresGradient = true)
    val at = a.transpose()

    // dC distinto por posicao, shape (3,2): [[10,20],[30,40],[50,60]]
    List(10.0, 20.0, 30.0, 40.0, 50.0, 60.0).zipWithIndex.foreach { case (v, i) =>
      at.gradient.accumulate(i, v)
    }
    at.backwardStep()

    // dA esperado (teoria, secao 5): [[10,30,50],[20,40,60]]
    a.gradient.toList shouldBe List(10.0, 30.0, 50.0, 20.0, 40.0, 60.0)
  }

  it should "force requiresGradient = false and disconnect from the graph inside noGrad" in {
    val a = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), Array(2, 3), requiresGradient = true)

    val at = Tensor.noGrad { a.transpose() }

    at.requiresGradient shouldBe false
    at.previous shouldBe empty
  }

  it should "copy the gradient without permuting it" in {
    // regressao: `mapping` traduz indice canonico -> posicao fisica e so vale
    // pra `data`. O `gradient` ja e canonico, entao aplicar o mapping nele
    // embaralhava o gradiente de qualquer tensor nao contiguo. Ninguem lia
    // esse array hoje, mas era uma armadilha armada pro proximo chamador.
    val t = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), Array(2, 3))
    val transposed = t.transpose(0, 1)

    // valores bem separados: uma permutacao muda a lista de forma obvia
    (0 until transposed.size).foreach(i => transposed.gradient.accumulate(i, Math.pow(10, i)))

    val c = transposed.contiguous

    c.shape.toList shouldBe transposed.shape.toList
    c.gradient.toList shouldBe transposed.gradient.toList
  }

  "contiguous" should "reorder the underlying data to canonical row-major layout after a transpose" in {
    val t = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), Array(2, 3))
    val c = t.transpose(0, 1).contiguous

    c.rank shouldBe 2
    c.size shouldBe 6
    c.get(0, 0) shouldBe 1.0
    c.get(0, 1) shouldBe 4.0
    c.get(1, 0) shouldBe 2.0
    c.get(1, 1) shouldBe 5.0
    c.get(2, 0) shouldBe 3.0
    c.get(2, 1) shouldBe 6.0
  }

  "broadcastTo" should "stretch a dimension of size 1 to match the target shape, repeating the same data" in {
    val t = Tensor.make(Array(10.0, 20.0, 30.0), Array(1, 3))
    val b = t.broadcastTo(Shape(2, 3))

    b.rank shouldBe 2
    b.size shouldBe 6
    b.get(0, 0) shouldBe 10.0; b.get(0, 1) shouldBe 20.0; b.get(0, 2) shouldBe 30.0
    b.get(1, 0) shouldBe 10.0; b.get(1, 1) shouldBe 20.0; b.get(1, 2) shouldBe 30.0
  }

  it should "left-pad rank before stretching, when the target has more dimensions" in {
    val t = Tensor.make(Array(1.0, 2.0, 3.0), Array(3))
    val b = t.broadcastTo(Shape(2, 3))

    b.rank shouldBe 2
    b.get(0, 1) shouldBe 2.0
    b.get(1, 1) shouldBe 2.0
  }

  it should "reject a target shape that is not broadcast-compatible" in {
    val t = Tensor.make(Array(1.0, 2.0), Array(2))
    an[IllegalArgumentException] should be thrownBy t.broadcastTo(Shape(3))
  }

  "randn com um rng semeado" should "produzir exatamente os mesmos valores" in {
    // sem semente injetavel nenhuma inicializacao de modelo e reproduzivel, e
    // "rodei de novo e deu diferente" fica indistinguivel de "mudei algo".
    val a = Tensor.randn(Array(3, 4), rng = new scala.util.Random(7))
    val b = Tensor.randn(Array(3, 4), rng = new scala.util.Random(7))
    val c = Tensor.randn(Array(3, 4), rng = new scala.util.Random(8))

    a.data.toList shouldBe b.data.toList
    a.data.toList should not be c.data.toList
  }

  "isContiguous" should "reconhecer um tensor transposto duas vezes" in {
    // as strides voltam a ser as canonicas, mas a instancia e outra: uma
    // comparacao por referencia diria que nao, e forcaria uma copia inutil
    // justamente no ida-e-volta da divisao em cabecas da Etapa 12.
    val t = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), Array(2, 3))

    val idaEVolta = t.transpose(0, 1).transpose(0, 1)

    t.isContiguous shouldBe true
    t.transpose(0, 1).isContiguous shouldBe false
    idaEVolta.isContiguous shouldBe true

    // e por isso `contiguous` devolve o proprio tensor, sem copiar
    (idaEVolta.contiguous eq idaEVolta) shouldBe true
  }

  "Strides.toArray" should "devolver uma copia defensiva" in {
    // regressao: antes devolvia o array interno, ao contrario de Shape.toArray.
    // Quem quer o array cru sem copia agora pede por `unsafeValues`, que diz isso.
    val t = Tensor.make(Array(1.0, 2.0, 3.0, 4.0), Array(2, 2))
    val copia = t.shape.canonicalStrides.toArray
    copia(0) = 999

    t.shape.canonicalStrides.toArray.toList shouldBe List(2, 1)
  }

  // ---- oneHot (Etapa 16) ----
  // Fabrica de nucleo: uma linha por indice, um 1.0 por linha. A perda da
  // Etapa 16 a usa pra selecionar o logit do alvo sem precisar de um `gather`.

  "Tensor.oneHot" should "put the 1.0 in the column named by each index" in {
    // regressao: a primeira versao comparava a LINHA com o indice
    // (`i == indices(i)`) em vez da coluna, o que fazia cada linha sair
    // inteira de zeros ou inteira de uns.
    val out = Tensor.oneHot(Array(2, 0, 3), numClasses = 4)

    out.shape.toArray shouldBe Array(3, 4)

    val esperado = List(
      List(0.0, 0.0, 1.0, 0.0),
      List(1.0, 0.0, 0.0, 0.0),
      List(0.0, 0.0, 0.0, 1.0)
    )

    for i <- 0 until 3; j <- 0 until 4 do
      withClue(s"posicao ($i, $j): ") {
        out.get(i, j) shouldBe esperado(i)(j)
      }
  }

  it should "give every row exactly one 1.0, whatever the indices" in {
    val out = Tensor.oneHot(Array(0, 0, 4, 1, 4), numClasses = 5)

    for i <- 0 until 5 do
      withClue(s"linha $i: ") {
        (0 until 5).map(j => out.get(i, j)).sum shouldBe 1.0
        (0 until 5).count(j => out.get(i, j) == 1.0) shouldBe 1
      }
  }

  it should "repeat a row when its index repeats" in {
    val out = Tensor.oneHot(Array(3, 3), numClasses = 4)

    (0 until 4).foreach(j => out.get(0, j) shouldBe out.get(1, j))
  }

  it should "be a constant: no gradient required" in {
    // ela e uma mascara, nao um parametro -- se nascesse com requiresGradient
    // viraria passageiro na lista que o otimizador percorre.
    Tensor.oneHot(Array(1, 0), numClasses = 3).requiresGradient shouldBe false
  }

  it should "reject an index outside [0, numClasses)" in {
    // sem esta checagem, um indice fora da faixa produz uma linha toda zero
    // em silencio -- mesmo modo de falha do bug do i/j. O `indexSelect` valida
    // do mesmo jeito.
    an[IllegalArgumentException] should be thrownBy Tensor.oneHot(Array(0, 3), numClasses = 3)
    an[IllegalArgumentException] should be thrownBy Tensor.oneHot(Array(-1), numClasses = 3)
  }

  "toArray" should "return the values in canonical order" in {
    val t = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), Array(2, 3))

    t.toArray shouldBe Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
  }

  it should "follow the shape, not the buffer, on a transposed view" in {
    // `data` continua na ordem fisica original; so as strides mudaram. Ler o
    // buffer direto devolveria a ordem errada.
    val t = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), Array(2, 3)).transpose()

    t.shape.toArray shouldBe Array(3, 2)
    t.toArray shouldBe Array(1.0, 4.0, 2.0, 5.0, 3.0, 6.0)
  }

  it should "return a copy, so mutating the result leaves the tensor alone" in {
    val t = Tensor.make(Array(1.0, 2.0), Array(2))
    val copia = t.toArray
    copia(0) = 99.0

    t.get(0) shouldBe 1.0
  }

  "updateData" should "overwrite the values in canonical order" in {
    val t = Tensor.make(Array(1.0, 2.0, 3.0, 4.0), Array(2, 2))
    t.updateData(Array(10.0, 20.0, 30.0, 40.0))

    t.get(0, 0) shouldBe 10.0
    t.get(0, 1) shouldBe 20.0
    t.get(1, 0) shouldBe 30.0
    t.get(1, 1) shouldBe 40.0
  }

  it should "not touch the gradient" in {
    val t = Tensor.make(Array(1.0, 2.0), Array(2), true)
    t.gradient.accumulate(0, 0.5)
    t.updateData(Array(9.0, 9.0))

    t.gradient.toArray shouldBe Array(0.5, 0.0)
  }

  it should "reject an array whose length is not the tensor size" in {
    val t = Tensor.make(Array(1.0, 2.0, 3.0, 4.0), Array(2, 2))

    an[IllegalArgumentException] should be thrownBy t.updateData(Array(1.0, 2.0))
    an[IllegalArgumentException] should be thrownBy t.updateData(Array(1.0, 2.0, 3.0, 4.0, 5.0))
  }

  it should "reject a transposed view" in {
    // `data` e indexado pelas strides reais e o array recebido e canonico: numa
    // view os dois divergem, e a escrita embaralharia o tensor (bug 2026-08-21).
    val t = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), Array(2, 3)).transpose()

    an[IllegalArgumentException] should be thrownBy t.updateData(Array.fill(6)(0.0))
  }

  it should "reject a broadcast view" in {
    // pior que a transposta: varias posicoes do shape anunciado apontam para o
    // mesmo elemento fisico, entao nem existe um array de `size` valores.
    val t = Tensor.make(Array(1.0, 2.0, 3.0), Array(1, 3)).broadcastTo(Shape(2, 3))

    an[IllegalArgumentException] should be thrownBy t.updateData(Array.fill(6)(0.0))
  }

  it should "accept a tensor made contiguous again after a transpose" in {
    val t = Tensor.make(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), Array(2, 3)).transpose().contiguous

    t.updateData(Array(7.0, 8.0, 9.0, 10.0, 11.0, 12.0))

    t.get(0, 0) shouldBe 7.0
    t.get(2, 1) shouldBe 12.0
  }
end TensorSpec
