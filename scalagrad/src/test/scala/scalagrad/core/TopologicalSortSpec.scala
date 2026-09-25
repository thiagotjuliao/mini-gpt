package scalagrad.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Nó de grafo mínimo para desenvolver `topologicalSort` isoladamente do `Tensor`.
  * Identidade por referência (não `case class`) pelo mesmo motivo do `Tensor`:
  * dois nós distintos podem ter o mesmo `name`, e igualdade estrutural os
  * colapsaria num `Set`, mascarando bugs de deduplicação.
  */
final class Node(val name: String, val previous: Set[Node] = Set()):
  override def toString: String = name

/** Implemente aqui. Depois de validar contra os testes abaixo, transplante
  * a lógica para dentro de `object Tensor` trocando `Node` por `Tensor`.
  */
object TopoSort:
  def sort(root: Node): List[Node] =
    @scala.annotation.tailrec
    def visit(
        nodes: List[Node],
        visited: Set[Node],
        added: Set[Node],
        sorted: Vector[Node]
    ): List[Node] =
      if nodes.isEmpty then sorted.toList
      else
        val node = nodes.head
        val parents = node.previous.toList

        if parents.forall(visited.contains(_)) then
          if !added.contains(node) then
            visit(nodes.tail, visited + node, added + node, sorted :+ node)
          else visit(nodes.tail, visited + node, added, sorted)
        else visit(parents ::: nodes, visited + node, added, sorted)

    visit(List(root), Set(), Set(), Vector())

class TopologicalSortSpec extends AnyFlatSpec with Matchers:

  "topologicalSort" should "visit every reachable node exactly once" in {
    val a = new Node("a")
    val b = new Node("b")
    val root = new Node("root", Set(a, b))

    val order = TopoSort.sort(root)

    order.toSet shouldBe Set(a, b, root)
    order.size shouldBe 3 // detecta duplicatas, não só membros distintos
  }

  it should "place the root last" in {
    val a = new Node("a")
    val b = new Node("b")
    val root = new Node("root", Set(a, b))

    val order = TopoSort.sort(root)

    order.last shouldBe root
  }

  it should "place every node after all of its dependencies (previous)" in {
    val a = new Node("a")
    val b = new Node("b")
    val root = new Node("root", Set(a, b))

    val order = TopoSort.sort(root)
    val position = order.zipWithIndex.toMap

    for node <- order; dep <- node.previous do
      withClue(s"${dep.name} deveria vir antes de ${node.name}") {
        position(dep) should be < position(node)
      }
  }

  // grafo "diamante": d é dependência compartilhada por x e y.
  //
  //     a   d   b
  //      \ / \ /
  //       x   y
  //        \ /
  //        loss
  it should "not duplicate a node reachable via two different paths" in {
    val a = new Node("a")
    val d = new Node("d")
    val b = new Node("b")
    val x = new Node("x", Set(a, d))
    val y = new Node("y", Set(b, d))
    val loss = new Node("loss", Set(x, y))

    val order = TopoSort.sort(loss)

    order.count(_ eq d) shouldBe 1
    order.size shouldBe 6
  }

  it should "keep the dependency invariant even with a shared node (diamond graph)" in {
    val a = new Node("a")
    val d = new Node("d")
    val b = new Node("b")
    val x = new Node("x", Set(a, d))
    val y = new Node("y", Set(b, d))
    val loss = new Node("loss", Set(x, y))

    val order = TopoSort.sort(loss)
    val position = order.zipWithIndex.toMap

    // em particular, isso falha se `d` for colocado antes de `x` ou de `y`
    for node <- order; dep <- node.previous do position(dep) should be < position(node)
    order.last shouldBe loss
  }

  it should "not stack overflow on a long chain (stack-safety check)" in {
    val depth = 100000
    val chain = (1 until depth).foldLeft(new Node("0")) { (prev, i) =>
      new Node(i.toString, Set(prev))
    }

    val order = TopoSort.sort(chain)

    order.size shouldBe depth
    order.last shouldBe chain
  }

  // aqui quem é compartilhado não é uma folha, e sim um nó intermediário (n):
  //
  //   leafZ
  //     |
  //     n
  //    / \
  //   x   y
  //    \ /
  //   root
  it should "not duplicate a non-leaf node reachable via two different paths" in {
    val leafZ = new Node("leafZ")
    val n = new Node("n", Set(leafZ))
    val x = new Node("x", Set(n))
    val y = new Node("y", Set(n))
    val root = new Node("root", Set(x, y))

    val order = TopoSort.sort(root)

    order.count(_ eq n) shouldBe 1
    order.size shouldBe 5
  }
end TopologicalSortSpec
