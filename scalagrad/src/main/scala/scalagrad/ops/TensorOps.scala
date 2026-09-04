package scalagrad.ops

/** Ponto único de import da API de operações (`import scalagrad.ops.tensor.*`).
  * A implementação de cada categoria mora no seu próprio arquivo (`UnaryOps`,
  * `BinaryOps`, `ReduceOps`, `MatmulOps`) — misturados aqui via `extends`/`with`
  * pra que todos os `extension` fiquem disponíveis num só import, sem duplicar
  * nada.
  */
object tensor
    extends UnaryOps
    with BinaryOps
    with ReduceOps
    with MatmulOps
    with SoftmaxOps
    with IndexOps
