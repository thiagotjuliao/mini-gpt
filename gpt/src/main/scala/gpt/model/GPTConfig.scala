package gpt.model

/** A forma do modelo, num valor só.
  *
  * Existe porque o checkpoint da Etapa 18 precisa guardar a configuração junto
  * dos pesos: sem ela, carregar um arquivo exige lembrar de cabeça com que
  * dimensões o modelo foi criado, e errar produz um erro de tamanho de
  * parâmetro em vez de uma mensagem útil
  * (ver theory/18-training-loop/18-training-loop.md §6).
  */
final case class GPTConfig(
    vocabSize: Int,
    dModel: Int,
    nHeads: Int,
    nLayers: Int,
    contextLength: Int,
    expansion: Int = 4
)
