package gpt.train

import gpt.model.{GPT, GPTConfig}
import gpt.optim.AdamW
import scalagrad.core.Tensor

import java.io.{
  BufferedInputStream,
  BufferedOutputStream,
  DataInputStream,
  DataOutputStream,
  File,
  FileInputStream,
  FileOutputStream
}
import scala.util.{Random, Using}

/** Serialização binária dos parâmetros do modelo e do estado do otimizador.
  *
  * Formato, em ordem: um magic number, a versão, a `GPTConfig`, o `t` do
  * otimizador, a quantidade de parâmetros e um marcador de "tem momentos".
  * Depois, por parâmetro na ordem de `model.parameters`: o tamanho, os valores,
  * e — quando há momentos — os arrays `m` e `v`.
  *
  * A configuração entra para que `loadModel` possa reconstruir o modelo sozinho,
  * e para que carregar num modelo de forma diferente dê uma mensagem que nomeia
  * as duas configurações, em vez de reclamar do tamanho de um parâmetro.
  *
  * `m` e `v` entram junto de propósito: retomar um treino sem eles reinicia o
  * momento em zero, e a correção de viés volta a agir como se fosse o primeiro
  * passo (theory/18-training-loop §6).
  */
object Checkpoint {
  private val Magic = 0x6d696e69 // "mini"
  private val Version = 2

  private def writeConfig(out: DataOutputStream, c: GPTConfig): Unit = {
    out.writeInt(c.vocabSize)
    out.writeInt(c.dModel)
    out.writeInt(c.nHeads)
    out.writeInt(c.nLayers)
    out.writeInt(c.contextLength)
    out.writeInt(c.expansion)
  }

  private def readConfig(in: DataInputStream): GPTConfig =
    GPTConfig(
      vocabSize = in.readInt(),
      dModel = in.readInt(),
      nHeads = in.readInt(),
      nLayers = in.readInt(),
      contextLength = in.readInt(),
      expansion = in.readInt()
    )

  /** A configuração guardada no arquivo, sem carregar os pesos. */
  def configOf(file: File): GPTConfig =
    Using.resource(new DataInputStream(new BufferedInputStream(new FileInputStream(file)))) { in =>
      readHeader(in)._1
    }

  private def readHeader(in: DataInputStream): (GPTConfig, Int, Int, Boolean) = {
    val magic = in.readInt()
    require(magic == Magic, f"Not a mini-gpt checkpoint: unexpected magic 0x$magic%08x.")

    val version = in.readInt()
    require(version == Version, s"Unsupported checkpoint version $version, expected $Version.")

    val config = readConfig(in)
    val t = in.readInt()
    val count = in.readInt()
    val hasMoments = in.readBoolean()

    (config, t, count, hasMoments)
  }

  def save(file: File, model: GPT, optimizer: AdamW): Unit = {
    val parameters = model.parameters
    val moments = optimizer.state
    val hasMoments = parameters.forall(moments.contains)

    Using.resource(
      new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)))
    ) { out =>
      out.writeInt(Magic)
      out.writeInt(Version)
      writeConfig(out, model.config)
      out.writeInt(optimizer.t)
      out.writeInt(parameters.length)
      out.writeBoolean(hasMoments)

      parameters.foreach { p =>
        out.writeInt(p.size)
        p.toArray.foreach(out.writeDouble)

        if hasMoments then
          val (m, v) = moments(p)
          m.foreach(out.writeDouble)
          v.foreach(out.writeDouble)
      }
    }
  }

  /** Constrói o modelo a partir da configuração guardada no arquivo e carrega
    * os pesos nele. É o caminho para gerar texto de um checkpoint sem precisar
    * lembrar com que dimensões o modelo foi treinado.
    */
  def loadModel(file: File, rng: Random = new Random()): (GPT, AdamW) = {
    val model = new GPT(configOf(file), rng)
    val optimizer = load(file, model, new AdamW(model.parameters))

    model -> optimizer
  }

  /** Restaura no `model` e no `optimizer` recebidos, que precisam ter sido
    * construídos com a mesma configuração. Devolve o otimizador com `t` e os
    * momentos do arquivo; o modelo é atualizado in-place, via `updateData`.
    */
  def load(file: File, model: GPT, optimizer: AdamW): AdamW = {
    val parameters = model.parameters

    Using.resource(
      new DataInputStream(new BufferedInputStream(new FileInputStream(file)))
    ) { in =>
      val (config, t, count, hasMoments) = readHeader(in)

      // Conferir a configuração antes dos tamanhos dá a mensagem útil: diz qual
      // modelo o arquivo espera, em vez de reclamar do parâmetro 7.
      require(
        config == model.config,
        s"This checkpoint was saved from $config, but the model given is ${model.config}."
      )

      require(
        count == parameters.length,
        s"Checkpoint holds $count parameters, but this model has ${parameters.length}."
      )

      val restored = parameters.map { p =>
        val size = in.readInt()
        require(
          size == p.size,
          s"Checkpoint holds $size values for a parameter of size ${p.size}. " +
            "The model configuration must match the one that was saved."
        )

        p.updateData(Array.fill(size)(in.readDouble()))

        val moments =
          if hasMoments then
            val m = Array.fill(size)(in.readDouble())
            val v = Array.fill(size)(in.readDouble())
            Some(p -> (m, v))
          else None

        moments
      }

      new AdamW(
        optimizer.parameters,
        optimizer.lr,
        optimizer.beta1,
        optimizer.beta2,
        optimizer.eps,
        optimizer.weightDecay,
        t,
        restored.flatten.toMap
      )
    }
  }
}
