package gpt.train

import gpt.loss.CrossEntropy
import gpt.model.{GPT, GPTConfig}
import gpt.optim.AdamW
import scalagrad.core.Tensor
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import java.io.{DataOutputStream, File, FileOutputStream}

class CheckpointSpec extends AnyFlatSpec with Matchers:

  private val vocabSize = 6
  private val dModel = 8
  private val contextLength = 4

  private def modelo(dim: Int = dModel): GPT =
    GPT(vocabSize, dim, nHeads = 2, nLayers = 1, contextLength = contextLength)

  private def otimizador(m: GPT): AdamW = new AdamW(m.parameters, lr = 1e-3)

  private val entradas = Tensor.make(Array(1.0, 2.0, 3.0, 0.0), Array(1, contextLength))
  private val alvos = Tensor.make(Array(2.0, 3.0, 0.0, 1.0), Array(1, contextLength))

  private def umPasso(m: GPT, opt: AdamW): AdamW =
    val loss = CrossEntropy(m.forward(entradas), alvos)
    opt.zeroGrad()
    loss.backward()
    opt.step()

  private def arquivoTemporario(): File =
    val f = File.createTempFile("mini-gpt-checkpoint", ".bin")
    f.deleteOnExit()
    f

  private def logits(m: GPT): Array[Double] = Tensor.noGrad(m.forward(entradas)).toArray

  "a checkpoint" should "restore the parameters into a different model instance" in {
    val original = modelo()
    val opt = umPasso(original, otimizador(original))

    val arquivo = arquivoTemporario()
    Checkpoint.save(arquivo, original, opt)

    val restaurado = modelo()
    // dois modelos nascem com inicializacao aleatoria distinta
    logits(restaurado) should not be logits(original)

    Checkpoint.load(arquivo, restaurado, otimizador(restaurado))

    restaurado.parameters.zip(original.parameters).foreach { (r, o) =>
      r.toArray shouldBe o.toArray
    }
    logits(restaurado) shouldBe logits(original)
  }

  it should "restore the optimizer step counter" in {
    val m = modelo()
    val opt = umPasso(m, umPasso(m, umPasso(m, otimizador(m))))
    opt.t shouldBe 3

    val arquivo = arquivoTemporario()
    Checkpoint.save(arquivo, m, opt)

    val outro = modelo()
    Checkpoint.load(arquivo, outro, otimizador(outro)).t shouldBe 3
  }

  it should "restore the moments, so resuming continues instead of restarting" in {
    val original = modelo()
    val opt = umPasso(original, umPasso(original, otimizador(original)))

    val arquivo = arquivoTemporario()
    Checkpoint.save(arquivo, original, opt)

    val retomado = modelo()
    val optRetomado = Checkpoint.load(arquivo, retomado, otimizador(retomado))

    // o mesmo passo seguinte, dos dois lados, tem que dar exatamente o mesmo
    // resultado -- e so da se `m`, `v` e `t` tiverem voltado junto
    umPasso(original, opt)
    umPasso(retomado, optRetomado)

    retomado.parameters.zip(original.parameters).foreach { (r, o) =>
      r.toArray shouldBe o.toArray
    }
  }

  it should "differ from a resume that dropped the moments" in {
    val original = modelo()
    val opt = umPasso(original, umPasso(original, otimizador(original)))

    val arquivo = arquivoTemporario()
    Checkpoint.save(arquivo, original, opt)

    val semMomentos = modelo()
    Checkpoint.load(arquivo, semMomentos, otimizador(semMomentos))

    // um otimizador zerado trata o proximo passo como se fosse o primeiro
    umPasso(original, opt)
    umPasso(semMomentos, otimizador(semMomentos))

    val iguais = semMomentos.parameters
      .zip(original.parameters)
      .forall((r, o) => r.toArray.sameElements(o.toArray))

    iguais shouldBe false
  }

  it should "reject a model whose configuration does not match" in {
    val original = modelo()
    val arquivo = arquivoTemporario()
    Checkpoint.save(arquivo, original, otimizador(original))

    val diferente = modelo(dim = 16)

    an[IllegalArgumentException] should be thrownBy
      Checkpoint.load(arquivo, diferente, otimizador(diferente))
  }

  it should "reject a file that is not a checkpoint" in {
    val arquivo = arquivoTemporario()
    val out = new DataOutputStream(new FileOutputStream(arquivo))
    out.writeInt(42)
    out.writeInt(42)
    out.close()

    val m = modelo()
    an[IllegalArgumentException] should be thrownBy Checkpoint.load(arquivo, m, otimizador(m))
  }

  it should "be saveable before any step, with no moments yet" in {
    val m = modelo()
    val arquivo = arquivoTemporario()

    Checkpoint.save(arquivo, m, otimizador(m))

    val outro = modelo()
    val opt = Checkpoint.load(arquivo, outro, otimizador(outro))

    opt.t shouldBe 0
    logits(outro) shouldBe logits(m)
  }

  "the config" should "travel inside the checkpoint" in {
    val m = modelo()
    val arquivo = arquivoTemporario()
    Checkpoint.save(arquivo, m, otimizador(m))

    Checkpoint.configOf(arquivo) shouldBe GPTConfig(vocabSize, dModel, 2, 1, contextLength)
  }

  "loadModel" should "rebuild the model from the file, with no config given" in {
    val original = modelo()
    val opt = umPasso(original, otimizador(original))

    val arquivo = arquivoTemporario()
    Checkpoint.save(arquivo, original, opt)

    val (restaurado, optRestaurado) = Checkpoint.loadModel(arquivo)

    restaurado.config shouldBe original.config
    optRestaurado.t shouldBe opt.t
    logits(restaurado) shouldBe logits(original)
  }

  it should "name both configurations when they do not match" in {
    val original = modelo()
    val arquivo = arquivoTemporario()
    Checkpoint.save(arquivo, original, otimizador(original))

    val diferente = modelo(dim = 16)
    val erro = intercept[IllegalArgumentException] {
      Checkpoint.load(arquivo, diferente, otimizador(diferente))
    }

    // as duas configuracoes aparecem na mensagem, e nao um "parametro 7 tem
    // tamanho errado" que nao diz qual modelo o arquivo esperava
    erro.getMessage should include("GPTConfig(6,8,2,1,4,4)")
    erro.getMessage should include("GPTConfig(6,16,2,1,4,4)")
  }
end CheckpointSpec
