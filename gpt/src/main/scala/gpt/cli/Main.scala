package gpt.cli

import gpt.data.{BatchSampler, Tokenizer}
import gpt.generate.{Generator, Greedy, SamplingStrategy, Temperature, TopK}
import gpt.model.{GPT, GPTConfig}
import gpt.optim.AdamW
import gpt.train.{Checkpoint, Trainer, TrainingConfig}

import java.io.File
import scala.io.{Source, StdIn}
import scala.util.{Random, Using}

/** Ponto de entrada do projeto: treinar um modelo no corpus e conversar com ele.
  *
  * ```
  * sbt "gpt/runMain gpt.cli.train"
  * sbt "gpt/runMain gpt.cli.chat"
  * ```
  *
  * O vocabulário não é serializado no checkpoint: ele é derivado do corpus por
  * `corpus.distinct.sorted`, que é determinístico. Por isso o `chat` reconstrói
  * o mesmo tokenizador lendo o mesmo arquivo, e trocar de corpus exige treinar
  * de novo.
  *
  * Os defaults são calibrados para terminar em tempo humano nesta implementação,
  * que é Scala puro sem BLAS: medido em 2026-09-01, cerca de 0,5 s por passo com
  * `dModel = 64`, 2 camadas, contexto 32 e lote 8. Dobrar o `dModel` custa perto
  * de 4x, porque o termo dominante do MLP e das projeções é `dModel²`.
  */
private def readCorpus(path: String): String =
  Using.resource(Source.fromFile(new File(path), "UTF-8"))(_.mkString)

private def intArg(args: Seq[String], name: String, default: Int): Int =
  args.sliding(2).collectFirst { case Seq(`name`, v) => v.toInt }.getOrElse(default)

private def doubleArg(args: Seq[String], name: String, default: Double): Double =
  args.sliding(2).collectFirst { case Seq(`name`, v) => v.toDouble }.getOrElse(default)

private def stringArg(args: Seq[String], name: String, default: String): String =
  args.sliding(2).collectFirst { case Seq(`name`, v) => v }.getOrElse(default)

@main def train(args: String*): Unit = {
  val corpusPath = stringArg(args, "--corpus", "gpt/data/corpus.txt")
  val checkpointPath = stringArg(args, "--checkpoint", "gpt/data/model.bin")
  val steps = intArg(args, "--steps", 3000)
  val batchSize = intArg(args, "--batch", 8)
  val contextLength = intArg(args, "--context", 32)
  val dModel = intArg(args, "--dmodel", 64)
  val nHeads = intArg(args, "--heads", 4)
  val nLayers = intArg(args, "--layers", 2)
  val lrMax = doubleArg(args, "--lr", 1e-3)
  val seed = intArg(args, "--seed", 42)

  val text = readCorpus(corpusPath)
  val tokenizer = Tokenizer.charLevel(text)
  val tokens = tokenizer.encode(text)
  val rng = new Random(seed)

  val config = GPTConfig(tokenizer.vocabSize, dModel, nHeads, nLayers, contextLength)
  val model = new GPT(config, rng)

  println(s"corpus     $corpusPath — ${tokens.length} tokens, vocabulário ${tokenizer.vocabSize}")
  println(s"modelo     $config")
  println(s"parâmetros ${model.parameters.map(_.size).sum}")
  println(f"perda inicial esperada (chute uniforme): ${Math.log(tokenizer.vocabSize)}%.4f")
  println()

  val (trainSampler, validationSampler) =
    BatchSampler.split(tokens, contextLength, validationFraction = 0.1, randomizer = rng)

  val checkpointFile = new File(checkpointPath)
  val trainingConfig = TrainingConfig(
    steps = steps,
    batchSize = batchSize,
    lrMax = lrMax,
    lrMin = lrMax / 10,
    warmupSteps = Math.max(1, steps / 20),
    logInterval = Math.max(1, steps / 100),
    evalInterval = Math.max(1, steps / 10),
    evalBatches = 5,
    checkpointInterval = Math.max(1, steps / 10),
    checkpointFile = Some(checkpointFile)
  )

  val startedAt = System.currentTimeMillis()
  val result = Trainer.train(
    model,
    new AdamW(model.parameters, lr = lrMax),
    trainSampler,
    trainingConfig,
    Some(validationSampler)
  )
  val elapsed = (System.currentTimeMillis() - startedAt) / 1000.0

  Checkpoint.save(checkpointFile, model, result.optimizer)

  println()
  println(f"treino terminado em $elapsed%.1f s (${elapsed / steps * 1000}%.0f ms por passo)")
  println(f"perda: ${result.losses.head}%.4f no primeiro passo, ${result.losses.last}%.4f no último")
  println(s"checkpoint salvo em $checkpointPath")
  println()
  println("amostra do que o modelo aprendeu:")
  println("-" * 60)

  val generator = new Generator(model, rng)
  print(generator.generate("Capitu", 300, tokenizer, TopK(20, 0.8)))
  println()
  println("-" * 60)
  println(s"""para conversar: sbt "gpt/runMain gpt.cli.chat"""")
}

@main def chat(args: String*): Unit = {
  val corpusPath = stringArg(args, "--corpus", "gpt/data/corpus.txt")
  val checkpointPath = stringArg(args, "--checkpoint", "gpt/data/model.bin")
  val seed = intArg(args, "--seed", 7)

  val checkpointFile = new File(checkpointPath)
  if !checkpointFile.exists() then
    println(s"Não achei o checkpoint em $checkpointPath.")
    println(s"""Treine primeiro: sbt "gpt/runMain gpt.cli.train"""")
  else
    val tokenizer = Tokenizer.charLevel(readCorpus(corpusPath))
    val rng = new Random(seed)
    val (model, _) = Checkpoint.loadModel(checkpointFile, rng)

    require(
      model.vocabSize == tokenizer.vocabSize,
      s"O checkpoint foi treinado com vocabulário de ${model.vocabSize} símbolos, " +
        s"e este corpus tem ${tokenizer.vocabSize}. Use o mesmo corpus do treino."
    )

    println(s"modelo ${model.config}, contexto de ${model.contextLength} caracteres")
    println("escreva um começo de frase e o modelo continua. `:ajuda` lista os comandos.")
    println()

    chatLoop(new Generator(model, rng), tokenizer, ChatState())
}

/** O que atravessa a conversa: os ajustes de amostragem e o texto já escrito.
  *
  * O histórico existe para o modelo continuar de onde parou, inclusive do que
  * ele mesmo gerou. Como o contexto é curto, o que passa de `contextLength` é
  * esquecido de qualquer forma — a janela deslizante da Etapa 19 §2 cuida disso.
  */
private case class ChatState(
    temperature: Double = 0.8,
    topK: Int = 20,
    maxNewTokens: Int = 200,
    history: Vector[Int] = Vector.empty
) {
  def strategy: SamplingStrategy =
    if temperature <= 0.01 then Greedy
    else if topK > 0 then TopK(topK, temperature)
    else Temperature(temperature)

  def summary: String = {
    val sampling =
      if temperature <= 0.01 then "greedy"
      else if topK > 0 then f"top-k $topK, T = $temperature%.2f"
      else f"T = $temperature%.2f"

    s"[$sampling, $maxNewTokens tokens]"
  }
}

private val helpText =
  """comandos:
    |  :temp <n>    temperatura; 0 vira greedy, 0.8 é o padrão, acima de 1.2 delira
    |  :topk <n>    quantos candidatos considerar; 0 desliga o corte
    |  :tokens <n>  quantos caracteres gerar por resposta
    |  :limpar      esquece a conversa e recomeça do zero
    |  :ajuda       esta lista
    |  :sair        encerra""".stripMargin

@annotation.tailrec
private def chatLoop(generator: Generator, tokenizer: Tokenizer, state: ChatState): Unit = {
  print(s"${state.summary} > ")
  Console.out.flush()

  Option(StdIn.readLine()).map(_.trim) match {
    case None | Some(":sair") | Some(":quit") =>
      println("até mais.")

    case Some("") =>
      chatLoop(generator, tokenizer, state)

    case Some(":ajuda") =>
      println(helpText)
      chatLoop(generator, tokenizer, state)

    case Some(command) if command.startsWith(":") =>
      chatLoop(generator, tokenizer, applyCommand(command, state))

    case Some(text) =>
      chatLoop(generator, tokenizer, respond(generator, tokenizer, state, text))
  }
}

private def applyCommand(command: String, state: ChatState): ChatState = {
  val parts = command.split("\\s+")
  val value = parts.lift(1)

  (parts.head, value) match {
    case (":temp", Some(v)) =>
      v.toDoubleOption.fold(invalidValue(v, state))(t => state.copy(temperature = t))

    case (":topk", Some(v)) =>
      v.toIntOption.fold(invalidValue(v, state))(k => state.copy(topK = k))

    case (":tokens", Some(v)) =>
      v.toIntOption.fold(invalidValue(v, state))(n => state.copy(maxNewTokens = n))

    case (":limpar", _) =>
      println("(esqueci a conversa)")
      state.copy(history = Vector.empty)

    case _ =>
      println(s"não conheço `$command`.")
      println(helpText)
      state
  }
}

private def invalidValue(value: String, state: ChatState): ChatState = {
  println(s"`$value` não é um número.")
  state
}

private def respond(
    generator: Generator,
    tokenizer: Tokenizer,
    state: ChatState,
    text: String
): ChatState = {
  // O tokenizador é de caractere e lança em símbolo desconhecido. Filtrar aqui
  // é mais gentil que recusar a linha inteira por causa de um emoji.
  val known = text.filter(tokenizer.alphabet.contains)
  val dropped = text.length - known.length

  if dropped > 0 then
    println(s"(ignorei $dropped caractere(s) fora do vocabulário do corpus)")

  if known.isEmpty then
    println("(preciso de pelo menos um caractere que o corpus contenha)")
    state
  else
    val input = state.history ++ tokenizer.encode(known).toVector
    print(known)
    Console.out.flush()

    val output = generator.generate(
      input.toArray,
      state.maxNewTokens,
      state.strategy,
      token => {
        print(tokenizer.decode(Array(token)))
        Console.out.flush()
      }
    )

    println()
    println()

    state.copy(history = output.toVector)
}
