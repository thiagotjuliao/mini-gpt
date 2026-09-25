ThisBuild / scalaVersion := "3.9.0"

// Flags do compilador: a base comum a todos os projetos (project-templates).
// Nomes e descricao de cada flag em project/CompilerFlags.scala.
ThisBuild / scalacOptions ++= CompilerFlags.base

lazy val root = rootProject
  .aggregate(scalagrad, gpt)
  .settings(
    name := "mini-gpt-scala"
  )

lazy val scalagrad = project
  .in(file("scalagrad"))
  .settings(
    name := "scalagrad",
    libraryDependencies ++= Seq(
      "org.scalatest" %% "scalatest" % "3.2.19" % Test
    )
  )

lazy val gpt = project
  .in(file("gpt"))
  .dependsOn(scalagrad)
  .settings(
    name := "gpt",
    libraryDependencies ++= Seq(
      "org.scalatest" %% "scalatest" % "3.2.19" % Test
    ),
    // Sem isto o `gpt.cli.chat` recebe EOF na primeira leitura e encerra na
    // hora: o sbt guarda o proprio stdin e nao o repassa a um `run` nao forkado.
    // `StdoutOutput` mantem a impressao progressiva token a token.
    Compile / run / fork := true,
    Compile / run / connectInput := true,
    outputStrategy := Some(StdoutOutput),
    // O fork roda a partir do diretorio do submodulo por padrao, e ai
    // `gpt/data/corpus.txt` viraria `gpt/gpt/data/corpus.txt`.
    Compile / run / baseDirectory := (ThisBuild / baseDirectory).value,
    // O corpus e portugues, e sem isto a JVM escreve os acentos em latin-1:
    // num terminal UTF-8 cada `ç` e `ã` do texto gerado vira lixo.
    Compile / run / javaOptions ++= Seq(
      "-Dfile.encoding=UTF-8",
      "-Dstdout.encoding=UTF-8",
      "-Dstderr.encoding=UTF-8"
    )
  )
