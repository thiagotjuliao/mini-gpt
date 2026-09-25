/** Scala compiler flags, one `val` per flag, and the groups a build picks from.
  *
  * A build starts from `base` and adds what it wants:
  *
  * {{{
  * scalacOptions ++= CompilerFlags.base
  * scalacOptions ++= CompilerFlags.lint                              // more warnings
  * scalacOptions ++= CompilerFlags.base :+ CompilerFlags.explain     // one flag at a time
  * }}}
  *
  * This file is part of the project-templates contract: identical in every
  * project, so a flag means the same thing everywhere. A flag a project wants
  * that is missing here is added here — in project-templates first.
  */
object CompilerFlags:

  // --- Java platform --------------------------------------------------------

  /** The Java release the code targets: an LTS, and the one CI runs on. */
  val javaRelease: Int = 21

  /** Emit bytecode for `javaRelease` and check calls against its API, whatever
    * newer JDK happens to run the compiler.
    */
  val javaOutputVersion: String = s"-java-output-version:$javaRelease"

  // --- Warnings -------------------------------------------------------------

  /** Say where each deprecated API is used, not just that one is. */
  val deprecation: String = "-deprecation"

  /** Warn on language features that need an explicit import (implicit
    * conversions, postfix operators, ...).
    */
  val feature: String = "-feature"

  /** Say where a type test cannot be checked at runtime because of erasure. */
  val unchecked: String = "-unchecked"

  /** Warn on imports nothing uses. */
  val unusedImports: String = "-Wunused:imports"

  /** Warn on everything unused: imports, privates, locals, parameters. */
  val unusedAll: String = "-Wunused:all"

  /** Warn when a non-`Unit` value is silently discarded to fit a `Unit`. */
  val valueDiscard: String = "-Wvalue-discard"

  /** Warn on a statement whose non-`Unit` result is thrown away. */
  val nonUnitStatement: String = "-Wnonunit-statement"

  /** Check that no field is read before it is initialized. */
  val safeInit: String = "-Wsafe-init"

  /** Warn on patterns that can never match the value they are tested on. */
  val implausiblePatterns: String = "-Wimplausible-patterns"

  /** Every warning category the compiler knows. */
  val all: String = "-Wall"

  /** Turn every warning into an error. */
  val fatalWarnings: String = "-Werror"

  // --- Language and diagnostics ---------------------------------------------

  /** Opt into the next Scala 3 semantics, e.g. the `Matchable` checks. */
  val sourceFuture: String = "-source:future"

  /** Print the long explanation with each error, not only its first line. */
  val explain: String = "-explain"

  /** Allow `n` nested inline expansions instead of the default 32 — type-level
    * recursion (Peano arithmetic, tuple folds, derivation) needs more.
    */
  def maxInlines(n: Int): String = s"-Xmax-inlines:$n"

  // --- Groups ---------------------------------------------------------------

  /** What every project compiles with. */
  val base: Seq[String] =
    Seq(deprecation, feature, unchecked, unusedImports, javaOutputVersion)

  /** More warnings that catch real bugs without stopping the build. */
  val lint: Seq[String] =
    Seq(unusedAll, valueDiscard, nonUnitStatement, safeInit, implausiblePatterns)

  /** Everything, and fatal: for projects where a warning is a defect. */
  val strict: Seq[String] =
    Seq(sourceFuture, all, fatalWarnings)
end CompilerFlags
