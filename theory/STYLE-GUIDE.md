# Guia de estilo — capítulos de `theory/`

**Leia este arquivo antes de escrever ou revisar qualquer documento de `theory/`.** Ele existe para que os 19 capítulos formem um livro, e não uma coleção de anotações com formatos diferentes.

O padrão nasceu da reescrita-piloto da Etapa 3 (2026-08-10), depois de uma auditoria que mediu os 9 documentos existentes. Os números-alvo deste guia não são opinião: são o que a auditoria mediu como diferença entre o material antigo e o piloto aprovado.

> **`theory/00-overview/` é o ponto de entrada do material.** Ele não é uma etapa: reúne o que é um modelo de linguagem, o mapa das 19 etapas, as convenções de notação e o glossário consolidado. Por isso a descrição dos quatro blocos destacados e o bloco de notação canônica moram **lá**, e os capítulos de etapa não os repetem — apontam de volta. Todo termo novo definido num capítulo deve ser acrescentado ao glossário do overview.

---

## 1. Para quem estamos escrevendo

> **O leitor-alvo.** Um calouro. Sabe programar. Sabe cálculo básico: derivada, derivada parcial, regra da cadeia. **Não** sabe nada de aprendizado de máquina, redes neurais ou LLMs. Nunca ouviu falar de transformer, atenção, logit ou embedding.

Esse leitor deve conseguir ler os 19 capítulos em ordem e sair sabendo construir um GPT do zero. Todo o resto deste guia decorre disso.

Duas consequências práticas.

**Nada de conhecimento presumido.** Se um termo de ML aparece, ele é definido ali, na primeira vez. Não vale "como todos sabem", nem adiar a definição para um capítulo posterior.

**Rigor não se negocia.** Escrever para iniciante não significa simplificar a matemática. Significa desempacotar a explicação. As derivações continuam completas; o que muda é o tamanho dos passos entre elas.

---

## 2. Esqueleto obrigatório do capítulo

Nesta ordem. Seções marcadas com ★ são obrigatórias.

```
# Etapa N — Título

## Antes de começar
  ### O que você vai construir          ★  tabela ou lista do que existirá ao final
  ### O que você precisa saber antes    ★  pré-requisitos, com ponteiro pra onde revisar
  ### Onde esta etapa se encaixa        ★  o "porquê", de preferência com uma analogia
  ### Objetivos de aprendizagem         ★  "ao terminar você deve conseguir..." (3 a 5 itens)

## Convenções de notação                    quando o capítulo introduz notação nova

## §1 ... §N                              ★  o conteúdo, numerado
## §último — Para onde isso leva           ★  ligação com a próxima etapa

## Cartão de referência                   ★  tabela-resumo + "as lições que se repetem"
```

**A numeração `§N` é contrato.** Comentários no código apontam para seções específicas (`// ver theory/03-elementary-operations/03-elementary-operations.md §4`). Ao reescrever um capítulo, mantenha a numeração das seções de conteúdo. Material novo de abertura entra como seção **não numerada**, antes do `§1`.

---

## 3. Os quatro blocos destacados

Use blockquote com rótulo em negrito. Sem emoji — a casa é sóbria.

### Definição

Toda vez que um termo aparece pela primeira vez.

```markdown
> **Definição — elemento a elemento.** Uma operação é elemento a elemento quando
> `C[i]` depende apenas de `A[i]` e `B[i]` — a mesma posição, nos dois operandos.
```

Regra dura: **se um termo é usado sem definição em algum ponto, é defeito.** Ver §6.

### Armadilha

Um erro que **realmente aconteceu** neste projeto. Fonte: `HISTORY.md`.

Estes são o material mais valioso do livro, porque marcam onde a intuição falha de verdade. Todo bloco de armadilha tem três partes, nesta ordem:

1. **O que foi feito de errado** — concreto, com o código ou a fórmula.
2. **Por que passou despercebido** — quase sempre "o forward continua funcionando".
3. **Lição geral** — a regra transferível, marcada em negrito.

```markdown
> **Armadilha.** A primeira versão do backward do `exp` usou a entrada em vez da saída.
>
> Por que é perigoso: o forward continua perfeito, os testes de forward passam, nada
> trava. Só os gradientes saem errados — e gradiente errado não avisa.
>
> **Lição geral:** sempre pergunte de quais valores a derivada depende.
```

Não invente armadilhas. Se não houve um bug real naquele ponto, não force o bloco.

### Confira você mesmo

Exercício curto no meio do texto, com resposta recolhida. De 2 a 6 por capítulo, proporcional ao tamanho — sempre logo depois da seção que ele testa.

Prefira poucos e bons a muitos e forçados. Um exercício só vale a pena se errar a resposta revelar um mal-entendido real. Pergunta cuja resposta é "está escrito duas linhas acima" não ensina nada.

```markdown
> **Confira você mesmo.** [pergunta]
>
> <details><summary>Resposta</summary>
>
> [resposta, explicando o porquê — não só o resultado]
> </details>
```

Estes são **separados** do `exercises.html`, que continua sendo a avaliação final do capítulo.

### Guia visual

Ponteiro para o `.html` da pasta, colocado no início da seção que ele ilustra — não todos amontoados no topo do arquivo.

```markdown
> **Guia visual.** Alinhamento de formatos e a view de stride 0: [`broadcasting.html`](broadcasting.html).
```

---

## 4. Prosa: metas mensuráveis

| Métrica | Meta | Original (média dos 9) | Reescritos |
|---|---|---|---|
| palavras por frase | **11 a 20** | 29,6 | 11,7 a 12,7 |
| p90 (10% das frases passam de) | **≤ 25** | 46 a 63 | 19 a 22 |
| parênteses / 1.000 palavras | **≤ 20** | 52,1 | 9 a 19 |

O piso de 11 existe só para evitar prosa entrecortada demais. O que realmente importa é o teto: frase longa é o que faz o leitor reler.

Como atingir isso, em quatro regras:

1. **Um período, uma ideia.** Se a frase tem ponto-e-vírgula ou dois travessões, quase sempre eram duas ou três frases.
2. **Parêntese só para aparte dispensável.** Se o conteúdo importa, vira frase própria. Se não importa, corta.
3. **Varie o ritmo.** A meta é média, não uniformidade. Uma frase curta depois de uma longa é o que faz a conclusão aterrissar.
4. **Não mexa no vocabulário nem no rigor** para atingir a meta. O alvo é a sintaxe, nunca o conteúdo.

Medir é obrigatório antes de dar um capítulo por pronto. Script no §9.

---

## 5. Exemplos numéricos

**Toda subseção de conteúdo fecha com um exemplo numérico verificado. Sem exceção** — inclusive as que parecem simples demais para merecer (`clamp`, `reshape`, `neg`).

Quatro regras.

**Valores distintos entre si.** Nunca use um gradiente uniforme (`dC` todo `1`). Valores iguais escondem erros de indexação, porque qualquer posição trocada dá o mesmo resultado. Prefira `dC = [1, 10, 100]` a `dC = [1, 1, 1]`.

**Verificação independente quando possível.** Depois de aplicar a fórmula, confirme o mesmo número por outro caminho: uma derivação alternativa, ou conferindo posição a posição. Exemplos do que funciona bem — Etapa 2 confere o grafo contra derivação simbólica; Etapa 6 confere a fórmula simplificada contra o Jacobiano completo.

**Reuse os mesmos valores dentro do capítulo.** Quando o mesmo `A` atravessa várias seções, o leitor consegue comparar resultados entre elas. A Etapa 6 faz isso bem, com `x = [1,2,3]` do começo ao fim.

**Confira as contas de verdade.** Rode um script. A auditoria encontrou um erro de fator 1000 num documento que ensina precisão numérica.

---

## 6. Terminologia

**Definir na primeira aparição, sempre.** Não vale usar um termo no capítulo 5 e defini-lo no 8. A auditoria encontrou quatro casos: `neurônio` (usado na Etapa 5, definido na 8), `logit`, `transformer` e `atenção` (usados e nunca definidos).

Antes de dar um capítulo por pronto, procure os termos de ML que ele usa e confirme que cada um foi definido — ali ou num capítulo **anterior**.

**Ponteiro para frente é permitido; dívida não.** Escrever "você vai construir isso na Etapa 16" é bom, porque orienta. Escrever `∂L/∂A` sem nunca ter dito o que é `L` é dívida.

---

## 7. Notação canônica

Idêntica em todos os capítulos. Quem introduzir notação nova acrescenta aqui.

| Símbolo | Significa |
|---|---|
| `A`, `B`, `C` | tensores no forward |
| `dA` | `∂L/∂A` — gradiente da **perda** em relação a `A` |
| `dC` | o gradiente que chega de cima, vindo da operação seguinte |
| `∂C/∂A` | a derivada **local** da operação, isolada do grafo |
| `L` | a perda: **um único número**, que mede o quão errado o modelo está |
| `N(μ, σ²)` | distribuição normal de média `μ` e **variância** `σ²` |

Três armadilhas de notação que o guia resolve de uma vez:

- `dA` **não** é "a derivada de `A`". É a derivada de `L` em relação a `A`.
- O gradiente **sempre** tem o formato do tensor a que se refere, nunca o da saída da operação.
- Não misture estilos no mesmo documento. Escolha `dA` e fique nele — não alterne com `a.grad`, `dy` ou `A.gradient` na mesma explicação.

---

## 8. Idioma, arquivos e referências cruzadas

**Idioma.** Prosa em português. Identificadores, nomes de arquivo e de diretório em inglês. Isso vale para o conteúdo dos documentos e para os títulos — só os *nomes* de arquivo são em inglês, o texto dentro deles é português.

**Arquivos por pasta de etapa:**

```
theory/NN-nome-em-ingles/
├── NN-nome-em-ingles.md      ★  o capítulo
├── nome-do-tema.html            guia visual, quando o assunto for visual
└── exercises.html            ★  quiz de múltipla escolha
```

**`exercises.html`:** baseline de 10 questões, mais em capítulos densos (a Etapa 3 tem 20). Níveis fácil / médio / difícil / desafio, feedback imediato, placar. Mesma paleta *blueprint/blueline* dos demais. Sem dependências externas. Gerado junto com o capítulo, sem esperar pedido.

**Referências cruzadas.** Aponte para a seção específica (`Etapa 3 §4`), nunca para o documento inteiro. Não mande o leitor para o `HISTORY.md` como material de estudo — ele é registro cronológico, não didático. Se um bug do `HISTORY.md` é instrutivo, traga o conteúdo para dentro de um bloco de Armadilha.

---

## 9. Checklist antes de dar um capítulo por pronto

Estrutura:

- [ ] Todas as seções ★ do §2 presentes, na ordem
- [ ] Numeração `§N` preservada, se for reescrita de um capítulo existente
- [ ] "Para onde isso leva" aponta corretamente para a etapa seguinte
- [ ] Cartão de referência com tabela-resumo e as lições que se repetem

Conteúdo:

- [ ] Toda subseção fecha com exemplo numérico verificado
- [ ] Nenhum exemplo usa gradiente uniforme
- [ ] Pelo menos uma verificação independente no capítulo
- [ ] Todas as contas conferidas por script
- [ ] Todo termo de ML definido na primeira aparição
- [ ] Notação bate com o §7 deste guia
- [ ] Blocos de Armadilha correspondem a bugs reais do `HISTORY.md`
- [ ] De 2 a 6 blocos "Confira você mesmo", proporcional ao tamanho

Prosa:

- [ ] Média entre 13 e 20 palavras por frase
- [ ] p90 ≤ 25 palavras
- [ ] Parênteses ≤ 20 por 1.000 palavras

Sincronia com o código:

- [ ] Assinaturas de função conferidas contra o código real
- [ ] Nomes de parâmetro conferidos (`keepDim`, não `keepdim`)
- [ ] Nenhuma seção descreve decisão em aberto que já foi tomada

Companheiros:

- [ ] `exercises.html` criado ou atualizado
- [ ] Guia visual criado, se o assunto for visual
- [ ] Todos os `.html` validados: nada fora do `viewBox`, sem erro de console

### Script de medição

Duas sutilezas importam para o medidor não mentir. Blocos de código viram **separador** de frase, não vazio — senão o texto antes e depois deles é colado numa frase falsa e gigante. E linha terminada em dois-pontos encerra a frase, pelo mesmo motivo.

```bash
cd theory && python3 -c "
import re, statistics, glob
for f in sorted(glob.glob('*/*.md')):
    if f.startswith('_'): continue
    t = open(f, encoding='utf-8').read()
    t = re.sub(r'\`\`\`.*?\`\`\`', '.\n', t, flags=re.S)   # separador, nunca vazio
    t = '\n'.join(l for l in t.split('\n') if not l.strip().startswith(('|','#','>')))
    t = re.sub(r':\s*\n', '.\n', t)
    s = [x for x in re.split(r'(?<=[.!?])\s+', t) if len(x.split()) > 3]
    l = [len(x.split()) for x in s]
    if not l: continue
    w = sum(l); p90 = sorted(l)[int(len(l)*0.9)]; m = statistics.mean(l); par = t.count('(')/w*1000
    flag = 'OK ' if (11 <= m <= 20 and p90 <= 25 and par <= 20) else 'REV'
    print(f'{flag} {f:<52} media={m:5.1f}  p90={p90:3}  par/1k={par:5.1f}')
"
```

Um `REV` isolado não é veredito automático. Confira as frases apontadas antes de reescrever — o medidor erra para mais quando o capítulo tem muitos blocos de código curtos.

---

## 10. O que não mudar

A auditoria de 2026-08-10 identificou o que o material já fazia bem. Preserve.

**Derivar em vez de postular.** O backward do `matmul` não é entregue pronto: sai da definição, em quatro passos. O mesmo vale para a simplificação `O(N²) → O(N)` do softmax, para a GELU e para o embedding. É o que distingue este material de um tutorial comum.

**Verificações independentes.** Confirmar o mesmo resultado por dois caminhos diferentes é o hábito mais valioso do material.

**Ligação explícita entre etapas.** Toda etapa fecha apontando para a próxima, e fecha ganchos que etapas anteriores deixaram abertos.

**Honestidade sobre decisões de projeto.** Quando algo foi decidido de um jeito em vez de outro, o documento explica o trade-off — divisão por zero sem proteção, `matmul` limitado a 2D e 3D, tokenização por caractere. Isso ensina engenharia, não só matemática.
