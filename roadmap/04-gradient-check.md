## Etapa 4 — Verificação Numérica de Gradientes (Gradient Check)

### Por que existe

É impossível verificar por inspeção se um backward está correto para operações não-triviais como matmul ou softmax. Um erro de sinal em um gradiente pode parecer que treina, mas converge para um mínimo errado ou diverge lentamente. O gradient check é o teste unitário dos gradientes: usa a definição de derivada para calcular numericamente o gradiente e compara com o gradiente analítico que implementamos.

### O que implementar

**Gradiente numérico (diferença central)**

Para cada parâmetro `p[i]` de um tensor, perturbamos ligeiramente seu valor em `+ε` e `-ε`, computamos o forward pass nos dois casos, e estimamos a derivada:

```
grad_numérico[i] = (f(p[i] + ε) - f(p[i] - ε)) / (2ε)
```

Usando `ε = 1e-5`. A diferença central é mais precisa que a diferença simples porque cancela termos de segunda ordem.

**Erro relativo**

A comparação entre gradiente numérico e analítico usa erro relativo:

```
erro = |grad_analítico - grad_numérico| / max(|grad_analítico|, |grad_numérico|, 1e-8)
```

O denominador previne divisão por zero quando ambos são próximos de zero. Um erro relativo abaixo de `1e-5` é considerado correto. Entre `1e-3` e `1e-5` é suspeito. Acima de `1e-3` indica bug.

**Como usar**

Criar uma função genérica `gradCheck(f: Tensor => Tensor, input: Tensor)` que:
1. Faz o forward e backward analítico
2. Para cada elemento do input, faz as duas perturbações numéricas
3. Compara e reporta o erro relativo máximo

Deve ser rodado para cada nova operação implementada, com inputs aleatórios de formas variadas. Especialmente importante para: `matmul`, `softmax`, `layerNorm`, `attention`.

### Por que essa etapa importa

Muitos bugs em backpropagation são silenciosos — o código roda, a perda cai (ou não), mas os gradientes estão errados. Gradient check é a única forma confiável de garantir que a matemática está correta antes de subir para camadas mais complexas. Descobrir um bug no backward de `matmul` depois de ter implementado o transformer completo é muito mais caro do que descobrir agora.
