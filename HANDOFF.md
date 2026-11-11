# Handoff

## Agente atual
Claude Code

## Branch
fase/0-regua-de-erros

## Trilha
grande

## Plano/fase
QG: brain/plans/detector-de-cifras-refino/phase-0-regua-de-erros.md (tarefa 104)

## Feito
- plano anterior (organização) concluído: PRs #1 a #4
- fase 0 do refino: `cifras/diagnostico.py` (classes de erro dos acordes e invariantes de forma) com 17 testes sintéticos
- fase 0 do refino: `benchmark_guitarset.py` grava as classes de erro de cada trecho e aceita `--resultados <arquivo>`
- invariantes medidas nas 9 cifras antigas do laboratório: 18 de 83 seções fora de frase, 5 seções curtas, 81 de 114 voltas com cifra diferente do ciclo

## Falta
- rodar o benchmark do GuitarSet de novo (180 trechos, perto de 2 horas) com `--resultados fase0.jsonl` e gravar a linha de base das classes de erro
- medir as invariantes nas cifras geradas pelo código atual (as do laboratório são de 08 a 10/11)
- escrever o Resultado e as metas no plano do QG; revisão do Codex; tirar o PR do rascunho

## Próximo comando
python cifras/benchmark_guitarset.py D:/PROJETOS/moises-rea/dados/guitarset --resultados fase0.jsonl (continua de onde parou); depois python cifras/diagnostico.py cifras/saida

## Bloqueios
nenhum

## Atualizado em
2026-11-11 07:28
