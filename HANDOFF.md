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
- linha de base de 11/11 no GuitarSet (180 trechos): root 81,31%, majmin 77,87%; sem_acorde_indevido 3,12% do tempo (funk 9,89%, jazz 5,10%); certo + qualidade = root
- invariantes nas 9 cifras de `cifras/saida` com o código atual: 24 de 81 seções fora de frase, 6 curtas, 23 compassos "N" no meio, 80 de 111 voltas com cifra diferente
- resultado completo e metas gravados no plano do QG
- revisão do Codex no commit db68ad3 (Revise, 5 achados) tratada: retomada mede de novo trecho sem classes, sobreposição de intervalos (inclusive um trecho dentro de outro, achado da segunda rodada no commit 1d753df), lista de graus com terça, silêncio escrito de outras formas, pontas só isentas para introdução e outro

## Falta
- nada nesta fase além da revisão cruzada e do merge

## Próximo comando
revisão cruzada do PR #5 por um agente diferente do autor (Codex); depois do merge, fase 1 do plano (QG: brain/plans/detector-de-cifras-refino/phase-1-sem-acorde-indevido.md), branch fase/1-sem-acorde-indevido a partir da main

## Bloqueios
nenhum

## Atualizado em
2026-11-11 08:50
