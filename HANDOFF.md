# Handoff

## Agente atual
Claude Code

## Branch
fase/4-letras

## Trilha
grande

## Plano/fase
QG: brain/plans/detector-de-cifras-refino/phase-4-letras.md (tarefa 108). Ordem combinada com o usuário em 11/11: fases 3, 4 e 5 antes das fases 1 e 2

## Feito
- plano anterior (organização) concluído: PRs #1 a #4
- fase 0 (régua de erros) integrada: PR #5; fase 3 (frases) integrada: PR #6
- fase 4, em andamento: `cifras/letters.py` (letra por frase, blocos de frases que andam juntas, modulação) com 11 testes sintéticos verdes; `analyze.py` grava `croma_compassos` na cifra salva, para refazer as camadas de forma sem o áudio
- fase 4, medição nas 7 músicas locais (croma da mixagem inteira): janelas de 4 compassos com as mesmas fundamentais contra janelas diferentes separam bem em 3 músicas (AUC perto de 0,95), médio em 2 (0,85 a 0,88) e mal em 2 (0,79 e 0,80); os valores absolutos mudam de música para música, então o limiar fixo de 0,9 do `letters.py` não serve em música real

## Falta
- `letters.py` ainda não é chamado pelo `analyze.py` nem tem PR
- trocar o limiar fixo `SAME_LETTER` por um limiar por música (ou por agrupamento sem limiar) e medir nas 7 músicas
- testar croma só do acompanhamento (sem voz, pelo Demucs que o projeto já usa) como entrada da semelhança
- fraqueza da fase 3 achada aqui: numa forma AABA tocada duas vezes (A de 8 compassos sem repetição interna), `phrases.novelty` põe picos fora da fronteira real e as frases desalinham. Ideia de conserto: novidade pela mudança do padrão de repetição de cada compasso (em que distâncias ele volta), média de 4 compassos antes e depois. O teste que provava isso (`test_aaba_twice_joins_the_bridge_too`, forma esperada `A A B A A A B A`) foi tirado da suíte até o conserto

## Próximo comando
na branch fase/4-letras: consertar `phrases.novelty` (ver Falta), repor o teste AABA duas vezes, escolher o limiar por música e medir com `python <scratch>/sim_frase.py`; só então ligar `find_letters` no `analyze.py` e abrir o PR em rascunho

## Bloqueios
nenhum

## Atualizado em
2026-11-11 15:05
