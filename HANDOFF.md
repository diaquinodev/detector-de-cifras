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
- fase 4, segunda medição: corte por música (Otsu) sobre pares de frases, com croma cru, centrado e a soma dos dois. Em todas as 7 músicas o corte aceita dezenas de pares que a cifra diz serem diferentes (ex.: Sutilmente 13 certos e 72 a 107 falsos; Flake 24 certos e 60 falsos). Conclusão: o croma da mixagem não basta para dar letras em música real. A referência usada foi a cifra atual, que tem erros

## Falta
- decisão do usuário sobre o novo rumo (ver Próximo comando)
- `letters.py` fica na branch, sem PR e sem ser chamado: serve nos casos sintéticos, não em música real
- fraqueza da fase 3 (`phrases.novelty` desalinha frases numa forma AABA tocada duas vezes): conserto e teste ainda por fazer

## Próximo comando
esperar o usuário confirmar o novo rumo proposto em 11/11: (A) uma cifra por volta usando os ciclos que o detector já acha pela cifra (`cycles.py`), com voto entre as voltas; (B) editor na página cujas correções viram gabarito (`importar_anotacao.py` e `avaliar.py` já leem anotações). O usuário já respondeu que quer as correções usadas para melhorar o detector. Letras só pelo áudio ficam paradas até existir gabarito

## Bloqueios
nenhum

## Atualizado em
2026-11-11 15:46
