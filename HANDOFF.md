# Handoff

## Agente atual
Claude Code

## Branch
fase/1-estilo-ruff

## Trilha
grande

## Plano/fase
QG: brain/plans/detector-de-cifras/phase-1-estilo-ruff.md

## Feito
- fase 0 integrada (PR #1)
- commit 1: ruff configurado no pyproject (regras do kit, `cifras/vendor` fora) e `ruff format` em 28 arquivos; pytest verde (95 testes, 678 subtestes)

- commit 2: `ruff check --fix` (95 correções: imports não usados e ordem); os 3 nomes que outros arquivos pegavam pelo `analyze` (`MAX_AUDIO_SECONDS`, `PITCHES_SHARP`, `interval_bass`) agora vêm da origem (`constants`, `btc_align`)

## Falta
- commit 3: correções manuais (E501, B905, B904, B007)
- commit 4: ruff no CI; PR draft e revisão do Codex

## Próximo comando
python -m ruff check . --output-format concise  (33 avisos manuais: E501, B905, B904, B007)

## Bloqueios
nenhum

## Atualizado em
2026-11-10 17:15
