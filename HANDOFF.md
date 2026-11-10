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

## Falta
- commit 2: `ruff check --fix` (imports não usados e ordenados), conferindo reexportações
- commit 3: correções manuais (E501, B905, B904, B007)
- commit 4: ruff no CI; PR draft e revisão do Codex

## Próximo comando
python -m ruff check . --select F401 --output-format concise  (conferir cada import antes do --fix)

## Bloqueios
nenhum

## Atualizado em
2026-11-10 17:00
