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

- commit 3: correções manuais (linhas longas, `zip(strict=False)`, `raise ... from`, `_root`); AST igual à de antes em todos os 10 arquivos, descontadas só essas mudanças; `ruff check` limpo

## Falta
- commit 4: ruff no CI; PR draft e revisão do Codex

## Próximo comando
acrescentar ruff check e ruff format --check ao .github/workflows/ci.yml e ruff ao requirements-test.txt

## Bloqueios
nenhum

## Atualizado em
2026-11-10 17:45
