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

- commit 4: `ruff check` e `ruff format --check` no CI, `ruff==0.17.3` fixado no requirements-test.txt; verificação nova no AGENTS.md e no README

- PR #2 aberto; 1ª revisão do Codex (Revise) tratada: `analyze` volta a expor `MAX_AUDIO_SECONDS`, `PITCHES_SHARP` e `interval_bass`, com teste que trava a interface

## Falta
- nova revisão do Codex no commit atual do PR #2

## Próximo comando
revisão cruzada do PR #2 pelo Codex; com Accept, marcar Ready for review

## Bloqueios
nenhum

## Atualizado em
2026-11-10 18:20
