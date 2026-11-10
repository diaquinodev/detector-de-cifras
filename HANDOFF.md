# Handoff

## Agente atual
Claude Code

## Branch
fase/0-importar-codigo

## Trilha
grande

## Plano/fase
QG: brain/plans/detector-de-cifras/phase-0-repositorio-e-importacao.md

## Feito
- main com a estrutura do kit (camada comum) e .gitignore que barra áudio, pesos e saídas
- código de `cifras/` e pesquisa (`docs/pesquisa/`) copiados do laboratório `D:\PROJETOS\moises-rea`, sem mudar código
- pyproject (pytest), requirements-test.txt, CI com pytest, README e AGENTS.md do projeto
- `python -m pytest`: 95 testes e 676 subtestes verdes, igual ao laboratório
- revisões do Codex (Revise, 3x) tratadas: LICENSE (MIT), LICENSE-APACHE-2.0 (trecho do tensor2tensor) e PROCEDENCIA.md no vendor do BTC; fixture fictícia em `cifras/tests/fixtures/` para o `SavedChartsTest` conferir algo no CI; .gitignore barra GuitarSet e *.jams em qualquer pasta

## Falta
- nova revisão do Codex no commit atual do PR #1
- fase 1 (ruff) e fase 2 (pesos reproduzíveis)

## Próximo comando
usuário: merge do PR da fase 0; depois "continuar" para a fase 1 (estilo com ruff)

## Bloqueios
nenhum

## Atualizado em
2026-11-10 16:30
