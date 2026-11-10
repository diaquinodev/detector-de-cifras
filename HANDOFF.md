# Handoff

## Agente atual
Claude Code

## Branch
fase/2-pesos-reproduziveis

## Trilha
grande

## Plano/fase
QG: brain/plans/detector-de-cifras/phase-2-pesos-reproduziveis.md

## Feito
- fases 0 e 1 integradas (PRs #1 e #2)
- `scripts/baixar_modelos.py`: baixa Beat This! e BTC com SHA-256 fixo e grava o BTC "safe" sem usar o modo inseguro do PyTorch; 8 testes sem rede em `cifras/tests/test_baixar_modelos.py`
- prova em cópia limpa: download real, segunda execução sem baixar, `analyze.py` em "My Girl" usando os dois modelos
- código da fase 0 e código atual geram cifra e JSON idênticos para "My Girl" (o ruff não mudou o resultado)
- licenças com fonte: Beat This! (README do projeto: código e pesos MIT) e BTC (repositório original MIT)
- CI instala torch de CPU para rodar os testes da conversão

## Falta
- PR draft da fase 2 e revisão do Codex

## Próximo comando
abrir o PR draft da fase 2 e pedir revisão cruzada ao Codex

## Bloqueios
nenhum

## Atualizado em
2026-11-10 20:30
