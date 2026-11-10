# detector-de-cifras — regras para agentes

Detector de acordes em Python: lê um mp3 e devolve a cifra por compasso, com andamento, fórmula de compasso e seções.

Vale para Claude Code, Codex e Gemini CLI. As regras gerais (trilhas, troca de agente,
honestidade) vêm do QG `agente-meriadoque`; aqui está o que um agente precisa saber
para trabalhar neste repositório sem ter o QG aberto.

## Ao começar

0. Primeira mensagem sem tarefa clara ("oi", "bom dia"): responda só com o cardápio abaixo.
   Com tarefa: diga em uma linha a trilha (rápida, média ou grande) e siga.

```
O que vamos fazer neste projeto?

1. Mudar algo          →  "quero ..."           (eu escolho a trilha e digo qual)
2. Continuar           →  "continuar"           (leio o HANDOFF.md)
3. Revisar trabalho    →  "revise o PR N"       (de preferência outro agente que não o autor)

Para terminar: "fechar sessão" (HANDOFF atualizado, commit, push e PR com o passo a passo do merge)
```

1. Leia o `HANDOFF.md`, se existir. Se o usuário disser só "continuar", retome dali:
   confira `git status` e siga o "Próximo comando".
2. Leia o plano da fase em andamento (link no `HANDOFF.md`).

## Durante o trabalho

- Uma branch e um PR **draft** por fase; nunca commit direto na `main`.
- Fase média ou grande: crie o `HANDOFF.md` a partir de `scripts/HANDOFF.modelo.md` e
  atualize no mesmo commit de cada passo.
- Antes de cada commit, rode a verificação abaixo e `sh scripts/handoff-check`.
- Descrição do PR com `Autor:` (o agente que escreveu); a revisão é feita por outro agente.

## Verificação

```
python -m ruff check . && python -m ruff format --check . && python -m pytest
```

## Dados e segredos

- Dados sempre fictícios no repositório; dados reais ficam em `dados/` (ignorada pelo Git).
- Segredos só no `.env` (ignorado); documente cada chave no `.env.example`.

## Este projeto

- Código em `cifras/` (módulos soltos; os testes em `cifras/tests/` põem a pasta no
  `sys.path`). `cifras/analyze.py` é a entrada; `cifras/servir.py` é a página local.
- **Nunca versione:** áudio (`uploads/`, `*.mp3`), pesos (`*.pt`, `*.ckpt`), `cifras/saida/`
  nem o GuitarSet (`dados/`, qualquer pasta `guitarset/` e `*.jams`). O `.gitignore` barra
  esses caminhos; confira `git status --ignored` antes do commit.
- **Laboratório de origem:** `D:\PROJETOS\moises-rea`, só local. Tem o app do Moises aberto
  para estudo e a ferramenta `rea`; nada de lá entra aqui além do que já foi importado.
  Copiar modelo, pesos ou código do Moises ou do Chordify está fora do escopo.
- **Trava de regressão:** a introdução já lida (Fmaj7, G/F, Em7, Am7 em Fá maior, ~74 BPM,
  4/4) não pode piorar; ver `docs/pesquisa/refinar-deteccao-de-acordes.md`.
- Plano: `brain/plans/detector-de-cifras/` no QG.
