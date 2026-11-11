# Handoff

## Agente atual
Claude Code

## Branch
fase/5-voto-entre-voltas

## Trilha
grande

## Plano/fase
QG: brain/plans/detector-de-cifras-refino/phase-5-cifra-unica-por-letra.md (tarefa 109). Rumo confirmado pelo usuário em 11/11: (1) uma cifra por volta usando os ciclos lidos na cifra; (2) editor na página cujas correções viram gabarito

## Feito
- plano anterior (organização) concluído: PRs #1 a #4
- fase 0 (régua de erros) integrada: PR #5; fase 3 (frases) integrada: PR #6
- fase 4 (letras só pelo áudio) parada: a branch `fase/4-letras` guarda `letters.py` e as duas medições, sem PR; o croma da mixagem não separa trechos iguais de diferentes em música real
- fase 5: `cycles._settle_laps` no lugar de `_harmonize_outliers`. As voltas de um ciclo votam a cifra de cada posição; a volta antes de a voz entrar vale em dobro (`QUIET_LAP_WEIGHT`); um acorde de passagem lido em poucas voltas cede para a cifra do ciclo; a diferença que fica ganha `variacao_do_ciclo` no compasso
- fase 5: a régua (`diagnostico.form_invariants`) conta à parte a volta cujas diferenças estão todas marcadas (`voltas_com_variacao_marcada`)
- fase 5: 153 testes verdes no Windows; em "Flake" as cinco voltas do verso saem `Dm | F | Bb | F` e só o final (`C`) fica diferente

- fase 5, medição nas 7 músicas locais (mesma régua, antes e depois): das 107 voltas, as escritas igual ao ciclo foram de 30 para 41; as diferentes sem marca caíram de 77 para 34 (32%); 32 ficaram diferentes com marca. A meta do plano (até 10% diferentes) não foi atingida

## Falta
- GuitarSet rodando em segundo plano (`--resultados fase5.jsonl`, cerca de 70 min); comparar com `fase0.resumo.json`
- olhar as 34 voltas ainda diferentes e as 32 marcadas: quantas são leitura errada e quantas são variação de verdade (só gabarito responde; vem do editor)
- decidir `repeats.lock_repeated_phrases`: o plano manda apagar, mas ela compara o áudio e não a cifra; só sai se a medição mostrar que não faz falta
- PR draft, revisão cruzada pelo Codex

## Próximo comando
python cifras/benchmark_guitarset.py D:/PROJETOS/moises-rea/dados/guitarset --resultados fase5.jsonl --resumo, comparar majmin e classes com a fase 0 e abrir o PR draft

## Bloqueios
nenhum

## Atualizado em
2026-11-11 19:28
