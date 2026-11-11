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

- fase 5, medição nas 7 músicas locais (mesma régua, antes e depois): das 107 voltas, as escritas igual ao ciclo foram de 30 para 41; as diferentes sem marca caíram de 77 para 38 (36%); 28 ficaram diferentes com marca (a marca só conta quando diz o acorde que a régua esperava). A meta do plano (até 10% diferentes) não foi atingida

- fase 5, GuitarSet (180 trechos, mesmo comando da fase 0): maior/menor 77,87% -> 78,18%; fundamental 81,31% -> 81,52%; com sétimas 65,81% -> 66,52%. Nenhum estilo caiu (voz e violão 91,10 -> 91,63; rock 90,04 -> 90,19; bossa 80,25 -> 80,67; jazz 65,16 -> 65,55; funk 58,09 -> 58,13). 36 trechos melhoraram e 11 pioraram (o pior perdeu 6,7 pontos); compassos ajustados por regra foram de 72 para 174

- revisão do Codex no PR #7 (Revise, 3 achados) atendida: a entrada da voz medida pelo perfil também pesa no voto; ciclo de 2 compassos fica de fora, dito no texto e testado; a marca só conta na régua quando diz o acorde esperado. Segunda revisão (Revise, 1 achado): o teste da voz pelo perfil agora só passa com uma única volta em dobro. 159 testes verdes

## Falta
- nada nesta fase além da revisão. Ficou fora, de propósito: `repeats.lock_repeated_phrases` continua (o plano mandava apagar; ela compara o áudio, não a cifra, e não foi medido se faz falta); a marca `variacao_do_ciclo` ainda não aparece na página; a meta de até 10% de voltas diferentes não foi atingida (36%)

## Próximo comando
nova revisão cruzada do PR #7 pelo Codex, no commit com as correções (autor: Claude Code)

## Bloqueios
nenhum

## Atualizado em
2026-11-11 20:38
