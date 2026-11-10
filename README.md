# detector-de-cifras

Detector de acordes em Python: lê um mp3 e devolve a cifra por compasso, com andamento,
fórmula de compasso e seções.

A base é o método publicado por Mauch e Dixon (NNLS chroma, ISMIR 2010): espectro em
frequência logarítmica, afinação global, mínimos quadrados não negativos sobre um dicionário
de notas e um croma de baixo separado. Por cima disso entram uma grade de compasso e regras
de harmonia. A pesquisa que embasa cada decisão, com as fontes, está em
[`docs/pesquisa/`](docs/pesquisa/).

## Como rodar

```
pip install -r requirements.txt
python cifras/analyze.py uploads/minha-musica.mp3
```

A cifra é impressa na tela e salva em `cifras/saida/<música>.txt` e `.json`. Para a página
local, que lista as músicas e mostra a cifra, rode `python cifras/servir.py` e abra
<http://127.0.0.1:8765>.

Ponha os seus áudios em `uploads/`. Essa pasta, os mp3 e as saídas ficam fora do Git: músicas
comerciais nunca entram no repositório.

## Pesos dos modelos

Dois detectores opcionais usam redes treinadas de terceiros. Os pesos **não estão no
repositório** (tamanho e licença). Quando um peso falta, o código usa o caminho clássico no
lugar dele (`cifras/pulse.py`, `detect_pulse`; `cifras/analyze.py`, `overlay_btc`).

| Arquivo esperado | Modelo | Origem |
|---|---|---|
| `cifras/vendor/btc-chord/btc_model_large_voca_safe.pt` | BTC, acordes (ISMIR 2019) | [jayg996/BTC-ISMIR19](https://github.com/jayg996/BTC-ISMIR19) (MIT), empacotado em [puar-playground/btc-chord](https://huggingface.co/puar-playground/btc-chord) |
| `cifras/vendor/beat-this/final0.ckpt` | Beat This!, pulsação e tempo forte | [CPJKU/beat_this](https://github.com/CPJKU/beat_this) [confirmar licença] |

O BTC usa uma versão "safe" do checkpoint, que o PyTorch abre com `weights_only=True`. Um
script que baixa e prepara os dois arquivos é a fase 2 do plano.

## Avaliação

- `python cifras/benchmark_guitarset.py dados/guitarset --resumo` mede o detector no
  GuitarSet. Baixe o dataset para `dados/guitarset/` (a pasta `dados/` fica fora do Git).
- `python cifras/avaliar.py <pasta-de-anotacoes>` compara as cifras salvas com um gabarito.

## Verificação

```
python -m pytest
```

Os testes só precisam de `numpy`, `scipy` e `pytest` (`requirements-test.txt`).
