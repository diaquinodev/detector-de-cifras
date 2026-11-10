# Procedência do código do BTC

- **Original:** [jayg996/BTC-ISMIR19](https://github.com/jayg996/BTC-ISMIR19), licença MIT,
  Copyright (c) 2019 Jonggwon Park. Texto integral em [`LICENSE`](LICENSE).
- **Empacotamento:** [puar-playground/btc-chord](https://huggingface.co/puar-playground/btc-chord)
  no Hugging Face. O cartão do modelo declara `license: mit`; o repositório não tem arquivo
  de licença próprio (conferido em 2026-11-10).
- **Trecho de terceiros:** `_gen_timing_signal` em `btc_src/transformer_modules.py` é adaptado
  do [tensor2tensor](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py)
  (Apache-2.0), como já estava no original; o arquivo é idêntico ao do empacotamento.

## Alterações locais (em relação ao empacotamento)

- `modeling_btc.py`: carrega `btc_model_large_voca_safe.pt` com `weights_only=True`; aceita
  `tuning_cents` em `predict`; devolve evidência por quadro; corrige o preenchimento quando
  o número de quadros já é múltiplo do tamanho da janela.
- `btc_src/features.py`: parâmetro `tuning` repassado ao `librosa.cqt`.
- `btc_src/btc_model.py` e `btc_src/transformer_modules.py`: sem alteração.

Os pesos (`*.pt`) não estão no repositório; ver o README da raiz.
