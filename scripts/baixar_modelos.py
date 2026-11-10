"""Baixa e prepara os pesos dos dois modelos opcionais.

    python scripts/baixar_modelos.py

- Beat This! (pulsação e tempo forte): baixa ``final0.ckpt`` e confere o SHA-256.
- BTC (acordes): baixa o checkpoint original, confere o SHA-256 e grava a versão
  "safe", que o PyTorch abre com ``weights_only=True``.

O checkpoint original do BTC guarda ``mean`` e ``std`` como escalares do numpy, que o
modo seguro do PyTorch recusa. A conversão abre o arquivo ainda no modo seguro,
liberando só esses tipos do numpy, e regrava os dois números como tensores. O arquivo
nunca é aberto com ``weights_only=False`` (que permitiria executar código do arquivo).

Rodar de novo é seguro: o que já está no lugar e confere é mantido.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
VENDOR = RAIZ / "cifras" / "vendor"

BEAT_THIS_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt"
BEAT_THIS_SHA256 = "8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331"
BEAT_THIS_DESTINO = VENDOR / "beat-this" / "final0.ckpt"

# Repositório original (MIT); o empacotamento no Hugging Face traz o mesmo arquivo.
BTC_URL = (
    "https://raw.githubusercontent.com/jayg996/BTC-ISMIR19/master/test/btc_model_large_voca.pt"
)
BTC_SHA256 = "1673d23f8f9a55ae7f9e8b80a51da616debb22675b8d8b67ea6ce0ef37b0ab51"
BTC_DESTINO = VENDOR / "btc-chord" / "btc_model_large_voca_safe.pt"
# Resumo do conteúdo (tensores, mean e std) do arquivo "safe" esperado; ver content_digest.
BTC_CONTEUDO_SHA256 = "e84c4886a9b3731ddde2457c78596ab45a670f2343dc2e87f06a98f3f6659fb4"

BLOCO = 1 << 20
TIMEOUT_S = 60  # sem resposta do servidor por esse tempo, desiste em vez de travar


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(BLOCO):
            digest.update(block)
    return digest.hexdigest()


def _temporario(destino: Path) -> Path:
    """Arquivo exclusivo ao lado do destino: duas execuções ao mesmo tempo não se pisam."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    handle, nome = tempfile.mkstemp(
        dir=destino.parent, prefix=destino.name + ".", suffix=".parcial"
    )
    os.close(handle)
    return Path(nome)


def download(url: str, destino: Path, sha256: str) -> bool:
    """Baixa ``url`` para ``destino`` e confere o SHA-256. Devolve False se já estava certo.

    Escreve num arquivo ao lado e só renomeia depois de conferir: um download cortado ou
    adulterado nunca fica com o nome final.
    """
    if destino.is_file() and sha256_of(destino) == sha256:
        return False
    parcial = _temporario(destino)
    try:
        with (
            urllib.request.urlopen(url, timeout=TIMEOUT_S) as resposta,
            parcial.open("wb") as saida,
        ):
            while block := resposta.read(BLOCO):
                saida.write(block)
        obtido = sha256_of(parcial)
        if obtido != sha256:
            raise SystemExit(
                f"{destino.name}: o SHA-256 baixado ({obtido}) não é o esperado ({sha256}). "
                "O arquivo foi descartado."
            )
        os.replace(parcial, destino)
    finally:
        parcial.unlink(missing_ok=True)
    return True


def _numpy_permitidos() -> list:
    """Os únicos tipos fora do PyTorch que a conversão aceita: escalar float do numpy."""
    import numpy as np

    core = getattr(np, "_core", None) or np.core
    escalar = core.multiarray.scalar
    return [
        (escalar, "numpy.core.multiarray.scalar"),
        (escalar, "numpy._core.multiarray.scalar"),
        np.dtype,
        type(np.dtype("float32")),
        type(np.dtype("float64")),
    ]


def content_digest(checkpoint: dict) -> str:
    """SHA-256 dos nomes, tipos, formas e bytes dos tensores, mais mean e std.

    O hash do arquivo muda com a versão do PyTorch que grava; o do conteúdo não.
    """
    digest = hashlib.sha256()
    itens = [*checkpoint["model"].items(), ("mean", checkpoint["mean"]), ("std", checkpoint["std"])]
    for nome, tensor in itens:
        digest.update(f"{nome}|{tensor.dtype}|{tuple(tensor.shape)}|".encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def convert_btc(original: Path, destino: Path, esperado: str | None = None) -> str:
    """Grava a versão "safe" do checkpoint do BTC e devolve o resumo do conteúdo.

    Com ``esperado``, o arquivo só ganha o nome final se o conteúdo tiver esse resumo.
    """
    import torch

    with torch.serialization.safe_globals(_numpy_permitidos()):
        checkpoint = torch.load(original, map_location="cpu", weights_only=True)
    seguro = {
        "model": checkpoint["model"],
        "mean": torch.as_tensor(checkpoint["mean"]),
        "std": torch.as_tensor(checkpoint["std"]),
    }
    parcial = _temporario(destino)
    try:
        torch.save(seguro, parcial)
        # Prova que o arquivo gravado abre no modo seguro puro, sem nenhuma liberação.
        resumo = content_digest(torch.load(parcial, map_location="cpu", weights_only=True))
        if esperado is not None and resumo != esperado:
            raise SystemExit(
                f"{destino.name}: o conteúdo convertido ({resumo}) não é o esperado ({esperado}). "
                "O arquivo foi descartado."
            )
        os.replace(parcial, destino)
    finally:
        parcial.unlink(missing_ok=True)
    return resumo


def btc_ok(destino: Path, esperado: str) -> bool:
    if not destino.is_file():
        return False
    import torch

    try:
        checkpoint = torch.load(destino, map_location="cpu", weights_only=True)
        return content_digest(checkpoint) == esperado
    except Exception:
        return False


def prepare_btc(
    url: str = BTC_URL,
    sha256: str = BTC_SHA256,
    destino: Path = BTC_DESTINO,
    esperado: str = BTC_CONTEUDO_SHA256,
) -> bool:
    """Baixa e converte o BTC. Devolve False se o destino já estava certo."""
    if btc_ok(destino, esperado):
        return False
    destino.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destino.parent, prefix=".btc-original-") as pasta:
        original = Path(pasta) / "btc_model_large_voca.pt"
        download(url, original, sha256)
        convert_btc(original, destino, esperado)
    return True


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("Beat This! (cerca de 78 MB)...")
    novo = download(BEAT_THIS_URL, BEAT_THIS_DESTINO, BEAT_THIS_SHA256)
    print(f"  {'baixado' if novo else 'já estava certo'}: {BEAT_THIS_DESTINO.relative_to(RAIZ)}")
    print("BTC (cerca de 12 MB)...")
    novo = prepare_btc()
    print(
        f"  {'baixado e convertido' if novo else 'já estava certo'}: "
        f"{BTC_DESTINO.relative_to(RAIZ)}"
    )


if __name__ == "__main__":
    main()
