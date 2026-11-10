"""Testes de scripts/baixar_modelos.py, sem rede: as URLs são arquivos locais fictícios."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import baixar_modelos  # noqa: E402

TEM_TORCH = importlib.util.find_spec("torch") is not None
SEM_TORCH = "precisa do torch (pip install -r requirements.txt)"


class _Explosivo:
    """Um objeto que, aberto no modo inseguro, criaria a pasta ``alvo``."""

    def __init__(self, alvo: Path):
        self.alvo = alvo

    def __reduce__(self):
        return (os.mkdir, (str(self.alvo),))


class DownloadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.origem = self.tmp / "origem.bin"
        self.origem.write_bytes(b"pesos ficticios")
        self.sha = hashlib.sha256(b"pesos ficticios").hexdigest()
        self.destino = self.tmp / "vendor" / "modelo.ckpt"

    def test_downloads_once_and_keeps_a_file_that_already_matches(self):
        self.assertTrue(baixar_modelos.download(self.origem.as_uri(), self.destino, self.sha))
        self.assertEqual(self.destino.read_bytes(), b"pesos ficticios")
        self.assertFalse(baixar_modelos.download(self.origem.as_uri(), self.destino, self.sha))

    def test_replaces_a_file_whose_hash_is_wrong(self):
        self.destino.parent.mkdir(parents=True)
        self.destino.write_bytes(b"cortado")
        self.assertTrue(baixar_modelos.download(self.origem.as_uri(), self.destino, self.sha))
        self.assertEqual(self.destino.read_bytes(), b"pesos ficticios")

    def test_discards_a_download_with_the_wrong_hash(self):
        with self.assertRaises(SystemExit):
            baixar_modelos.download(self.origem.as_uri(), self.destino, "0" * 64)
        self.assertEqual(list(self.destino.parent.iterdir()), [])


@unittest.skipUnless(TEM_TORCH, SEM_TORCH)
class ConvertBtcTest(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch

        self.torch = torch
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.original = self.tmp / "original.pt"
        self.destino = self.tmp / "vendor" / "safe.pt"
        self.pesos = {"camada.peso": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
        torch.save({"model": self.pesos, "mean": 1.5, "std": np.float64(0.25)}, self.original)

    def test_the_fixture_is_refused_by_the_plain_safe_mode(self):
        with self.assertRaises(Exception):  # noqa: B017
            self.torch.load(self.original, map_location="cpu", weights_only=True)

    def test_the_converted_file_opens_in_safe_mode_with_the_same_values(self):
        resumo = baixar_modelos.convert_btc(self.original, self.destino)
        seguro = self.torch.load(self.destino, map_location="cpu", weights_only=True)
        self.assertTrue(self.torch.equal(seguro["model"]["camada.peso"], self.pesos["camada.peso"]))
        self.assertEqual(float(seguro["mean"]), 1.5)
        self.assertEqual(float(seguro["std"]), 0.25)
        self.assertEqual(seguro["std"].dtype, self.torch.float64)
        self.assertEqual(resumo, baixar_modelos.content_digest(seguro))

    def test_the_digest_changes_when_one_weight_changes(self):
        antes = baixar_modelos.convert_btc(self.original, self.destino)
        self.pesos["camada.peso"][0, 0] = 9.0
        self.torch.save({"model": self.pesos, "mean": 1.5, "std": 0.25}, self.original)
        self.assertNotEqual(antes, baixar_modelos.convert_btc(self.original, self.destino))

    def test_refuses_a_checkpoint_that_would_run_code(self):
        alvo = self.tmp / "codigo-executado"
        self.torch.save({"model": self.pesos, "mean": _Explosivo(alvo), "std": 0.25}, self.original)
        with self.assertRaises(Exception):  # noqa: B017
            baixar_modelos.convert_btc(self.original, self.destino)
        self.assertFalse(alvo.exists())
        self.assertFalse(self.destino.exists())

    def test_prepare_is_idempotent_and_rejects_unexpected_content(self):
        sha = baixar_modelos.sha256_of(self.original)
        esperado = baixar_modelos.convert_btc(self.original, self.tmp / "referencia.pt")
        url = self.original.as_uri()
        self.assertTrue(baixar_modelos.prepare_btc(url, sha, self.destino, esperado))
        self.assertFalse(baixar_modelos.prepare_btc(url, sha, self.destino, esperado))
        outro = self.tmp / "vendor2" / "safe.pt"
        with self.assertRaises(SystemExit):
            baixar_modelos.prepare_btc(url, sha, outro, "0" * 64)
        self.assertEqual(list(outro.parent.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
