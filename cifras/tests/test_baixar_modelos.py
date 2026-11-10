"""Testes de scripts/baixar_modelos.py, sem rede: as URLs são arquivos locais fictícios."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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

    def test_leaves_nothing_behind_when_the_connection_breaks_mid_download(self):
        class Cortada:
            def __init__(self):
                self.blocos = [b"pesos "]

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self, _size):
                if self.blocos:
                    return self.blocos.pop()
                raise ConnectionResetError("conexão cortada")

        with mock.patch("urllib.request.urlopen", return_value=Cortada()) as aberto:
            with self.assertRaises(ConnectionResetError):
                baixar_modelos.download("https://exemplo.invalid/x", self.destino, self.sha)
        self.assertEqual(aberto.call_args.kwargs["timeout"], baixar_modelos.TIMEOUT_S)
        self.assertEqual(list(self.destino.parent.iterdir()), [])

    def test_two_partial_files_never_share_a_name(self):
        um, outro = (
            baixar_modelos._temporario(self.destino),
            baixar_modelos._temporario(self.destino),
        )
        self.assertNotEqual(um, outro)
        self.assertEqual(um.parent, self.destino.parent)


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

    def test_accepts_a_float32_numpy_mean(self):
        import numpy as np

        self.torch.save(
            {"model": self.pesos, "mean": np.float32(1.5), "std": np.float64(0.25)}, self.original
        )
        baixar_modelos.convert_btc(self.original, self.destino)
        seguro = self.torch.load(self.destino, map_location="cpu", weights_only=True)
        self.assertEqual(seguro["mean"].dtype, self.torch.float32)
        self.assertEqual(float(seguro["mean"]), 1.5)

    def test_unexpected_content_never_gets_the_final_name(self):
        with self.assertRaises(SystemExit):
            baixar_modelos.convert_btc(self.original, self.destino, "0" * 64)
        self.assertEqual(list(self.destino.parent.iterdir()), [])

    def test_the_digest_changes_when_one_weight_changes(self):
        antes = baixar_modelos.convert_btc(self.original, self.destino)
        self.pesos["camada.peso"][0, 0] = 9.0
        self.torch.save({"model": self.pesos, "mean": 1.5, "std": 0.25}, self.original)
        self.assertNotEqual(antes, baixar_modelos.convert_btc(self.original, self.destino))

    def test_refuses_a_global_outside_the_allowlist_and_runs_nothing(self):
        # Prova que um global fora da lista (aqui os.mkdir) é recusado e não roda. Não prova
        # que a lista em si é inexplorável: ela tem só o escalar do numpy e dtypes float.
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
