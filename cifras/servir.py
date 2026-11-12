"""Serve a cifra já medida e o áudio numa página local.

A análise continua em analyze.py. Esta página só mostra o resultado e,
se pedido, mede a faixa de novo.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import traceback
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import analyze
import correcoes
from constants import MAX_AUDIO_SECONDS

ROOT = analyze.ROOT
OUT = ROOT / "cifras" / "saida"
WEB = ROOT / "cifras" / "web"
PAGE = WEB / "index.html"
STYLES = WEB / "styles.css"
UPLOADS = ROOT / "uploads"
SKIPPED = {
    "moises-app-x64",
    "moises-setup-unpacked",
    "rea",
    "node_modules",
    "cifras",
    "__pycache__",
}
AUDIO = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac"}
MAX_AUDIO_BYTES = 200 * 1024 * 1024
MAX_VIDEO_SECONDS = 15 * 60
ANALYSIS_LOCK = threading.Lock()
HOST = "127.0.0.1"
PORT = 8765
ALLOWED_HOSTS = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}
ALLOWED_ORIGINS = {f"http://{host}" for host in ALLOWED_HOSTS}
# Baixar áudio do YouTube fora do player oficial vai contra os termos da
# plataforma; a rota só existe quando quem roda o servidor liga isso.
YOUTUBE_ENABLED = os.environ.get("CIFRA_PERMITIR_YOUTUBE") == "1"


def host_allowed(host: str | None) -> bool:
    return (host or "").strip().lower() in ALLOWED_HOSTS


def origin_allowed(origin: str | None) -> bool:
    return (origin or "").strip().lower() in ALLOWED_ORIGINS


def content_type_is(header: str | None, expected: str) -> bool:
    return (header or "").split(";", 1)[0].strip().lower() == expected


def audio_files() -> list[Path]:
    found = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [
            name for name in dirnames if name not in SKIPPED and not name.startswith(".")
        ]
        for name in filenames:
            path = Path(dirpath) / name
            if path.suffix.lower() in AUDIO:
                found.append(path)
    return sorted(found, key=lambda path: path.stem.lower())


def resolve_track(raw: str) -> Path:
    text = unquote(raw or "").strip().replace("\\", "/")
    if not text or text.startswith("/") or ":" in text:
        raise ValueError("Faixa inválida.")
    candidate = (ROOT / text).resolve()
    if (
        ROOT not in candidate.parents
        or not candidate.is_file()
        or candidate.suffix.lower() not in AUDIO
    ):
        raise ValueError("Essa faixa não está no projeto.")
    if any(part in SKIPPED for part in candidate.parts):
        raise ValueError("Essa pasta fica de fora.")
    return candidate


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def safe_upload_name(raw: str) -> str:
    original = Path(raw or "audio").name
    suffix = Path(original).suffix.lower()
    if suffix not in AUDIO:
        raise ValueError("Envie um arquivo MP3, WAV ou FLAC.")
    stem = (
        unicodedata.normalize("NFKD", Path(original).stem).encode("ascii", "ignore").decode("ascii")
    )
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "-", stem).strip(" .-_") or "audio"
    candidate = UPLOADS / f"{stem}{suffix}"
    counter = 2
    while candidate.exists():
        candidate = UPLOADS / f"{stem}-{counter}{suffix}"
        counter += 1
    return candidate.name


def youtube_url(raw: str) -> str:
    text = (raw or "").strip()
    parsed = urlparse(text)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or host not in {
        "youtube.com",
        "www.youtube.com",
        "music.youtube.com",
        "youtu.be",
    }:
        raise ValueError("Cole um link válido do YouTube ou YouTube Music.")
    return text


def chart_for(path: Path) -> dict | None:
    payload = OUT / f"{path.stem}.json"
    if not payload.is_file():
        return None
    data = json.loads(payload.read_text(encoding="utf-8"))
    bars = data.get("compassos") or []
    cursor = 0
    sections = []
    for section in data.get("secoes") or []:
        packed = []
        for _bar in section.get("bars") or []:
            if cursor >= len(bars):
                break
            source = bars[cursor]
            cursor += 1
            packed.append(
                {
                    "bar": source.get("bar"),
                    "chords": source.get("chords") or ["N"],
                    "events": source.get("events") or [],
                    "events_modelo": source.get("events_modelo"),
                    "chords_modelo": source.get("chords_modelo"),
                    "origem": source.get("origem"),
                    "regra": source.get("regra"),
                    "confidence": source.get("confidence"),
                    "start": source.get("start"),
                    "end": source.get("end"),
                }
            )
        sections.append(
            {
                "name": section.get("name"),
                "start": section.get("start"),
                "end": section.get("end"),
                "padrao": section.get("padrao"),
                "graus": section.get("graus"),
                "cadencia": section.get("cadencia"),
                "bars": packed,
            }
        )
    text_path = OUT / f"{path.stem}.txt"
    pulse = data.get("pulsacao") or {}
    meter = data.get("compasso") or {}
    if meter.get("formula_confirmada"):
        meter_label = meter.get("formula")
    elif meter.get("tempos_por_compasso"):
        meter_label = f"{meter['tempos_por_compasso']} tempos/compasso"
    else:
        meter_label = meter.get("formula")
    return {
        "nome": path.stem,
        "faixa": relative(path),
        "duracao_s": data.get("duracao_s"),
        "tom": data.get("tom"),
        "bpm": pulse.get("bpm"),
        "grade_recolhida": pulse.get("grade_recolhida_da_colcheia"),
        "compasso": meter_label,
        "ajustados_por_regra": data.get("compassos_ajustados_por_regra"),
        "detector_estrutura": data.get("detector_estrutura"),
        "padroes_harmonicos": data.get("padroes_harmonicos") or [],
        "voz": data.get("voz"),
        "texto": text_path.read_text(encoding="utf-8") if text_path.is_file() else "",
        "repeticoes": correcoes.valid_repeats(correcoes.read(path.stem), len(bars)),
        "secoes": sections,
    }


CHORD_TEXT = re.compile(
    r"^(?:N|N\.C\.|[A-G](?:#|b)?[A-Za-z0-9#b°Δ+\-()]{0,16}(?:/[A-G](?:#|b)?)?)$"
)


def apply_chord_edit(
    data: dict, bar_number: int, event_index: int, chord: str, source: str = "cifra"
) -> None:
    if not CHORD_TEXT.match(chord or ""):
        raise ValueError("Acorde inválido.")
    if event_index < 0:
        raise ValueError("Acorde não encontrado.")
    events_key = "events_modelo" if source == "modelo" else "events"
    chords_key = "chords_modelo" if source == "modelo" else "chords"
    for bar in data.get("compassos") or []:
        if int(bar.get("bar") or 0) != int(bar_number):
            continue
        events = list(bar.get(events_key) or [])
        if not events:
            events = [
                {"chord": item, "start": bar.get("start"), "end": bar.get("end")}
                for item in (bar.get(chords_key) or ["N"])
            ]
        if event_index >= len(events):
            raise ValueError("Acorde não encontrado.")
        events[event_index]["chord"] = chord
        events[event_index]["origem"] = "edicao"
        bar[events_key] = events
        bar[chords_key] = [event.get("chord") or "N" for event in events]
        bar["origem"] = "edicao"
        return
    raise ValueError("Compasso não encontrado.")


def save_chord_edit(
    track: Path, bar_number: int, event_index: int, chord: str, source: str = "cifra"
) -> None:
    path = OUT / f"{track.stem}.json"
    if not path.is_file():
        raise ValueError("Esta faixa ainda não foi medida.")
    data = json.loads(path.read_text(encoding="utf-8"))
    apply_chord_edit(data, bar_number, event_index, chord, source)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_repeat(track: Path, body: dict) -> None:
    """Grava as barras de repetição da faixa: a lista inteira, ou uma a pôr ou tirar."""
    path = OUT / f"{track.stem}.json"
    if not path.is_file():
        raise ValueError("Esta faixa ainda não foi medida.")
    bar_count = len(json.loads(path.read_text(encoding="utf-8")).get("compassos") or [])
    data = correcoes.read(track.stem)
    if "repeticoes" in body:
        correcoes.set_repeats(data, bar_count, body.get("repeticoes"))
    elif body.get("remover"):
        correcoes.remove_repeat(data, int(body.get("inicio")))
    else:
        correcoes.add_repeat(
            data, bar_count, int(body.get("inicio")), int(body.get("fim")), int(body.get("vezes"))
        )
    correcoes.write(track.stem, data)


def save_result(path: Path, result: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in result.items() if key != "cifra"}
    (OUT / f"{path.stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / f"{path.stem}.txt").write_text(result["cifra"], encoding="utf-8")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s\n" % (fmt % args))

    def do_GET(self) -> None:
        if not host_allowed(self.headers.get("Host")):
            self._send_json({"erro": "Host não permitido."}, 403)
            return
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_bytes(PAGE.read_bytes(), "text/html; charset=utf-8")
            return
        if parsed.path == "/styles.css":
            self._send_bytes(STYLES.read_bytes(), "text/css; charset=utf-8")
            return
        if parsed.path == "/api/config":
            self._send_json({"youtube": YOUTUBE_ENABLED, "max_audio_min": MAX_AUDIO_SECONDS // 60})
            return
        if parsed.path == "/api/musicas":
            songs = [
                {
                    "nome": path.stem,
                    "faixa": relative(path),
                    "medida": (OUT / f"{path.stem}.json").is_file(),
                }
                for path in audio_files()
            ]
            self._send_json({"musicas": songs})
            return
        if parsed.path == "/api/cifra":
            try:
                track = resolve_track(parse_qs(parsed.query).get("faixa", [""])[0])
            except ValueError as error:
                self._send_json({"erro": str(error)}, 400)
                return
            chart = chart_for(track)
            if chart is None:
                self._send_json(
                    {"erro": "Esta faixa ainda não foi medida.", "faixa": relative(track)}, 404
                )
                return
            self._send_json(chart)
            return
        if parsed.path == "/audio":
            try:
                track = resolve_track(parse_qs(parsed.query).get("faixa", [""])[0])
            except ValueError as error:
                self._send_json({"erro": str(error)}, 400)
                return
            self._send_audio(track)
            return
        self._send_json({"erro": "Não encontrado."}, 404)

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_POST(self) -> None:
        # Uma recusa pode sair antes de ler o corpo; o que sobrar no socket não
        # pode virar o próximo pedido da mesma conexão.
        self.close_connection = True
        if not host_allowed(self.headers.get("Host")) or not origin_allowed(
            self.headers.get("Origin")
        ):
            self._send_json({"erro": "Pedido de outra origem recusado."}, 403)
            return
        parsed = urlparse(self.path)
        if parsed.path == "/api/upload":
            self._upload(parsed)
            return
        if parsed.path == "/api/youtube":
            self._youtube()
            return
        if parsed.path == "/api/analisar":
            self._analyze()
            return
        if parsed.path == "/api/cifra":
            self._edit_chord()
            return
        if parsed.path == "/api/repeticao":
            self._edit_repeat()
            return
        self._send_json({"erro": "Não encontrado."}, 404)

    def _content_length(self) -> int | None:
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            return None
        return length if length >= 0 else None

    def _json_body(self) -> dict:
        if not content_type_is(self.headers.get("Content-Type"), "application/json"):
            raise ValueError("Envie JSON.")
        length = self._content_length()
        if length is None or length > 1024 * 1024:
            raise ValueError("Pedido grande demais ou sem tamanho.")
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(body, dict):
            raise ValueError("Envie um objeto JSON.")
        return body

    def _edit_chord(self) -> None:
        try:
            body = self._json_body()
            track = resolve_track(str(body.get("faixa") or ""))
            bar_number = int(body.get("compasso"))
            event_index = int(body.get("indice"))
            source = "modelo" if body.get("fonte") == "modelo" else "cifra"
            save_chord_edit(
                track, bar_number, event_index, str(body.get("acorde") or "").strip(), source
            )
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._send_json({"erro": str(error)}, 400)
            return
        chart = chart_for(track)
        self._send_json(chart or {"erro": "A cifra não foi salva."}, 200 if chart else 500)

    def _edit_repeat(self) -> None:
        try:
            body = self._json_body()
            track = resolve_track(str(body.get("faixa") or ""))
            save_repeat(track, body)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._send_json({"erro": str(error)}, 400)
            return
        chart = chart_for(track)
        self._send_json(chart or {"erro": "A cifra não foi encontrada."}, 200 if chart else 500)

    def _analyze(self) -> None:
        try:
            body = self._json_body()
            track = resolve_track(str(body.get("faixa") or ""))
        except (ValueError, json.JSONDecodeError) as error:
            self._send_json({"erro": str(error)}, 400)
            return
        if not ANALYSIS_LOCK.acquire(blocking=False):
            self._send_json({"erro": "Já existe uma música em análise. Aguarde ela terminar."}, 409)
            return
        try:
            result = analyze.analyze(track)
            save_result(track, result)
        except (Exception, SystemExit) as error:
            traceback.print_exc()
            self._send_json({"erro": str(error)}, 500)
            return
        finally:
            ANALYSIS_LOCK.release()
        chart = chart_for(track)
        self._send_json(chart or {"erro": "A medição não gerou cifra."}, 200 if chart else 500)

    def _upload(self, parsed) -> None:
        length = self._content_length()
        if not length or length > MAX_AUDIO_BYTES:
            self._send_json({"erro": "O áudio deve ter no máximo 200 MB."}, 413)
            return
        if not content_type_is(self.headers.get("Content-Type"), "application/octet-stream"):
            self._send_json({"erro": "Envie o arquivo como application/octet-stream."}, 415)
            return
        target = None
        try:
            name = parse_qs(parsed.query).get("nome", [""])[0]
            UPLOADS.mkdir(parents=True, exist_ok=True)
            target = UPLOADS / safe_upload_name(name)
            remaining = length
            with target.open("xb") as output:
                while remaining > 0:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("O envio parou antes do fim.")
                    output.write(chunk)
                    remaining -= len(chunk)
        except (ValueError, OSError) as error:
            if target is not None and target.exists():
                target.unlink(missing_ok=True)
            self._send_json({"erro": str(error)}, 400)
            return
        self._send_json({"faixa": relative(target), "nome": target.stem}, 201)

    def _youtube(self) -> None:
        if not YOUTUBE_ENABLED:
            self._send_json(
                {
                    "erro": "Download do YouTube desligado. Ele vai contra os termos da "
                    "plataforma; suba um arquivo seu."
                },
                403,
            )
            return
        import imageio_ffmpeg

        try:
            body = self._json_body()
            url = youtube_url(str(body.get("url") or ""))
        except (ValueError, json.JSONDecodeError) as error:
            self._send_json({"erro": str(error)}, 400)
            return

        UPLOADS.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--no-playlist",
            "--match-filter",
            f"duration <= {MAX_VIDEO_SECONDS}",
            "--max-filesize",
            "200M",
            "--no-overwrites",
            "--restrict-filenames",
            "-x",
            "--audio-format",
            "mp3",
            "--audio-quality",
            "0",
            "--ffmpeg-location",
            imageio_ffmpeg.get_ffmpeg_exe(),
            "--print",
            "after_move:filepath",
            "-o",
            str(UPLOADS / "%(title)s [%(id)s].%(ext)s"),
            url,
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=20 * 60, check=False
            )
        except subprocess.TimeoutExpired:
            self._send_json({"erro": "O download do YouTube excedeu 20 minutos."}, 504)
            return
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip().splitlines()
            message = detail[-1] if detail else "Não foi possível baixar esse vídeo."
            self._send_json({"erro": message}, 422)
            return
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        try:
            target = Path(lines[-1]).resolve()
            if (
                UPLOADS.resolve() not in target.parents
                or not target.is_file()
                or target.suffix.lower() not in AUDIO
            ):
                raise ValueError
        except (IndexError, ValueError):
            self._send_json(
                {"erro": "O YouTube baixou o áudio, mas não informou o arquivo final."}, 500
            )
            return
        self._send_json({"faixa": relative(target), "nome": target.stem}, 201)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status)

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_audio(self, path: Path) -> None:
        size = path.stat().st_size
        start, end = 0, size - 1
        status = 200
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            spec = range_header.split("=", 1)[1].split(",", 1)[0].strip()
            left, _, right = spec.partition("-")
            try:
                if left:
                    start = int(left)
                if right:
                    end = int(right)
            except ValueError:
                self._send_json({"erro": "Intervalo inválido."}, 416)
                return
            if start >= size or start < 0:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            end = min(end, size - 1)
            status = 206
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", AUDIO[path.suffix.lower()])
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        try:
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = handle.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Cifra local em http://{HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
