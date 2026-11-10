"""Seções, perfil vocal e texto da cifra."""

from __future__ import annotations

import sys

import numpy as np
from constants import (
    BAR_OFFSET_PRINT_MS,
    CHEST_BAND,
    CHEST_HOLD_RATIO,
    CHEST_LOOKBACK,
    CHEST_MIN_WINDOW,
    CHEST_PUSH_RATIO,
    CHORUS_LOOKAHEAD,
    ENVELOPE_WINDOW_S,
    EPS,
    FALLBACK_CHORUS_MIN,
    FALLBACK_VOCAB_BARS,
    FINAL_REST_BARS,
    INSTRUMENTAL_PERCENTILE,
    PRESENCE_BAND,
    PUSH_LAND_BARS,
    PUSH_LAND_HITS,
    PUSH_LAND_MIN,
    SAMPLE_RATE,
    SINGING_FLOOR_FRACTION,
    SINGING_PERCENTILE,
    SINGING_PERCENTILE_GAIN,
    VOICE_REENTRY_QUIET,
)
from scipy import signal
from scipy.ndimage import uniform_filter1d


def band_envelope(samples: np.ndarray, low: float, high: float) -> np.ndarray:
    sos = signal.butter(4, [low, high], btype="band", fs=SAMPLE_RATE, output="sos")
    filtered = signal.sosfiltfilt(sos, samples)
    window = max(1, int(ENVELOPE_WINDOW_S * SAMPLE_RATE))
    power = uniform_filter1d(filtered**2, size=window)
    return np.sqrt(np.maximum(power, 0.0))


def bar_level(envelope: np.ndarray, start: float, end: float) -> float:
    left = max(0, int(start * SAMPLE_RATE))
    right = min(envelope.size, max(left + 1, int(end * SAMPLE_RATE)))
    return float(np.median(envelope[left:right]))


def vocal_profile(samples: np.ndarray, bars: list[dict]) -> dict:
    """Presença é o brilho da voz. Corpo é o registro grave, acima do baixo."""
    presence_env = band_envelope(samples, *PRESENCE_BAND)
    chest_env = band_envelope(samples, *CHEST_BAND)
    presence = np.array([bar_level(presence_env, bar["start"], bar["end"]) for bar in bars])
    chest = np.array([bar_level(chest_env, bar["start"], bar["end"]) for bar in bars])
    return {"presence": presence, "chest": chest}


def singing_threshold(presence: np.ndarray) -> float:
    peak = float(np.max(presence)) if presence.size else 0.0
    early = presence[: max(4, presence.size // 2)]
    active = early[early > peak * SINGING_FLOOR_FRACTION]
    if active.size < 3:
        return peak
    return float(np.percentile(active, SINGING_PERCENTILE) * SINGING_PERCENTILE_GAIN)


def first_sustained(presence: np.ndarray, threshold: float) -> int:
    for index in range(presence.size - 1):
        previous = 0.0 if index == 0 else presence[index - 1]
        if previous < threshold <= presence[index] and presence[index + 1] >= threshold:
            return index
    return 0


def final_rest(presence: np.ndarray, threshold: float) -> int:
    quiet = 0
    for index in range(presence.size - 1, -1, -1):
        if presence[index] < threshold:
            quiet += 1
            continue
        break
    if quiet >= FINAL_REST_BARS:
        return presence.size - quiet
    return presence.size


def instrumental_floor(presence: np.ndarray, onset: int) -> float:
    peak = float(np.max(presence)) if presence.size else 0.0
    bed = presence[: max(onset, 1)]
    active = bed[bed > peak * SINGING_FLOOR_FRACTION]
    if active.size < 2:
        return float(np.percentile(presence, INSTRUMENTAL_PERCENTILE)) if presence.size else 0.0
    return float(np.median(active))


def voice_reentries(
    presence: np.ndarray, floor: float, start_at: int, min_quiet: int = VOICE_REENTRY_QUIET
) -> list[int]:
    found = []
    quiet = 0
    for index in range(start_at, presence.size):
        if presence[index] < floor:
            quiet += 1
            continue
        if quiet >= min_quiet and index > start_at:
            found.append(index)
        quiet = 0
    return found


def chest_pushes(chest: np.ndarray, sung: np.ndarray) -> list[tuple[int, int]]:
    pushes = []
    index = 0
    while index < chest.size:
        window = chest[max(0, index - CHEST_LOOKBACK) : index]
        if window.size < CHEST_MIN_WINDOW or not sung[index]:
            index += 1
            continue
        baseline = float(np.median(window))
        if baseline <= EPS or chest[index] < baseline * CHEST_PUSH_RATIO:
            index += 1
            continue
        end = index
        while end + 1 < chest.size and chest[end + 1] >= baseline * CHEST_HOLD_RATIO:
            end += 1
        pushes.append((index, end))
        index = end + 1
    return pushes


def push_lands(presence: np.ndarray, end: int, threshold: float) -> bool:
    following = presence[end + 1 : end + PUSH_LAND_BARS]
    if following.size < PUSH_LAND_MIN:
        return False
    return int(np.sum(following >= threshold)) >= PUSH_LAND_HITS


def chorus_from_push(roots: list[str], start: int, end: int) -> int | None:
    landing = end + 1
    if landing >= len(roots):
        return None
    return landing


def fallback_chorus(roots: list[str], onset: int) -> int | None:
    vocabulary = set(roots[onset : onset + FALLBACK_VOCAB_BARS])
    best = None
    index = onset
    while index < len(roots):
        if roots[index] in vocabulary:
            index += 1
            continue
        end = index
        while end + 1 < len(roots) and roots[end + 1] == roots[index]:
            end += 1
        length = end - index + 1
        if length >= FALLBACK_CHORUS_MIN and (best is None or length > best[0]):
            best = (length, index)
        index = end + 1
    return None if best is None else best[1]


def sections_from_voice(
    bars: list[dict], presence: np.ndarray, chest: np.ndarray
) -> tuple[list[dict], dict, list[dict]]:
    roots = [bar.get("root") or "?" for bar in bars]
    threshold = singing_threshold(presence)
    onset = first_sustained(presence, threshold)
    floor = instrumental_floor(presence, onset)
    rest = final_rest(presence, floor)
    sung = presence >= threshold
    candidates = []
    from_voice = False
    for start, end in chest_pushes(chest, sung):
        if end + 1 <= onset or not push_lands(presence, end, threshold):
            continue
        landing = chorus_from_push(roots, start, end)
        if landing is not None and landing < rest:
            candidates.append(landing)
            from_voice = True
    if not candidates:
        fallback = fallback_chorus(roots, onset)
        if fallback is not None and fallback < rest:
            candidates.append(fallback)
    landings = candidates[:1]
    reentries = []
    if landings:
        reentries = [
            index
            for index in voice_reentries(presence, floor, landings[0])
            if index < rest and index not in landings
        ]
        for candidate in candidates[1:]:
            if any(entry < candidate for entry in reentries):
                landings.append(candidate)
                break
    names = []
    state = "introdução"
    chorus_notes: set[str] = set()
    for index, root in enumerate(roots):
        if index >= rest:
            state = "outro"
            chorus_notes = set()
        elif index < onset:
            state = "introdução"
        elif index in landings:
            state = "refrão"
            chorus_notes = {note for note in roots[index : index + CHORUS_LOOKAHEAD] if note != "?"}
        elif index == onset or index in reentries:
            state = "verso"
            chorus_notes = set()
        elif state == "refrão" and chorus_notes:
            ahead = [note for note in roots[index : index + CHORUS_LOOKAHEAD] if note != "?"]
            if len(ahead) >= CHORUS_LOOKAHEAD and chorus_notes.isdisjoint(ahead):
                state = "ponte"
                chorus_notes = set()
        names.append(state)
    cuts = set(landings) | set(reentries) | {rest}
    merged = []
    for index, bar in enumerate(bars):
        name = names[index]
        if merged and merged[-1]["name"] == name and index not in cuts:
            merged[-1]["end"] = bar["end"]
            merged[-1]["bars"].append(bar)
            merged[-1]["repeats"] += 1
            continue
        merged.append(
            {"name": name, "start": bar["start"], "end": bar["end"], "repeats": 1, "bars": [bar]}
        )
    sections = []
    for item in merged:
        sections.append(
            {
                "name": item["name"],
                "start": item["start"],
                "end": item["end"],
                "repeats": item["repeats"],
                "bars": [
                    {"bar": bar["bar"], "chords": bar["chords"], "confidence": bar["confidence"]}
                    for bar in item["bars"]
                ],
            }
        )
    voice = {
        "entrada_verso_s": None if onset <= 0 else round(float(bars[onset]["start"]), 3),
        "entrada_refrao_s": None if not landings else round(float(bars[landings[0]]["start"]), 3),
        "refrao_pela_dinamica": from_voice and bool(landings),
    }
    return sections, voice, []


def estimate_sections(
    samples: np.ndarray,
    bars: list[dict],
    key: dict | None = None,
    ai_bar_labels: list[str] | None = None,
    vocal_onset_bar: int | None = None,
    bar_chroma: np.ndarray | None = None,
) -> tuple[list[dict], dict, list[dict]]:
    profile = vocal_profile(samples, bars)
    presence = profile["presence"]
    chest = profile["chest"]
    try:
        from harmonic_structure import map_sections_with_theory

        key_info = key or {"name": "C", "mode": "maior", "tonic": 0}
        return map_sections_with_theory(
            bars,
            key_info,
            vocal_presence=presence,
            vocal_chest=chest,
            ai_bar_labels=ai_bar_labels,
            vocal_onset_bar=vocal_onset_bar,
            bar_chroma=bar_chroma,
        )
    except Exception as error:
        sys.stderr.write(f"Mapa harmônico indisponível; usando heurística simples: {error}\n")
    return sections_from_voice(bars, presence, chest)


def clock(seconds: float) -> str:
    whole = max(0, int(round(seconds)))
    return f"{whole // 60}:{whole % 60:02d}"


def render_header(
    song: str, pulse: dict, meter: dict, key: dict, voice: dict, ruled: int
) -> list[str]:
    lines = [
        song,
        f"Tom: {key['name']} {key['mode']}",
    ]
    if key.get("modulacao"):
        mod = key["modulacao"]
        lines.append(
            f"Modulação: sobe {mod['semitones']} semitom(ns) para {mod['new_name_flat']} / {mod['new_name_sharp']} "
            f"{mod.get('new_mode', 'maior')} em {clock(mod['start_s'])} (compasso {mod['start_bar'] + 1})."
        )
    if key.get("correcao_tom"):
        fix = key["correcao_tom"]
        lines.append(
            f"O áudio apontava {fix['antes']}, mas só {round(fix['encaixe_antes'] * 100)}% dos acordes cabem nele; "
            f"{round(fix['encaixe_depois'] * 100)}% cabem em {key['name']} {key['mode']}."
        )
    if key.get("rival"):
        rival = key["rival"]
        lines.append(
            f"O grave puxava {rival['name']} {rival['mode']}. O centro fica uma quinta abaixo."
        )
    elif key.get("quinto"):
        fifth = key["quinto"]
        lines.append(
            f"{fifth['name']} {fifth['mode']} marca {round(fifth['relacao'] * 100)}%. O perfil de presença não troca o centro."
        )
    lines.append(f"Pulsação: {pulse['bpm']} BPM")
    if meter.get("source") == "beat_this":
        lines.append(
            f"Compasso: {meter['beats_per_bar']} tempos por compasso. A rede não separa 6/8 de 2/4, "
            "então a fórmula não é confirmada."
        )
        grid = "Grade: Beat This! marca pulsos e tempos fortes; um Viterbi segura o compasso."
        if pulse.get("inferred_downbeats"):
            grid += f" {pulse['inferred_downbeats']} tempo(s) 1 vieram da continuidade, sem marca da rede."
        if pulse.get("rejected_downbeats"):
            grid += (
                f" {pulse['rejected_downbeats']} marca(s) da rede fora do tempo 1 foram ignoradas."
            )
        if pulse.get("irregular_bars"):
            grid += f" {pulse['irregular_bars']} compasso(s) têm outra duração."
        if pulse.get("extrapolated_beats"):
            grid += f" {pulse['extrapolated_beats']} pulso(s) nas pontas foram estendidos no mesmo andamento."
        lines.append(grid)
    else:
        lines.append(
            f"Compasso: {meter['numerator']}/{meter['denominator']} (estimativa heurística, não confirmada)"
        )
    if meter.get("phase_from_harmony"):
        lines.append("O tempo 1 acompanha a troca de acorde. O grave sozinho marcava outro tempo.")
    if pulse.get("compasso_ms") and abs(int(pulse["compasso_ms"])) >= BAR_OFFSET_PRINT_MS:
        shift = int(pulse["compasso_ms"])
        if shift < 0:
            lines.append(
                f"A troca do compasso acompanha o grave, {-shift} ms antes do ataque agudo."
            )
        else:
            lines.append(
                f"A troca do compasso acompanha o grave, {shift} ms depois do ataque agudo."
            )
    if pulse["collapsed_eighths"]:
        lines.append("A grade fina caiu na colcheia. O pulso abaixo é o tempo acentuado.")
    if voice.get("entrada_verso_s") is not None:
        if voice.get("detector", "").startswith("all-in-one"):
            lines.append(
                f"Estrutura: All-In-One marca o primeiro verso em {clock(voice['entrada_verso_s'])}."
            )
        else:
            lines.append(f"A voz entra em {clock(voice['entrada_verso_s'])} e abre o verso.")
    if voice.get("entrada_refrao_s") is not None and voice.get("refrao_pela_dinamica"):
        lines.append(
            f"O refrão entra em {clock(voice['entrada_refrao_s'])}, quando a dinâmica da voz sobe e se sustenta."
        )
    lines.append(f"Afinação: {key['reference']:.1f} Hz")
    if key.get("detector") == "btc":
        lines.append(
            "Acordes: estimativa do Transformer BTC (170 classes), com a afinação estimada aplicada; "
            "ainda sem avaliação contra cifras de referência."
        )
    else:
        lines.append(
            "Acordes: gabarito NNLS local (fallback). O harmônico volta para a nota que o gerou."
        )
    if key.get("detector") == "btc":
        lines.append(
            "Cifra: uma troca que o modelo sustenta guarda o instante dele quando cai fora do tempo; "
            "um piscar curto continua no tempo."
        )
    else:
        lines.append("Cifra: cada troca de acorde cai no tempo mais próximo dentro do compasso.")
    lines.append("A barra é o baixo estimado quando ele não é a fundamental.")
    if ruled:
        lines.append(f"{ruled} compasso(s) foram ajustados por regra; o modelo dizia outra coisa.")
    return lines


def render_patterns(patterns: list[dict] | None) -> list[str]:
    if not patterns:
        return []
    lines = []
    repeated = [p for p in patterns if p.get("repeticoes", 0) >= 2]
    for pat in repeated[:3]:
        chords_str = " | ".join(pat.get("acordes") or [])
        romans_str = " – ".join(pat.get("graus") or [])
        lines.append(
            f"{pat['id']} ({pat['repeticoes']}x, {pat['compassos_por_ciclo']} compassos): "
            f"{chords_str}  [{romans_str} · {pat.get('cadencia', '')}]"
        )
    return lines


def render_sections(sections: list[dict]) -> list[str]:
    lines = []
    for section in sections:
        meta = []
        if section.get("padrao"):
            meta.append(str(section["padrao"]))
        if section.get("graus"):
            meta.append(str(section["graus"]))
        if section.get("cadencia") and section.get("cadencia") != "silêncio":
            meta.append(str(section["cadencia"]))
        suffix = f"  [{' · '.join(meta)}]" if meta else ""
        lines.append(
            f"{section['name']}  {clock(section['start'])}–{clock(section['end'])}{suffix}"
        )
        row = []
        for bar in section["bars"]:
            row.append(" ".join(bar["chords"]))
            if len(row) == 4:
                lines.append("  " + " | ".join(row))
                row = []
        if row:
            lines.append("  " + " | ".join(row))
        lines.append("")
    return lines


def render(
    song: str,
    pulse: dict,
    meter: dict,
    key: dict,
    sections: list[dict],
    voice: dict,
    patterns: list[dict] | None = None,
    ruled: int = 0,
) -> str:
    lines = render_header(song, pulse, meter, key, voice, ruled)
    lines.extend(render_patterns(patterns))
    lines.append("")
    lines.extend(render_sections(sections))
    return "\n".join(lines).rstrip() + "\n"
