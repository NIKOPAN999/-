from __future__ import annotations

import time
from pathlib import Path

from .core import map_alignment, parse_textgrid, read_json, words_from_text, write_json
from .media import execute


class Aligners:
    def __init__(self, config):
        self.config = config
        self.whisper_model = None

    def run(self, backend, row, directory, media):
        directory = Path(directory)
        output = directory / f"alignment_{backend}.json"
        if output.exists():
            return read_json(output)
        words = words_from_text(row["raw_text"])
        transcript = " ".join(w["normalized"] for w in words)
        begin = time.perf_counter()
        if backend == "whisperx":
            from whisperx.alignment import align, load_align_model
            from whisperx.audio import load_audio
            if self.whisper_model is None:
                model_path = Path(self.config["align_model_path"])
                if not (model_path / "config.json").is_file():
                    raise FileNotFoundError("Alignment model missing; run download-models explicitly on server")
                self.whisper_model = load_align_model(language_code="en", device=self.config["device"],
                                                      model_name=str(model_path))
                self.whisper_model[0].eval()
            audio = load_audio(str(directory / "audio.wav"))
            model, metadata = self.whisper_model
            result = align([dict(start=0.0, end=media["audio_duration"], text=transcript)],
                           model, metadata, audio, self.config["device"], return_char_alignments=False)
            emitted = result["word_segments"]
            # WhisperX can interpolate missing boundaries. A missing score is not
            # acoustic alignment evidence; do not silently accept it as valid.
            emitted = [dict(w, start=None, end=None) if w.get("score") is None else w for w in emitted]
        elif backend == "mfa":
            task = directory / "mfa"
            task.mkdir(exist_ok=True)
            text_path = task / "audio.lab"
            text_path.write_text(transcript, encoding="utf-8")
            target = task / "audio.TextGrid"
            execute([self.config["mfa_bin"], "align_one", directory / "audio.wav", text_path,
                     self.config["mfa_dictionary"], self.config["mfa_acoustic"], target,
                     "--output_format", "long_textgrid", "--single_speaker", "--clean",
                     "--temporary_directory", task / "temp", "--num_jobs", "1"],
                    directory / "mfa.log", timeout=1800)
            emitted = parse_textgrid(target)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        rows = map_alignment(words, emitted, media["audio_duration"])
        result = dict(backend=backend, sample_id=row["sample_id"], raw_text=row["raw_text"],
                      normalized_text=transcript, coordinate="seconds from prepared audio.wav start",
                      elapsed_seconds=time.perf_counter() - begin, words=rows, emitted=emitted,
                      valid_words=sum(w["valid"] for w in rows), total_words=len(rows),
                      note="Elapsed includes model loading on the first sample; not a speed benchmark")
        write_json(output, result)
        return result
