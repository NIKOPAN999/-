from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path

import numpy as np

from .core import read_json, write_json, sha256


def execute(args, log, timeout=3600):
    """No shell interpolation. Every external invocation is logged."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as out:
        out.write("\nCOMMAND " + json.dumps([str(a) for a in args]) + "\n")
        out.flush()
        subprocess.run([str(a) for a in args], check=True, stdout=out,
                       stderr=subprocess.STDOUT, timeout=timeout)


def probe(config, source, selector):
    cmd = [config["ffprobe"], "-v", "error", "-select_streams", selector,
           "-show_streams", "-show_frames", "-show_entries",
           "stream=start_time,time_base,duration,sample_rate,width,height:"
           "frame=best_effort_timestamp_time,pkt_duration_time,duration_time,nb_samples",
           "-of", "json", str(source)]
    return json.loads(subprocess.check_output(cmd, timeout=120))


def prepare(config, row, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = Path(config["data_root"]) / row["relative_path"]
    if sha256(source) != row["sha256"]:
        raise ValueError("Source video changed since manifest generation")
    output = directory / "media.json"
    if output.exists():
        media = read_json(output)
        if media["source_sha256"] == row["sha256"] and sha256(directory / "audio.wav") == media["wav_sha256"]:
            return media
        raise ValueError("Stale media cache; use a new work directory")
    video, audio = probe(config, source, "v:0"), probe(config, source, "a:0")
    if not video.get("frames") or not audio.get("frames"):
        raise ValueError("Missing decodable video or audio frames")
    pts = np.array([float(f["best_effort_timestamp_time"]) for f in video["frames"]])
    ap = np.array([float(f["best_effort_timestamp_time"]) for f in audio["frames"]])
    if not np.isfinite(pts).all() or not np.isfinite(ap).all() or np.any(np.diff(pts) <= 0):
        raise ValueError("Non-finite or non-increasing video/audio timestamps")
    sr = int(audio["streams"][0]["sample_rate"])
    ad = np.array([int(f["nb_samples"]) / sr for f in audio["frames"]])
    # A simple affine WAV-time mapping is unsafe in discontinuous streams.
    if len(ap) > 1 and np.max(np.abs(ap[1:] - (ap[:-1] + ad[:-1]))) > 0.02:
        raise ValueError("Audio has discontinuities; requires an explicit piecewise timestamp map")
    origin = min(float(pts[0]), float(ap[0]))
    execute([config["ffmpeg"], "-nostdin", "-y", "-v", "error", "-i", source,
             "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
             directory / "audio.wav"], directory / "commands.log")
    with wave.open(str(directory / "audio.wav")) as wav:
        duration = wav.getnframes() / wav.getframerate()
    if abs(duration - float(ad.sum())) > 0.10:
        raise ValueError("Decoded WAV duration differs from source audio frame durations")
    last = video["frames"][-1]
    step = float(last.get("duration_time") or last.get("pkt_duration_time") or 0)
    if step <= 0:
        if len(pts) < 2:
            raise ValueError("Cannot infer the final video frame duration")
        step = float(np.median(np.diff(pts)))
    ends = np.r_[pts[1:], pts[-1] + step]
    # Frame coverage is [its PTS, next PTS), not nominal FPS.
    intervals = np.column_stack([pts - origin, ends - origin])
    media = dict(source_sha256=row["sha256"], source_origin_seconds=origin,
                 wav_sha256=sha256(directory / "audio.wav"), sample_rate=16000,
                 audio_duration=duration, audio_offset_seconds=float(ap[0] - origin),
                 video_pts=(pts - origin).tolist(), video_intervals=intervals.tolist(),
                 source_video_pts=pts.tolist(), video_frames=len(pts),
                 clip_duration=max(float(ends[-1] - origin), float(ap[0] - origin + duration)),
                 time_coordinate="source presentation seconds minus source_origin_seconds",
                 original_video_stream=video["streams"][0], original_audio_stream=audio["streams"][0])
    write_json(output, media)
    return media


def decode_frames(config, row, directory, media):
    directory = Path(directory)
    frames = directory / "frames"
    frames.mkdir(exist_ok=True)
    marker = frames / "complete.json"
    if marker.exists():
        if len(list(frames.glob("*.png"))) == media["video_frames"]:
            return frames
        raise ValueError("Frame cache is incomplete")
    # On retries, matching names are overwritten; source/config changes use another run directory.
    execute([config["ffmpeg"], "-nostdin", "-y", "-v", "error", "-i",
             Path(config["data_root"]) / row["relative_path"], "-map", "0:v:0", "-an",
             "-fps_mode", "passthrough", "-start_number", "1", frames / "%08d.png"],
            directory / "commands.log")
    count = len(list(frames.glob("*.png")))
    if count != media["video_frames"]:
        raise ValueError(f"Decoded frame count {count} != ffprobe count {media['video_frames']}")
    write_json(marker, dict(count=count, source_sha256=row["sha256"]))
    return frames
