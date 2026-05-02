"""
VLM Best Image Picker — ComfyUI custom node.

Single node that scores a set of candidate images via an Ollama-served VLM
and returns the best one along with a Markdown ranking log.

Two input modes (mutually exclusive, IMAGE batch wins if both given):

1. ``image_dir`` (STRING) — scan a directory on disk.
2. ``images`` (IMAGE) — score an in-graph IMAGE batch (e.g. crops produced
   upstream by another node). ``filenames`` (optional) gives display names;
   otherwise falls back to ``image_0``, ``image_1``, …

Both modes support ``start_index`` / ``max_count`` to slice the candidate
list (useful when iterating chunks of a large directory or batch).

The scoring criteria are entirely controlled by the ``prompt`` widget; the
model is expected to respond with a JSON object containing at least a
``score`` (integer 0-10). Optional booleans ``frontal``, ``fullbody`` are
used for tie-breaking when multiple candidates share the top score.
"""
import base64
import fnmatch
import io
import json
import os
import re
import time
import urllib.error
import urllib.request

import numpy as np
import torch
from PIL import Image


def _ollama_generate(url, model, prompt, images_b64, timeout, num_predict=200, keep_alive=None):
    """Raw HTTP call to Ollama's /api/generate. Avoids the official python
    SDK because some versions raise ResponseError('') on transient 502s
    during model cold-load instead of returning a usable error."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": num_predict},
    }
    if images_b64:
        payload["images"] = images_b64
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return json.loads(data).get("response", "")


SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

DEFAULT_PROMPT = (
    "请按以下维度对这张图片打分（0-10）：人物正面、全身可见、主推服装无遮挡、构图饱满、画面清晰。"
    '直接输出 JSON 不带其他文字：{"score":整数0-10,"frontal":true/false,"fullbody":true/false,"reason":"15字内"}'
)


def _list_images(image_dir):
    if not os.path.isdir(image_dir):
        return []
    return sorted(
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(SUPPORTED_EXTS)
    )


def _pil_to_tensor(img):
    img = img.convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None,]


def _tensor_to_pil(t):
    """Convert a single [H, W, C] float tensor in 0-1 to a PIL RGB image."""
    arr = (t.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        return Image.fromarray(arr, "RGBA").convert("RGB")
    return Image.fromarray(arr, "RGB")


def _pil_to_b64(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _extract_json(text):
    if not text:
        return None
    match = re.search(r"\{[^{}]*\}", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except Exception:
        return None


def _split_lines_or_commas(text):
    return [p.strip() for p in re.split(r"[,\n]+", text or "") if p.strip()]


class VLMBestImagePicker:
    """Pick the best image from a directory or IMAGE batch using an Ollama-served VLM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": DEFAULT_PROMPT,
                        "tooltip": "Scoring prompt. The model should output JSON with at least 'score' (0-10).",
                    },
                ),
                "url": (
                    "STRING",
                    {"default": "http://127.0.0.1:11434"},
                ),
                "model": (
                    "STRING",
                    {"default": "qwen2.5vl:7b"},
                ),
                "timeout_per_image": (
                    "INT",
                    {"default": 60, "min": 10, "max": 600, "step": 5},
                ),
                "tie_break": (
                    ["first", "frontal_fullbody", "shortest_reason"],
                    {
                        "default": "frontal_fullbody",
                        "tooltip": (
                            "Tie-break rule when multiple candidates share the top score. "
                            "first=stable order; frontal_fullbody=prefer frontal+fullbody=true; "
                            "shortest_reason=shortest reason wins."
                        ),
                    },
                ),
            },
            "optional": {
                "image_dir": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Absolute path to a directory of candidate images. Used when 'images' is not connected.",
                    },
                ),
                "images": (
                    "IMAGE",
                    {
                        "tooltip": "IMAGE batch input. If connected, image_dir is ignored. Each batch item is scored as one candidate.",
                    },
                ),
                "filenames": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Optional display names for the IMAGE batch (one per line or comma-separated). "
                            "If shorter than the batch, missing names fall back to image_0, image_1, …"
                        ),
                    },
                ),
                "start_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 99999,
                        "step": 1,
                        "tooltip": "Skip the first N candidates (after ignore_files filter, before max_count cap).",
                    },
                ),
                "max_count": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 99999,
                        "step": 1,
                        "tooltip": "Score at most N candidates (0 = no limit).",
                    },
                ),
                "ignore_files": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Filenames or glob patterns to skip (image_dir mode only — IMAGE batch ignores this). "
                            "One per line or comma-separated. Supports wildcards: '0*' matches anything starting with 0, "
                            "'*.png' matches all PNGs. Case-insensitive, basename-only match."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("best_image", "best_filename", "log_md", "best_index", "scores_json")
    FUNCTION = "pick_best"
    CATEGORY = "VLM"

    def pick_best(
        self,
        prompt,
        url,
        model,
        timeout_per_image,
        tie_break,
        image_dir="",
        images=None,
        filenames="",
        start_index=0,
        max_count=0,
        ignore_files="",
    ):
        # Build the candidate list. Each candidate is a dict carrying either
        # a PIL image (batch mode) or a path on disk (dir mode), plus its
        # original index in the post-filter list (for stable best_index).
        if images is not None and len(images) > 0:
            mode = "batch"
            names = _split_lines_or_commas(filenames)
            candidates = []
            for i in range(int(images.shape[0])):
                pil = _tensor_to_pil(images[i])
                fname = names[i] if i < len(names) else f"image_{i}"
                candidates.append(
                    {"_filename": fname, "_pil": pil, "_path": None, "_index_orig": i}
                )
            source_label = f"<IMAGE batch · {len(candidates)} item(s)>"
        else:
            mode = "dir"
            files = _list_images(image_dir)
            if not files:
                raise RuntimeError(
                    f"No images found in image_dir={image_dir!r} (and no IMAGE batch provided)."
                )
            patterns = [p.lower() for p in _split_lines_or_commas(ignore_files)]

            def _is_ignored(path):
                base = os.path.basename(path).lower()
                return any(fnmatch.fnmatch(base, pat) for pat in patterns)

            if patterns:
                kept = [p for p in files if not _is_ignored(p)]
                skipped = [os.path.basename(p) for p in files if _is_ignored(p)]
                print(
                    f"[VLMBestImagePicker] patterns={patterns} → skipped {len(skipped)}: {skipped}"
                )
                files = kept
                if not files:
                    raise RuntimeError(
                        f"All images filtered out by ignore_files patterns: {patterns}"
                    )
            candidates = [
                {
                    "_filename": os.path.basename(p),
                    "_pil": None,
                    "_path": p,
                    "_index_orig": i,
                }
                for i, p in enumerate(files)
            ]
            source_label = image_dir

        total_before_slice = len(candidates)
        if start_index:
            candidates = candidates[start_index:]
        if max_count:
            candidates = candidates[:max_count]
        if not candidates:
            raise RuntimeError(
                f"No candidates after applying start_index={start_index}, max_count={max_count} "
                f"to {total_before_slice} item(s)."
            )

        # Warmup: pre-load the model into VRAM with keep_alive. The first
        # request after a cold start can return 502 if the HTTP layer becomes
        # briefly unreachable while the GGUF is being mmapped.
        for attempt in range(3):
            try:
                _ollama_generate(url, model, "", [], timeout=120, num_predict=1, keep_alive="30m")
                break
            except Exception as e:
                print(f"[VLMBestImagePicker] warmup attempt {attempt + 1} failed: {e!r}")
                time.sleep(5 + attempt * 5)

        def _call_with_retry(img_b64, retries=2):
            last_err = None
            for k in range(retries + 1):
                try:
                    return _ollama_generate(url, model, prompt, [img_b64], timeout=timeout_per_image)
                except Exception as e:
                    last_err = e
                    if k < retries:
                        time.sleep(2 + k * 3)
            raise last_err

        results = []
        for idx, c in enumerate(candidates):
            t0 = time.time()
            if c["_pil"] is not None:
                img_b64 = _pil_to_b64(c["_pil"], fmt="PNG")
            else:
                with open(c["_path"], "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode()
            try:
                raw = _call_with_retry(img_b64)
            except Exception as e:
                raw = f'{{"score": -1, "reason": "ERR: {e!r}"}}'

            wall = round(time.time() - t0, 1)
            parsed = _extract_json(raw) or {"score": 0, "reason": "parse fail"}
            parsed.setdefault("frontal", False)
            parsed.setdefault("fullbody", False)
            parsed.setdefault("reason", "")
            parsed["_filename"] = c["_filename"]
            parsed["_path"] = c["_path"]
            parsed["_pil"] = c["_pil"]
            parsed["_wall_s"] = wall
            parsed["_index_orig"] = c["_index_orig"]
            results.append(parsed)
            print(
                f"[VLMBestImagePicker] [{idx + 1}/{len(candidates)}] "
                f"{c['_filename']} ({wall}s) -> {raw[:160]}"
            )

        def sort_key(r):
            base = -float(r.get("score", 0) or 0)
            if tie_break == "frontal_fullbody":
                penalty = 0 if (r.get("frontal") and r.get("fullbody")) else 1
                return (base, penalty)
            if tie_break == "shortest_reason":
                return (base, len(r.get("reason", "") or ""))
            return (base, 0)

        ranked = sorted(results, key=sort_key)
        best = ranked[0]
        best_idx = best["_index_orig"]

        if best["_pil"] is not None:
            best_pil = best["_pil"]
        else:
            best_pil = Image.open(best["_path"])
        best_tensor = _pil_to_tensor(best_pil)

        slice_note = ""
        if start_index or max_count:
            slice_note = (
                f" (sliced from {total_before_slice} via "
                f"start_index={start_index}, max_count={max_count})"
            )
        lines = [
            "## VLM Best Image Picker",
            f"- Source: `{source_label}`",
            f"- Mode: `{mode}`",
            f"- Model: `{model}`",
            f"- Candidates scored: {len(candidates)}{slice_note}",
            f"- Total time: {round(sum(r['_wall_s'] for r in results), 1)}s",
            "",
            (
                f"**Selected: `{best['_filename']}`** "
                f"(score={best.get('score')}, "
                f"frontal={best.get('frontal')}, fullbody={best.get('fullbody')})"
            ),
            "",
            "| Rank | File | Score | Frontal | Fullbody | Reason | Time |",
            "|---|---|---|---|---|---|---|",
        ]
        for i, r in enumerate(ranked, 1):
            lines.append(
                f"| {i} | {r['_filename']} | {r.get('score')} | "
                f"{'Y' if r.get('frontal') else 'N'} | "
                f"{'Y' if r.get('fullbody') else 'N'} | "
                f"{str(r.get('reason', ''))[:30]} | {r['_wall_s']}s |"
            )
        log_md = "\n".join(lines)

        # Drop non-serializable PIL refs before dumping JSON.
        clean_results = [{k: v for k, v in r.items() if k != "_pil"} for r in results]
        scores_json = json.dumps(clean_results, ensure_ascii=False, indent=2)

        return (best_tensor, best["_filename"], log_md, best_idx, scores_json)


NODE_CLASS_MAPPINGS = {
    "VLMBestImagePicker": VLMBestImagePicker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VLMBestImagePicker": "VLM·选最佳图 (Best Picker)",
}
