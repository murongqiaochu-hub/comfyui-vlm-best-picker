"""
VLM Best Image Picker — ComfyUI custom node.

Single node that scans a directory of candidate images, scores each via an
Ollama-served VLM, and outputs the best one along with a Markdown ranking log.

The scoring criteria are entirely controlled by the `prompt` widget; the model
is expected to respond with a JSON object containing at least a `score`
(integer 0-10). Optional booleans `frontal`, `fullbody` are used for
tie-breaking when multiple candidates share the top score.
"""
import base64
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


class VLMBestImagePicker:
    """Pick the best image from a directory using an Ollama-served VLM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_dir": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Absolute path to a directory containing candidate images.",
                    },
                ),
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
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("best_image", "best_filename", "log_md", "best_index", "scores_json")
    FUNCTION = "pick_best"
    CATEGORY = "VLM"

    def pick_best(self, image_dir, prompt, url, model, timeout_per_image, tie_break):
        files = _list_images(image_dir)
        if not files:
            raise RuntimeError(f"No images found in: {image_dir}")

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
        for idx, path in enumerate(files):
            t0 = time.time()
            with open(path, "rb") as f:
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
            parsed["_filename"] = os.path.basename(path)
            parsed["_path"] = path
            parsed["_wall_s"] = wall
            results.append(parsed)
            print(
                f"[VLMBestImagePicker] [{idx + 1}/{len(files)}] "
                f"{os.path.basename(path)} ({wall}s) -> {raw[:160]}"
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
        best_idx = files.index(best["_path"])

        best_pil = Image.open(best["_path"])
        best_tensor = _pil_to_tensor(best_pil)

        lines = [
            "## VLM Best Image Picker",
            f"- Directory: `{image_dir}`",
            f"- Model: `{model}`",
            f"- Candidates: {len(files)}",
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

        scores_json = json.dumps(results, ensure_ascii=False, indent=2)

        return (best_tensor, best["_filename"], log_md, best_idx, scores_json)


NODE_CLASS_MAPPINGS = {
    "VLMBestImagePicker": VLMBestImagePicker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VLMBestImagePicker": "VLM·选最佳图 (Best Picker)",
}
