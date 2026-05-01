# comfyui-vlm-best-picker

A ComfyUI custom node that picks the **best image** from a directory of candidates by scoring each one with a vision-language model (VLM) served via [Ollama](https://ollama.com/).

Single node, single run. No batch-blowing context limits, no in-graph for-loops.

## Use case

You have a folder of N candidate images (product photos, screenshots, frames, etc.) and you want the workflow to automatically pick the most suitable one based on a *natural-language* scoring rubric you write. The node iterates over every file in the directory, queries the VLM once per image, parses the JSON score, and returns:

- `best_image` — IMAGE tensor of the top-scoring file
- `best_filename` — string filename
- `log_md` — Markdown ranking table (good for ShowText nodes)
- `best_index` — integer position in the sorted directory listing
- `scores_json` — full JSON dump of every candidate's parsed result

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/murongqiaochu-hub/comfyui-vlm-best-picker.git
# restart ComfyUI
```

No extra Python dependencies — the node uses stdlib `urllib` plus `numpy/torch/Pillow` already shipped with ComfyUI.

## Prereqs

- Ollama server reachable at `url` (default `http://127.0.0.1:11434`)
- A vision-capable model pulled, e.g.:
  ```bash
  ollama pull qwen2.5vl:7b
  ```

  Other vision models that work: `qwen2.5vl:3b`, `llava`, `gemma3` (if multimodal variant), `minicpm-v`. Avoid `qwen3-vl` for now — the current Ollama release ships it without a chat template, so thinking tokens swamp the output budget and the JSON never lands.

## Node inputs

| Input | Type | Default | Notes |
|---|---|---|---|
| `image_dir` | STRING | `""` | Absolute directory path. Reads `.jpg/.jpeg/.png/.webp/.bmp`. |
| `prompt` | STRING (multiline) | (Chinese e-commerce default) | Must instruct the model to output JSON with at least `score`. |
| `url` | STRING | `http://127.0.0.1:11434` | Ollama HTTP endpoint. |
| `model` | STRING | `qwen2.5vl:7b` | Any Ollama vision model tag. |
| `timeout_per_image` | INT | 60 | Seconds. Crank up for very large images / slow GPUs. |
| `tie_break` | enum | `frontal_fullbody` | `first` / `frontal_fullbody` / `shortest_reason`. |
| `ignore_files` | STRING (multiline) | `""` | Filenames or glob patterns to skip, one per line or comma-separated. Supports wildcards (`0*`, `*.png`, `IMG_47??.JPG`). Case-insensitive, basename-only. Skipped files do not call the model and do not appear in rankings. |

## Prompt contract

The model must respond with at least:

```json
{"score": 0-10}
```

The default prompt also asks for `frontal`, `fullbody`, and `reason` for nicer tie-breaking and logs, but those are optional — the node fills in defaults if missing. Replace the prompt with whatever rubric fits your task; only `score` is load-bearing.

If parsing fails for an image (model returned non-JSON), it gets `score=0` and `reason="parse fail"` and is ranked last.

## Speed

On an RTX 4060 Ti 16GB with `qwen2.5vl:7b` Q4_K_M:

- ~12 s per image (first call adds ~8 s for model load)
- 30 candidates ≈ 4 minutes
- Most of the time is the vision encoder; text generation itself is < 1 s

## License

MIT
