# textsnap

> **Snap any image, screenshot, or webpage into plaintext. No GPU. No cloud. One command.**

![textsnap demo](demo-textsnap.jpg)

![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platforms](https://img.shields.io/badge/platforms-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)

```
textsnap screenshot.png
```

That's it. You get a `.txt` next to your shell, recognized on your CPU, from a screenshot, a photo, an image URL, or even a webpage.

---

## Why textsnap

- ⚡ **Runs on CPU.** A 0.9B PaddleOCR-VL-1.5 vision-language model, quantized to q4 ONNX, parses full pages on a plain laptop. No CUDA. No M-series-only tricks. Plain old cores, pinned to your physical-core count.
- 🖼 **Images, screenshots, URLs, webpages.** Point it at a local file, a direct image URL, or a full article URL — it isolates the main content and OCRs the most prominent image. Or OCR straight from your clipboard with no argument at all — and get the text put *back* on your clipboard, ready to paste.
- 📴 **Offline after first run.** ~890 MB of ONNX downloads once to your cache and stays there. No API keys. No quotas. Your images never leave your machine.
- 🎒 **Portable.** Drop the model files next to the script and the whole folder becomes a self-contained, copy-anywhere tool — no install, no download, no flags.
- 🪶 **One file.** The whole tool is a single Python module. Dependencies install themselves on first run if missing.
- 📝 **Markdown or plaintext.** Default output is the model's native markdown (tables, headings, structure preserved). Add `--plaintext` to flatten it.

---

## Quickstart

```
# Install
pip install textsnap

# Snap something
textsnap screenshot.png
textsnap https://example.com/article --plaintext
textsnap photo.jpg -o ~/notes/receipt.txt
```

The first run downloads the model (~890 MB). Every run after is offline.

---

## What it handles

| Source            | Example                                  |
| ----------------- | ---------------------------------------- |
| Clipboard         | `textsnap` *(no argument)*               |
| Local image file  | `textsnap path/to/img.png`               |
| Direct image URL  | `textsnap https://example.com/x.png`     |
| Webpage URL       | `textsnap https://example.com/article`   |

Local files cover anything Pillow can decode: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.gif`, `.tiff`, and friends. For webpage URLs, textsnap uses readability to isolate the main content, then picks the most prominent image on the page and OCRs that.

---

## Clipboard in, clipboard out

Run `textsnap` with **no argument** and it reads the image currently on your clipboard. The recognized text is then copied **straight back to the clipboard**, so a screenshot-to-text round trip is just: snap → `textsnap` → paste.

The `.txt` file is still written as well (and its path still printed to stdout), so nothing about scripting changes — the clipboard copy is a pure convenience layered on top.

Clipboard-out uses your platform's native tool — `pbcopy` (macOS), `clip` (Windows), or `wl-copy` / `xclip` / `xsel` (Linux) — so it needs no extra Python package. If none of those is installed, textsnap simply skips the clipboard copy; the `.txt` file is always there regardless. (Run with `-v` to see whether the copy succeeded.)

---

## Portable mode

By default textsnap downloads its model files to an OS cache directory (`~/.cache/textsnap/`). But if it finds the model files **sitting next to the script**, it uses those directly — no download, no `--model-dir` flag, no setup at all.

"Next to the script" means a layout like:

```
textsnap/
├── textsnap.py
├── onnx/
│   ├── vision_encoder_q4.onnx
│   ├── decoder_q4.onnx
│   └── embedding.onnx
└── tokenizer.json
```

Drop those files in, and you can copy the entire `textsnap/` folder to any machine — a USB stick, an air-gapped box, a fresh laptop — and run it immediately, fully offline, with zero install steps.

Model-directory resolution order:

1. `--model-dir DIR` — if you pass it explicitly, it always wins.
2. **Portable** — model files found next to the script.
3. **OS cache** — `~/.cache/textsnap/`, downloading on first run if needed.

> Like `--model-dir`, portable-mode files are **not** SHA-256 verified — files you placed there yourself are trusted by definition. Integrity verification applies to files textsnap *downloads*. See [Security](#security).

---

## Install

```
pip install textsnap
```

Installs two equivalent commands on your `PATH`: **`textsnap`** (canonical) and **`ocr`** (alias, for when the name slips your mind).

To install from a local source checkout instead:

```
pip install .
```

For a reproducible install with exact pinned dependency versions:

```
pip install -r requirements-lock.txt
pip install .
```

> **Clipboard note.** Reading images *from* the clipboard relies on Pillow's `ImageGrab`; on Linux you may need `xclip` or `wl-clipboard` installed. Writing recognized text *back* to the clipboard uses `pbcopy` / `clip` / `wl-copy` / `xclip` / `xsel`. macOS and Windows work out of the box.

---

## Usage

```
# Clipboard (no argument) — text is also copied back to the clipboard
textsnap

# Local image file
textsnap path/to/screenshot.png

# Direct image URL
textsnap "https://example.com/diagram.png"

# Webpage — OCRs the most prominent image on the page
textsnap "https://example.com/article"

# Flatten the model's markdown to plain text
textsnap input.png --plaintext

# Custom output path
textsnap input.png -o ./out/extracted.txt

# Raise the token cap for very dense pages
textsnap dense-page.png --max-tokens 4096

# Trade accuracy for speed by shrinking the image budget
textsnap input.png --max-pixels 250000

# Use a local model directory instead of downloading
textsnap input.png --model-dir ~/models/paddleocr-vl
```

---

## Output

Plaintext, UTF-8. Default location is `./textsnaps/` (created if missing) under the current working directory; override with `-o`. The filename is derived from the image filename stem (`receipt_ocr.txt`), or from the webpage slug for URL inputs.

textsnap is quiet by default, Unix-style: the **only** thing printed to stdout is the path to the file it wrote, so it composes cleanly —

```
OUT=$(textsnap receipt.png)   # capture the path
textsnap receipt.png | xargs cat   # print the recognized text
```

When the input is the clipboard, the recognized text is *also* placed on the clipboard — see [Clipboard in, clipboard out](#clipboard-in-clipboard-out).

Pass `-v` to send progress diagnostics (input type, image size, decode speed, token counts) to **stderr**; stdout stays just the path either way.

Default file output is the model's **native markdown** — it preserves tables, headings, and document structure:

```
# Quarterly Report

| Region | Revenue |
| ------ | ------- |
| EMEA   | $1.2M   |
| APAC   | $0.9M   |
```

With **`--plaintext`**, markdown is flattened to bare text:

```
Quarterly Report

Region Revenue
EMEA $1.2M
APAC $0.9M
```

---

## Flags

| Flag                  | Description                                                          |
| --------------------- | -------------------------------------------------------------------- |
| `-o`, `--output`      | Output `.txt` path. Default: `./textsnaps/<name>_ocr.txt`.           |
| `-v`, `--verbose`     | Print progress diagnostics to stderr. Off by default.                |
| `--plaintext`         | Flatten the model's native markdown to plain text.                   |
| `--model-dir`         | Use ONNX/config files from this directory. Overrides portable mode and the OS cache. |
| `--max-tokens`        | Cap generated tokens. Default `2048`. Raise it for very dense pages. |
| `--max-pixels`        | Image pixel budget fed to the vision encoder. Default is the model's maximum. Lower trades accuracy for speed; too low makes the model hallucinate. The image is only ever shrunk, never enlarged. |
| `--no-verify`         | Skip SHA-256 verification of downloaded model files (not advised).   |
| `--generate-checksums`| Download the pinned model files, write a fresh manifest, and exit.   |

An environment variable, `TEXTSNAP_DECODE_THREADS`, overrides the decoder's intra-op thread count if you want to tune CPU decode for a specific machine. Left unset, textsnap picks a sensible default based on your physical core count.

---

## Security

textsnap auto-downloads ~890 MB of model weights from the Hugging Face Hub on first run, so it treats those files as untrusted until proven otherwise:

- **Pinned model revision.** Downloads are pinned to a specific repo revision, so a moved or retagged `main` can't silently swap the weights.
- **SHA-256 verification.** Every downloaded file is hashed and checked against known-good digests before it's loaded. A mismatch aborts the run with a clear error rather than executing unverified weights. Digests live in [`model_checksums.sha256`](model_checksums.sha256) and are also embedded in the script as a fallback, so verification works whether you install from source or from a wheel.
- **Pinned dependencies.** [`requirements-lock.txt`](requirements-lock.txt) pins exact dependency versions for reproducible installs; the file documents how to add per-wheel `--hash` entries with `pip-compile --generate-hashes` for full supply-chain pinning.

Verification applies to files textsnap **downloads**. Model files you supply yourself — via `--model-dir` or [portable mode](#portable-mode) — are trusted as-is and not re-hashed; you are responsible for their provenance.

Regenerate the checksum manifest after a deliberate model-revision bump:

```
textsnap --generate-checksums
```

To bypass verification (for local experimentation with a modified model), pass `--no-verify`.

---

## How it works

1. **Load.** From the clipboard, a local file, a direct image URL, or — for a webpage URL — the most prominent image inside the page's main content (readability + a prominence heuristic).
2. **Preprocess.** The image is run through PaddleOCR-VL's Qwen2-VL-style smart-resize and patchify, producing the pixel-value tensor and grid the vision encoder expects. Smart-resize bounds the image to the model's pixel budget (tunable with `--max-pixels`) and snaps it to the patch grid — textsnap does not pre-shrink beyond that, since starving the encoder of resolution makes the model hallucinate rather than degrade gracefully.
3. **Recognize.** Three ONNX components run on CPU: a vision encoder (q4), a token-embedding model (fp32), and an autoregressive decoder (q4) with a wired-up KV cache bound via ONNX Runtime IOBinding to avoid copying the cache each step. Greedy decode, guarded against runaway repetition by an n-gram block (it refuses to re-emit an n-gram it has already produced) plus a loop detector that trims any cycle that slips through.
4. **Format.** Native markdown by default; `--plaintext` reduces it to bare text.

No image is sent anywhere. No state is kept between runs except the cached model.

---

## Model & cache

The PaddleOCR-VL-1.5 ONNX components are downloaded on first run to `~/.cache/textsnap/`:

- `onnx/vision_encoder_q4.onnx` — vision encoder + spatial-merge projector
- `onnx/decoder_q4.onnx` — autoregressive decoder
- `onnx/embedding.onnx` — token embeddings (fp32; no q4 variant exists)
- `tokenizer.json`, `config.json`

Together ~890 MB. To use your own copy, either point `--model-dir` at a directory containing the same `onnx/` files plus `tokenizer.json` and `config.json`, or place those files next to the script for [portable mode](#portable-mode).

---

## Notes & limits

- **First run is the slow one** — it downloads ~890 MB. After that, textsnap is fully offline.
- **CPU decode is sequential.** Dense, full-page documents take longer than a short screenshot. textsnap pins thread counts to your physical cores and prints a live tokens/sec readout so a slow run is visibly alive, not hung.
- **`--max-tokens` caps the output.** Very dense pages can hit the default 2048-token cap and truncate; raise it if the tail of a page is missing.
- **`--max-pixels` is a speed/accuracy dial.** Lowering it speeds up the vision encoder but feeds the model a coarser image; set it too low and recognition quality drops sharply. The default (the model's full budget) is the safe choice.
- **Webpage inputs OCR one image** — the most prominent one in the main content, not the whole rendered page.
- **Greedy decoding** can occasionally loop on repetitive layouts; an n-gram block prevents most loops outright and a detector trims any that remain.

---

## License

MIT for this project — see [LICENSE](LICENSE).

The model is **PaddleOCR-VL-1.5**, distributed under Apache-2.0 by PaddlePaddle; textsnap pulls the ONNX export from [`onnx-community/PaddleOCR-VL-1.5-ONNX`](https://huggingface.co/onnx-community/PaddleOCR-VL-1.5-ONNX). See the [original model card](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) for model terms. Powered by [onnxruntime](https://onnxruntime.ai/) and [huggingface_hub](https://github.com/huggingface/huggingface_hub).
