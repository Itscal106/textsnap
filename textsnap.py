#!/usr/bin/env python3
"""
textsnap.py - Lean CPU OCR using PaddleOCR-VL-1.5 ONNX (q4).

Snap any image, screenshot, or webpage into plaintext. No GPU. No cloud.
One command.

Usage:
    textsnap                          # OCR image from clipboard
    textsnap path/to/img.jpg          # OCR a local image file
    textsnap https://.../x.png        # OCR a direct image URL
    textsnap https://example.com/page # OCR the biggest image on a webpage

Options:
    --plaintext     Strip markdown -> plain text (default output is the model's
                    native markdown).
    -o, --output    Output .txt path. Default: ./textsnaps/<name>_ocr.txt.
    --model-dir DIR Use ONNX/config files from DIR instead of downloading.
    --max-tokens N  Cap generated tokens (default 2048).
    --max-pixels N  Image pixel budget for the vision encoder (default is the
                    model's max). Lower trades accuracy for speed.

Output:
    Plaintext, UTF-8. Default location is ./textsnaps/ (created if missing)
    under the current working directory; override with -o. The filename is
    "<name>_ocr.txt", where <name> is the image filename stem (for image
    inputs) or the webpage slug (for HTML inputs).

    When the input image comes from the clipboard (textsnap run with no
    arguments), the OCR text is ALSO copied back to the clipboard so it can
    be pasted immediately -- the .txt file is still written as well.

Model files:
    The 3 ONNX components (~890 MB) are auto-downloaded on first run and
    cached in ~/.cache/textsnap. The vision encoder and decoder use the q4
    variants; the embedding model ships fp32 only (no q4 exists in the repo).

    Portable mode: if the model files are found next to this script
    (./onnx/*.onnx + ./tokenizer.json), they are used directly -- no
    download, no --model-dir flag, no setup. Copy the textsnap folder
    together with its model files to any machine and run it offline.
"""

import sys
import os
import io
import re
import hashlib
import subprocess
import argparse
from pathlib import Path
from urllib.parse import urlparse, unquote

# --------------------------------------------------------------------------
# Logging: all diagnostics go to stderr and are silent unless -v is passed.
# stdout is reserved for the one thing a Unix pipe wants -- the output path.
# --------------------------------------------------------------------------
VERBOSE = False


def log(*args, **kwargs):
    """Print a [textsnap] diagnostic to stderr, but only when verbose."""
    if VERBOSE:
        kwargs.setdefault("file", sys.stderr)
        print(*args, **kwargs)

# --------------------------------------------------------------------------
# Thread env vars: must be set BEFORE numpy / onnxruntime import their native
# math backends (OpenMP / MKL / OpenBLAS), otherwise these are ignored.
# We pin them to physical-core count so the BLAS pool does not fight ORT's
# own intra-op thread pool (double-booking cores = cache thrash, slower).
# --------------------------------------------------------------------------
def _early_core_estimate():
    try:
        phys = set()
        cur = {}
        with open("/proc/cpuinfo") as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    cur[k.strip()] = v.strip()
                elif line.strip() == "":
                    if "physical id" in cur and "core id" in cur:
                        phys.add((cur["physical id"], cur["core id"]))
                    cur = {}
        if phys:
            return len(phys)
    except Exception:
        pass
    return max(1, (os.cpu_count() or 4) // 2)


_NTHREADS = str(_early_core_estimate())
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _NTHREADS)
os.environ.setdefault("OMP_WAIT_POLICY", "ACTIVE")   # keep threads hot

# --------------------------------------------------------------------------
# 0. Dependency bootstrap -- install everything inline so the script "just runs"
# --------------------------------------------------------------------------
REQUIRED = {
    # import name : pip spec
    "numpy": "numpy",
    "PIL": "pillow",
    "onnxruntime": "onnxruntime",
    "huggingface_hub": "huggingface_hub",
    "requests": "requests",
    "tokenizers": "tokenizers",
    "psutil": "psutil",
    "bs4": "beautifulsoup4",
    "readability": "readability-lxml",
    "lxml": "lxml",
}


def _ensure_deps():
    missing = []
    for mod, pkg in REQUIRED.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        # This runs before argparse, so it can't honor -v. Send it to stderr
        # (never stdout) -- it only appears on a first run with missing deps.
        print(f"[textsnap] Installing missing packages: {', '.join(missing)}",
              file=sys.stderr)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", *missing]
        )
    # Clipboard support is optional/platform dependent; install lazily later.


_ensure_deps()

import numpy as np  # noqa: E402
from PIL import Image, ImageGrab  # noqa: E402
import onnxruntime as ort  # noqa: E402
import requests  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

# --------------------------------------------------------------------------
# 1. Constants from PaddleOCR-VL-1.5 config (verified against HF repo)
# --------------------------------------------------------------------------
HF_REPO = "onnx-community/PaddleOCR-VL-1.5-ONNX"
# Pin the model revision so a moved/retagged 'main' can't silently swap weights
# out from under the checksums in model_checksums.sha256.
HF_REVISION = "main"
CACHE_DIR = Path(os.path.expanduser("~/.cache/textsnap"))

# SHA-256 manifest of known-good model files. Shipped alongside the script;
# also located next to the module after install. Empty/absent -> verification
# is skipped with a warning (never blocks a run).
CHECKSUM_MANIFEST = "model_checksums.sha256"

# Embedded fallback digests for onnx-community/PaddleOCR-VL-1.5-ONNX @ main.
# Used when the external model_checksums.sha256 is not found (e.g. a wheel
# install that didn't carry the data file). The external file, if present,
# takes precedence -- it's the source of truth and is easy to regenerate.
EMBEDDED_CHECKSUMS = {
    "onnx/vision_encoder_q4.onnx":
        "d737d600be1bd90ec1e3b537ffe1645a6d780de688904ca4301353df6086f46e",
    "onnx/decoder_q4.onnx":
        "87858a011c3f5ae8b373ec7298fba781dfe3ceb49828a803a197becdee26853c",
    "onnx/embedding.onnx":
        "91b1babbe9dbc44f2b59f8462cbf27dd1520a88b1b85695e342b05e5b4a50004",
    "tokenizer.json":
        "c8a215a59183d0d0781adc33bacd3ce6162716f7fd568fb30234a74d69803a7d",
    "config.json":
        "164809b94c8dd5b352cb9a0b9964572844398faeab27f9a6e1dd7d1a984410c8",
}

PATCH_SIZE = 14
MERGE_SIZE = 2
FACTOR = PATCH_SIZE * MERGE_SIZE          # 28
TEMPORAL_PATCH = 1
IMAGE_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
IMAGE_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
MIN_PIXELS = 112896
MAX_PIXELS = 1003520

IMAGE_TOKEN_ID = 100295
VISION_START_ID = 101305
VISION_END_ID = 101306
EOS_TOKEN_ID = 2
PAD_TOKEN_ID = 0

# Decoder architecture (for KV-cache tensor shapes)
NUM_LAYERS = 18
NUM_KV_HEADS = 2
HEAD_DIM = 128
HIDDEN_SIZE = 1024
MROPE_SECTION = [16, 24, 24]

# The OCR prompt. PaddleOCR-VL is trained for document parsing; this is the
# generic full-page parse instruction used by the reference pipeline.
OCR_PROMPT = "OCR:"

# ONNX filenames (q4 where available; embedding is fp32-only)
ONNX_FILES = [
    "onnx/vision_encoder_q4.onnx",
    "onnx/vision_encoder_q4.onnx_data",   # may or may not exist; handled below
    "onnx/decoder_q4.onnx",
    "onnx/embedding.onnx",
    "onnx/embedding.onnx_data",
]
TOKENIZER_FILE = "tokenizer.json"


# --------------------------------------------------------------------------
# 2. Input detection: figure out what the positional arg is
# --------------------------------------------------------------------------
def detect_input(arg):
    """
    Returns (kind, value) where kind is one of:
        'clipboard'  -> value is None
        'file'       -> value is a Path
        'image_url'  -> value is a URL string
        'html_url'   -> value is a URL string
    """
    if arg is None:
        return "clipboard", None

    p = Path(arg)
    if p.exists() and p.is_file():
        return "file", p

    parsed = urlparse(arg)
    if parsed.scheme in ("http", "https"):
        # Probe the URL: Content-Type header is the source of truth.
        ctype = ""
        try:
            head = requests.head(arg, allow_redirects=True, timeout=15)
            ctype = head.headers.get("Content-Type", "").lower()
            # Some servers don't answer HEAD usefully; fall back to a ranged GET.
            if not ctype or head.status_code >= 400:
                g = requests.get(arg, stream=True, timeout=15,
                                 headers={"Range": "bytes=0-0"})
                ctype = g.headers.get("Content-Type", "").lower()
                g.close()
        except requests.RequestException:
            pass

        if ctype.startswith("image/"):
            return "image_url", arg
        if "html" in ctype or "xml" in ctype:
            return "html_url", arg
        # Ambiguous / no content-type: fall back to extension heuristic.
        ext = Path(parsed.path).suffix.lower()
        if ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"):
            return "image_url", arg
        return "html_url", arg

    # Not a URL, not an existing file.
    raise SystemExit(f"[textsnap] '{arg}' is neither an existing file nor a "
                     f"http(s) URL.")


# --------------------------------------------------------------------------
# 3. Load image from each input kind
# --------------------------------------------------------------------------
def _download_bytes(url):
    r = requests.get(url, timeout=60, headers={"User-Agent": "textsnap/1.0"})
    r.raise_for_status()
    return r.content


def load_from_clipboard():
    try:
        img = ImageGrab.grabclipboard()
    except Exception as e:
        raise SystemExit(f"[textsnap] Could not read clipboard: {e}\n"
                         "On Linux you may need 'xclip' or 'wl-clipboard' "
                         "installed.")
    if img is None:
        raise SystemExit("[textsnap] No image found in clipboard.")
    if isinstance(img, list):
        # Clipboard held file path(s) rather than raw image data.
        paths = [Path(x) for x in img if Path(x).is_file()]
        if not paths:
            raise SystemExit("[textsnap] Clipboard holds no usable image.")
        img = Image.open(paths[0])
    return img.convert("RGB"), "clipboard"


def load_from_file(path):
    return Image.open(path).convert("RGB"), path.stem


def copy_text_to_clipboard(text):
    """Best-effort: put `text` on the system clipboard. Returns True on
    success, False otherwise. Never raises -- clipboard-out is a convenience,
    not a contract, so a failure here must not fail the run.

    Tries platform-native tools so it works without extra Python deps:
      macOS   -> pbcopy
      Windows -> clip
      Linux   -> wl-copy (Wayland) or xclip / xsel (X11)
    """
    data = text.encode("utf-8")
    if sys.platform == "darwin":
        cmds = [["pbcopy"]]
    elif sys.platform.startswith("win"):
        cmds = [["clip"]]
    else:
        cmds = [["wl-copy"], ["xclip", "-selection", "clipboard"],
                ["xsel", "--clipboard", "--input"]]
    for cmd in cmds:
        try:
            p = subprocess.run(cmd, input=data,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            if p.returncode == 0:
                return True
        except FileNotFoundError:
            continue   # tool not installed -- try the next one
        except Exception:
            continue
    return False


def load_from_image_url(url):
    data = _download_bytes(url)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    stem = Path(unquote(urlparse(url).path)).stem or "image"
    return img, stem


def load_from_html_url(url):
    """Use readability-lxml to isolate main content, then pick the most
    prominent <img> from that cleaned region."""
    from readability import Document
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin

    html = requests.get(url, timeout=60,
                         headers={"User-Agent": "textsnap/1.0"}).text
    doc = Document(html)
    main_html = doc.summary()           # de-fluffed main content
    title = doc.short_title() or urlparse(url).netloc

    soup = BeautifulSoup(main_html, "lxml")
    candidates = soup.find_all("img")

    # If readability stripped all images, fall back to the full page.
    if not candidates:
        soup = BeautifulSoup(html, "lxml")
        candidates = soup.find_all("img")

    def score(tag):
        """Prominence heuristic: prefer explicit large dimensions, then
        document order (earlier = more prominent)."""
        w = h = 0
        for attr in ("width", "height"):
            v = tag.get(attr, "")
            m = re.search(r"\d+", str(v))
            if m and attr == "width":
                w = int(m.group())
            if m and attr == "height":
                h = int(m.group())
        return w * h

    # Build ordered list of (score, order_index, src)
    scored = []
    for i, tag in enumerate(candidates):
        src = (tag.get("src") or tag.get("data-src")
               or tag.get("data-original") or "")
        if not src:
            continue
        # Skip obvious non-content images.
        if src.startswith("data:"):
            continue
        if re.search(r"(sprite|icon|logo|avatar|pixel|tracking|spacer|"
                     r"blank|1x1)", src, re.I):
            continue
        scored.append((score(tag), -i, urljoin(url, src)))

    if not scored:
        raise SystemExit("[textsnap] No usable image found on the page.")

    # Highest declared area wins; ties broken by earliest appearance.
    scored.sort(reverse=True)

    # If no element declared a size (all score 0), download top few and
    # measure real pixels to pick the biggest.
    if scored[0][0] == 0:
        best_img, best_px, best_src = None, -1, None
        for _, _, src in scored[:8]:
            try:
                data = _download_bytes(src)
                im = Image.open(io.BytesIO(data))
                px = im.width * im.height
                if px > best_px:
                    best_img, best_px, best_src = im, px, src
            except Exception:
                continue
        if best_img is None:
            raise SystemExit("[textsnap] Could not download any page image.")
        img = best_img.convert("RGB")
    else:
        src = scored[0][2]
        img = Image.open(io.BytesIO(_download_bytes(src))).convert("RGB")

    slug = re.sub(r"[^\w\-]+", "_", title).strip("_").lower()[:60] or "webpage"
    return img, slug


# --------------------------------------------------------------------------
# 4. (Resizing is handled entirely by smart_resize() in section 5, which
#    bounds the image to MAX_PIXELS / MIN_PIXELS and snaps to the patch grid.
#    There is deliberately no separate "cap the longest side" step -- pre-
#    shrinking on top of smart_resize only destroys resolution the vision
#    encoder could have used.)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# 5. PaddleOCR-VL image preprocessing (Qwen2-VL style smart_resize + patchify)
#    Mirrors image_processing_paddleocr_vl.py from the HF repo.
# --------------------------------------------------------------------------
def smart_resize(height, width, factor=FACTOR,
                 min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    import math
    if height < factor:
        width = round((width * factor) / height)
        height = factor
    if width < factor:
        height = round((height * factor) / width)
        width = factor
    if max(height, width) / min(height, width) > 200:
        raise ValueError("aspect ratio too extreme")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def preprocess_image(img, max_pixels=MAX_PIXELS):
    """Returns (pixel_values, grid_thw).

    pixel_values is produced in the canonical rank-5 layout
        (num_patches, channel, temporal_patch, patch, patch)
    which matches the PaddleOCR-VL / Qwen2-VL ONNX vision encoder export.
    run_ocr() adapts this to rank 2 or rank 4 if the loaded graph declares
    a different rank.
    """
    w, h = img.size
    rh, rw = smart_resize(h, w, max_pixels=max_pixels)
    img = img.resize((rw, rh), Image.BICUBIC)

    arr = np.asarray(img, dtype=np.float32) / 255.0       # rescale
    arr = (arr - IMAGE_MEAN) / IMAGE_STD                  # normalize
    arr = arr.transpose(2, 0, 1)                          # HWC -> CHW
    arr = arr[np.newaxis, ...]                            # (1, 3, H, W)

    # temporal tiling (temporal_patch_size == 1 here, so tile to 1)
    patches = np.tile(arr, (TEMPORAL_PATCH, 1, 1, 1))

    channel = patches.shape[1]
    grid_t = patches.shape[0] // TEMPORAL_PATCH
    grid_h = rh // PATCH_SIZE
    grid_w = rw // PATCH_SIZE

    patches = patches.reshape(
        grid_t, TEMPORAL_PATCH, channel,
        grid_h, PATCH_SIZE, grid_w, PATCH_SIZE,
    )
    patches = patches.transpose(0, 3, 5, 2, 1, 4, 6)
    # rank-5: (num_patches, channel, temporal_patch, patch, patch)
    pixel_values = patches.reshape(
        grid_t * grid_h * grid_w, channel, TEMPORAL_PATCH,
        PATCH_SIZE, PATCH_SIZE,
    ).astype(np.float32)
    return pixel_values, (grid_t, grid_h, grid_w)


def fit_pixel_values(pixel_values, declared_shape):
    """Reshape pixel_values (produced as (N, C, T, P, P)) to match the
    vision encoder's declared input shape.

    declared_shape is session_input.shape, a list mixing ints and strings
    (symbolic dims). The PaddleOCR-VL-1.5 ONNX export declares:
        [1, 'num_patches', 3, 14, 14]
    i.e. a leading batch axis, patches on axis 1, channel on axis 2, no
    temporal axis. We collapse the temporal axis (size 1) and re-place the
    batch axis as needed.
    """
    n = pixel_values.shape[0]
    c = pixel_values.shape[1]
    # Drop the temporal axis -> (N, C, P, P)
    pv = pixel_values.reshape(n, c, PATCH_SIZE, PATCH_SIZE)

    if not declared_shape:
        return pv

    rank = len(declared_shape)

    if rank == 5:
        # [batch, num_patches, C, P, P]  -> add leading batch axis
        return pv[np.newaxis, ...]                       # (1, N, C, P, P)
    if rank == 4:
        # [num_patches, C, P, P]
        return pv                                        # (N, C, P, P)
    if rank == 3:
        # [num_patches, C, P*P]  (rare)
        return pv.reshape(n, c, PATCH_SIZE * PATCH_SIZE)
    if rank == 2:
        # [num_patches, C*P*P]  (Qwen2-VL flattened style)
        return pv.reshape(n, -1)
    return pv


# --------------------------------------------------------------------------
# 6. Model download + integrity verification
# --------------------------------------------------------------------------
def _sha256_file(path, chunk=1 << 20):
    """Stream a file through SHA-256 without loading it all into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _find_checksum_manifest():
    """Locate model_checksums.sha256: next to this module, in CWD, or in the
    cache dir. Returns a Path or None."""
    here = Path(__file__).resolve().parent
    for cand in (here / CHECKSUM_MANIFEST,
                 Path.cwd() / CHECKSUM_MANIFEST,
                 CACHE_DIR / CHECKSUM_MANIFEST):
        if cand.is_file():
            return cand
    return None


def _load_checksums(manifest_path):
    """Parse a `sha256sum`-style manifest into {repo_path: sha256}."""
    sums = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        digest, name = parts[0].lower(), parts[1].strip().lstrip("*")
        sums[name] = digest
    return sums


def verify_files(file_map, checksums):
    """Verify downloaded files against the manifest.

    file_map: {repo_relative_name: local_path}
    checksums: {repo_relative_name: expected_sha256}

    Hard-fails (SystemExit) on any mismatch. Files with no manifest entry are
    reported but not fatal -- the manifest is the source of truth for *what*
    is pinned, and a partial manifest is still useful.
    """
    verified = 0
    for name, path in file_map.items():
        expected = checksums.get(name)
        if not expected:
            log(f"[textsnap]   (no pinned checksum for {name} -- skipped)")
            continue
        actual = _sha256_file(path)
        if actual != expected:
            raise SystemExit(
                f"[textsnap] CHECKSUM MISMATCH for {name}\n"
                f"             expected {expected}\n"
                f"             got      {actual}\n"
                f"[textsnap] Refusing to use a model file that does not match "
                f"the pinned digest. Delete {path} and re-run, or update "
                f"{CHECKSUM_MANIFEST} if you intend to use a new revision."
            )
        verified += 1
        log(f"[textsnap]   verified {name}")
    return verified


def _looks_like_model_dir(d):
    """True if `d` contains a usable model set (the 3 ONNX graphs + tokenizer).

    Used for 'portable mode': if the model files sit next to the script, we
    use them directly -- no download, no --model-dir, no setup. This lets a
    user copy the whole textsnap folder (script + model files) to any machine
    and run it offline immediately.
    """
    d = Path(d)
    needed = [
        d / "onnx" / "vision_encoder_q4.onnx",
        d / "onnx" / "decoder_q4.onnx",
        d / "onnx" / "embedding.onnx",
        d / "tokenizer.json",
    ]
    return all(p.is_file() for p in needed)


def _portable_model_dir():
    """Return the directory next to the textsnap script if it holds a model
    set, else None. Tries the module dir and (for frozen/symlinked installs)
    the resolved executable dir."""
    candidates = []
    try:
        candidates.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    # Frozen build (PyInstaller etc.): sys.executable is the binary.
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent)
    for c in candidates:
        if _looks_like_model_dir(c):
            return c
    return None


def get_model_dir(override=None, verify=True):
    if override:
        d = Path(override)
        if not d.exists():
            raise SystemExit(f"[textsnap] --model-dir {d} does not exist.")
        return d

    # Portable mode: model files sitting next to the script take precedence
    # over the OS cache. No flag needed -- copy the folder, run it anywhere.
    portable = _portable_model_dir()
    if portable is not None:
        log(f"[textsnap] Portable mode: using model files next to the "
            f"script ({portable}). Integrity check skipped -- locally "
            f"placed files are trusted, same as --model-dir.")
        return portable

    from huggingface_hub import hf_hub_download

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log("[textsnap] Ensuring ONNX model files are cached "
          "(~890 MB on first run)...")

    # Required files. The *_data sidecars only exist for models whose weights
    # exceed the 2 GB protobuf limit; we try them but tolerate 404s.
    required = [
        ("onnx/vision_encoder_q4.onnx", True),
        ("onnx/decoder_q4.onnx", True),
        ("onnx/embedding.onnx", True),
        ("tokenizer.json", True),
        ("config.json", True),
    ]
    optional_sidecars = [
        "onnx/vision_encoder_q4.onnx_data",
        "onnx/vision_encoder_q4.onnx.data",
        "onnx/decoder_q4.onnx_data",
        "onnx/decoder_q4.onnx.data",
        "onnx/embedding.onnx_data",
        "onnx/embedding.onnx.data",
    ]

    downloaded = {}   # repo-relative name -> local path, for verification
    for fname, _req in required:
        local = hf_hub_download(repo_id=HF_REPO, filename=fname,
                                revision=HF_REVISION,
                                local_dir=str(CACHE_DIR))
        downloaded[fname] = local
    for fname in optional_sidecars:
        try:
            hf_hub_download(repo_id=HF_REPO, filename=fname,
                            revision=HF_REVISION,
                            local_dir=str(CACHE_DIR))
        except Exception:
            pass   # sidecar not present for this file -- fine

    # ---- integrity check -------------------------------------------------
    if not verify:
        log("[textsnap] WARNING: --no-verify set -- skipping model integrity "
            "check.")
        return CACHE_DIR

    manifest = _find_checksum_manifest()
    if manifest is not None:
        checksums = _load_checksums(manifest)
        source = manifest.name
    else:
        checksums = dict(EMBEDDED_CHECKSUMS)
        source = "embedded digests"
        log(f"[textsnap] {CHECKSUM_MANIFEST} not found -- using embedded "
            f"digests.")

    if not checksums:
        log(f"[textsnap] WARNING: no checksums available -- skipping model "
            f"integrity verification.")
    else:
        log(f"[textsnap] Verifying model files against {source}...")
        n = verify_files(downloaded, checksums)
        log(f"[textsnap] Integrity OK ({n} files verified).")

    return CACHE_DIR


# --------------------------------------------------------------------------
# 7. ONNX session helpers -- introspect graph I/O so we bind by pattern,
#    not by hard-coded names (robust to export naming differences).
# --------------------------------------------------------------------------
def _physical_cores():
    """Best-effort physical (not logical/hyperthread) core count."""
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except Exception:
        pass
    # Linux: parse /proc/cpuinfo for distinct core ids.
    try:
        phys = set()
        cur = {}
        with open("/proc/cpuinfo") as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    cur[k.strip()] = v.strip()
                elif line.strip() == "":
                    if "physical id" in cur and "core id" in cur:
                        phys.add((cur["physical id"], cur["core id"]))
                    cur = {}
        if phys:
            return len(phys)
    except Exception:
        pass
    # Fallback: assume hyperthreading, halve logical count.
    log = os.cpu_count() or 4
    return max(1, log // 2)


_PHYS_CORES = _physical_cores()


def make_session(path, role="generic"):
    """Build an ORT session tuned for the model's access pattern.

    role:
        'vision'  -> big single parallel forward pass; use all phys cores.
        'decoder' -> autoregressive, latency-bound; oversubscription hurts,
                     so cap intra-op threads (<=4, env-overridable) and
                     disable mem_pattern (seq length grows every step).
        'embed'   -> trivial lookup; minimal threads.
    """
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.enable_cpu_mem_arena = True
    # mem_pattern pre-plans buffers assuming STATIC shapes. The decoder's
    # sequence axis grows by one every token, so the plan is invalidated each
    # step and ORT just pays the planning overhead for nothing. Enable it only
    # where shapes are actually static (vision encoder, embedding lookup).
    so.enable_mem_pattern = True

    if role == "vision":
        so.intra_op_num_threads = _PHYS_CORES
        so.inter_op_num_threads = 1
    elif role == "decoder":
        # Per-token decode is dominated by many SMALL GEMMs (batch=1, one
        # token). Beyond a handful of threads, cross-core synchronization
        # costs more than the parallel work saved -- classic oversubscription.
        # Default to <=4; let a deployment override via TEXTSNAP_DECODE_THREADS
        # since the sweet spot is CPU-dependent.
        _env = os.environ.get("TEXTSNAP_DECODE_THREADS")
        if _env and _env.isdigit() and int(_env) > 0:
            so.intra_op_num_threads = int(_env)
        else:
            so.intra_op_num_threads = max(1, min(_PHYS_CORES, 4))
        so.inter_op_num_threads = 1
        # Dynamic (growing) seq length defeats mem_pattern -- turn it off here.
        so.enable_mem_pattern = False
    elif role == "embed":
        so.intra_op_num_threads = min(_PHYS_CORES, 2)
        so.inter_op_num_threads = 1
    else:
        so.intra_op_num_threads = _PHYS_CORES
        so.inter_op_num_threads = 1

    return ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])


def _find(names, *keywords):
    """Return the first name containing all keywords (case-insensitive)."""
    for n in names:
        low = n.lower()
        if all(k in low for k in keywords):
            return n
    return None


# --------------------------------------------------------------------------
# 8. (Position IDs are handled internally by the decoder graph -- the ONNX
#    export has no position_ids input, so no mRoPE construction is needed
#    here. The decoder derives positions from attention_mask + cache length.)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# 9. The OCR inference pipeline
# --------------------------------------------------------------------------
def run_ocr(img, model_dir, max_tokens=2048, max_pixels=MAX_PIXELS):
    tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))

    vis_path = Path(model_dir) / "onnx" / "vision_encoder_q4.onnx"
    dec_path = Path(model_dir) / "onnx" / "decoder_q4.onnx"
    emb_path = Path(model_dir) / "onnx" / "embedding.onnx"

    log("[textsnap] Loading ONNX sessions...")
    vis = make_session(vis_path, role="vision")
    dec = make_session(dec_path, role="decoder")
    emb = make_session(emb_path, role="embed")

    # ---- preprocess image ----
    pixel_values, grid_thw = preprocess_image(img, max_pixels=max_pixels)
    grid_t, grid_h, grid_w = grid_thw

    # ---- vision encoder (run FIRST, before building the prompt) ----------
    # The encoder includes the spatial-merge projector, so its output row
    # count IS the number of image tokens the decoder expects. We must build
    # the prompt with exactly that many IMAGE_TOKEN_ID placeholders -- never
    # predict the count and truncate, which silently corrupts the visual
    # features and makes the model ignore the image.
    vis_in = {i.name: None for i in vis.get_inputs()}
    vin = list(vis_in.keys())
    pv_name = _find(vin, "pixel") or vin[0]
    grid_name = _find(vin, "grid")

    pv_input = next(i for i in vis.get_inputs() if i.name == pv_name)
    pixel_values = fit_pixel_values(pixel_values, pv_input.shape)

    feed = {pv_name: pixel_values}
    if grid_name:
        feed[grid_name] = np.array([[grid_t, grid_h, grid_w]], dtype=np.int64)
    log(f"[textsnap] Running vision encoder... "
          f"(pixel_values shape {pixel_values.shape})")
    image_embeds = vis.run(None, feed)[0]
    image_embeds = image_embeds.reshape(-1, HIDDEN_SIZE).astype(np.float32)

    # The actual, authoritative image-token count.
    n_image_tokens = image_embeds.shape[0]
    predicted = (grid_t * grid_h * grid_w) // (MERGE_SIZE * MERGE_SIZE)
    log(f"[textsnap] Image tokens: {n_image_tokens} "
          f"(grid {grid_t}x{grid_h}x{grid_w}, predicted {predicted})")

    # ---- build the prompt token sequence --------------------------------
    # Chat template:
    #   "<bos>User: <IMAGE_START><img placeholders><IMAGE_END>{prompt}\n
    #    Assistant:\n"
    prefix = "<|begin_of_sentence|>User: "
    pre_ids = tok.encode(prefix, add_special_tokens=False).ids
    img_open = tok.encode("<|IMAGE_START|>", add_special_tokens=False).ids
    img_close = tok.encode("<|IMAGE_END|>", add_special_tokens=False).ids
    suffix = f"{OCR_PROMPT}\nAssistant:\n"
    suf_ids = tok.encode(suffix, add_special_tokens=False).ids

    input_ids = np.array(
        [pre_ids + img_open + [IMAGE_TOKEN_ID] * n_image_tokens
         + img_close + suf_ids],
        dtype=np.int64,
    )
    seq_len = input_ids.shape[1]

    # ---- token embeddings ----
    emb_in = [i.name for i in emb.get_inputs()]
    emb_ids_name = _find(emb_in, "input") or emb_in[0]
    inputs_embeds = emb.run(None, {emb_ids_name: input_ids})[0]
    inputs_embeds = inputs_embeds.astype(np.float32)   # (1, seq, hidden)

    # ---- splice image embeddings into the placeholder positions ----------
    # By construction the counts now match exactly -- assert, do not patch.
    mask = (input_ids[0] == IMAGE_TOKEN_ID)
    if int(mask.sum()) != image_embeds.shape[0]:
        raise SystemExit(
            f"[textsnap] internal error: placeholder count "
            f"{int(mask.sum())} != image embed rows {image_embeds.shape[0]}"
        )
    inputs_embeds[0, mask, :] = image_embeds

    # ---- decoder: prefill + autoregressive decode ----
    dec_inputs = [i.name for i in dec.get_inputs()]
    dec_outputs = [o.name for o in dec.get_outputs()]

    emb_name = _find(dec_inputs, "inputs_embeds") or _find(dec_inputs, "embed")
    mask_name = _find(dec_inputs, "attention", "mask")
    logits_name = _find(dec_outputs, "logits") or dec_outputs[0]

    # --- KV cache wiring ---------------------------------------------------
    # The decoder declares 18 layers of past_key_values.{i}.{key,value} as
    # inputs and present.{i}.{key,value} as outputs. They MUST be paired by
    # numeric layer index. A plain lexical sort would order them
    # 0,1,10,11,...,17,2,3,... and feed layer 2's cache into layer 10 -- a
    # silent correctness bug that also wrecks performance. We sort by the
    # integer index parsed from the name.
    def _layer_idx(name):
        m = re.search(r"\.(\d+)\.", name)
        return int(m.group(1)) if m else -1

    past_names = sorted(
        [n for n in dec_inputs
         if "past" in n.lower() or "cache" in n.lower()],
        key=lambda n: (_layer_idx(n), "value" in n.lower()),
    )
    present_names = sorted(
        [n for n in dec_outputs if n.lower().startswith("present")
         or "present" in n.lower()],
        key=lambda n: (_layer_idx(n), "value" in n.lower()),
    )
    if len(past_names) != len(present_names):
        raise SystemExit("[textsnap] KV cache input/output count mismatch "
                          f"({len(past_names)} vs {len(present_names)})")

    # Map each present output -> the past input it feeds next step, by index.
    present_to_past = dict(zip(present_names, past_names))

    # KV cache dtype from the declared input type.
    kv_dtype = np.float32
    for inp in dec.get_inputs():
        if inp.name in past_names and "float16" in inp.type:
            kv_dtype = np.float16
            break

    def empty_cache():
        """Zero-length past KV tensors for the prefill pass."""
        return {
            name: np.zeros((1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=kv_dtype)
            for name in past_names
        }

    # --- IOBinding: bind I/O once, avoid marshaling the whole KV cache
    #     through Python dicts on every single token. -----------------------
    # Without binding, each dec.run() copies ~2*NUM_LAYERS cache tensors IN
    # and the same number OUT, per token -- thousands of array copies across
    # the C++/Python boundary over a full decode. IOBinding lets ORT keep the
    # present.* outputs on its own allocator and we hand those OrtValues
    # straight back as the next step's past.* inputs: zero-copy cache feedback.
    def decoder_step(embeds, attn_mask, past):
        """past: dict name -> OrtValue (or ndarray, for the prefill seed).

        Returns (logits ndarray, new_past dict name->OrtValue).
        embeds must already be float32; attn_mask must already be int64.
        """
        io = dec.io_binding()
        io.bind_cpu_input(emb_name, embeds)
        io.bind_cpu_input(mask_name, attn_mask)
        for name, val in past.items():
            if isinstance(val, np.ndarray):
                io.bind_cpu_input(name, val)
            else:
                # An OrtValue from the previous step -- bind it directly,
                # no copy back into Python.
                io.bind_ortvalue_input(name, val)
        # Let ORT allocate every output on its own CPU allocator so the
        # present.* tensors can be reused as next-step inputs without a copy.
        for oname in dec_outputs:
            io.bind_output(oname, "cpu")
        dec.run_with_iobinding(io)

        out_vals = io.get_outputs()             # list of OrtValue
        named = dict(zip(dec_outputs, out_vals))
        # logits is small relative to the cache and we need it on the host
        # for argmax -- materialize just this one.
        logits = named[logits_name].numpy()
        # present.* stay as OrtValues; re-key them to the past.* names they
        # feed next step. No numpy() call -> no copy.
        new_past = {present_to_past[p]: named[p] for p in present_names}
        return logits, new_past

    log(f"[textsnap] Decoding on {_PHYS_CORES} cores "
          f"(KV-cache enabled, IOBinding on, {len(past_names)//2} layers, "
          f"max {max_tokens} tokens)...")

    import time
    t0 = time.time()

    # --- Pre-allocate the attention mask once -----------------------------
    # The mask is all-ones and only grows; allocate it full-size up front and
    # feed a contiguous slice each step instead of np.ones()-ing every token.
    attn_buf = np.ones((1, seq_len + max_tokens), dtype=np.int64)

    # inputs_embeds is already float32 (set in section 9 above); no per-step
    # cast needed -- decoder_step now requires the correct dtype from us.

    # --- Prefill: process the full prompt once, populate the cache ---------
    logits, past = decoder_step(inputs_embeds, attn_buf[:, :seq_len],
                                empty_cache())
    next_id = int(np.argmax(logits[0, -1]))
    t_prefill = time.time() - t0
    log(f"[textsnap] Prefill done in {t_prefill:.1f}s; generating...")

    # --- Decode: one token at a time, feeding back the growing cache ------
    generated = []
    total = seq_len
    stop_reason = "max_tokens"
    last_print = time.time()

    def _looping(seq, min_run=40, max_period=60):
        """Detect a greedy-decoding repetition loop.

        Three cheap checks:
          * a single token id repeated >= min_run times in a row;
          * a short cycle (period 2..12) repeating for >= min_run tokens;
          * a long block (period up to max_period) repeating >= 3 times --
            this catches sentence-level loops (e.g. the same line of a page
            emitted over and over) that the short-period scan cannot see.
        Greedy argmax on dense or low-quality input frequently falls into
        these; without this guard it runs to max_tokens every time.
        """
        if len(seq) < min_run:
            return False
        tail = seq[-min_run:]
        if len(set(tail)) == 1:
            return True
        for period in range(2, 13):
            if len(seq) < period * 6:
                continue
            window = seq[-period * 6:]
            if all(window[i] == window[i % period]
                   for i in range(len(window))):
                return True
        # Long-period: does the last `period` tokens repeat >=3x back-to-back?
        for period in range(13, max_period + 1):
            if len(seq) < period * 3:
                continue
            window = seq[-period * 3:]
            if all(window[i] == window[i % period]
                   for i in range(len(window))):
                return True
        return False

    # --- no-repeat-ngram banning ------------------------------------------
    # Greedy decoding has no randomness, so once the model starts repeating
    # an n-gram it will repeat it forever -- the run only ends at max_tokens
    # (expensive) or when _looping() trips (after the loop already wasted
    # many tokens). Banning is preventive: before committing a token, if it
    # would complete an n-gram that already occurred, we forbid it and take
    # the next-best token instead. This is the standard no_repeat_ngram_size
    # from HF generate(). It stops a runaway at the FIRST repeat, not the
    # 50th, which is the single biggest wall-time win on hard inputs.
    NO_REPEAT_NGRAM = 4

    def _banned_next_tokens(seq):
        """Token ids that would complete a previously-seen NO_REPEAT_NGRAM-gram
        if appended to `seq`. Returns a set (usually empty or tiny)."""
        n = NO_REPEAT_NGRAM
        if len(seq) < n - 1:
            return ()
        prefix = tuple(seq[-(n - 1):])
        banned = set()
        # Scan all prior n-grams; small n + a few thousand tokens is cheap.
        for i in range(len(seq) - n + 1):
            if tuple(seq[i:i + n - 1]) == prefix:
                banned.add(seq[i + n - 1])
        return banned

    def _pick_next(logit_row, seq):
        """argmax over the vocab, with previously-seen n-grams masked out."""
        banned = _banned_next_tokens(seq)
        if not banned:
            return int(np.argmax(logit_row))
        # Copy only when we actually need to mutate -- keeps the common
        # (no-ban) path allocation-free.
        row = logit_row.copy()
        for t in banned:
            if 0 <= t < row.shape[0]:
                row[t] = -np.inf
        return int(np.argmax(row))

    for step in range(max_tokens):
        if next_id == EOS_TOKEN_ID:
            stop_reason = "EOS"
            break
        generated.append(next_id)
        total += 1

        # Repetition guard -- backstop for any loop the n-gram ban doesn't
        # prevent. _looping() is O(n) in the generated length; running it
        # every token makes the whole decode O(n^2), so check every 12th.
        if len(generated) % 12 == 0 and _looping(generated):
            stop_reason = "repetition loop"
            # Trim the looped tail so it doesn't pollute the output.
            while len(generated) > 1 and _looping(generated):
                generated.pop()
            break

        # Embed only the single new token (cheap lookup). The embedding graph
        # already emits float32; decoder_step needs float32 -- no cast.
        tok_embed = emb.run(
            None, {emb_ids_name: np.array([[next_id]], dtype=np.int64)}
        )[0]
        if tok_embed.dtype != np.float32:
            tok_embed = tok_embed.astype(np.float32)

        # attention_mask spans the full context so far (cached + new). Reuse
        # the pre-allocated all-ones buffer; just hand over a longer slice.
        logits, past = decoder_step(
            tok_embed, attn_buf[:, :total], past)
        # n-gram-aware token selection -- prevents greedy runaway loops.
        next_id = _pick_next(logits[0, -1], generated)

        # Live progress -- so a slow run is visibly alive, not hung.
        now = time.time()
        if now - last_print > 2.0:
            el = now - t0
            r = len(generated) / el if el > 0 else 0
            log(f"[textsnap]   ... {len(generated)} tokens "
                  f"({r:.1f} tok/s)", flush=True)
            last_print = now

    dt = time.time() - t0
    n = len(generated)
    rate = n / dt if dt > 0 else 0
    log(f"[textsnap] Stopped: {stop_reason}. "
          f"{n} tokens in {dt:.1f}s ({rate:.1f} tok/s)")
    if stop_reason == "max_tokens":
        log("[textsnap] NOTE: hit the token cap -- output may be "
              "truncated. Raise --max-tokens if the page is very dense.")

    text = tok.decode(generated, skip_special_tokens=True)
    return text.strip()


# --------------------------------------------------------------------------
# 10. Output formatting
# --------------------------------------------------------------------------
def to_plaintext(md):
    """Lightweight markdown -> plain text reduction."""
    t = md
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)          # images
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)      # links -> text
    t = re.sub(r"`{1,3}([^`]*)`{1,3}", r"\1", t)        # code spans
    t = re.sub(r"^#{1,6}\s*", "", t, flags=re.M)        # headings
    t = re.sub(r"(\*\*|__|\*|_)", "", t)                # bold/italic
    t = re.sub(r"^\s*[-*+]\s+", "", t, flags=re.M)      # bullet markers
    t = re.sub(r"^\s*>\s?", "", t, flags=re.M)          # blockquotes
    t = re.sub(r"^\s*\|", "", t, flags=re.M)            # table pipes (leading)
    t = re.sub(r"\|", " ", t)                           # remaining pipes
    t = re.sub(r"^\s*[-:|\s]+\s*$", "", t, flags=re.M)  # table rule rows
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# --------------------------------------------------------------------------
# 11. main
# --------------------------------------------------------------------------
def generate_checksums(dest=None):
    """Download the pinned model files and write a fresh model_checksums.sha256.

    Used to regenerate the manifest after deliberately moving to a new model
    revision. Writes next to this module by default.
    """
    from huggingface_hub import hf_hub_download

    files = [
        "onnx/vision_encoder_q4.onnx",
        "onnx/decoder_q4.onnx",
        "onnx/embedding.onnx",
        "tokenizer.json",
        "config.json",
    ]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# textsnap model checksums for {HF_REPO} @ {HF_REVISION}",
        "# Regenerate with: textsnap --generate-checksums",
    ]
    for fname in files:
        local = hf_hub_download(repo_id=HF_REPO, filename=fname,
                                revision=HF_REVISION, local_dir=str(CACHE_DIR))
        digest = _sha256_file(local)
        lines.append(f"{digest}  {fname}")
        print(f"{digest}  {fname}", file=sys.stderr)

    if dest is None:
        dest = Path(__file__).resolve().parent / CHECKSUM_MANIFEST
    dest = Path(dest)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(dest)


def main():
    ap = argparse.ArgumentParser(
        prog="textsnap",
        description="Lean CPU OCR via PaddleOCR-VL-1.5 ONNX (q4).",
    )
    ap.add_argument("input", nargs="?", default=None,
                    help="image file, image URL, or webpage URL. "
                         "Omit to read from the clipboard.")
    ap.add_argument("-o", "--output", default=None,
                    help="output .txt path. "
                         "Default: ./textsnaps/<name>_ocr.txt.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print progress diagnostics to stderr. "
                         "By default only the output path is printed.")
    ap.add_argument("--plaintext", action="store_true",
                    help="output plain text instead of native markdown.")
    ap.add_argument("--model-dir", default=None,
                    help="use ONNX/config files from this directory. "
                         "If omitted, textsnap uses model files found next "
                         "to the script (portable mode), else the OS cache "
                         "(downloading on first run).")
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="max generated tokens (default 2048).")
    ap.add_argument("--max-pixels", type=int, default=MAX_PIXELS,
                    help=f"image pixel budget fed to the vision encoder "
                         f"(default {MAX_PIXELS}). Lower = faster but less "
                         f"accurate; too low makes the model hallucinate. "
                         f"The image is only ever shrunk, never enlarged.")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip SHA-256 verification of downloaded model files "
                         "(not recommended).")
    ap.add_argument("--generate-checksums", action="store_true",
                    help="download the pinned model files, write a fresh "
                         "model_checksums.sha256, and exit.")
    args = ap.parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    if args.generate_checksums:
        VERBOSE = True
        generate_checksums()
        return

    kind, value = detect_input(args.input)
    log(f"[textsnap] Input type: {kind}")

    if kind == "clipboard":
        img, stem = load_from_clipboard()
    elif kind == "file":
        img, stem = load_from_file(value)
    elif kind == "image_url":
        img, stem = load_from_image_url(value)
    elif kind == "html_url":
        img, stem = load_from_html_url(value)
    else:
        raise SystemExit("[textsnap] unreachable input kind")

    log(f"[textsnap] Source image: {img.size[0]}x{img.size[1]}")
    # NOTE: do NOT pre-shrink here. preprocess_image() runs smart_resize(),
    # which bounds the image to MAX_PIXELS and snaps to the patch grid. An
    # extra cap on top of that just discards resolution the model could have
    # used -- and for text-dense screenshots, too-low resolution makes this
    # VLM hallucinate confident garbage rather than degrade gracefully.
    # The pixel budget is tunable via --max-pixels for an explicit
    # speed/accuracy trade; it is not silently forced.

    model_dir = get_model_dir(args.model_dir, verify=not args.no_verify)
    result = run_ocr(img, model_dir, max_tokens=args.max_tokens,
                     max_pixels=args.max_pixels)

    if args.plaintext:
        result = to_plaintext(result)

    if args.output:
        out_path = Path(args.output)
        if out_path.parent and not out_path.parent.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path.cwd() / "textsnaps"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}_ocr.txt"

    out_path.write_text(result, encoding="utf-8")
    log(f"[textsnap] Wrote {out_path}  ({len(result)} chars)")

    # Clipboard-in -> clipboard-out: if the image came from the clipboard,
    # put the OCR text straight back so the user can paste it immediately.
    # The .txt file is still written (the stdout-path contract is unchanged);
    # this is an added convenience. Best-effort -- a failure never aborts.
    if kind == "clipboard":
        if copy_text_to_clipboard(result):
            log("[textsnap] OCR text copied back to the clipboard.")
        else:
            log("[textsnap] Could not copy to clipboard "
                "(no pbcopy/clip/wl-copy/xclip/xsel found); "
                "text saved to the file above.")

    # The one line stdout is for: the path, bare, so `OUT=$(textsnap x.png)`
    # and `textsnap x.png | xargs cat` just work.
    print(out_path)


if __name__ == "__main__":
    main()
