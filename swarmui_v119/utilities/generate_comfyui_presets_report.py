#!/usr/bin/env python3
"""
ComfyUI Presets Download Report Generator

Scans:
- ../ComfyUI_Presets/*.json (ComfyUI workflow presets)
- utilities/model_catalog_data.py (model & bundle catalog used by the downloader app)
- utilities/model_sizes.json (cached model/bundle sizes)
- last_settings.json (last used base path + folder structure flags)

Generates:
- A single modern HTML dashboard inside ../ComfyUI_Presets/

This script is intentionally self-contained (no external deps).
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PRESETS_DIR = REPO_ROOT / "ComfyUI_Presets"

LAST_SETTINGS_FILE = REPO_ROOT / "last_settings.json"
MODEL_SIZES_FILE = SCRIPT_DIR / "model_sizes.json"

DEFAULT_OUTPUT_HTML = PRESETS_DIR / "ComfyUI_Presets_Report.html"

# Cache of HuggingFace snapshot repo file listings (used to make bundle coverage
# robust when bundles include "snapshot" models that contain many files).
SNAPSHOT_REPO_FILES_CACHE = SCRIPT_DIR / "snapshot_repo_files_cache.json"
SNAPSHOT_REPO_FILES_CACHE_TTL_SECONDS = 30 * 24 * 60 * 60  # 30 days
SNAPSHOT_REPO_FILES_CACHE_VERSION = 2

# Extensions that frequently appear as on-disk model assets referenced in ComfyUI workflows.
MODEL_FILE_EXTS = (
    ".safetensors",
    ".ckpt",
    ".pt",
    ".pth",
    ".gguf",
    ".onnx",
    ".bin",
)

# Files that should NOT be shown in the report and should NOT affect bundle coverage.
# These are typically auto-downloaded by ComfyUI/SwarmUI or otherwise not part of your bundle catalog.
EXCLUDED_MODEL_BASENAMES_LC: Set[str] = {
    "rife47.pth",  # auto-downloaded interpolation model used by some Wan workflows
}


def _now_local_str() -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "Unknown"


def _norm_ref_path(s: str) -> str:
    # Normalize workflow references like "Flux\\ae.safetensors" -> "Flux/ae.safetensors"
    s = (s or "").strip()
    s = s.replace("\\", "/")
    s = re.sub(r"/+", "/", s)
    return s


def _looks_like_model_file(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return False
    s_norm = _norm_ref_path(s).lower()
    return any(s_norm.endswith(ext) for ext in MODEL_FILE_EXTS)


def _is_excluded_model_ref(ref_norm: str) -> bool:
    """
    Exclude by basename (case-insensitive).
    Accepts either a normalized ref ("Foo/bar.safetensors") or raw ("Foo\\bar.safetensors").
    """
    base = Path(_norm_ref_path(ref_norm)).name.lower()
    return base in EXCLUDED_MODEL_BASENAMES_LC


def _bytes_to_gb(num_bytes: int) -> float:
    try:
        return round(float(num_bytes) / (1024**3), 2)
    except Exception:
        return 0.0


def _fmt_gb(gb: Optional[float]) -> str:
    if gb is None:
        return "—"
    try:
        if gb <= 0:
            return "—"
        return f"{gb:.2f} GB"
    except Exception:
        return "—"


def _escape(s: Any) -> str:
    return html.escape("" if s is None else str(s), quote=True)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _walk_strings(obj: Any) -> Iterable[str]:
    """Recursively yield all string values inside a JSON-like object."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_strings(v)


@dataclass(frozen=True)
class CatalogModelKey:
    category: str
    sub_category: str
    name: str

    def to_size_key(self) -> str:
        return f"{self.category}::{self.sub_category}::{self.name}"


@dataclass
class CatalogModel:
    key: CatalogModelKey
    repo_id: Optional[str]
    filename_in_repo: Optional[str]
    save_filename: Optional[str]
    target_dir_key: Optional[str]
    is_snapshot: bool
    info: Optional[str] = None


@dataclass
class Bundle:
    bundle_key: str  # eg "SwarmUI Bundles::bundle_0"
    category: str
    name: str
    info: Optional[str]
    model_refs: List[CatalogModelKey]
    save_files: Set[str]  # normalized save_filename values
    size_gb: Optional[float] = None
    model_count: int = 0
    # Coverage helpers (computed later)
    cover_exact_lc: Set[str] = field(default_factory=set)  # normalized save paths, lowercased
    cover_base_lc: Set[str] = field(default_factory=set)  # basenames, lowercased
    snapshot_repo_ids: Set[str] = field(default_factory=set)  # repo_ids of snapshot entries in this bundle


@dataclass
class FileHit:
    path: Path
    size_bytes: int
    base_root: Path

    @property
    def size_gb(self) -> float:
        return _bytes_to_gb(self.size_bytes)


@dataclass
class RequiredFile:
    ref_norm: str  # normalized, using forward slashes
    ref_original: str
    node_types: Set[str]
    # Catalog matches (can be >1 when multiple catalog entries save to same filename)
    catalog_keys: List[CatalogModelKey]
    target_dir_keys: Set[str]
    expected_size_gb: Optional[float]
    # Download status
    present: bool
    expected_paths: List[Path]
    hits: List[FileHit]


@dataclass
class PresetReport:
    preset_file: Path
    preset_name: str
    parse_error: Optional[str]
    required: List[RequiredFile]


def _load_last_settings() -> Tuple[Optional[Path], bool, bool, bool]:
    """
    Returns: (base_path, comfy_ui_structure, forge_structure, lowercase_folders)
    """
    try:
        if LAST_SETTINGS_FILE.exists():
            raw = _read_json(LAST_SETTINGS_FILE)
            path_val = raw.get("path")
            base_path = Path(path_val) if path_val else None
            comfy = bool(raw.get("comfy_ui_structure", False))
            forge = bool(raw.get("forge_structure", False))
            lowercase = bool(raw.get("lowercase_folders", False))
            if base_path and base_path.is_dir():
                return base_path, comfy, forge, lowercase
    except Exception:
        pass
    return None, False, False, False


def _default_base_path() -> Path:
    # Mirror Downloader_Gradio_App.py default on Windows: os.getcwd()/SwarmUI/Models
    return REPO_ROOT / "SwarmUI" / "Models"


def _find_latest_amazing_presets_file(repo_root: Path) -> Optional[Path]:
    """
    Find the latest Amazing_SwarmUI_Presets_v*.json by numeric version.
    Checks repo root and repo_root/older_presets.
    """
    candidates: List[Path] = []
    try:
        candidates.extend(list(repo_root.glob("Amazing_SwarmUI_Presets_v*.json")))
    except Exception:
        pass
    try:
        candidates.extend(list((repo_root / "older_presets").glob("Amazing_SwarmUI_Presets_v*.json")))
    except Exception:
        pass

    best: Tuple[int, Path] | None = None
    pat = re.compile(r"^Amazing_SwarmUI_Presets_v(\d+)\.json$", re.IGNORECASE)
    for p in candidates:
        m = pat.match(p.name)
        if not m:
            continue
        try:
            v = int(m.group(1))
        except Exception:
            continue
        if best is None or v > best[0]:
            best = (v, p)
    return best[1] if best else None


def _normalize_title_key(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "").strip()).lower()


def _load_amazing_preset_descriptions(repo_root: Path) -> Tuple[Optional[Path], Dict[str, str]]:
    """
    Load title -> description from the latest Amazing_SwarmUI_Presets_v*.json.
    Returns (file_path, map) where map keys are normalized (lower, collapsed whitespace).
    """
    path = _find_latest_amazing_presets_file(repo_root)
    if not path or not path.exists():
        return None, {}
    try:
        raw = _read_json(path)
    except Exception:
        return path, {}

    out: Dict[str, str] = {}

    def _add(title: Any, desc: Any) -> None:
        if not isinstance(title, str):
            return
        if not isinstance(desc, str) or not desc.strip():
            return
        k = _normalize_title_key(title)
        out[k] = desc.strip()

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                _add(item.get("title"), item.get("description"))
    elif isinstance(raw, dict):
        # Some formats may use {"Title": {...preset...}} style dict.
        for k, v in raw.items():
            if isinstance(v, dict) and ("title" in v or "description" in v):
                _add(v.get("title", k), v.get("description"))
            elif isinstance(v, dict):
                # If key is title and value contains description
                _add(k, v.get("description"))

    return path, out


def _description_for_preset_name(
    preset_name: str,
    desc_map: Dict[str, str],
) -> Optional[str]:
    """
    Best-effort matching:
    - exact title match
    - strip trailing date suffix like ' - 260101'
    - strip trailing ' - 1 Images Input'/' - 2 Images Input'/etc.
    """
    name = (preset_name or "").strip()
    if not name:
        return None

    candidates: List[str] = [name]

    # Strip date suffix (6-8 digits) e.g. " - 260101" or " - 20260101"
    m = re.match(r"^(.*?)(?:\s*-\s*\d{6,8})\s*$", name)
    if m:
        candidates.append(m.group(1).strip())

    # Strip " - N Images Input" (ComfyUI variants)
    m2 = re.match(r"^(.*?)(?:\s*-\s*\d+\s+Images?\s+Input)\s*$", name, flags=re.IGNORECASE)
    if m2:
        candidates.append(m2.group(1).strip())

    # Try candidates in order (exact match)
    normalized_candidates: List[str] = []
    for c in candidates:
        k = _normalize_title_key(c)
        if k and k not in normalized_candidates:
            normalized_candidates.append(k)
        if k in desc_map:
            return desc_map[k]

    # Fallback: conservative substring matching (avoids wrong descriptions).
    # Helps map things like "FLUX Dev" -> "FLUX Dev Models" without matching unrelated titles.
    best_desc: Optional[str] = None
    best_score: int = -1

    for pn in normalized_candidates:
        if not pn or len(pn) < 4:
            continue
        for title_key, desc in desc_map.items():
            if not isinstance(title_key, str) or not isinstance(desc, str):
                continue

            score: Optional[int] = None
            if title_key == pn:
                score = 1000
            elif title_key.startswith(pn):
                score = 900
            elif pn in title_key:
                score = 800
            elif title_key in pn:
                score = 750

            if score is None:
                continue

            # Prefer closer length matches (fewer extra words)
            score -= abs(len(title_key) - len(pn))

            if score > best_score:
                best_score = score
                best_desc = desc

    return best_desc


def _get_current_subdirs(is_comfy_ui_structure: bool, is_forge_structure: bool, lowercase_folders: bool) -> Dict[str, str]:
    """
    Keep in sync with Downloader_Gradio_App.py (BASE_SUBDIRS + get_current_subdirs).
    """
    base_subdirs = {
        "vae": "VAE",
        "VAE": "VAE",
        "diffusion_models": "diffusion_models",
        "Stable-Diffusion": "Stable-Diffusion",
        "clip": "clip",
        "text_encoders": "clip",
        "clip_vision": "clip_vision",
        "yolov8": "yolov8",
        "style_models": "style_models",
        "Lora": "Lora",
        "upscale_models": "upscale_models",
        "LLM": "LLM",
        "Joy_caption": "Joy_caption",
        "clip_vision_google_siglip": "clip_vision/google--siglip-so400m-patch14-384",
        "LLM_unsloth_llama": "LLM/unsloth--Meta-Llama-3.1-8B-Instruct",
        "Joy_caption_monster_joy": "Joy_caption/cgrkzexw-599808",
        "controlnet": "controlnet",
        "model_patches": "controlnet",  # SwarmUI default
    }

    current = dict(base_subdirs)

    if is_comfy_ui_structure:
        current["Lora"] = "loras"
        current["loras"] = "loras"
        current["clip"] = "text_encoders"
        current["text_encoders"] = "text_encoders"
        current["vae"] = "vae"
        current["VAE"] = "vae"
        current["Embeddings"] = "embeddings"
        current["embeddings"] = "embeddings"
        current["Stable-Diffusion"] = "checkpoints"
        current["checkpoints"] = "checkpoints"
        current["diffusion_models"] = "diffusion_models"
        current["model_patches"] = "model_patches"
    elif is_forge_structure:
        current["Stable-diffusion"] = "Stable-diffusion"
        current["Stable-Diffusion"] = "Stable-diffusion"
        current["diffusion_models"] = "Stable-diffusion"

        current["vae"] = "VAE"
        current["VAE"] = "VAE"

        current["Lora"] = "Lora"
        current["lora"] = "Lora"
        current["loras"] = "Lora"

        current["clip"] = "text_encoder"
        current["text_encoder"] = "text_encoder"
        current["text_encoders"] = "text_encoder"
        current["clip_vision"] = "text_encoder"
        current["t5"] = "text_encoder"
        current["umt5"] = "text_encoder"

        current["controlnet"] = "ControlNet"
        current["ControlNet"] = "ControlNet"
        current["model_patches"] = "ControlNet"

        current["controlnetpreprocessor"] = "ControlNetPreprocessor"
        current["ControlNetPreprocessor"] = "ControlNetPreprocessor"
        current["preprocessor"] = "ControlNetPreprocessor"

        current["upscale_models"] = "ESRGAN"
        current["ESRGAN"] = "ESRGAN"
        current["RealESRGAN"] = "ESRGAN"
        current["BSRGAN"] = "ESRGAN"
        current["DAT"] = "ESRGAN"
        current["SwinIR"] = "ESRGAN"
        current["ScuNET"] = "ESRGAN"
        current["upscalers"] = "ESRGAN"

        current["embeddings"] = "embeddings"
        current["embedding"] = "embeddings"
        current["textual_inversion"] = "embeddings"

        current["diffusers"] = "diffusers"
        current["diffusion"] = "diffusers"

        current["Codeformer"] = "Codeformer"
        current["GFPGAN"] = "GFPGAN"
        current["BLIP"] = "BLIP"
        current["deepbooru"] = "deepbooru"
        current["hypernetworks"] = "hypernetworks"

    if lowercase_folders:
        current = {k: (v.lower() if isinstance(v, str) else v) for k, v in current.items()}
    return current


def _os_relpath_from_ref(norm_ref: str) -> str:
    # Convert "Flux/ae.safetensors" -> OS-specific relpath components.
    return norm_ref.replace("/", os.sep)


def _safe_stat_size(path: Path) -> Optional[int]:
    try:
        if path.is_file():
            return path.stat().st_size
    except Exception:
        return None
    return None


def _build_file_index(base_roots: Sequence[Path]) -> Tuple[Dict[str, List[FileHit]], Dict[str, List[FileHit]]]:
    """
    Returns:
      - by_basename_lower: { "file.safetensors": [FileHit,...] }
      - by_relpath_lower: { "vae/flux/ae.safetensors": [FileHit,...] } (relative to base_root)
    """
    by_basename: Dict[str, List[FileHit]] = {}
    by_relpath: Dict[str, List[FileHit]] = {}

    for base_root in base_roots:
        if not base_root or not base_root.exists():
            continue
        try:
            for root, _, files in os.walk(str(base_root)):
                for fname in files:
                    f_lower = fname.lower()
                    if not any(f_lower.endswith(ext) for ext in MODEL_FILE_EXTS):
                        continue
                    full_path = Path(root) / fname
                    size = _safe_stat_size(full_path)
                    if size is None:
                        continue
                    hit = FileHit(path=full_path, size_bytes=size, base_root=base_root)

                    by_basename.setdefault(f_lower, []).append(hit)

                    try:
                        rel = full_path.relative_to(base_root)
                        rel_norm = str(rel).replace("\\", "/").lower()
                        by_relpath.setdefault(rel_norm, []).append(hit)
                    except Exception:
                        # If relative_to fails (shouldn't), skip relpath indexing
                        pass
        except Exception:
            # If a base root can't be walked, ignore it
            continue

    return by_basename, by_relpath


def _load_catalog_and_bundles() -> Tuple[List[CatalogModel], List[Bundle]]:
    # Ensure repo root is importable (utilities/* imports expect this).
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from utilities.model_catalog_data import models_structure  # type: ignore

    models: List[CatalogModel] = []
    bundles: List[Bundle] = []

    for cat_name, cat_data in (models_structure or {}).items():
        # Models
        subcats = cat_data.get("sub_categories") if isinstance(cat_data, dict) else None
        if isinstance(subcats, dict):
            for sub_name, sub_data in subcats.items():
                if not isinstance(sub_data, dict):
                    continue
                sub_target_key = sub_data.get("target_dir_key")
                for model_info in sub_data.get("models", []) or []:
                    if not isinstance(model_info, dict):
                        continue
                    name = model_info.get("name")
                    if not name:
                        continue
                    model_key = CatalogModelKey(category=cat_name, sub_category=sub_name, name=str(name))
                    repo_id = model_info.get("repo_id")
                    filename_in_repo = model_info.get("filename_in_repo")
                    save_filename = model_info.get("save_filename") or model_info.get("filename_in_repo")
                    target_key = model_info.get("target_dir_key") or sub_target_key
                    is_snapshot = bool(model_info.get("is_snapshot", False))
                    info = model_info.get("info")
                    models.append(
                        CatalogModel(
                            key=model_key,
                            repo_id=repo_id,
                            filename_in_repo=filename_in_repo,
                            save_filename=save_filename,
                            target_dir_key=target_key,
                            is_snapshot=is_snapshot,
                            info=info,
                        )
                    )

        # Bundles
        if isinstance(cat_data, dict) and "bundles" in cat_data and isinstance(cat_data.get("bundles"), list):
            for i, bundle_info in enumerate(cat_data.get("bundles") or []):
                if not isinstance(bundle_info, dict):
                    continue
                b_name = bundle_info.get("name") or f"Bundle {i+1}"
                b_key = f"{cat_name}::bundle_{i}"
                b_info = bundle_info.get("info")
                model_refs: List[CatalogModelKey] = []
                for ref in bundle_info.get("models_to_download", []) or []:
                    if (
                        isinstance(ref, (list, tuple))
                        and len(ref) == 3
                        and all(isinstance(x, str) for x in ref)
                    ):
                        model_refs.append(CatalogModelKey(category=ref[0], sub_category=ref[1], name=ref[2]))
                bundles.append(
                    Bundle(
                        bundle_key=b_key,
                        category=cat_name,
                        name=str(b_name),
                        info=b_info if isinstance(b_info, str) else None,
                        model_refs=model_refs,
                        save_files=set(),
                        size_gb=None,
                        model_count=0,
                    )
                )

    return models, bundles


def _load_size_data() -> Tuple[Dict[str, Any], Dict[str, Any], Optional[str]]:
    if not MODEL_SIZES_FILE.exists():
        return {}, {}, None
    try:
        raw = _read_json(MODEL_SIZES_FILE)
        return raw.get("models", {}) or {}, raw.get("bundles", {}) or {}, raw.get("fetch_date")
    except Exception:
        return {}, {}, None


def _load_snapshot_repo_files_cache() -> Dict[str, Any]:
    """
    Cache format:
      {
        "repos": {
          "<repo_id>": {
            "fetched_at": <unix seconds>,
            "fetched_date": "<local date str>",
            "basenames": ["file1.safetensors", ...],
            "source": "hf" | "local"
          }
        }
      }
    """
    try:
        if SNAPSHOT_REPO_FILES_CACHE.exists():
            raw = _read_json(SNAPSHOT_REPO_FILES_CACHE)
            if (
                isinstance(raw, dict)
                and raw.get("version") == SNAPSHOT_REPO_FILES_CACHE_VERSION
                and isinstance(raw.get("repos"), dict)
            ):
                return raw
    except Exception:
        pass
    # Cache schema changed? Start fresh.
    return {"version": SNAPSHOT_REPO_FILES_CACHE_VERSION, "repos": {}}


def _save_snapshot_repo_files_cache(cache: Dict[str, Any]) -> None:
    try:
        if not isinstance(cache, dict):
            cache = {"version": SNAPSHOT_REPO_FILES_CACHE_VERSION, "repos": {}}
        cache.setdefault("version", SNAPSHOT_REPO_FILES_CACHE_VERSION)
        cache.setdefault("repos", {})
        SNAPSHOT_REPO_FILES_CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        # Cache is best-effort; report generation should still succeed.
        pass


def _snapshot_cache_entry_is_fresh(entry: Dict[str, Any]) -> bool:
    try:
        fetched_at = float(entry.get("fetched_at", 0))
    except Exception:
        fetched_at = 0.0
    if fetched_at <= 0:
        return False
    return (time.time() - fetched_at) <= SNAPSHOT_REPO_FILES_CACHE_TTL_SECONDS


def _try_list_hf_repo_files(repo_id: str, hf_token: Optional[str]) -> Optional[List[str]]:
    """
    Best-effort HuggingFace repo file listing.
    Returns None if huggingface_hub isn't available or if listing fails.
    """
    try:
        from huggingface_hub import HfApi  # type: ignore

        api = HfApi(token=hf_token)
        return api.list_repo_files(repo_id=repo_id)
    except Exception:
        return None


def _model_basenames_from_repo_files(
    repo_files: Sequence[str],
    *,
    required_basenames_lc: Optional[Set[str]] = None,
) -> Set[str]:
    basenames: Set[str] = set()
    for f in repo_files:
        if not isinstance(f, str):
            continue
        if not _looks_like_model_file(f):
            continue
        base = Path(_norm_ref_path(f)).name.lower()
        if required_basenames_lc is not None and base not in required_basenames_lc:
            continue
        basenames.add(base)
    return basenames


def _scan_local_dir_for_model_basenames(
    dir_path: Path,
    *,
    required_basenames_lc: Optional[Set[str]] = None,
) -> Set[str]:
    basenames: Set[str] = set()
    try:
        if not dir_path.exists():
            return basenames
        for root, _, files in os.walk(str(dir_path)):
            for fname in files:
                fl = fname.lower()
                if not any(fl.endswith(ext) for ext in MODEL_FILE_EXTS):
                    continue
                if required_basenames_lc is not None and fl not in required_basenames_lc:
                    continue
                basenames.add(fl)
    except Exception:
        return basenames
    return basenames


def _get_snapshot_repo_basenames_lc(
    repo_id: str,
    *,
    required_basenames_lc: Optional[Set[str]],
    hf_token: Optional[str],
    cache: Dict[str, Any],
    local_dirs: Sequence[Path],
    memory: Dict[str, Set[str]],
) -> Tuple[Set[str], bool]:
    """
    Returns (basenames_lc, cache_dirty).
    basenames are filtered to MODEL_FILE_EXTS and lowercased.
    """
    if not repo_id:
        return set(), False

    if repo_id in memory:
        full = memory[repo_id]
        if required_basenames_lc is None:
            return full, False
        return {b for b in full if b in required_basenames_lc}, False

    repos = cache.setdefault("repos", {})
    entry = repos.get(repo_id)
    if isinstance(entry, dict) and _snapshot_cache_entry_is_fresh(entry):
        cached = entry.get("basenames") or entry.get("files") or []
        full = {str(x).lower() for x in cached if isinstance(x, str)}
        memory[repo_id] = full
        if required_basenames_lc is None:
            return full, False
        return {b for b in full if b in required_basenames_lc}, False

    # Try to list from HuggingFace (best accuracy)
    repo_files = _try_list_hf_repo_files(repo_id, hf_token)
    if repo_files:
        full = _model_basenames_from_repo_files(repo_files, required_basenames_lc=None)
        repos[repo_id] = {
            "fetched_at": time.time(),
            "fetched_date": _now_local_str(),
            "basenames": sorted(full),
            "source": "hf",
        }
        memory[repo_id] = full
        if required_basenames_lc is None:
            return full, True
        return {b for b in full if b in required_basenames_lc}, True

    # Fallback: scan local target dirs (only covers what is currently on disk)
    full: Set[str] = set()
    for d in local_dirs:
        full |= _scan_local_dir_for_model_basenames(d, required_basenames_lc=None)

    if full:
        repos[repo_id] = {
            "fetched_at": time.time(),
            "fetched_date": _now_local_str(),
            "basenames": sorted(full),
            "source": "local",
        }
        memory[repo_id] = full
        if required_basenames_lc is None:
            return full, True
        return {b for b in full if b in required_basenames_lc}, True

    memory[repo_id] = set()
    return set(), False


def _extract_required_files_from_preset(preset_json: Any) -> Dict[str, Tuple[str, Set[str]]]:
    """
    Returns mapping:
      ref_norm -> (first_seen_original, set(node_types))
    """
    found: Dict[str, Tuple[str, Set[str]]] = {}

    # Prefer node widgets_values (more likely to contain model paths).
    if isinstance(preset_json, dict) and isinstance(preset_json.get("nodes"), list):
        for node in preset_json.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            node_type = node.get("type")
            node_type_str = str(node_type) if node_type else "UnknownNode"
            wv = node.get("widgets_values")
            if not isinstance(wv, list):
                continue
            for v in wv:
                if not isinstance(v, str):
                    continue
                if not _looks_like_model_file(v):
                    continue
                ref_norm = _norm_ref_path(v)
                if _is_excluded_model_ref(ref_norm):
                    continue
                if ref_norm not in found:
                    found[ref_norm] = (v, {node_type_str})
                else:
                    found[ref_norm][1].add(node_type_str)

    # Fallback: scan all strings (catches some workflows that embed paths elsewhere).
    for s in _walk_strings(preset_json):
        if not _looks_like_model_file(s):
            continue
        ref_norm = _norm_ref_path(s)
        if _is_excluded_model_ref(ref_norm):
            continue
        if ref_norm not in found:
            found[ref_norm] = (s, set())

    return found


def _choose_expected_size_gb(
    model_keys: Sequence[CatalogModelKey],
    size_by_model_key: Dict[str, Any],
) -> Optional[float]:
    best: Optional[float] = None
    for mk in model_keys:
        entry = size_by_model_key.get(mk.to_size_key()) or {}
        gb = entry.get("size_gb")
        try:
            gb_f = float(gb)
        except Exception:
            continue
        if gb_f <= 0:
            continue
        if best is None or gb_f > best:
            best = gb_f
    return best


def _greedy_bundle_cover(required_files: Set[str], bundle_files: List[Tuple[str, Set[str]]]) -> List[str]:
    """
    Greedy set cover: returns bundle keys chosen to cover as many required_files as possible.
    """
    remaining = set(required_files)
    chosen: List[str] = []
    while remaining:
        best_key = None
        best_cover = 0
        best_set: Set[str] = set()
        for b_key, b_files in bundle_files:
            cover = len(remaining & b_files)
            if cover > best_cover:
                best_cover = cover
                best_key = b_key
                best_set = b_files
        if not best_key or best_cover <= 0:
            break
        chosen.append(best_key)
        remaining -= best_set
    return chosen


def generate_report(
    base_path: Optional[Path] = None,
    comfy_ui_structure: Optional[bool] = None,
    forge_structure: Optional[bool] = None,
    lowercase_folders: Optional[bool] = None,
    output_html: Path = DEFAULT_OUTPUT_HTML,
) -> Path:
    if not PRESETS_DIR.exists():
        raise FileNotFoundError(f"Missing presets folder: {PRESETS_DIR}")

    # Preset descriptions (from latest Amazing_SwarmUI_Presets_v*.json)
    amazing_presets_file, preset_desc_map = _load_amazing_preset_descriptions(REPO_ROOT)

    # 1) Settings / paths
    saved_base, saved_comfy, saved_forge, saved_lower = _load_last_settings()
    base_path = base_path or saved_base or _default_base_path()
    comfy_ui_structure = saved_comfy if comfy_ui_structure is None else bool(comfy_ui_structure)
    forge_structure = saved_forge if forge_structure is None else bool(forge_structure)
    lowercase_folders = saved_lower if lowercase_folders is None else bool(lowercase_folders)

    base_path = Path(base_path)

    # Also scan common in-repo model roots to be more helpful, even if user used another base path.
    extra_roots: List[Path] = []
    swarm_default = REPO_ROOT / "SwarmUI" / "Models"
    comfy_default = REPO_ROOT / "ComfyUI" / "models"
    for candidate in (swarm_default, comfy_default):
        if candidate.exists() and candidate.is_dir() and candidate.resolve() != base_path.resolve():
            extra_roots.append(candidate)

    scan_roots = [base_path] + extra_roots

    # Folder mapping (must match the downloader app).
    subdir_map = _get_current_subdirs(comfy_ui_structure, forge_structure, lowercase_folders)

    # Snapshot repo file cache (used for bundle coverage when bundles include snapshot models).
    snapshot_cache = _load_snapshot_repo_files_cache()
    snapshot_cache_dirty = False
    snapshot_repo_basenames_mem: Dict[str, Set[str]] = {}
    try:
        # Optional: use the downloader's HF token resolution if available.
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from utilities.hf_token_manager import resolve_hf_token  # type: ignore

        hf_token_for_listing = resolve_hf_token().token
    except Exception:
        hf_token_for_listing = None

    # 2) Load catalog + bundles + sizes
    catalog_models, bundles = _load_catalog_and_bundles()
    size_by_model_key, size_by_bundle_key, sizes_fetch_date = _load_size_data()

    catalog_by_model_key: Dict[str, CatalogModel] = {}
    by_save_norm: Dict[str, List[CatalogModelKey]] = {}
    by_basename_lower: Dict[str, List[CatalogModelKey]] = {}

    for m in catalog_models:
        catalog_by_model_key[m.key.to_size_key()] = m
        if m.save_filename:
            save_norm = _norm_ref_path(str(m.save_filename))
            by_save_norm.setdefault(save_norm, []).append(m.key)
            base = Path(save_norm).name.lower()
            by_basename_lower.setdefault(base, []).append(m.key)

    # Local fallback directories for snapshot repos (per repo_id).
    # NOTE: Snapshot downloads go to the model's target_dir (no save_filename), so we can only
    # infer contents from disk if that directory exists.
    snapshot_local_dirs_by_repo_id: Dict[str, List[Path]] = {}
    for m in catalog_models:
        if not m.is_snapshot or not m.repo_id:
            continue
        target_dir_key = m.target_dir_key or "diffusion_models"
        subdir = subdir_map.get(str(target_dir_key), str(target_dir_key))
        for root in scan_roots:
            p = Path(root) / subdir
            lst = snapshot_local_dirs_by_repo_id.setdefault(m.repo_id, [])
            # De-dupe (Path equality is fine here)
            if p not in lst:
                lst.append(p)

    # Resolve bundles to the actual saved filenames they would download.
    for b in bundles:
        save_files: Set[str] = set()
        cover_exact_lc: Set[str] = set()
        cover_base_lc: Set[str] = set()
        snapshot_repo_ids: Set[str] = set()
        for mk in b.model_refs:
            cm = catalog_by_model_key.get(mk.to_size_key())
            if not cm:
                continue
            if cm.is_snapshot and cm.repo_id:
                snapshot_repo_ids.add(cm.repo_id)
            if cm.save_filename:
                save_norm = _norm_ref_path(str(cm.save_filename))
                save_files.add(save_norm)
                cover_exact_lc.add(save_norm.lower())
                cover_base_lc.add(Path(save_norm).name.lower())
        b.save_files = save_files
        b.model_count = len(save_files)
        b.cover_exact_lc = cover_exact_lc
        b.cover_base_lc = cover_base_lc
        b.snapshot_repo_ids = snapshot_repo_ids
        b_size = size_by_bundle_key.get(b.bundle_key) or {}
        if b_size and b_size.get("total_size_gb"):
            try:
                b.size_gb = float(b_size.get("total_size_gb"))
            except Exception:
                b.size_gb = None

    # 3) Index local files once (fast lookups later)
    files_by_basename, files_by_relpath = _build_file_index(scan_roots)

    # 4) Scan presets
    preset_reports: List[PresetReport] = []
    all_required_files_norm: Set[str] = set()

    preset_paths = sorted(PRESETS_DIR.glob("*.json"), key=lambda p: p.name.lower())
    for preset_path in preset_paths:
        preset_name = preset_path.stem
        try:
            preset_json = _read_json(preset_path)
        except Exception as e:
            preset_reports.append(
                PresetReport(
                    preset_file=preset_path,
                    preset_name=preset_name,
                    parse_error=f"{type(e).__name__}: {e}",
                    required=[],
                )
            )
            continue

        required_map = _extract_required_files_from_preset(preset_json)
        required_list: List[RequiredFile] = []

        for ref_norm, (ref_original, node_types) in sorted(required_map.items(), key=lambda x: x[0].lower()):
            all_required_files_norm.add(ref_norm)

            # Catalog match by full normalized save_filename, then by basename
            cat_keys = list(by_save_norm.get(ref_norm) or [])
            if not cat_keys:
                base = Path(ref_norm).name.lower()
                # If basename maps to multiple models, we still include all keys (for bundle coverage).
                cat_keys = list(by_basename_lower.get(base) or [])

            # Collect target_dir_keys from catalog matches
            target_keys: Set[str] = set()
            for mk in cat_keys:
                cm = catalog_by_model_key.get(mk.to_size_key())
                if cm and cm.target_dir_key:
                    target_keys.add(str(cm.target_dir_key))

            expected_size_gb = _choose_expected_size_gb(cat_keys, size_by_model_key) if cat_keys else None

            expected_paths: List[Path] = []
            hits: List[FileHit] = []

            # Expected path checks for catalog-known files
            if cat_keys:
                # Build expected absolute paths across scan roots (base_path + any extra roots).
                for mk in cat_keys:
                    cm = catalog_by_model_key.get(mk.to_size_key())
                    if not cm or not cm.save_filename:
                        continue
                    save_norm = _norm_ref_path(str(cm.save_filename))
                    save_os_rel = _os_relpath_from_ref(save_norm)
                    target_dir_key = cm.target_dir_key or "diffusion_models"
                    subdir = subdir_map.get(str(target_dir_key), str(target_dir_key))
                    for root in scan_roots:
                        exp = Path(root) / subdir / save_os_rel
                        expected_paths.append(exp)
                        sz = _safe_stat_size(exp)
                        if sz is not None:
                            hits.append(FileHit(path=exp, size_bytes=sz, base_root=root))

            # Fallback search: by exact relpath under root, then basename under any root
            if not hits:
                # Try relpath match (case-insensitive via lower index)
                for root in scan_roots:
                    for subdir_guess in ("",):  # relpath index is relative to root already
                        _ = subdir_guess  # placeholder for potential future expansion
                    rel_norm_lower = ref_norm.lower()
                    rel_hits = files_by_relpath.get(rel_norm_lower) or []
                    hits.extend(rel_hits)
                if not hits:
                    base = Path(ref_norm).name.lower()
                    hits.extend(files_by_basename.get(base) or [])

            # De-duplicate hits by absolute path
            unique_hits: Dict[str, FileHit] = {}
            for h in hits:
                unique_hits[str(h.path.resolve()).lower()] = h
            hits = list(unique_hits.values())

            required_list.append(
                RequiredFile(
                    ref_norm=ref_norm,
                    ref_original=ref_original,
                    node_types=set(node_types),
                    catalog_keys=cat_keys,
                    target_dir_keys=target_keys,
                    expected_size_gb=expected_size_gb,
                    present=bool(hits),
                    expected_paths=expected_paths,
                    hits=sorted(hits, key=lambda x: str(x.path).lower()),
                )
            )

        preset_reports.append(
            PresetReport(
                preset_file=preset_path,
                preset_name=preset_name,
                parse_error=None,
                required=required_list,
            )
        )

    # 5) Aggregate bundle coverage
    bundle_files_list = [(b.bundle_key, b.save_files) for b in bundles]

    # 6) Render HTML
    def preset_status(pr: PresetReport) -> str:
        if pr.parse_error:
            return "error"
        if not pr.required:
            return "unknown"
        present = sum(1 for r in pr.required if r.present)
        if present == len(pr.required):
            return "complete"
        if present == 0:
            return "missing"
        return "partial"

    def preset_missing_gb(pr: PresetReport) -> float:
        total = 0.0
        for r in pr.required:
            if r.present:
                continue
            if r.expected_size_gb:
                total += float(r.expected_size_gb)
        return round(total, 2)

    total_presets = len(preset_reports)
    total_errors = sum(1 for p in preset_reports if p.parse_error)
    total_complete = sum(1 for p in preset_reports if preset_status(p) == "complete")
    total_partial = sum(1 for p in preset_reports if preset_status(p) == "partial")
    total_missing = sum(1 for p in preset_reports if preset_status(p) == "missing")

    unique_models_total = len(all_required_files_norm)
    unique_models_present = 0
    unique_models_missing = 0
    unique_missing_gb = 0.0

    # Build unique model map (for "All models" section)
    all_models_map: Dict[str, Dict[str, Any]] = {}
    for pr in preset_reports:
        for r in pr.required:
            m = all_models_map.setdefault(
                r.ref_norm,
                {
                    "ref_norm": r.ref_norm,
                    "ref_original": r.ref_original,
                    "present": r.present,
                    "expected_size_gb": r.expected_size_gb,
                    "hits": r.hits,
                    "presets": set(),
                    "catalog_keys": set(),
                    "target_dir_keys": set(),
                    "node_types": set(),
                },
            )
            m["presets"].add(pr.preset_name)
            for ck in r.catalog_keys:
                m["catalog_keys"].add(ck.to_size_key())
            for tk in r.target_dir_keys:
                m["target_dir_keys"].add(tk)
            for nt in r.node_types:
                m["node_types"].add(nt)
            # If any preset reports it present, treat as present globally
            if r.present:
                m["present"] = True
                m["hits"] = r.hits
            # Prefer a known expected size if one is available
            if (m.get("expected_size_gb") in (None, 0, 0.0)) and r.expected_size_gb:
                m["expected_size_gb"] = r.expected_size_gb

    for m in all_models_map.values():
        if m.get("present"):
            unique_models_present += 1
        else:
            unique_models_missing += 1
            if m.get("expected_size_gb"):
                unique_missing_gb += float(m["expected_size_gb"])
    unique_missing_gb = round(unique_missing_gb, 2)

    # Helper: bundle coverage for a given required model ref
    bundle_keys_by_file: Dict[str, List[str]] = {}
    for b in bundles:
        for f in b.save_files:
            bundle_keys_by_file.setdefault(f, []).append(b.bundle_key)
    for f, lst in bundle_keys_by_file.items():
        lst.sort()

    # Helper: bundle label
    bundle_label_by_key = {b.bundle_key: b.name for b in bundles}
    bundle_size_by_key = {b.bundle_key: b.size_gb for b in bundles}

    # Build HTML parts
    out_title = "ComfyUI Presets • Download Status Dashboard"

    # Search text is computed per preset based on what is rendered (preset name + models + fully covering bundles).

    # Sorting helpers
    def preset_present_count(pr: PresetReport) -> int:
        return sum(1 for r in pr.required if r.present)

    def preset_total_count(pr: PresetReport) -> int:
        return len(pr.required)

    # Render
    html_parts: List[str] = []
    html_parts.append("<!doctype html>")
    html_parts.append("<html lang='en'>")
    html_parts.append("<head>")
    html_parts.append("<meta charset='utf-8' />")
    html_parts.append("<meta name='viewport' content='width=device-width, initial-scale=1' />")
    html_parts.append(f"<title>{_escape(out_title)}</title>")
    html_parts.append("<style>")
    html_parts.append(
        """
:root{
  --bg:#0b1020;
  --panel:#121a31;
  --panel2:#0f172b;
  --muted:#93a4c7;
  --text:#e8eefc;
  --border:#223055;
  --good:#2dd4bf;
  --warn:#fbbf24;
  --bad:#fb7185;
  --info:#60a5fa;
  --shadow: 0 20px 60px rgba(0,0,0,.35);
  --radius: 16px;
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}
*{ box-sizing:border-box; }
body{
  margin:0;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
  background: radial-gradient(1200px 600px at 20% -10%, rgba(96,165,250,.28), transparent 70%),
              radial-gradient(1000px 500px at 80% 0%, rgba(45,212,191,.18), transparent 65%),
              var(--bg);
  color:var(--text);
}
a{ color:var(--info); text-decoration:none; }
a:hover{ text-decoration:underline; }
.wrap{ max-width:1180px; margin: 0 auto; padding: 26px 18px 64px; }
.topbar{
  position: sticky;
  top: 0;
  z-index: 20;
  backdrop-filter: blur(10px);
  background: rgba(11,16,32,.70);
  border-bottom: 1px solid rgba(34,48,85,.6);
}
.topbar-inner{ max-width:1180px; margin: 0 auto; padding: 16px 18px; display:flex; gap:12px; align-items:center; flex-wrap:wrap;}
.title{
  display:flex; flex-direction:column; gap:2px; flex:1 1 auto;
}
.title h1{ margin:0; font-size: 18px; letter-spacing:.2px; }
.title .sub{ color:var(--muted); font-size: 12px; }
.controls{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; justify-content:flex-end; }
input[type="search"]{
  width:min(520px, 76vw);
  padding: 10px 12px;
  border-radius: 12px;
  border: 1px solid var(--border);
  background: rgba(18,26,49,.9);
  color: var(--text);
  outline: none;
}
input[type="search"]:focus{ border-color: rgba(96,165,250,.75); box-shadow: 0 0 0 4px rgba(96,165,250,.15); }
.chip{
  display:inline-flex; align-items:center; gap:8px;
  padding: 8px 10px;
  border-radius: 999px;
  border:1px solid var(--border);
  background: rgba(18,26,49,.75);
  color: var(--muted);
  font-size: 12px;
  user-select:none;
}
.chip input{ accent-color: var(--info); }
.grid{
  display:grid;
  grid-template-columns: repeat(12, 1fr);
  gap: 12px;
  margin-top: 18px;
}
.card{
  grid-column: span 3;
  background: linear-gradient(180deg, rgba(18,26,49,.95), rgba(15,23,43,.92));
  border: 1px solid rgba(34,48,85,.8);
  border-radius: var(--radius);
  padding: 14px 14px;
  box-shadow: var(--shadow);
}
.card .k{ color:var(--muted); font-size: 12px; }
.card .v{ font-size: 20px; margin-top: 6px; }
.card .hint{ color:var(--muted); font-size: 11px; margin-top: 6px; }
@media (max-width: 980px){ .card{ grid-column: span 6; } }
@media (max-width: 520px){ .card{ grid-column: span 12; } }

.section{ margin-top: 18px; }
.section h2{ margin: 18px 0 10px; font-size: 14px; color: var(--muted); font-weight: 600; letter-spacing: .2px; }

.preset{
  border:1px solid rgba(34,48,85,.85);
  border-radius: var(--radius);
  background: rgba(18,26,49,.72);
  box-shadow: var(--shadow);
  margin: 10px 0;
  padding: 14px 14px;
}
.preset summary{
  list-style:none;
  cursor:pointer;
  padding: 14px 14px;
  display:flex;
  gap: 12px;
  align-items:center;
}
.preset summary::-webkit-details-marker{ display:none; }
.badge{
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
  border:1px solid rgba(255,255,255,.08);
  background: rgba(255,255,255,.06);
  color: var(--muted);
  white-space: nowrap;
}
.badge.complete{ color: var(--good); border-color: rgba(45,212,191,.35); background: rgba(45,212,191,.10); }
.badge.partial{ color: var(--warn); border-color: rgba(251,191,36,.35); background: rgba(251,191,36,.10); }
.badge.missing{ color: var(--bad); border-color: rgba(251,113,133,.35); background: rgba(251,113,133,.10); }
.badge.error{ color: var(--bad); border-color: rgba(251,113,133,.35); background: rgba(251,113,133,.10); }
.badge.unknown{ color: var(--muted); }

.preset-title{ flex: 1 1 auto; min-width: 240px; }
.preset-title .name{ font-weight: 650; font-size: 14px; }
.preset-title .meta{ color: var(--muted); font-size: 12px; margin-top: 4px; }
.bar{
  height: 10px;
  width: 220px;
  border-radius: 999px;
  background: rgba(255,255,255,.06);
  border:1px solid rgba(34,48,85,.8);
  overflow:hidden;
}
.bar > span{ display:block; height:100%; background: linear-gradient(90deg, rgba(96,165,250,.95), rgba(45,212,191,.95)); width: 0%; }
.summary-right{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; justify-content:flex-end; }
.mini{ color: var(--muted); font-size: 12px; white-space: nowrap; }

.details{
  padding: 0 14px 14px;
  border-top: 1px solid rgba(34,48,85,.65);
}
.row{
  display:flex;
  gap: 10px;
  align-items:center;
  flex-wrap:wrap;
  margin: 10px 0;
}
.pill{
  font-size: 12px;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(34,48,85,.85);
  background: rgba(15,23,43,.75);
  color: var(--muted);
}
.pill strong{ color: var(--text); font-weight: 650; }
.table{
  width: 100%;
  border-collapse: collapse;
  overflow:hidden;
  border-radius: 12px;
  border: 1px solid rgba(34,48,85,.75);
  background: rgba(15,23,43,.60);
}
.table th, .table td{
  padding: 10px 10px;
  border-bottom: 1px solid rgba(34,48,85,.5);
  font-size: 12px;
  vertical-align: top;
}
.table th{
  text-align:left;
  color: var(--muted);
  font-weight: 650;
  background: rgba(18,26,49,.55);
}
.table tr:last-child td{ border-bottom: none; }
.mono{ font-family: var(--mono); }
.ok{ color: var(--good); }
.no{ color: var(--bad); }
.dim{ color: var(--muted); }
.bundle-list{ display:flex; flex-wrap:wrap; gap:8px; }
.bundle-tag{
  display:inline-flex; gap:8px; align-items:center;
  padding: 7px 10px;
  border-radius: 12px;
  border: 1px solid rgba(34,48,85,.85);
  background: rgba(18,26,49,.55);
  font-size: 12px;
  color: var(--muted);
}
.bundle-tag .bname{ color: var(--text); font-weight: 650; }
.bundle-tag .bmeta{ color: var(--muted); }

/* New always-expanded preset layout (no accordions) */
.preset-header{
  display:flex;
  align-items:flex-start;
  justify-content: space-between;
  gap: 12px;
  flex-wrap:wrap;
  padding-bottom: 12px;
  border-bottom: 1px solid rgba(34,48,85,.55);
  margin-bottom: 12px;
}
.preset-title{
  display:flex;
  flex-direction:column;
  gap: 6px;
  min-width: 260px;
  flex: 1 1 auto;
}
.preset-title .name{
  font-weight: 750;
  font-size: 16px;
  letter-spacing: .2px;
}
.preset-title .meta{
  color: var(--muted);
  font-size: 13px;
}
.preset-title .desc{
  color: rgba(232,238,252,.82);
  font-size: 13px;
  line-height: 1.35;
  max-width: 860px;
}
.tag{
  font-size: 12px;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(34,48,85,.75);
  background: rgba(18,26,49,.55);
  color: var(--muted);
  white-space: nowrap;
}
.cols{
  display:grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
}
@media (max-width: 900px){
  .cols{ grid-template-columns: 1fr; }
}
.block{
  border: 1px solid rgba(34,48,85,.75);
  border-radius: 14px;
  background: rgba(15,23,43,.55);
  padding: 12px 12px;
}
.block h3{
  margin: 0 0 10px 0;
  font-size: 13px;
  font-weight: 750;
  color: var(--muted);
  letter-spacing: .2px;
}
.list{
  display:flex;
  flex-direction:column;
  gap: 8px;
}
.bundle-item{
  padding: 10px 10px;
  border-radius: 12px;
  border: 1px solid rgba(34,48,85,.75);
  background: rgba(18,26,49,.55);
}
.bundle-item .bname{
  font-weight: 750;
  font-size: 14px;
}
.bundle-item .bmeta{
  margin-top: 4px;
  font-size: 13px;
  color: var(--muted);
}
.model-item{
  display:flex;
  align-items:center;
  justify-content: space-between;
  gap: 10px;
  padding: 10px 10px;
  border-radius: 12px;
  border: 1px solid rgba(34,48,85,.75);
  background: rgba(18,26,49,.35);
}
.model-left{
  display:flex;
  align-items:center;
  gap: 10px;
  min-width: 0;
}
.dot{
  width: 9px;
  height: 9px;
  border-radius: 999px;
  background: rgba(255,255,255,.18);
  border: 1px solid rgba(255,255,255,.12);
  flex: 0 0 auto;
}
.dot.ok{ background: rgba(45,212,191,.85); border-color: rgba(45,212,191,.35); }
.dot.no{ background: rgba(251,113,133,.85); border-color: rgba(251,113,133,.35); }
.model-name{
  font-family: var(--mono);
  font-size: 13px;
  color: var(--text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.model-subpath{
  margin-top: 3px;
  font-family: var(--mono);
  font-size: 11px;
  color: var(--muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.model-right{
  display:flex;
  gap: 10px;
  align-items:center;
  flex: 0 0 auto;
}
.size{
  font-family: var(--mono);
  font-size: 12px;
  color: var(--muted);
  white-space: nowrap;
}
.errorbox{
  border: 1px solid rgba(251,113,133,.45);
  background: rgba(251,113,133,.10);
  color: var(--bad);
  border-radius: 14px;
  padding: 10px 12px;
  font-size: 13px;
}
.note{
  color: var(--muted);
  font-size: 13px;
}
.footer{
  margin-top: 26px;
  color: var(--muted);
  font-size: 12px;
}
.codebox{
  font-family: var(--mono);
  background: rgba(0,0,0,.25);
  border:1px solid rgba(34,48,85,.75);
  border-radius: 12px;
  padding: 10px 12px;
  color: var(--muted);
  overflow:auto;
}
        """
    )
    html_parts.append("</style>")
    html_parts.append("</head>")
    html_parts.append("<body>")

    # Topbar
    html_parts.append("<div class='topbar'><div class='topbar-inner'>")
    html_parts.append("<div class='title'>")
    html_parts.append(f"<h1>{_escape(out_title)}</h1>")
    html_parts.append(
        f"<div class='sub'>Generated {_escape(_now_local_str())} • Presets: {_escape(total_presets)} • Models referenced: {_escape(unique_models_total)}</div>"
    )
    html_parts.append("</div>")
    html_parts.append("<div class='controls'>")
    html_parts.append("<input id='search' type='search' placeholder='Search presets, models, bundles…' />")
    html_parts.append("<label class='chip'><input id='onlyIncomplete' type='checkbox' /> Incomplete only</label>")
    html_parts.append("<label class='chip'><input id='onlyMissingModels' type='checkbox' /> Missing models only</label>")
    html_parts.append("</div>")
    html_parts.append("</div></div>")

    # Summary cards
    html_parts.append("<div class='wrap'>")
    html_parts.append("<div class='grid'>")
    html_parts.append(
        f"<div class='card'><div class='k'>Presets</div><div class='v'>{total_presets}</div><div class='hint'>Complete {total_complete} • Partial {total_partial} • Missing {total_missing} • Errors {total_errors}</div></div>"
    )
    html_parts.append(
        f"<div class='card'><div class='k'>Unique models</div><div class='v'>{unique_models_total}</div><div class='hint'>Present {unique_models_present} • Missing {unique_models_missing}</div></div>"
    )
    html_parts.append(
        f"<div class='card'><div class='k'>Missing size (best-effort)</div><div class='v'>{_escape(_fmt_gb(unique_missing_gb))}</div><div class='hint'>Uses utilities/model_sizes.json when available</div></div>"
    )
    ui_mode = "ComfyUI" if comfy_ui_structure else ("Forge" if forge_structure else "SwarmUI")
    html_parts.append(
        f"<div class='card'><div class='k'>Scan roots</div><div class='v'>{_escape(ui_mode)}</div><div class='hint'>{_escape(str(base_path))}</div></div>"
    )
    html_parts.append("</div>")  # grid

    # Settings box
    html_parts.append("<div class='section'>")
    html_parts.append("<h2>Scan configuration</h2>")
    html_parts.append("<div class='codebox'>")
    html_parts.append(f"Base path: {html.escape(str(base_path))}<br/>")
    if extra_roots:
        html_parts.append(f"Extra scan roots: {html.escape(', '.join(str(p) for p in extra_roots))}<br/>")
    html_parts.append(f"Folder structure: {html.escape(ui_mode)}<br/>")
    html_parts.append(f"Lowercase folders: {html.escape(str(bool(lowercase_folders)))}<br/>")
    if amazing_presets_file:
        html_parts.append(f"Preset descriptions: {html.escape(str(amazing_presets_file.name))}<br/>")
    html_parts.append(f"Model sizes file: {html.escape(str(MODEL_SIZES_FILE))}<br/>")
    html_parts.append(f"Model sizes fetched: {html.escape(str(sizes_fetch_date or 'Unknown'))}")
    html_parts.append("</div>")
    html_parts.append("</div>")

    # Presets
    html_parts.append("<div class='section'>")
    html_parts.append("<h2>Presets</h2>")

    def _model_display_size(r: RequiredFile) -> str:
        # Prefer expected size from catalog. Fallback to disk size (first hit).
        if r.expected_size_gb and r.expected_size_gb > 0:
            return _fmt_gb(r.expected_size_gb)
        if r.hits:
            return _fmt_gb(r.hits[0].size_gb)
        return "—"

    def _bundle_fully_covers_preset(
        bundle: Bundle,
        *,
        required_refs: Sequence[str],
        required_basenames_lc: Set[str],
    ) -> bool:
        """
        Robust bundle coverage check:
        - Exact match via downloader save_filename (normalized, case-insensitive)
        - Basename match (for presets that reference just 'foo.safetensors')
        - Snapshot expansion (bundle contains snapshot models that include many files)
        """
        nonlocal snapshot_cache_dirty
        for rf in required_refs:
            rf_lc = _norm_ref_path(rf).lower()
            base_lc = Path(rf_lc).name
            if rf_lc in bundle.cover_exact_lc or base_lc in bundle.cover_base_lc:
                continue

            # Snapshot models: treat as covering any file present in the repo listing.
            covered_by_snapshot = False
            for repo_id in bundle.snapshot_repo_ids:
                basenames_lc, dirty = _get_snapshot_repo_basenames_lc(
                    repo_id,
                    required_basenames_lc=required_basenames_lc,
                    hf_token=hf_token_for_listing,
                    cache=snapshot_cache,
                    local_dirs=snapshot_local_dirs_by_repo_id.get(repo_id, []),
                    memory=snapshot_repo_basenames_mem,
                )
                if dirty:
                    snapshot_cache_dirty = True
                if base_lc in basenames_lc:
                    covered_by_snapshot = True
                    break
            if not covered_by_snapshot:
                return False
        return True

    for pr in preset_reports:
        st = preset_status(pr)
        present_n = preset_present_count(pr)
        total_n = preset_total_count(pr)
        miss_gb = preset_missing_gb(pr)

        required_refs = [r.ref_norm for r in pr.required]
        required_basenames_lc = {Path(r.ref_norm).name.lower() for r in pr.required}
        req_files = {r.ref_norm for r in pr.required}

        # Only show bundles that fully cover the preset (covers N/N).
        full_cover_bundles: List[Bundle] = []
        if required_refs:
            for b in bundles:
                if _bundle_fully_covers_preset(
                    b, required_refs=required_refs, required_basenames_lc=required_basenames_lc
                ):
                    full_cover_bundles.append(b)
        full_cover_bundles.sort(
            key=lambda b: (
                b.size_gb is None,
                float(b.size_gb or 0.0),
                b.name.lower(),
            )
        )

        # Search text should match what the user actually sees on the page.
        search_parts: List[str] = [pr.preset_name.lower()]
        for r in pr.required:
            search_parts.append(r.ref_norm.lower())
            search_parts.append(Path(r.ref_norm).name.lower())
        for b in full_cover_bundles:
            search_parts.append(b.name.lower())
        desc_text = _description_for_preset_name(pr.preset_name, preset_desc_map) or ""
        if desc_text:
            search_parts.append(desc_text.lower())
        data_search = " ".join(sorted(set(search_parts)))

        html_parts.append(
            f"<div class='preset' data-name='{_escape(pr.preset_name.lower())}' data-status='{_escape(st)}' data-missing='{_escape(str(miss_gb))}' data-search='{_escape(data_search)}'>"
        )
        html_parts.append("<div class='preset-header'>")
        html_parts.append(f"<span class='badge {_escape(st)}'>{_escape(st.upper())}</span>")
        html_parts.append("<div class='preset-title'>")
        html_parts.append(f"<div class='name'>{_escape(pr.preset_name)}</div>")
        if desc_text:
            desc_html = "<br/>".join(_escape(line) for line in desc_text.splitlines())
            html_parts.append(f"<div class='desc'>{desc_html}</div>")
        meta = f"{present_n}/{total_n} models present"
        if miss_gb > 0:
            meta += f" • Missing ≈ {_fmt_gb(miss_gb)}"
        html_parts.append(f"<div class='meta'>{_escape(meta)}</div>")
        html_parts.append("</div>")
        html_parts.append(f"<span class='tag mono'>{_escape(pr.preset_file.name)}</span>")
        html_parts.append("</div>")  # preset-header

        if pr.parse_error:
            html_parts.append(f"<div class='errorbox'>Parse error: {_escape(pr.parse_error)}</div>")
            html_parts.append("</div>")
            continue

        if not pr.required:
            html_parts.append("<div class='note'>No model references detected in this preset.</div>")
            html_parts.append("</div>")
            continue

        html_parts.append("<div class='cols'>")

        # Bundles column
        html_parts.append("<div class='block'>")
        html_parts.append("<h3>BUNDLES (FULL COVERAGE ONLY)</h3>")
        html_parts.append("<div class='list'>")
        if not req_files:
            html_parts.append("<div class='note'>No models detected, so bundle coverage is not applicable.</div>")
        elif not full_cover_bundles:
            html_parts.append("<div class='note'>No bundle fully covers this preset.</div>")
        else:
            needed = len(req_files)
            for b in full_cover_bundles:
                size_txt = _fmt_gb(b.size_gb) if b.size_gb else "—"
                html_parts.append("<div class='bundle-item'>")
                html_parts.append(f"<div class='bname'>{_escape(b.name)}</div>")
                html_parts.append(f"<div class='bmeta'>• covers {needed}/{needed} • {_escape(size_txt)}</div>")
                html_parts.append("</div>")
        html_parts.append("</div></div>")  # list + block

        # Models column
        html_parts.append("<div class='block'>")
        html_parts.append("<h3>MODELS</h3>")
        html_parts.append("<div class='list'>")
        required_sorted = sorted(pr.required, key=lambda r: Path(r.ref_norm).name.lower())
        for r in required_sorted:
            base = Path(r.ref_norm).name
            subpath = r.ref_norm
            dot_cls = "ok" if r.present else "no"
            size_txt = _model_display_size(r)

            html_parts.append("<div class='model-item'>")
            html_parts.append("<div class='model-left'>")
            html_parts.append(f"<span class='dot {dot_cls}'></span>")
            html_parts.append("<div style='min-width:0;'>")
            html_parts.append(f"<div class='model-name'>{_escape(base)}</div>")
            if _norm_ref_path(base) != _norm_ref_path(subpath):
                html_parts.append(f"<div class='model-subpath'>{_escape(subpath)}</div>")
            html_parts.append("</div>")
            html_parts.append("</div>")
            html_parts.append("<div class='model-right'>")
            html_parts.append(f"<span class='size'>{_escape(size_txt)}</span>")
            html_parts.append("</div>")
            html_parts.append("</div>")
        html_parts.append("</div></div>")  # list + block

        html_parts.append("</div>")  # cols
        html_parts.append("</div>")  # preset

    html_parts.append("</div>")  # section

    # Footer
    html_parts.append("<div class='footer'>")
    html_parts.append(
        "Tip: Search for a model filename (e.g. <span class='mono'>clip_l.safetensors</span>) or a bundle name to filter presets instantly."
    )
    html_parts.append("</div>")

    html_parts.append("</div>")  # wrap

    # JS filtering
    html_parts.append("<script>")
    html_parts.append(
        """
const $ = (sel) => document.querySelector(sel);
const presets = Array.from(document.querySelectorAll('.preset')).filter(x => x.hasAttribute('data-status'));
function applyFilters(){
  const q = ($('#search').value || '').trim().toLowerCase();
  const onlyIncomplete = $('#onlyIncomplete').checked;
  const onlyMissingModels = $('#onlyMissingModels').checked;
  for(const el of presets){
    const status = el.getAttribute('data-status') || 'unknown';
    const search = (el.getAttribute('data-search') || '').toLowerCase();
    let ok = true;
    if(q && !search.includes(q)) ok = false;
    if(onlyIncomplete && status === 'complete') ok = false;
    if(onlyMissingModels){
      // hide presets that have zero missing models (missing GB == 0 and not error/unknown)
      const miss = parseFloat(el.getAttribute('data-missing') || '0') || 0;
      if(status === 'complete' || (status !== 'error' && miss <= 0)) ok = false;
    }
    el.style.display = ok ? '' : 'none';
  }
}
$('#search').addEventListener('input', applyFilters);
$('#onlyIncomplete').addEventListener('change', applyFilters);
$('#onlyMissingModels').addEventListener('change', applyFilters);
        """
    )
    html_parts.append("</script>")

    html_parts.append("</body></html>")

    # Persist snapshot repo file cache (best-effort).
    if snapshot_cache_dirty:
        _save_snapshot_repo_files_cache(snapshot_cache)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text("\n".join(html_parts), encoding="utf-8")
    return output_html


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a single HTML dashboard showing ComfyUI preset download status vs downloader bundles/models."
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default=None,
        help="Base model folder to scan (defaults to last_settings.json or ./SwarmUI/Models).",
    )
    parser.add_argument(
        "--comfy-ui-structure",
        action="store_true",
        help="Use ComfyUI folder structure mapping (loras/text_encoders/vae/checkpoints).",
    )
    parser.add_argument(
        "--forge-structure",
        action="store_true",
        help="Use Forge/A1111 folder structure mapping.",
    )
    parser.add_argument(
        "--lowercase-folders",
        action="store_true",
        help="Assume folder names are lowercased (matches downloader 'Lowercase Folders').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_HTML),
        help="Output HTML path (default: ../ComfyUI_Presets/ComfyUI_Presets_Report.html).",
    )

    args = parser.parse_args(argv)

    # If user specified a structure flag, it should override saved settings.
    # If they specify neither, we'll defer to saved settings.
    comfy_override: Optional[bool] = None
    forge_override: Optional[bool] = None
    lower_override: Optional[bool] = None
    if args.comfy_ui_structure or args.forge_structure or args.lowercase_folders:
        comfy_override = bool(args.comfy_ui_structure)
        forge_override = bool(args.forge_structure)
        lower_override = bool(args.lowercase_folders)

    out = generate_report(
        base_path=Path(args.base_path) if args.base_path else None,
        comfy_ui_structure=comfy_override,
        forge_structure=forge_override,
        lowercase_folders=lower_override,
        output_html=Path(args.output),
    )
    print(f"Report generated: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


