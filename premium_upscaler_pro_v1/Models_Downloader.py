"""
Model Downloader for Ultimate Video/Image Upscalers Premium

Downloads all required models:

SeedVR2 Models (from MonsterMMORPG/Wan_GGUF):
- VAE model (ema_vae_fp16.safetensors)
- SeedVR2 3B model (seedvr2_ema_3b_fp16.safetensors)
- SeedVR2 7B model (seedvr2_ema_7b_fp16.safetensors)
- SeedVR2 7B Sharp model (seedvr2_ema_7b_sharp_fp16.safetensors)
- SeedVR2 7B FP8 mixed_block35 model (seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors)
- SeedVR2 7B Sharp FP8 mixed_block35 model (seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors)

RIFE Models (from MonsterMMORPG/Wan_GGUF/RIFE_Models):
- RIFE folders: 4.14, 4.15, 4.17, 4.18, 4.20, 4.21, 4.22, 4.25, 4.26

BestImageUpscalers (from MonsterMMORPG/BestImageUpscalers):
- All model files for best image upscaling
"""

from huggingface_hub import hf_hub_url, get_hf_file_metadata, list_repo_files
import os
import argparse
import requests
import threading
import time
import sys
import hashlib
from pathlib import Path
import concurrent.futures
from typing import List, Dict, Optional, Tuple
import shutil
import json
import urllib.parse
from contextlib import suppress
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Configuration - paths relative to script directory
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR_CANDIDATES = [
    "SECourses_Premium_Upscaler_Pro",
    "Ultimate_Video_Image_Upscalers_Premium",
]
PROJECT_DIR_NAME = next(
    (name for name in PROJECT_DIR_CANDIDATES if (SCRIPT_DIR / name).exists()),
    PROJECT_DIR_CANDIDATES[0],
)
PROJECT_DIR = SCRIPT_DIR / PROJECT_DIR_NAME
SEEDVR2_MODELS_DIR = PROJECT_DIR / "SeedVR2" / "models"
RIFE_MODELS_DIR = PROJECT_DIR / "RIFE" / "models"
IMAGE_UPSCALE_MODELS_DIR = PROJECT_DIR / "models"
CACHE_DIR = SCRIPT_DIR / "download_cache"

# Repository configuration
SEEDVR2_REPO_ID = "MonsterMMORPG/Wan_GGUF"
RIFE_REPO_ID = "MonsterMMORPG/Wan_GGUF"
RIFE_REPO_SUBDIR = "RIFE_Models"
RIFE_VERSION_FOLDERS = ["4.14", "4.15", "4.17", "4.18", "4.20", "4.21", "4.22", "4.25", "4.26"]
RIFE_REPO_PREFIXES = [f"{RIFE_REPO_SUBDIR}/{version}" for version in RIFE_VERSION_FOLDERS]
BESTIMAGEUPSCALE_REPO_ID = "MonsterMMORPG/BestImageUpscalers"

# SeedVR2 Model configurations
SEEDVR2_MODEL_CONFIGS = {
    "vae": {
        "filename": "ema_vae_fp16.safetensors",
        "name": "VAE Model (FP16)",
        "description": "Variational Autoencoder for SeedVR2",
        "size_hint": "~330 MB",
    },
    "3b": {
        "filename": "seedvr2_ema_3b_fp16.safetensors",
        "name": "SeedVR2 3B Model (FP16)",
        "description": "3 billion parameter model - faster, lower VRAM",
        "size_hint": "~6.5 GB",
    },
    "7b": {
        "filename": "seedvr2_ema_7b_fp16.safetensors",
        "name": "SeedVR2 7B Model (FP16)",
        "description": "7 billion parameter model - best quality",
        "size_hint": "~14 GB",
    },
    "7b_sharp": {
        "filename": "seedvr2_ema_7b_sharp_fp16.safetensors",
        "name": "SeedVR2 7B Sharp Model (FP16)",
        "description": "7 billion parameter model - sharpened variant",
        "size_hint": "~14 GB",
    },
    "7b_fp8_mixed_block35": {
        "filename": "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
        "name": "SeedVR2 7B FP8 Mixed Block35 Model",
        "description": "7 billion parameter model - FP8 e4m3fn mixed block35 variant",
        "size_hint": "~14 GB",
    },
    "7b_sharp_fp8_mixed_block35": {
        "filename": "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors",
        "name": "SeedVR2 7B Sharp FP8 Mixed Block35 Model",
        "description": "7 billion parameter sharpened model - FP8 e4m3fn mixed block35 variant",
        "size_hint": "~14 GB",
    },
}

ALL_SEEDVR2_MODEL_IDS = [
    "vae",
    "3b",
    "7b",
    "7b_sharp",
    "7b_fp8_mixed_block35",
    "7b_sharp_fp8_mixed_block35",
]

# RIFE configuration - downloads selected folders from Wan_GGUF/RIFE_Models
RIFE_CONFIG = {
    "name": "RIFE Models (4.14-4.26)",
    "description": "Frame interpolation models for RIFE",
    "repo_id": RIFE_REPO_ID,
    "repo_subdir": RIFE_REPO_SUBDIR,
    "versions": RIFE_VERSION_FOLDERS,
    "target_dir": RIFE_MODELS_DIR,
}

BESTIMAGEUPSCALE_CONFIG = {
    "name": "BestImageUpscalers Models",
    "description": "High-quality image upscaling models",
    "repo_id": BESTIMAGEUPSCALE_REPO_ID,
    "target_dir": IMAGE_UPSCALE_MODELS_DIR,
}

DOWNLOAD_CONFIG = {
    "num_connections": 16,
    "chunk_size": 10 * 1024 * 1024,  # 10MB buffer
    "max_retries": 5,
    "retry_delay": 2,
    "max_retry_delay": 30,
    "connect_timeout": 30,
    "read_timeout": 300,
}


class RobustDownloader:
    """Cross-platform robust file downloader with resume support and verification."""
    
    def __init__(self, config: Dict, skip_verify: bool = False):
        self.config = config
        self.skip_verify = skip_verify
        self.session = self._create_session()
        
        # Ensure cache directory exists
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Cache files
        self.sha_cache_file = CACHE_DIR / "sha256_cache.json"
        self.sha_cache = self._load_json_cache(self.sha_cache_file)
        self.verified_cache_file = CACHE_DIR / "verified_files_cache.json"
        self.verified_cache = self._load_json_cache(self.verified_cache_file)
        
        # Progress tracking
        self._progress_lock = threading.Lock()
        self._active_progress = False
        self._last_progress_len = 0
        
        # In-memory metadata cache (per run) to avoid repeated HF metadata calls
        self._hf_metadata_cache = {}

    def _create_session(self) -> requests.Session:
        """Create a configured requests session with retry logic."""
        session = requests.Session()
        
        # Use Retry from urllib3 for proper retry handling
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=self.config["max_retries"],
            backoff_factor=self.config["retry_delay"],
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET"],
        )
        
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=retry_strategy
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        return session

    def _load_json_cache(self, filepath: Path) -> Dict:
        """Load JSON cache file safely."""
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (IOError, json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not load cache {filepath.name}: {e}")
                return {}
        return {}

    def _save_json_cache(self, filepath: Path, data: Dict) -> None:
        """Save JSON cache file safely."""
        try:
            # Write to temp file first, then rename for atomic operation
            temp_file = filepath.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            # Atomic rename (works on both Windows and Linux)
            temp_file.replace(filepath)
        except (IOError, OSError) as e:
            logger.warning(f"Could not save cache {filepath.name}: {e}")

    def _get_terminal_width(self) -> int:
        """Get terminal width for progress display."""
        try:
            return shutil.get_terminal_size(fallback=(100, 20)).columns
        except Exception:
            return 100

    def clear_progress_line(self) -> None:
        """Clear the current progress line."""
        with self._progress_lock:
            if self._active_progress:
                width = self._get_terminal_width()
                sys.stdout.write("\r" + " " * width + "\r")
                sys.stdout.flush()
                self._last_progress_len = 0
                self._active_progress = False

    def show_progress_line(self, text: str) -> None:
        """Display progress on a single line."""
        with self._progress_lock:
            width = self._get_terminal_width()
            max_len = max(1, width - 1)
            if len(text) > max_len:
                text = text[:max_len]
            sys.stdout.write("\r" + text)
            extra_spaces = max(0, self._last_progress_len - len(text))
            if extra_spaces:
                sys.stdout.write(" " * extra_spaces)
                sys.stdout.write("\r" + text)
            sys.stdout.flush()
            self._last_progress_len = len(text)
            self._active_progress = True

    def finalize_progress_line(self, final_text: Optional[str] = None) -> None:
        """Finalize progress line with optional final message."""
        with self._progress_lock:
            if final_text is not None:
                width = self._get_terminal_width()
                max_len = max(1, width - 1)
                if len(final_text) > max_len:
                    final_text = final_text[:max_len]
                sys.stdout.write("\r" + final_text)
                extra_spaces = max(0, self._last_progress_len - len(final_text))
                if extra_spaces:
                    sys.stdout.write(" " * extra_spaces)
                    sys.stdout.write("\r" + final_text)
                sys.stdout.write("\n")
                sys.stdout.flush()
            else:
                if self._active_progress:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
            self._last_progress_len = 0
            self._active_progress = False

    def log(self, msg: str) -> None:
        """Log a message, clearing progress line if needed."""
        with self._progress_lock:
            if self._active_progress:
                width = self._get_terminal_width()
                sys.stdout.write("\r" + " " * width + "\r")
                sys.stdout.flush()
                self._last_progress_len = 0
                self._active_progress = False
            print(msg, flush=True)

    def is_file_verified(self, repo_id: str, filename: str, filepath: Path, expected_sha: str) -> bool:
        """Check if file was previously verified and hasn't changed."""
        if not expected_sha:
            return False
        cache_key = f"{repo_id}/{filename}"
        if cache_key not in self.verified_cache:
            return False
        cached_info = self.verified_cache[cache_key]
        if not filepath.exists():
            return False
        current_size = filepath.stat().st_size
        current_mtime = filepath.stat().st_mtime
        return (cached_info.get('sha256') == expected_sha and
                cached_info.get('size') == current_size and
                abs(cached_info.get('mtime', 0) - current_mtime) < 1.0)

    def mark_file_verified(self, repo_id: str, filename: str, filepath: Path, sha256: str) -> None:
        """Mark a file as verified in the cache."""
        cache_key = f"{repo_id}/{filename}"
        if filepath.exists():
            self.verified_cache[cache_key] = {
                'sha256': sha256,
                'size': filepath.stat().st_size,
                'mtime': filepath.stat().st_mtime,
                'verified_at': time.time()
            }
            self._save_json_cache(self.verified_cache_file, self.verified_cache)

    @staticmethod
    def _normalize_etag(etag: str) -> str:
        return etag.replace('"', '').replace('W/', '').strip()

    @classmethod
    def _extract_sha256_from_etag(cls, etag: str) -> Optional[str]:
        """Extract sha256 from HuggingFace etag when available.
        
        HF etags are often in the form 'sha256:<hash>' for LFS files, but not always.
        Returns a 64-hex sha256 string when it can be confidently extracted.
        """
        if not etag:
            return None
        etag_norm = cls._normalize_etag(etag)
        candidate = etag_norm[7:] if etag_norm.startswith('sha256:') else etag_norm
        candidate = candidate.strip()
        if len(candidate) == 64 and all(c in '0123456789abcdef' for c in candidate.lower()):
            return candidate
        return None

    def get_hf_file_info(self, repo_id: str, filename: str) -> Dict[str, Optional[object]]:
        """Best-effort HuggingFace metadata fetch (size, etag, sha256).
        
        This is more reliable than raw HTTP HEAD for some files, and avoids failing downloads
        when servers omit Content-Length.
        """
        cache_key = f"{repo_id}/{filename}"
        cached = self._hf_metadata_cache.get(cache_key)
        if isinstance(cached, dict):
            return cached
        
        info: Dict[str, Optional[object]] = {"size": None, "etag": None, "sha256": None}
        try:
            url = hf_hub_url(repo_id, filename)
            metadata = get_hf_file_metadata(url)
            size = getattr(metadata, "size", None)
            if isinstance(size, int):
                info["size"] = size
            etag = getattr(metadata, "etag", None)
            if isinstance(etag, str) and etag:
                info["etag"] = etag
                sha = self._extract_sha256_from_etag(etag)
                if sha:
                    info["sha256"] = sha
        except Exception as e:
            logger.debug(f"Could not get HF metadata for {repo_id}/{filename}: {e}")
        
        self._hf_metadata_cache[cache_key] = info
        return info

    def get_file_sha256(self, repo_id: str, filename: str) -> Optional[str]:
        """Get SHA256 hash for a file from HuggingFace."""
        cache_key = f"{repo_id}/{filename}"
        cached_val = self.sha_cache.get(cache_key)
        if isinstance(cached_val, str) and cached_val:
            return cached_val
        
        info = self.get_hf_file_info(repo_id, filename)
        sha256 = info.get("sha256") if isinstance(info, dict) else None
        if isinstance(sha256, str) and sha256:
            self.sha_cache[cache_key] = sha256
            self._save_json_cache(self.sha_cache_file, self.sha_cache)
            return sha256
        return None

    @staticmethod
    def format_bytes(bytes_val: float) -> str:
        """Format bytes to human readable string."""
        if bytes_val < 0:
            return "0 B"
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_val < 1024.0:
                return f"{bytes_val:.1f} {unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.1f} TB"

    @staticmethod
    def format_time(seconds: float) -> str:
        """Format seconds to human readable string."""
        if seconds < 0:
            return "0s"
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            mins = seconds // 60
            secs = seconds % 60
            return f"{mins}m {secs}s" if secs else f"{mins}m"
        else:
            hours = seconds // 3600
            mins = (seconds % 3600) // 60
            return f"{hours}h {mins}m" if mins else f"{hours}h"

    def print_progress(self, current: int, total: int, start_time: float, 
                       filename: str, speed_bytes_per_sec: Optional[float] = None) -> None:
        """Print download progress bar."""
        if total <= 0:
            return
        elapsed = max(0.001, time.time() - start_time)
        percent = min(100.0, (current / total * 100))
        if speed_bytes_per_sec is None:
            speed_bytes_per_sec = current / elapsed if elapsed > 0 else 0.0
        if speed_bytes_per_sec > 0 and total > current:
            eta = (total - current) / speed_bytes_per_sec
            eta_str = self.format_time(eta)
        else:
            eta_str = "Complete" if current >= total else "Unknown"
        bar_length = 30
        filled = int(percent * bar_length / 100)
        bar = "█" * max(0, min(filled, bar_length)) + "░" * max(0, bar_length - filled)
        speed_str = self.format_bytes(speed_bytes_per_sec) + "/s" if speed_bytes_per_sec else "0 B/s"
        line = (
            f"{filename}: [{bar}] {percent:.1f}% "
            f"({self.format_bytes(current)}/{self.format_bytes(total)}) "
            f"{speed_str} ETA: {eta_str}"
        )
        self.show_progress_line(line)

    def print_progress_unknown(self, current: int, start_time: float,
                               filename: str, speed_bytes_per_sec: Optional[float] = None) -> None:
        """Print progress when total size is unknown."""
        elapsed = max(0.001, time.time() - start_time)
        if speed_bytes_per_sec is None:
            speed_bytes_per_sec = current / elapsed if elapsed > 0 else 0.0
        speed_str = self.format_bytes(speed_bytes_per_sec) + "/s" if speed_bytes_per_sec else "0 B/s"
        line = f"{filename}: {self.format_bytes(current)} downloaded {speed_str}"
        self.show_progress_line(line)

    @staticmethod
    def _parse_total_size_from_content_range(content_range: Optional[str]) -> Optional[int]:
        """Parse total size from Content-Range header.
        
        Examples:
          - 'bytes 0-0/1234'
          - 'bytes */1234'
        """
        if not content_range or '/' not in content_range:
            return None
        total = content_range.split('/')[-1].strip()
        if total.isdigit():
            return int(total)
        return None

    def get_file_url(self, repo_id: str, filename: str) -> str:
        """Get download URL for a file, properly URL-encoded."""
        encoded_filename = urllib.parse.quote(filename, safe='/')
        return f"https://huggingface.co/{repo_id}/resolve/main/{encoded_filename}"

    def get_file_size(self, url: str, repo_id: Optional[str] = None, filename: Optional[str] = None) -> Optional[int]:
        """Get file size, best-effort.
        
        Some HF files (especially small text files or files served with chunked encoding)
        may not return Content-Length on HEAD/Range. We try:
        - HuggingFace metadata API (most reliable)
        - HTTP HEAD Content-Length
        - HTTP Range Content-Range
        - HTTP GET Content-Length (without downloading body)
        """
        if repo_id and filename:
            info = self.get_hf_file_info(repo_id, filename)
            size = info.get("size") if isinstance(info, dict) else None
            if isinstance(size, int):
                return size
        
        timeout = (self.config["connect_timeout"], self.config["read_timeout"])
        
        # HEAD
        try:
            with self.session.head(url, timeout=timeout, allow_redirects=True) as response:
                if response.status_code == 200:
                    content_length = response.headers.get('content-length')
                    if content_length and content_length.isdigit():
                        return int(content_length)
        except Exception as e:
            logger.debug(f"Could not get file size via HEAD: {e}")
        
        # Range GET
        try:
            headers = {'Range': 'bytes=0-0'}
            with self.session.get(url, headers=headers, timeout=timeout, stream=True) as response:
                if response.status_code == 206:
                    total = self._parse_total_size_from_content_range(response.headers.get('content-range'))
                    if total is not None:
                        return total
                # Some servers ignore Range but still provide Content-Length on GET
                if response.status_code == 200:
                    content_length = response.headers.get('content-length')
                    if content_length and content_length.isdigit():
                        return int(content_length)
        except Exception as e:
            logger.debug(f"Could not get file size via Range GET: {e}")
        
        # GET (headers only, no body read)
        try:
            with self.session.get(url, timeout=timeout, stream=True) as response:
                if response.status_code == 200:
                    content_length = response.headers.get('content-length')
                    if content_length and content_length.isdigit():
                        return int(content_length)
        except Exception as e:
            logger.debug(f"Could not get file size via GET: {e}")
        
        return None

    def verify_file_sha256(self, filepath: Path, expected_sha: str, filename: str = "") -> bool:
        """Verify file SHA256 hash."""
        if not expected_sha:
            return True
        display_name = filename or filepath.name
        self.log(f"[VERIFYING] Computing SHA256 for {display_name}...")
        try:
            sha256_hash = hashlib.sha256()
            file_size = filepath.stat().st_size
            bytes_read = 0
            start_time = time.time()
            last_update = 0.0
            chunk_size = 8 * 1024 * 1024  # 8MB for verification
            
            with open(filepath, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    sha256_hash.update(chunk)
                    bytes_read += len(chunk)
                    now = time.time()
                    if now - last_update >= 0.2:
                        percent = (bytes_read / file_size) * 100 if file_size > 0 else 0.0
                        speed = bytes_read / max(0.001, (now - start_time))
                        line = (
                            f"[VERIFYING] {display_name}: {percent:.1f}% "
                            f"({self.format_bytes(bytes_read)}/{self.format_bytes(file_size)}) "
                            f"{self.format_bytes(speed)}/s"
                        )
                        self.show_progress_line(line)
                        last_update = now
            
            computed_sha = sha256_hash.hexdigest()
            if computed_sha.lower() == expected_sha.lower():
                self.finalize_progress_line(f"[VERIFIED] SHA256 match: {computed_sha[:16]}...")
                return True
            else:
                self.finalize_progress_line(f"[ERROR] SHA256 mismatch!")
                self.log(f"  Expected: {expected_sha}")
                self.log(f"  Got:      {computed_sha}")
                return False
        except Exception as e:
            self.finalize_progress_line()
            self.log(f"[ERROR] Failed to verify SHA256: {e}")
            return False

    def download_chunk(self, url: str, start: int, end: int,
                       filepath: Path, chunk_id: int,
                       progress_callback=None) -> bool:
        """Download a single chunk of a file."""
        chunk_file = filepath.with_suffix(f'.part{chunk_id}')
        chunk_size_expected = end - start + 1
        
        # Check for existing partial download
        if chunk_file.exists():
            existing_size = chunk_file.stat().st_size
            if existing_size == chunk_size_expected:
                if progress_callback:
                    progress_callback(chunk_id, chunk_size_expected)
                return True
            elif existing_size > chunk_size_expected:
                chunk_file.unlink()
                resume_pos = 0
            else:
                resume_pos = existing_size
        else:
            resume_pos = 0

        timeout = (self.config["connect_timeout"], self.config["read_timeout"])
        max_retries = self.config["max_retries"]
        
        for attempt in range(max_retries):
            try:
                actual_start = start + resume_pos
                headers = {'Range': f'bytes={actual_start}-{end}'}
                with self.session.get(url, headers=headers, timeout=timeout, stream=True) as response:
                    # For parallel chunking we REQUIRE Range support (206)
                    if response.status_code != 206:
                        raise Exception(f"Server did not honor Range request (status {response.status_code})")
                    
                    mode = 'ab' if resume_pos > 0 else 'wb'
                    downloaded = resume_pos
                    
                    with open(chunk_file, mode) as f:
                        for data in response.iter_content(chunk_size=self.config["chunk_size"]):
                            if data:
                                f.write(data)
                                downloaded += len(data)
                                if progress_callback:
                                    progress_callback(chunk_id, downloaded)
                
                final_size = chunk_file.stat().st_size
                if final_size == chunk_size_expected:
                    if progress_callback:
                        progress_callback(chunk_id, chunk_size_expected)
                    return True
                elif final_size < chunk_size_expected:
                    resume_pos = final_size
                    raise Exception(f"Chunk incomplete: {final_size}/{chunk_size_expected}")
                else:
                    chunk_file.unlink()
                    resume_pos = 0
                    raise Exception(f"Chunk too large: {final_size}/{chunk_size_expected}")
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = min(self.config["retry_delay"] * (2 ** attempt),
                                self.config["max_retry_delay"])
                    time.sleep(delay)
                else:
                    self.log(f"Chunk {chunk_id} failed after {max_retries} attempts: {e}")
                    return False
        return False

    def supports_range(self, url: str) -> bool:
        """Check if server honors HTTP Range requests (required for parallel download)."""
        timeout = (self.config["connect_timeout"], self.config["read_timeout"])
        try:
            headers = {'Range': 'bytes=0-0'}
            with self.session.get(url, headers=headers, timeout=timeout, stream=True) as response:
                return response.status_code == 206
        except Exception:
            return False

    def merge_chunks(self, filepath: Path, num_chunks: int) -> bool:
        """Merge downloaded chunks into final file."""
        temp_file = filepath.with_suffix('.tmp')
        try:
            total_size = 0
            chunk_files = []
            for i in range(num_chunks):
                chunk_file = filepath.with_suffix(f'.part{i}')
                if not chunk_file.exists():
                    self.log(f"Error: Missing chunk {i}")
                    return False
                size = chunk_file.stat().st_size
                chunk_files.append((chunk_file, size))
                total_size += size

            buffer_size = 64 * 1024 * 1024  # 64MB buffer for merging
            merged_size = 0
            start_time = time.time()
            last_update = 0.0

            with open(temp_file, 'wb') as outfile:
                for chunk_file, chunk_size in chunk_files:
                    with open(chunk_file, 'rb') as infile:
                        bytes_copied = 0
                        while bytes_copied < chunk_size:
                            to_read = min(buffer_size, chunk_size - bytes_copied)
                            data = infile.read(to_read)
                            if not data:
                                break
                            outfile.write(data)
                            bytes_copied += len(data)
                            merged_size += len(data)
                            now = time.time()
                            if now - last_update >= 0.5:
                                percent = (merged_size / total_size) * 100 if total_size > 0 else 0.0
                                speed = merged_size / max(0.001, (now - start_time))
                                line = (
                                    f"[MERGING] {percent:.1f}% "
                                    f"({self.format_bytes(merged_size)}/{self.format_bytes(total_size)}) "
                                    f"Speed: {self.format_bytes(speed)}/s"
                                )
                                self.show_progress_line(line)
                                last_update = now

            elapsed = max(0.001, time.time() - start_time)
            self.finalize_progress_line(
                f"[MERGING] 100.0% - Completed in {self.format_time(elapsed)}"
            )

            # Atomic rename
            with suppress(OSError):
                filepath.unlink()
            temp_file.replace(filepath)

            # Clean up chunk files
            for chunk_file, _ in chunk_files:
                with suppress(OSError):
                    chunk_file.unlink()
            return True
            
        except Exception as e:
            self.finalize_progress_line()
            self.log(f"Error merging: {e}")
            with suppress(OSError):
                temp_file.unlink()
            return False

    def download_file(self, repo_id: str, filename: str, local_dir: Path, local_filename: Optional[str] = None) -> bool:
        """Download a single file with resume support and verification."""
        url = self.get_file_url(repo_id, filename)
        filepath = local_dir / (local_filename if local_filename is not None else filename)
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        expected_sha = None if self.skip_verify else self.get_file_sha256(repo_id, filename)
        file_size = self.get_file_size(url, repo_id=repo_id, filename=filename)

        # Check if file already exists and is complete/verified
        if filepath.exists():
            # If we have a reliable sha256 from HF, prefer sha-based verification even if size is unknown.
            if expected_sha:
                if self.is_file_verified(repo_id, filename, filepath, expected_sha):
                    self.log(f"[SKIP] {filename} already complete and verified (cached)")
                    return True
                if self.verify_file_sha256(filepath, expected_sha, filename):
                    self.mark_file_verified(repo_id, filename, filepath, expected_sha)
                    self.log(f"[SKIP] {filename} already complete and verified")
                    return True
                else:
                    self.log(f"[WARNING] {filename} failed verification, re-downloading")
                    with suppress(OSError):
                        filepath.unlink()
            else:
                # Size-only check if we can determine size
                if file_size is not None:
                    actual_size = filepath.stat().st_size
                    if actual_size == file_size:
                        self.log(f"[SKIP] {filename} already complete ({self.format_bytes(file_size)})")
                        return True
                    elif actual_size > file_size:
                        self.log(f"[WARNING] {filename} corrupted (size mismatch), re-downloading")
                        with suppress(OSError):
                            filepath.unlink()

        # Download (even if size is unknown)
        if file_size is not None:
            # Use parallel download for large files (>50MB), single for smaller
            if file_size > 50 * 1024 * 1024:
                success = self.download_parallel(url, filepath, filename, file_size)
            else:
                success = self.download_single(url, filepath, filename, file_size)
        else:
            self.log(f"[INFO] Size unknown for {filename}; using streaming download with resume")
            success = self.download_single_unknown_size(url, filepath, filename)

        # Verify after download
        if success and expected_sha:
            if not self.verify_file_sha256(filepath, expected_sha, filename):
                self.log(f"[ERROR] {filename} downloaded but failed SHA256 verification")
                with suppress(OSError):
                    filepath.unlink()
                return False
            else:
                self.mark_file_verified(repo_id, filename, filepath, expected_sha)
        return success

    def download_parallel(self, url: str, filepath: Path, filename: str, file_size: int) -> bool:
        """Download file using multiple parallel connections."""
        if not self.supports_range(url):
            self.log(f"[WARNING] {filename}: server did not honor Range requests, falling back to single connection")
            return self.download_single(url, filepath, filename, file_size)

        num_chunks = self.config["num_connections"]
        base_chunk_size = file_size // num_chunks
        chunks = []
        for i in range(num_chunks):
            start = i * base_chunk_size
            if i == num_chunks - 1:
                end = file_size - 1
            else:
                end = (i + 1) * base_chunk_size - 1
            chunks.append((i, start, end))

        self.log(f"[DOWNLOADING] {filename} ({self.format_bytes(file_size)}) using {num_chunks} connections")

        chunk_progress = {}
        progress_lock = threading.Lock()

        def update_progress(chunk_id: int, bytes_downloaded: int):
            with progress_lock:
                chunk_progress[chunk_id] = bytes_downloaded

        # Check existing progress
        for chunk_id, start, end in chunks:
            chunk_file = filepath.with_suffix(f'.part{chunk_id}')
            if chunk_file.exists():
                chunk_progress[chunk_id] = chunk_file.stat().st_size
            else:
                chunk_progress[chunk_id] = 0

        initial_bytes = sum(chunk_progress.values())
        if initial_bytes > 0:
            self.log(f"[RESUMING] Already downloaded: {self.format_bytes(initial_bytes)}")

        start_time = time.time()
        failed_chunks = []

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_chunks) as executor:
                futures = {}
                for chunk_id, start, end in chunks:
                    chunk_file = filepath.with_suffix(f'.part{chunk_id}')
                    expected_size = end - start + 1
                    if chunk_file.exists() and chunk_file.stat().st_size == expected_size:
                        continue
                    future = executor.submit(
                        self.download_chunk,
                        url, start, end, filepath, chunk_id,
                        update_progress
                    )
                    futures[future] = chunk_id

                if not futures:
                    self.log("[MERGING] All chunks already complete")
                    if self.merge_chunks(filepath, num_chunks):
                        if filepath.stat().st_size == file_size:
                            self.log(f"[OK] {filename} completed")
                            return True
                    return False

                last_update = 0.0
                last_bytes = initial_bytes

                while futures:
                    done, pending = concurrent.futures.wait(
                        list(futures.keys()),
                        timeout=0.5,
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for future in done:
                        chunk_id = futures.pop(future)
                        try:
                            success = future.result()
                            if not success:
                                failed_chunks.append(chunk_id)
                        except Exception as e:
                            self.log(f"Chunk {chunk_id} exception: {e}")
                            failed_chunks.append(chunk_id)

                    current_time = time.time()
                    if current_time - last_update >= 0.5 or not futures:
                        with progress_lock:
                            current_bytes = sum(chunk_progress.values())
                        time_delta = current_time - last_update if last_update > 0 else (current_time - start_time)
                        bytes_delta = current_bytes - last_bytes if last_update > 0 else (current_bytes - initial_bytes)
                        speed = bytes_delta / max(0.001, time_delta)
                        self.print_progress(current_bytes, file_size, start_time, filename, speed)
                        last_update = current_time
                        last_bytes = current_bytes

        except KeyboardInterrupt:
            self.clear_progress_line()
            self.log("[INTERRUPTED] Download interrupted by user")
            raise

        self.clear_progress_line()

        if failed_chunks:
            self.log(f"[ERROR] Failed chunks: {failed_chunks}")
            return False

        # Verify all chunks
        for chunk_id, start, end in chunks:
            chunk_file = filepath.with_suffix(f'.part{chunk_id}')
            expected_size = end - start + 1
            if not chunk_file.exists():
                self.log(f"[ERROR] Missing chunk {chunk_id}")
                return False
            if chunk_file.stat().st_size != expected_size:
                self.log(f"[ERROR] Chunk {chunk_id} incomplete")
                return False

        self.log(f"[MERGING] Merging {num_chunks} chunks...")
        if self.merge_chunks(filepath, num_chunks):
            final_size = filepath.stat().st_size
            if final_size == file_size:
                elapsed = max(0.001, time.time() - start_time)
                avg_speed = (file_size - initial_bytes) / elapsed if elapsed > 0 else 0
                self.log(f"[OK] {filename} completed in {self.format_time(elapsed)} "
                         f"- Average: {self.format_bytes(avg_speed)}/s")
                return True
            else:
                self.log(f"[ERROR] Final size mismatch: {final_size} != {file_size}")
                with suppress(OSError):
                    filepath.unlink()
                return False
        else:
            self.log(f"[ERROR] Merge failed")
            return False

    def download_single(self, url: str, filepath: Path, filename: str, file_size: int) -> bool:
        """Download file using single connection with resume support."""
        timeout = (self.config["connect_timeout"], self.config["read_timeout"])
        max_retries = self.config["max_retries"]
        
        for attempt in range(max_retries):
            try:
                resume_pos = 0
                if filepath.exists():
                    resume_pos = filepath.stat().st_size
                    if resume_pos >= file_size:
                        self.log(f"[OK] {filename} already complete")
                        return True

                # Try resume first (Range)
                response = None
                if resume_pos > 0:
                    headers = {'Range': f'bytes={resume_pos}-'}
                    response = self.session.get(url, headers=headers, timeout=timeout, stream=True)
                    if response.status_code != 206:
                        with suppress(Exception):
                            response.close()
                        response = None
                        self.log(f"[WARNING] Resume not supported, restarting")
                        resume_pos = 0
                
                # Fresh download
                if response is None:
                    response = self.session.get(url, timeout=timeout, stream=True)

                with response:
                    response.raise_for_status()
                    mode = 'ab' if resume_pos > 0 and response.status_code == 206 else 'wb'
                    if mode == 'wb':
                        resume_pos = 0
                    
                    downloaded = 0
                    start_time = time.time()
                    last_update = 0.0

                    if resume_pos > 0:
                        self.log(f"[RESUMING] {filename} from {self.format_bytes(resume_pos)}")
                    else:
                        self.log(f"[DOWNLOADING] {filename} ({self.format_bytes(file_size)})")

                    with open(filepath, mode) as f:
                        for chunk in response.iter_content(chunk_size=self.config["chunk_size"]):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                now = time.time()
                                if now - last_update >= 0.5:
                                    total = resume_pos + downloaded
                                    self.print_progress(total, file_size, start_time, filename)
                                    last_update = now

                total = resume_pos + downloaded
                self.print_progress(total, file_size, start_time, filename)
                self.clear_progress_line()

                if filepath.stat().st_size == file_size:
                    self.log(f"[OK] {filename} completed")
                    return True
                else:
                    self.log(f"[ERROR] Size mismatch")
                    continue

            except KeyboardInterrupt:
                self.finalize_progress_line()
                self.log("[INTERRUPTED] Download interrupted by user")
                raise
            except Exception as e:
                self.finalize_progress_line()
                self.log(f"[ERROR] Attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    delay = min(self.config["retry_delay"] * (2 ** attempt),
                                self.config["max_retry_delay"])
                    time.sleep(delay)
                else:
                    return False
        return False

    def download_single_unknown_size(self, url: str, filepath: Path, filename: str) -> bool:
        """Download file with resume support when remote size is unknown.
        
        Uses Range resume when supported; otherwise restarts from scratch.
        """
        timeout = (self.config["connect_timeout"], self.config["read_timeout"])
        max_retries = self.config["max_retries"]
        
        for attempt in range(max_retries):
            try:
                resume_pos = filepath.stat().st_size if filepath.exists() else 0
                use_range = resume_pos > 0
                headers = {'Range': f'bytes={resume_pos}-'} if use_range else {}
                
                with self.session.get(url, headers=headers, timeout=timeout, stream=True) as response:
                    # If we attempted resume but server rejects the range, restart cleanly.
                    if use_range and response.status_code == 416:
                        total = self._parse_total_size_from_content_range(response.headers.get('content-range'))
                        if total is not None and resume_pos == total:
                            self.log(f"[OK] {filename} already complete")
                            return True
                        self.log(f"[WARNING] {filename} resume range not satisfiable, restarting")
                        with suppress(OSError):
                            filepath.unlink()
                        continue
                    
                    if use_range and response.status_code != 206:
                        self.log(f"[WARNING] Resume not supported for {filename}, restarting")
                        with suppress(OSError):
                            filepath.unlink()
                        continue
                    
                    response.raise_for_status()
                    
                    # Try to infer total size from response headers when possible
                    total_size: Optional[int] = None
                    content_range = response.headers.get('content-range')
                    total_size = self._parse_total_size_from_content_range(content_range)
                    if total_size is None:
                        content_length = response.headers.get('content-length')
                        if content_length and content_length.isdigit():
                            if response.status_code == 206 and resume_pos > 0:
                                total_size = resume_pos + int(content_length)
                            elif response.status_code == 200:
                                total_size = int(content_length)
                    
                    mode = 'ab' if use_range else 'wb'
                    downloaded = 0
                    start_time = time.time()
                    last_update = 0.0
                    
                    if use_range:
                        self.log(f"[RESUMING] {filename} from {self.format_bytes(resume_pos)}")
                    else:
                        if total_size is not None:
                            self.log(f"[DOWNLOADING] {filename} ({self.format_bytes(total_size)})")
                        else:
                            self.log(f"[DOWNLOADING] {filename} (size unknown)")
                    
                    with open(filepath, mode) as f:
                        for chunk in response.iter_content(chunk_size=self.config["chunk_size"]):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                now = time.time()
                                if now - last_update >= 0.5:
                                    total_dl = resume_pos + downloaded
                                    if total_size is not None:
                                        self.print_progress(total_dl, total_size, start_time, filename)
                                    else:
                                        self.print_progress_unknown(total_dl, start_time, filename)
                                    last_update = now
                
                total_dl = resume_pos + downloaded
                if total_size is not None:
                    self.print_progress(total_dl, total_size, start_time, filename)
                else:
                    self.print_progress_unknown(total_dl, start_time, filename)
                self.clear_progress_line()
                
                # Validate if we could infer total size
                if total_size is not None:
                    final_size = filepath.stat().st_size if filepath.exists() else -1
                    if final_size != total_size:
                        self.log(f"[ERROR] Size mismatch")
                        continue
                
                self.log(f"[OK] {filename} completed")
                return True
                
            except KeyboardInterrupt:
                self.finalize_progress_line()
                self.log("[INTERRUPTED] Download interrupted by user")
                raise
            except Exception as e:
                self.finalize_progress_line()
                self.log(f"[ERROR] Attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    delay = min(self.config["retry_delay"] * (2 ** attempt),
                                self.config["max_retry_delay"])
                    time.sleep(delay)
                else:
                    return False
        return False

    def download_repo(
        self,
        repo_id: str,
        local_dir: Path,
        exclude_patterns: Optional[List[str]] = None,
        include_prefixes: Optional[List[str]] = None,
        strip_prefix: Optional[str] = None,
    ) -> Tuple[int, int]:
        """Download all files from a HuggingFace repository.
        
        Args:
            repo_id: HuggingFace repository ID (e.g., 'MonsterMMORPG/Wan_GGUF')
            local_dir: Local directory to download files to
            exclude_patterns: List of patterns to exclude (e.g., ['.gitattributes', 'README.md'])
            include_prefixes: Optional list of repo path prefixes to include
            strip_prefix: Optional repo path prefix to remove from local paths
        
        Returns:
            Tuple of (successful_count, failed_count)
        """
        if exclude_patterns is None:
            exclude_patterns = ['.gitattributes', '.gitignore']
        
        normalized_prefixes = [p.strip('/') for p in include_prefixes or [] if p and p.strip('/')]
        normalized_strip_prefix = strip_prefix.strip('/') if strip_prefix and strip_prefix.strip('/') else None
        
        self.log(f"[INFO] Listing files in repository: {repo_id}")
        
        try:
            files = list(list_repo_files(repo_id))
        except Exception as e:
            self.log(f"[ERROR] Failed to list repository files: {e}")
            return 0, 1
        
        # Filter out excluded files
        filtered_files = []
        for f in files:
            excluded = False
            for pattern in exclude_patterns:
                if pattern in f or f.endswith(pattern):
                    excluded = True
                    break
            if not excluded:
                if normalized_prefixes:
                    included = any(f == prefix or f.startswith(f"{prefix}/") for prefix in normalized_prefixes)
                    if not included:
                        continue
                filtered_files.append(f)
        
        self.log(f"[INFO] Found {len(filtered_files)} files to download")
        
        # Ensure directory exists
        local_dir.mkdir(parents=True, exist_ok=True)
        
        successful = 0
        failed = 0
        failed_files = []
        
        for i, filename in enumerate(filtered_files, 1):
            self.log(f"\n[{i}/{len(filtered_files)}] {filename}")
            local_filename = filename
            if normalized_strip_prefix:
                strip_token = f"{normalized_strip_prefix}/"
                if filename.startswith(strip_token):
                    local_filename = filename[len(strip_token):]
                elif filename == normalized_strip_prefix:
                    continue
            if self.download_file(repo_id, filename, local_dir, local_filename=local_filename):
                successful += 1
            else:
                failed += 1
                failed_files.append(filename)
        
        if failed_files:
            self.log(f"\n[WARNING] Failed to download {len(failed_files)} files:")
            for f in failed_files:
                self.log(f"  - {f}")
        
        return successful, failed


def get_model_choice() -> Tuple[List[str], bool, bool]:
    """Interactive model selection.
    
    Returns:
        Tuple of (model_ids list, include_rife boolean, include_best boolean)
    """
    print("\n" + "=" * 60)
    print("Ultimate Video/Image Upscaler Model Downloader")
    print("=" * 60)
    print(f"\nSeedVR2 Repository: {SEEDVR2_REPO_ID}")
    print(f"SeedVR2 Target: {SEEDVR2_MODELS_DIR}")
    print(f"\nRIFE Repository: {RIFE_REPO_ID}/{RIFE_REPO_SUBDIR}")
    print(f"RIFE Target: {RIFE_MODELS_DIR}")
    print(f"\nBestImageUpscalers Repository: {BESTIMAGEUPSCALE_REPO_ID}")
    print(f"BestImageUpscalers Target: {IMAGE_UPSCALE_MODELS_DIR}")
    print("\nAvailable models:")
    print()
    print("=" * 40)
    print("SeedVR2 Models:")
    print("=" * 40)
    print("1. VAE Model (~330 MB)")
    print("   Required for all SeedVR2 operations")
    print()
    print("2. SeedVR2 3B Model (~6.5 GB)")
    print("   Faster processing, lower VRAM requirement")
    print()
    print("3. SeedVR2 7B Model (~14 GB)")
    print("   Best quality, higher VRAM requirement")
    print()
    print("4. SeedVR2 7B Sharp Model (~14 GB)")
    print("   Sharpened variant of 7B model")
    print()
    print("5. ALL SeedVR2 MODELS (~63 GB total)")
    print("   Download all SeedVR2 models")
    print()
    print("12. SeedVR2 7B FP8 Mixed Block35 Model (~14 GB)")
    print("    FP8 e4m3fn mixed block35 variant")
    print()
    print("13. SeedVR2 7B Sharp FP8 Mixed Block35 Model (~14 GB)")
    print("    Sharpened FP8 e4m3fn mixed block35 variant")
    print()
    print("6. VAE + 3B Only (~7 GB)")
    print("   Recommended for GPUs with <16GB VRAM")
    print()
    print("7. VAE + 7B Only (~14.5 GB)")
    print("   Recommended for GPUs with 16GB+ VRAM")
    print()
    print("=" * 40)
    print("RIFE Models:")
    print("=" * 40)
    print("8. RIFE Models (4.14-4.26)")
    print("   Frame interpolation models")
    print()
    print("=" * 40)
    print("Image Upscale Models:")
    print("=" * 40)
    print("10. BestImageUpscalers Models")
    print("    High-quality image upscaling models")
    print()
    print("=" * 40)
    print("Combined Options:")
    print("=" * 40)
    print("9. ALL MODELS (SeedVR2 + RIFE)")
    print("   Download everything")
    print()
    print("11. ALL MODELS (SeedVR2 + RIFE + BestImageUpscalers)")
    print("    Download everything including best image upscalers")
    print()
    
    while True:
        try:
            choice = input("Please select option (1-13): ").strip()
            if choice == "1":
                return ["vae"], False, False
            elif choice == "2":
                return ["3b"], False, False
            elif choice == "3":
                return ["7b"], False, False
            elif choice == "4":
                return ["7b_sharp"], False, False
            elif choice == "5":
                return ALL_SEEDVR2_MODEL_IDS.copy(), False, False
            elif choice == "6":
                return ["vae", "3b"], False, False
            elif choice == "7":
                return ["vae", "7b"], False, False
            elif choice == "8":
                return [], True, False
            elif choice == "9":
                return ALL_SEEDVR2_MODEL_IDS.copy(), True, False
            elif choice == "10":
                return [], False, True
            elif choice == "11":
                return ALL_SEEDVR2_MODEL_IDS.copy(), True, True
            elif choice == "12":
                return ["7b_fp8_mixed_block35"], False, False
            elif choice == "13":
                return ["7b_sharp_fp8_mixed_block35"], False, False
            else:
                print("Invalid choice. Please enter 1-13.")
        except KeyboardInterrupt:
            print("\nDownload cancelled.")
            sys.exit(0)


def download_models(model_ids: Optional[List[str]] = None, 
                    skip_verify: bool = False,
                    dry_run: bool = False,
                    include_rife: bool = False,
                    include_best: bool = False) -> None:
    """Main download function."""
    if model_ids is None and not include_rife and not include_best:
        model_ids, include_rife, include_best = get_model_choice()
    
    if model_ids is None:
        model_ids = []
    
    # Validate model IDs
    invalid_ids = [m for m in model_ids if m not in SEEDVR2_MODEL_CONFIGS]
    if invalid_ids:
        print(f"Error: Unknown model ID(s): {invalid_ids}")
        print(f"Valid options: {list(SEEDVR2_MODEL_CONFIGS.keys())}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Download Configuration")
    print("=" * 60)
    
    if model_ids:
        print(f"\nSeedVR2 Repository: {SEEDVR2_REPO_ID}")
        print(f"SeedVR2 Target Directory: {SEEDVR2_MODELS_DIR}")
        print(f"SeedVR2 Models to download:")
        for model_id in model_ids:
            config = SEEDVR2_MODEL_CONFIGS[model_id]
            print(f"  - {config['name']} ({config['size_hint']})")
    
    if include_rife:
        print(f"\nRIFE Repository: {RIFE_REPO_ID}")
        print(f"RIFE Source Directory: {RIFE_REPO_SUBDIR}")
        print(f"RIFE Versions: {', '.join(RIFE_VERSION_FOLDERS)}")
        print(f"RIFE Target Directory: {RIFE_MODELS_DIR}")
        print(f"RIFE: Download selected version folders from repository")
    
    if include_best:
        print(f"\nBestImageUpscalers Repository: {BESTIMAGEUPSCALE_REPO_ID}")
        print(f"BestImageUpscalers Target Directory: {IMAGE_UPSCALE_MODELS_DIR}")
        print(f"BestImageUpscalers: Download all files from repository")
    
    print(f"\nSkip Verification: {skip_verify}")
    print()
    
    if dry_run:
        print("[DRY RUN] Would download the above files. Exiting.")
        return
    
    downloader = RobustDownloader(DOWNLOAD_CONFIG, skip_verify=skip_verify)
    
    total_successful = 0
    total_failed = 0
    failed_files = []
    
    # Download SeedVR2 models
    if model_ids:
        # Ensure target directory exists
        SEEDVR2_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "=" * 60)
        print("Downloading SeedVR2 Models")
        print("=" * 60)
        
        for i, model_id in enumerate(model_ids, 1):
            config = SEEDVR2_MODEL_CONFIGS[model_id]
            filename = config["filename"]
            
            print(f"\n{'=' * 60}")
            print(f"[{i}/{len(model_ids)}] {config['name']}")
            print(f"File: {filename}")
            print(f"Description: {config['description']}")
            print("=" * 60)
            
            if downloader.download_file(SEEDVR2_REPO_ID, filename, SEEDVR2_MODELS_DIR):
                total_successful += 1
            else:
                total_failed += 1
                failed_files.append(f"SeedVR2: {filename}")
    
    # Download RIFE models
    if include_rife:
        print("\n" + "=" * 60)
        print("Downloading RIFE Models (4.14-4.26)")
        print("=" * 60)
        print(f"Repository: {RIFE_REPO_ID}")
        print(f"Source Directory: {RIFE_REPO_SUBDIR}")
        print(f"Versions: {', '.join(RIFE_VERSION_FOLDERS)}")
        print(f"Target: {RIFE_MODELS_DIR}")
        print("=" * 60)
        
        rife_success, rife_fail = downloader.download_repo(
            RIFE_REPO_ID, 
            RIFE_MODELS_DIR,
            exclude_patterns=['.gitattributes', '.gitignore'],
            include_prefixes=RIFE_REPO_PREFIXES,
            strip_prefix=RIFE_REPO_SUBDIR,
        )
        total_successful += rife_success
        total_failed += rife_fail
        if rife_fail > 0:
            failed_files.append(f"RIFE: {rife_fail} file(s)")
    
    # Download BestImageUpscalers models
    if include_best:
        print("\n" + "=" * 60)
        print("Downloading BestImageUpscalers Models")
        print("=" * 60)
        print(f"Repository: {BESTIMAGEUPSCALE_REPO_ID}")
        print(f"Target: {IMAGE_UPSCALE_MODELS_DIR}")
        print("=" * 60)
        
        best_success, best_fail = downloader.download_repo(
            BESTIMAGEUPSCALE_REPO_ID,
            IMAGE_UPSCALE_MODELS_DIR,
            exclude_patterns=['.gitattributes', '.gitignore']
        )
        total_successful += best_success
        total_failed += best_fail
        if best_fail > 0:
            failed_files.append(f"BestImageUpscalers: {best_fail} file(s)")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("Download Summary")
    print("=" * 60)
    print(f"  Successful: {total_successful}")
    print(f"  Failed: {total_failed}")
    
    if total_failed == 0:
        print(f"\n✓ All downloads completed successfully!")
        
        if model_ids:
            print(f"\nSeedVR2 models saved to: {SEEDVR2_MODELS_DIR}")
            print(f"Downloaded SeedVR2 files:")
            for model_id in model_ids:
                filepath = SEEDVR2_MODELS_DIR / SEEDVR2_MODEL_CONFIGS[model_id]["filename"]
                if filepath.exists():
                    size = filepath.stat().st_size
                    print(f"  - {SEEDVR2_MODEL_CONFIGS[model_id]['filename']} ({RobustDownloader.format_bytes(size)})")
        
        if include_rife:
            print(f"\nRIFE models saved to: {RIFE_MODELS_DIR}")
        if include_best:
            print(f"\nBestImageUpscalers models saved to: {IMAGE_UPSCALE_MODELS_DIR}")
    else:
        print(f"\n⚠ Some downloads failed:")
        for f in failed_files:
            print(f"  - {f}")
        print(f"\nPlease re-run the script to retry failed downloads.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Download models for Ultimate Video/Image Upscalers Premium',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Models_Downloader.py                      # Interactive mode
  python Models_Downloader.py --all                # Download ALL models (SeedVR2 + RIFE)
  python Models_Downloader.py --seedvr2            # Download all SeedVR2 models only
  python Models_Downloader.py --rife               # Download RIFE models (4.14-4.26) only
  python Models_Downloader.py --vae                # Download VAE model only
  python Models_Downloader.py --3b                 # Download 3B model only
  python Models_Downloader.py --7b                 # Download 7B model only
  python Models_Downloader.py --7b-sharp           # Download 7B Sharp model only
  python Models_Downloader.py --7b-fp8-mixed-block35
                                            # Download 7B FP8 mixed block35 model only
  python Models_Downloader.py --7b-sharp-fp8-mixed-block35
                                            # Download 7B Sharp FP8 mixed block35 model only
  python Models_Downloader.py --vae --3b           # Download VAE and 3B models
  python Models_Downloader.py --vae --rife         # Download VAE and RIFE models
  python Models_Downloader.py --all --skip-verify  # Download all, skip SHA256 verification
  python Models_Downloader.py --all --dry-run      # Show what would be downloaded
  python Models_Downloader.py --best               # Download BestImageUpscalers models
  python Models_Downloader.py --all --best         # Download all including BestImageUpscalers
        """
    )
    parser.add_argument('--all', action='store_true',
                        help='Download ALL models (SeedVR2 + RIFE)')
    parser.add_argument('--seedvr2', action='store_true',
                        help='Download all SeedVR2 models (VAE + 3B + 7B + 7B Sharp + FP8 mixed block35 variants)')
    parser.add_argument('--rife', action='store_true',
                        help='Download RIFE models (4.14-4.26 folders from Wan_GGUF/RIFE_Models)')
    parser.add_argument('--vae', action='store_true',
                        help='Download VAE model (required for SeedVR2 operations)')
    parser.add_argument('--3b', dest='model_3b', action='store_true',
                        help='Download SeedVR2 3B model')
    parser.add_argument('--7b', dest='model_7b', action='store_true',
                        help='Download SeedVR2 7B model')
    parser.add_argument('--7b-sharp', dest='model_7b_sharp', action='store_true',
                        help='Download SeedVR2 7B Sharp model')
    parser.add_argument('--7b-fp8-mixed-block35', dest='model_7b_fp8_mixed_block35', action='store_true',
                        help='Download SeedVR2 7B FP8 mixed block35 model')
    parser.add_argument('--7b-sharp-fp8-mixed-block35', dest='model_7b_sharp_fp8_mixed_block35', action='store_true',
                        help='Download SeedVR2 7B Sharp FP8 mixed block35 model')
    parser.add_argument('--skip-verify', action='store_true',
                        help='Skip SHA256 verification (faster but less safe)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be downloaded without downloading')
    parser.add_argument('--best', action='store_true',
                        help='Download BestImageUpscalers models (all files from repo)')
    
    args = parser.parse_args()
    
    model_ids = []
    include_rife = False
    include_best = False
    
    if args.all:
        # Download everything
        model_ids = ALL_SEEDVR2_MODEL_IDS.copy()
        include_rife = True
        include_best = True
    else:
        if args.seedvr2:
            model_ids = ALL_SEEDVR2_MODEL_IDS.copy()
        else:
            if args.vae:
                model_ids.append("vae")
            if args.model_3b:
                model_ids.append("3b")
            if args.model_7b:
                model_ids.append("7b")
            if args.model_7b_sharp:
                model_ids.append("7b_sharp")
            if args.model_7b_fp8_mixed_block35:
                model_ids.append("7b_fp8_mixed_block35")
            if args.model_7b_sharp_fp8_mixed_block35:
                model_ids.append("7b_sharp_fp8_mixed_block35")
        
        if args.rife:
            include_rife = True
        if args.best:
            include_best = True
    
    # If no arguments, run interactive mode
    if not model_ids and not include_rife and not include_best:
        model_ids = None
    
    download_models(
        model_ids,
        skip_verify=args.skip_verify,
        dry_run=args.dry_run,
        include_rife=include_rife,
        include_best=include_best
    )


if __name__ == "__main__":
    main()
