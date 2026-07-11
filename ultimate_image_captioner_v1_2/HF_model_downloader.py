from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import requests
import shutil
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_REPO_ID = "MonsterMMORPG/CapFiles"
DEFAULT_REVISION = "main"
DEFAULT_TARGET_DIR = REPO_ROOT / "Ultimate_Image_Captioner_Pro"
QWEN3_VL_REPO_ID = "Qwen/Qwen3-VL-8B-Instruct"
QWEN3_VL_TARGET_SUBDIR = "model_files_qwen3_vl3_8b_instruct"

DOWNLOAD_STATUS_DOWNLOADED = "downloaded"
DOWNLOAD_STATUS_SKIPPED = "skipped"
DOWNLOAD_STATUS_FAILED = "failed"

DOWNLOAD_CONFIG = {
    "num_connections": 16,
    "parallel_threshold": 10 * 1024 * 1024,
    "chunk_size": 10 * 1024 * 1024,
    "max_retries": 5,
    "retry_delay": 2,
    "max_retry_delay": 30,
    "timeout": 300,
}


def _get_hf_token() -> Optional[str]:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )


def _create_hf_api() -> HfApi:
    token = _get_hf_token()
    if not token:
        return HfApi()
    try:
        return HfApi(token=token)
    except TypeError:
        return HfApi()


class RobustDownloader:
    def __init__(
        self,
        repo_id: str,
        revision: str,
        config: Dict[str, Any],
    ):
        self.repo_id = repo_id
        self.revision = revision
        self.config = config
        self.hf_token = _get_hf_token()

        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=max(20, int(config["num_connections"]) + 4),
            pool_maxsize=max(20, int(config["num_connections"]) + 4),
            max_retries=3,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update(
            {
                "User-Agent": "Ultimate-Image-Captioner-Pro-Downloader/1.0",
                "Accept-Encoding": "identity",
            }
        )
        if self.hf_token:
            self.session.headers.update({"Authorization": f"Bearer {self.hf_token}"})

        self.sha_cache_file = REPO_ROOT / "sha256_cache.json"
        self.sha_cache = self.load_sha_cache()
        self.verified_cache_file = REPO_ROOT / "verified_files_cache.json"
        self.verified_cache = self.load_verified_cache()

        self._progress_lock = threading.Lock()
        self._active_progress = False
        self._last_progress_len = 0
        self._file_meta: Dict[str, Dict[str, Any]] = {}
        self._range_support_cache: Dict[str, bool] = {}

    def _hf_api(self) -> HfApi:
        return _create_hf_api()

    def _hf_file_url(self, filename: str) -> str:
        return hf_hub_url(
            repo_id=self.repo_id,
            filename=filename,
            repo_type="model",
            revision=self.revision,
        )

    def _hf_file_metadata(self, filename: str):
        url = self._hf_file_url(filename)
        if not self.hf_token:
            return get_hf_file_metadata(url)
        try:
            return get_hf_file_metadata(url, token=self.hf_token)
        except TypeError:
            return get_hf_file_metadata(url)

    def prefetch_metadata(self, filenames: Sequence[str]) -> None:
        wanted = set(filenames)
        try:
            api = self._hf_api()
            info = api.model_info(
                repo_id=self.repo_id,
                revision=self.revision,
                files_metadata=True,
            )
            for file_info in getattr(info, "siblings", []) or []:
                filename = getattr(file_info, "rfilename", None)
                if not filename or filename not in wanted:
                    continue

                size = getattr(file_info, "size", None)
                sha256: Optional[str] = None
                lfs = getattr(file_info, "lfs", None)
                if lfs:
                    if isinstance(lfs, dict):
                        sha256 = lfs.get("sha256") or lfs.get("oid")
                        if size is None:
                            size = lfs.get("size")
                    else:
                        sha256 = getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)
                        if size is None:
                            size = getattr(lfs, "size", None)

                self._file_meta[filename] = {
                    "size": self._normalize_size(size),
                    "sha256": sha256 if isinstance(sha256, str) and len(sha256) == 64 else None,
                }
        except Exception as exc:
            self.log(f"Warning: Could not prefetch Hugging Face metadata: {exc}")

    @staticmethod
    def _normalize_size(value: Any) -> Optional[int]:
        if isinstance(value, int) and value >= 0:
            return value
        if isinstance(value, float) and value >= 0:
            return int(value)
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return None

    def _cache_key(self, filename: str) -> str:
        return f"{self.repo_id}@{self.revision}/{filename}"

    def supports_range(self, url: str) -> bool:
        if url in self._range_support_cache:
            return self._range_support_cache[url]

        ok = False
        response = None
        try:
            response = self.session.get(
                url,
                headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"},
                timeout=30,
                stream=True,
                allow_redirects=True,
            )
            ok = response.status_code == 206 and bool(response.headers.get("content-range"))
        except Exception:
            ok = False
        finally:
            if response is not None:
                response.close()

        self._range_support_cache[url] = ok
        return ok

    def _terminal_width(self) -> int:
        try:
            return shutil.get_terminal_size(fallback=(100, 20)).columns
        except Exception:
            return 100

    def _clear_progress_line_locked(self) -> None:
        width = self._terminal_width()
        clear_len = max(self._last_progress_len, width)
        sys.stdout.write("\r" + " " * clear_len + "\r")
        sys.stdout.flush()
        self._last_progress_len = 0
        self._active_progress = False

    def clear_progress_line(self) -> None:
        with self._progress_lock:
            if self._active_progress:
                self._clear_progress_line_locked()

    def show_progress_line(self, text: str) -> None:
        with self._progress_lock:
            max_len = max(1, self._terminal_width() - 1)
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
        with self._progress_lock:
            if final_text is not None:
                max_len = max(1, self._terminal_width() - 1)
                if len(final_text) > max_len:
                    final_text = final_text[:max_len]
                sys.stdout.write("\r" + final_text)
                extra_spaces = max(0, self._last_progress_len - len(final_text))
                if extra_spaces:
                    sys.stdout.write(" " * extra_spaces)
                    sys.stdout.write("\r" + final_text)
                sys.stdout.write("\n")
                sys.stdout.flush()
            elif self._active_progress:
                sys.stdout.write("\n")
                sys.stdout.flush()

            self._last_progress_len = 0
            self._active_progress = False

    def log(self, message: str) -> None:
        with self._progress_lock:
            if self._active_progress:
                self._clear_progress_line_locked()
            print(message, flush=True)

    def load_sha_cache(self) -> Dict[str, str]:
        if not self.sha_cache_file.exists():
            return {}
        try:
            data = json.loads(self.sha_cache_file.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def save_sha_cache(self) -> None:
        try:
            self.sha_cache_file.write_text(
                json.dumps(self.sha_cache, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except Exception as exc:
            self.log(f"Warning: Could not save SHA cache: {exc}")

    def load_verified_cache(self) -> Dict[str, Dict[str, Any]]:
        if not self.verified_cache_file.exists():
            return {}
        try:
            data = json.loads(self.verified_cache_file.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def save_verified_cache(self) -> None:
        try:
            self.verified_cache_file.write_text(
                json.dumps(self.verified_cache, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except Exception as exc:
            self.log(f"Warning: Could not save verified cache: {exc}")

    def is_file_verified(self, filename: str, filepath: str, expected_sha: str) -> bool:
        if not expected_sha or not os.path.exists(filepath):
            return False

        cached_info = self.verified_cache.get(self._cache_key(filename))
        if not cached_info:
            return False

        current_size = os.path.getsize(filepath)
        current_mtime = os.path.getmtime(filepath)
        return (
            cached_info.get("sha256") == expected_sha
            and cached_info.get("size") == current_size
            and abs(cached_info.get("mtime", 0) - current_mtime) < 1.0
        )

    def mark_file_verified(self, filename: str, filepath: str, sha256: str) -> None:
        if not os.path.exists(filepath):
            return

        self.verified_cache[self._cache_key(filename)] = {
            "sha256": sha256,
            "size": os.path.getsize(filepath),
            "mtime": os.path.getmtime(filepath),
            "verified_at": time.time(),
        }
        self.save_verified_cache()

    def get_file_sha256(self, filename: str) -> Optional[str]:
        cache_key = self._cache_key(filename)
        cached = self.sha_cache.get(cache_key)
        if isinstance(cached, str) and len(cached) == 64:
            return cached

        try:
            meta_sha = self._file_meta.get(filename, {}).get("sha256")
            if isinstance(meta_sha, str) and len(meta_sha) == 64:
                self.sha_cache[cache_key] = meta_sha
                self.save_sha_cache()
                return meta_sha

            metadata = self._hf_file_metadata(filename)
            etag = str(getattr(metadata, "etag", "")).replace('"', "").replace("W/", "")
            if len(etag) == 64:
                self.sha_cache[cache_key] = etag
                self.save_sha_cache()
                return etag
        except Exception as exc:
            self.log(f"Warning: Could not get SHA256 for {filename}: {exc}")

        return None

    @staticmethod
    def format_bytes(bytes_value: float) -> str:
        if bytes_value < 0:
            return "0 B"
        for unit in ["B", "KB", "MB", "GB"]:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} TB"

    @staticmethod
    def format_time(seconds: float) -> str:
        if seconds < 0:
            return "0s"
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds}s"
        if seconds < 3600:
            minutes = seconds // 60
            remaining = seconds % 60
            return f"{minutes}m" if remaining == 0 else f"{minutes}m {remaining}s"
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h" if minutes == 0 else f"{hours}h {minutes}m"

    def print_progress(
        self,
        current: int,
        total: int,
        start_time: float,
        filename: str,
        speed_bytes_per_sec: Optional[float] = None,
    ) -> None:
        if total <= 0:
            return

        elapsed = max(0.001, time.time() - start_time)
        percent = min(100.0, (current / total) * 100)
        if speed_bytes_per_sec is None:
            speed_bytes_per_sec = current / elapsed if elapsed > 0 else 0.0

        if speed_bytes_per_sec > 0 and total > current:
            eta = self.format_time((total - current) / speed_bytes_per_sec)
        else:
            eta = "Complete" if current >= total else "Unknown"

        bar_length = 40
        filled = int(percent * bar_length / 100)
        bar = "=" * max(0, min(filled, bar_length)) + "-" * max(0, bar_length - filled)
        speed = self.format_bytes(speed_bytes_per_sec) + "/s" if speed_bytes_per_sec else "0 B/s"
        line = (
            f"{filename}: [{bar}] {percent:.1f}% "
            f"({self.format_bytes(current)}/{self.format_bytes(total)}) "
            f"{speed} ETA: {eta}"
        )
        self.show_progress_line(line)

    def get_file_size(self, url: str, filename: str) -> Optional[int]:
        meta_size = self._normalize_size(self._file_meta.get(filename, {}).get("size"))
        if meta_size is not None:
            return meta_size

        try:
            metadata = self._hf_file_metadata(filename)
            size = self._normalize_size(getattr(metadata, "size", None))
            if size is not None:
                return size
        except Exception:
            pass

        response = None
        try:
            response = self.session.head(
                url,
                headers={"Accept-Encoding": "identity"},
                timeout=30,
                allow_redirects=True,
            )
            if response.status_code in (200, 206):
                size = self._size_from_headers(response.headers)
                if size is not None:
                    return size
        except Exception:
            pass
        finally:
            if response is not None:
                response.close()

        response = None
        try:
            response = self.session.get(
                url,
                headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"},
                timeout=30,
                stream=True,
                allow_redirects=True,
            )
            return self._size_from_headers(response.headers)
        except Exception as exc:
            self.log(f"Warning: Could not get file size for {filename}: {exc}")
            return None
        finally:
            if response is not None:
                response.close()

    @staticmethod
    def _size_from_headers(headers: requests.structures.CaseInsensitiveDict) -> Optional[int]:
        content_range = headers.get("content-range")
        if content_range and "/" in content_range:
            total = content_range.split("/")[-1]
            if total.isdigit():
                return int(total)

        content_length = headers.get("content-length")
        if content_length and str(content_length).isdigit():
            return int(content_length)
        return None

    def verify_file_sha256(self, filepath: str, expected_sha: str, filename: str) -> bool:
        if not expected_sha:
            return False

        self.log(f"[VERIFYING] Computing SHA256 for {filename}...")
        try:
            sha256_hash = hashlib.sha256()
            file_size = os.path.getsize(filepath)
            bytes_read = 0
            start_time = time.time()
            last_update = 0.0
            chunk_size = 8 * 1024 * 1024

            with open(filepath, "rb") as handle:
                while True:
                    chunk = handle.read(chunk_size)
                    if not chunk:
                        break
                    sha256_hash.update(chunk)
                    bytes_read += len(chunk)
                    now = time.time()
                    if now - last_update >= 0.2:
                        percent = (bytes_read / file_size) * 100 if file_size > 0 else 0.0
                        speed = bytes_read / max(0.001, now - start_time)
                        line = (
                            f"[VERIFYING] {filename}: {percent:.1f}% "
                            f"({self.format_bytes(bytes_read)}/{self.format_bytes(file_size)}) "
                            f"{self.format_bytes(speed)}/s"
                        )
                        self.show_progress_line(line)
                        last_update = now

            computed_sha = sha256_hash.hexdigest()
            if computed_sha == expected_sha:
                self.finalize_progress_line(f"[VERIFIED] SHA256 match: {computed_sha[:16]}...")
                return True

            self.finalize_progress_line("[ERROR] SHA256 mismatch")
            self.log(f"  Expected: {expected_sha}")
            self.log(f"  Got:      {computed_sha}")
            return False
        except Exception as exc:
            self.finalize_progress_line()
            self.log(f"[ERROR] Failed to verify SHA256 for {filename}: {exc}")
            return False

    def download_chunk(
        self,
        url: str,
        start: int,
        end: int,
        filepath: str,
        chunk_id: int,
        progress_callback=None,
    ) -> bool:
        chunk_file = os.path.normpath(f"{filepath}.part{chunk_id}")
        expected_size = end - start + 1

        if os.path.exists(chunk_file):
            existing_size = os.path.getsize(chunk_file)
            if existing_size == expected_size:
                if progress_callback:
                    progress_callback(chunk_id, expected_size)
                return True
            if existing_size > expected_size:
                os.remove(chunk_file)
                resume_pos = 0
            else:
                resume_pos = existing_size
        else:
            resume_pos = 0

        for attempt in range(self.config["max_retries"]):
            response = None
            try:
                actual_start = start + resume_pos
                response = self.session.get(
                    url,
                    headers={"Range": f"bytes={actual_start}-{end}", "Accept-Encoding": "identity"},
                    timeout=self.config["timeout"],
                    stream=True,
                    allow_redirects=True,
                )

                if response.status_code == 200:
                    raise RuntimeError("Server did not honor Range request")
                if response.status_code != 206:
                    raise RuntimeError(f"Bad status code: {response.status_code}")

                mode = "ab" if resume_pos > 0 else "wb"
                downloaded = resume_pos
                with open(chunk_file, mode) as handle:
                    for data in response.iter_content(chunk_size=self.config["chunk_size"]):
                        if data:
                            handle.write(data)
                            downloaded += len(data)
                            if progress_callback:
                                progress_callback(chunk_id, downloaded)

                final_size = os.path.getsize(chunk_file)
                if final_size == expected_size:
                    if progress_callback:
                        progress_callback(chunk_id, expected_size)
                    return True
                if final_size < expected_size:
                    resume_pos = final_size
                    raise RuntimeError(f"Chunk incomplete: {final_size}/{expected_size}")

                os.remove(chunk_file)
                resume_pos = 0
                raise RuntimeError(f"Chunk too large: {final_size}/{expected_size}")
            except Exception as exc:
                if attempt < self.config["max_retries"] - 1:
                    delay = min(
                        self.config["retry_delay"] * (2 ** attempt),
                        self.config["max_retry_delay"],
                    )
                    self.log(f"Chunk {chunk_id} attempt {attempt + 1} failed: {exc}. Retrying in {delay}s")
                    time.sleep(delay)
                else:
                    self.log(f"Chunk {chunk_id} failed after retries: {exc}")
                    return False
            finally:
                if response is not None:
                    response.close()

        return False

    def merge_chunks(self, filepath: str, num_chunks: int) -> bool:
        temp_file = os.path.normpath(f"{filepath}.tmp")
        try:
            total_size = 0
            chunk_files: List[Tuple[str, int]] = []
            for idx in range(num_chunks):
                chunk_file = os.path.normpath(f"{filepath}.part{idx}")
                if not os.path.exists(chunk_file):
                    self.log(f"[ERROR] Missing chunk {idx}")
                    return False
                size = os.path.getsize(chunk_file)
                chunk_files.append((chunk_file, size))
                total_size += size

            merged_size = 0
            start_time = time.time()
            last_update = 0.0
            with open(temp_file, "wb") as outfile:
                for chunk_file, chunk_size in chunk_files:
                    with open(chunk_file, "rb") as infile:
                        copied = 0
                        while copied < chunk_size:
                            data = infile.read(64 * 1024 * 1024)
                            if not data:
                                break
                            outfile.write(data)
                            copied += len(data)
                            merged_size += len(data)
                            now = time.time()
                            if now - last_update >= 0.5:
                                percent = (merged_size / total_size) * 100 if total_size > 0 else 0.0
                                speed = merged_size / max(0.001, now - start_time)
                                line = (
                                    f"[MERGING] {percent:.1f}% "
                                    f"({self.format_bytes(merged_size)}/{self.format_bytes(total_size)}) "
                                    f"{self.format_bytes(speed)}/s"
                                )
                                self.show_progress_line(line)
                                last_update = now

            self.clear_progress_line()
            if os.path.exists(filepath):
                os.remove(filepath)
            os.rename(temp_file, filepath)

            for chunk_file, _size in chunk_files:
                try:
                    os.remove(chunk_file)
                except Exception:
                    pass
            return True
        except Exception as exc:
            self.finalize_progress_line()
            self.log(f"[ERROR] Merge failed: {exc}")
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass
            return False

    def download_parallel(self, url: str, filepath: str, filename: str, file_size: int) -> bool:
        num_chunks = max(1, min(int(self.config["num_connections"]), file_size))
        base_chunk_size = file_size // num_chunks
        chunks = []
        for idx in range(num_chunks):
            start = idx * base_chunk_size
            end = file_size - 1 if idx == num_chunks - 1 else ((idx + 1) * base_chunk_size - 1)
            chunks.append((idx, start, end))

        self.log(
            f"[DOWNLOADING] {filename} ({self.format_bytes(file_size)}) using {num_chunks} connections"
        )

        chunk_progress: Dict[int, int] = {}
        progress_lock = threading.Lock()

        def update_progress(chunk_id: int, bytes_downloaded: int) -> None:
            with progress_lock:
                chunk_progress[chunk_id] = bytes_downloaded

        for chunk_id, _start, _end in chunks:
            chunk_file = os.path.normpath(f"{filepath}.part{chunk_id}")
            chunk_progress[chunk_id] = os.path.getsize(chunk_file) if os.path.exists(chunk_file) else 0

        initial_bytes = sum(chunk_progress.values())
        if initial_bytes > 0:
            self.log(f"[RESUMING] Already downloaded: {self.format_bytes(initial_bytes)}")

        start_time = time.time()
        failed_chunks: List[int] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_chunks) as executor:
            futures = {}
            for chunk_id, start, end in chunks:
                chunk_file = os.path.normpath(f"{filepath}.part{chunk_id}")
                expected_size = end - start + 1
                if os.path.exists(chunk_file) and os.path.getsize(chunk_file) == expected_size:
                    continue
                futures[
                    executor.submit(
                        self.download_chunk,
                        url,
                        start,
                        end,
                        filepath,
                        chunk_id,
                        update_progress,
                    )
                ] = chunk_id

            if not futures:
                self.log("[MERGING] All chunks already complete")
                if self.merge_chunks(filepath, num_chunks) and os.path.getsize(filepath) == file_size:
                    self.log(f"[OK] {filename} completed")
                    return True
                self.log("[ERROR] Size mismatch after merge")
                return False

            last_update = 0.0
            last_bytes = initial_bytes
            while futures:
                done, _pending = concurrent.futures.wait(
                    list(futures.keys()),
                    timeout=0.5,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    chunk_id = futures.pop(future)
                    try:
                        if not future.result():
                            failed_chunks.append(chunk_id)
                    except Exception as exc:
                        self.log(f"Chunk {chunk_id} exception: {exc}")
                        failed_chunks.append(chunk_id)

                current_time = time.time()
                if current_time - last_update >= 1.0 or not futures:
                    with progress_lock:
                        current_bytes = sum(chunk_progress.values())
                    time_delta = current_time - last_update if last_update > 0 else (current_time - start_time)
                    bytes_delta = current_bytes - last_bytes if last_update > 0 else (current_bytes - initial_bytes)
                    speed = bytes_delta / max(0.001, time_delta)
                    self.print_progress(current_bytes, file_size, start_time, filename, speed)
                    last_update = current_time
                    last_bytes = current_bytes

        self.clear_progress_line()

        if failed_chunks:
            self.log(f"[ERROR] Failed chunks: {failed_chunks}")
            return False

        for chunk_id, start, end in chunks:
            chunk_file = os.path.normpath(f"{filepath}.part{chunk_id}")
            expected_size = end - start + 1
            if not os.path.exists(chunk_file):
                self.log(f"[ERROR] Missing chunk {chunk_id}")
                return False
            actual_size = os.path.getsize(chunk_file)
            if actual_size != expected_size:
                self.log(f"[ERROR] Chunk {chunk_id} incomplete: {actual_size}/{expected_size}")
                return False

        self.log(f"[MERGING] Merging {num_chunks} chunks...")
        if self.merge_chunks(filepath, num_chunks):
            final_size = os.path.getsize(filepath)
            if final_size == file_size:
                elapsed = max(0.001, time.time() - start_time)
                avg_speed = max(0, file_size - initial_bytes) / elapsed
                self.log(
                    f"[OK] {filename} completed in {self.format_time(elapsed)} "
                    f"- Average: {self.format_bytes(avg_speed)}/s"
                )
                return True
            self.log(f"[ERROR] Final size mismatch: {final_size} != {file_size}")
            try:
                os.remove(filepath)
            except Exception:
                pass
            return False

        self.log("[ERROR] Merge failed")
        return False

    def download_single(self, url: str, filepath: str, filename: str, file_size: int) -> bool:
        for attempt in range(self.config["max_retries"]):
            response = None
            try:
                resume_pos = os.path.getsize(filepath) if os.path.exists(filepath) else 0
                if resume_pos > file_size:
                    self.log(f"[WARNING] {filename} is larger than expected, restarting")
                    os.remove(filepath)
                    resume_pos = 0
                elif resume_pos == file_size:
                    self.log(f"[OK] {filename} already complete")
                    return True

                headers = (
                    {"Range": f"bytes={resume_pos}-", "Accept-Encoding": "identity"}
                    if resume_pos > 0
                    else {"Accept-Encoding": "identity"}
                )
                response = self.session.get(
                    url,
                    headers=headers,
                    timeout=self.config["timeout"],
                    stream=True,
                    allow_redirects=True,
                )

                if resume_pos > 0 and response.status_code != 206:
                    self.log(f"[WARNING] Resume not supported for {filename}, restarting")
                    response.close()
                    resume_pos = 0
                    response = self.session.get(
                        url,
                        headers={"Accept-Encoding": "identity"},
                        timeout=self.config["timeout"],
                        stream=True,
                        allow_redirects=True,
                    )

                response.raise_for_status()

                mode = "ab" if resume_pos > 0 else "wb"
                downloaded = 0
                start_time = time.time()
                last_update = 0.0

                if resume_pos > 0:
                    self.log(f"[RESUMING] {filename} from {self.format_bytes(resume_pos)}")
                else:
                    self.log(f"[DOWNLOADING] {filename} ({self.format_bytes(file_size)})")

                with open(filepath, mode) as handle:
                    for chunk in response.iter_content(chunk_size=self.config["chunk_size"]):
                        if chunk:
                            handle.write(chunk)
                            downloaded += len(chunk)
                            now = time.time()
                            if now - last_update >= 0.5:
                                self.print_progress(resume_pos + downloaded, file_size, start_time, filename)
                                last_update = now

                self.print_progress(resume_pos + downloaded, file_size, start_time, filename)
                self.clear_progress_line()

                final_size = os.path.getsize(filepath)
                if final_size == file_size:
                    self.log(f"[OK] {filename} completed")
                    return True

                self.log(f"[ERROR] Size mismatch: expected {file_size}, got {final_size}")
                if final_size > file_size:
                    os.remove(filepath)
            except Exception as exc:
                self.finalize_progress_line()
                self.log(f"[ERROR] Attempt {attempt + 1}: {exc}")
                if attempt < self.config["max_retries"] - 1:
                    delay = min(
                        self.config["retry_delay"] * (2 ** attempt),
                        self.config["max_retry_delay"],
                    )
                    time.sleep(delay)
                else:
                    return False
            finally:
                if response is not None:
                    response.close()

        return False

    def download_unknown_size(
        self,
        url: str,
        filepath: str,
        filename: str,
        expected_sha: Optional[str],
    ) -> str:
        for attempt in range(self.config["max_retries"]):
            response = None
            try:
                resume_pos = os.path.getsize(filepath) if os.path.exists(filepath) else 0
                if resume_pos > 0:
                    self.log(f"[RESUMING] {filename} (unknown size) from {self.format_bytes(resume_pos)}")
                    headers = {"Range": f"bytes={resume_pos}-", "Accept-Encoding": "identity"}
                else:
                    self.log(f"[DOWNLOADING] {filename} (unknown size)")
                    headers = {"Accept-Encoding": "identity"}

                response = self.session.get(
                    url,
                    headers=headers,
                    timeout=self.config["timeout"],
                    stream=True,
                    allow_redirects=True,
                )

                if resume_pos > 0 and response.status_code == 416:
                    response.close()
                    if expected_sha:
                        if self.is_file_verified(filename, filepath, expected_sha):
                            self.log(f"[SKIP] {filename} already complete and verified (cached)")
                            return DOWNLOAD_STATUS_SKIPPED
                        if self.verify_file_sha256(filepath, expected_sha, filename):
                            self.mark_file_verified(filename, filepath, expected_sha)
                            self.log(f"[SKIP] {filename} already complete and verified")
                            return DOWNLOAD_STATUS_SKIPPED
                        self.log(f"[WARNING] {filename} failed verification, re-downloading")
                        os.remove(filepath)
                        resume_pos = 0
                    else:
                        self.log(f"[SKIP] {filename} already complete (range not satisfiable)")
                        return DOWNLOAD_STATUS_SKIPPED

                if resume_pos > 0 and response.status_code != 206:
                    self.log(
                        f"[WARNING] Resume not supported for {filename} "
                        f"(status {response.status_code}), restarting"
                    )
                    response.close()
                    resume_pos = 0
                    response = self.session.get(
                        url,
                        headers={"Accept-Encoding": "identity"},
                        timeout=self.config["timeout"],
                        stream=True,
                        allow_redirects=True,
                    )

                response.raise_for_status()

                downloaded = 0
                start_time = time.time()
                last_update = 0.0
                total_size = self._size_from_headers(response.headers)

                mode = "ab" if resume_pos > 0 else "wb"
                with open(filepath, mode) as handle:
                    for chunk in response.iter_content(chunk_size=self.config["chunk_size"]):
                        if chunk:
                            handle.write(chunk)
                            downloaded += len(chunk)
                            now = time.time()
                            if now - last_update >= 0.5:
                                if total_size:
                                    self.print_progress(resume_pos + downloaded, total_size, start_time, filename)
                                else:
                                    elapsed = max(0.001, now - start_time)
                                    speed = downloaded / elapsed
                                    line = (
                                        f"[DOWNLOADING] {filename}: {self.format_bytes(resume_pos + downloaded)} "
                                        f"@ {self.format_bytes(speed)}/s"
                                    )
                                    self.show_progress_line(line)
                                last_update = now

                final_size = os.path.getsize(filepath)
                elapsed = max(0.001, time.time() - start_time)
                avg_speed = final_size / elapsed
                self.finalize_progress_line(
                    f"[OK] {filename} completed ({self.format_bytes(final_size)}) "
                    f"in {self.format_time(elapsed)} - Avg: {self.format_bytes(avg_speed)}/s"
                )

                if expected_sha:
                    if not self.verify_file_sha256(filepath, expected_sha, filename):
                        self.log(f"[ERROR] {filename} downloaded but failed SHA256 verification")
                        os.remove(filepath)
                        return DOWNLOAD_STATUS_FAILED
                    self.mark_file_verified(filename, filepath, expected_sha)
                return DOWNLOAD_STATUS_DOWNLOADED
            except Exception as exc:
                self.finalize_progress_line()
                self.log(f"[ERROR] Attempt {attempt + 1}: {exc}")
                if attempt < self.config["max_retries"] - 1:
                    delay = min(
                        self.config["retry_delay"] * (2 ** attempt),
                        self.config["max_retry_delay"],
                    )
                    time.sleep(delay)
                else:
                    return DOWNLOAD_STATUS_FAILED
            finally:
                if response is not None:
                    response.close()

        return DOWNLOAD_STATUS_FAILED

    def download_file(self, filename: str, local_dir: Path) -> str:
        url = self._hf_file_url(filename)
        filepath = (local_dir / filename).resolve()
        filepath.parent.mkdir(parents=True, exist_ok=True)

        expected_sha = self.get_file_sha256(filename)
        if expected_sha:
            self.log(f"[INFO] {filename} expected SHA256: {expected_sha[:16]}...")
        else:
            self.log(f"[INFO] {filename} has no SHA256 metadata; using size checks")

        file_size = self.get_file_size(url, filename)

        if filepath.exists():
            actual_size = filepath.stat().st_size
            if file_size is not None:
                if actual_size == file_size:
                    if expected_sha and self.is_file_verified(filename, str(filepath), expected_sha):
                        self.log(
                            f"[SKIP] {filename} already complete and verified (cached) "
                            f"({self.format_bytes(file_size)})"
                        )
                        return DOWNLOAD_STATUS_SKIPPED
                    if expected_sha:
                        if self.verify_file_sha256(str(filepath), expected_sha, filename):
                            self.mark_file_verified(filename, str(filepath), expected_sha)
                            self.log(
                                f"[SKIP] {filename} already complete and verified "
                                f"({self.format_bytes(file_size)})"
                            )
                            return DOWNLOAD_STATUS_SKIPPED
                        self.log(f"[WARNING] {filename} failed verification, re-downloading")
                        filepath.unlink(missing_ok=True)
                    else:
                        self.log(f"[SKIP] {filename} already complete ({self.format_bytes(file_size)})")
                        return DOWNLOAD_STATUS_SKIPPED
                elif actual_size > file_size:
                    self.log(f"[WARNING] {filename} is larger than expected, re-downloading")
                    filepath.unlink(missing_ok=True)
                else:
                    self.log(
                        f"[RESUMING] {filename} from "
                        f"{self.format_bytes(actual_size)}/{self.format_bytes(file_size)}"
                    )
            elif expected_sha and actual_size > 0:
                if self.is_file_verified(filename, str(filepath), expected_sha):
                    self.log(
                        f"[SKIP] {filename} exists and verified (cached) "
                        f"({self.format_bytes(actual_size)})"
                    )
                    return DOWNLOAD_STATUS_SKIPPED
                if self.verify_file_sha256(str(filepath), expected_sha, filename):
                    self.mark_file_verified(filename, str(filepath), expected_sha)
                    self.log(f"[SKIP] {filename} exists and verified ({self.format_bytes(actual_size)})")
                    return DOWNLOAD_STATUS_SKIPPED
                self.log(f"[WARNING] {filename} failed verification, re-downloading")
                filepath.unlink(missing_ok=True)

        if file_size is None:
            self.log(f"[INFO] Could not determine size for {filename}; downloading without size info")
            return self.download_unknown_size(url, str(filepath), filename, expected_sha)

        if file_size > int(self.config["parallel_threshold"]) and self.supports_range(url):
            success = self.download_parallel(url, str(filepath), filename, file_size)
        else:
            if file_size > int(self.config["parallel_threshold"]):
                self.log(f"[INFO] Range requests not supported for {filename}; using single connection")
            success = self.download_single(url, str(filepath), filename, file_size)

        if success and expected_sha:
            if not self.verify_file_sha256(str(filepath), expected_sha, filename):
                self.log(f"[ERROR] {filename} downloaded but failed SHA256 verification")
                filepath.unlink(missing_ok=True)
                return DOWNLOAD_STATUS_FAILED
            self.mark_file_verified(filename, str(filepath), expected_sha)

        return DOWNLOAD_STATUS_DOWNLOADED if success else DOWNLOAD_STATUS_FAILED


def scan_repo_files(repo_id: str, revision: str) -> List[str]:
    try:
        api = _create_hf_api()
        return list(
            api.list_repo_files(
                repo_id=repo_id,
                repo_type="model",
                revision=revision,
            )
        )
    except Exception as exc:
        print(f"Error scanning repository {repo_id}@{revision}: {exc}")
        return []


def _resolve_target_dir(download_dir: Optional[str]) -> Path:
    if download_dir:
        return Path(download_dir).expanduser().resolve()
    return DEFAULT_TARGET_DIR.resolve()


def list_files(repo_id: str, revision: str) -> int:
    files = scan_repo_files(repo_id, revision)
    if not files:
        return 1

    print(f"Repository: {repo_id}@{revision}")
    print(f"Files: {len(files)}")
    for filename in files:
        print(f"  {filename}")
    return 0


def download_repository(
    download_dir: Optional[str] = None,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    dry_run: bool = False,
    num_connections: Optional[int] = None,
) -> int:
    target_dir = _resolve_target_dir(download_dir)
    files = scan_repo_files(repo_id, revision)
    if not files:
        print(f"[ERROR] No files found in {repo_id}@{revision}")
        return 1

    config = dict(DOWNLOAD_CONFIG)
    if num_connections is not None:
        config["num_connections"] = max(1, num_connections)

    print(f"Repository: {repo_id}@{revision}")
    print(f"Target directory: {target_dir}")
    print(f"Files: {len(files)}")
    print(f"Mode: {'dry run' if dry_run else 'download'}")

    if dry_run:
        print("\n[DRY RUN] Planned file targets:")
        for filename in files:
            print(f"  - {(target_dir / filename).resolve()}")
        return 0

    target_dir.mkdir(parents=True, exist_ok=True)

    downloader = RobustDownloader(repo_id=repo_id, revision=revision, config=config)
    downloader.prefetch_metadata(files)

    downloaded = 0
    skipped = 0
    failed = 0
    for filename in files:
        status = downloader.download_file(filename, target_dir)
        if status == DOWNLOAD_STATUS_DOWNLOADED:
            downloaded += 1
        elif status == DOWNLOAD_STATUS_SKIPPED:
            skipped += 1
        else:
            failed += 1

    print("\n" + "=" * 72)
    print("OVERALL SUMMARY")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Model directory: {target_dir}")

    if failed == 0:
        print("\n[SUCCESS] Ultimate Image Captioner Pro downloads completed successfully.")
        return 0

    print(f"\n[ERROR] {failed} file downloads failed.")
    return 1


def download_configured_repositories(local_dir: Optional[str] = None) -> int:
    target_dir = Path(local_dir).expanduser().resolve() if local_dir else DEFAULT_TARGET_DIR.resolve()
    targets = [
        (DEFAULT_REPO_ID, DEFAULT_REVISION, target_dir),
        (QWEN3_VL_REPO_ID, DEFAULT_REVISION, target_dir / QWEN3_VL_TARGET_SUBDIR),
    ]

    failed_repositories = 0
    for index, (repo_id, revision, download_dir) in enumerate(targets, start=1):
        print("\n" + "#" * 72)
        print(f"DOWNLOAD TARGET {index}/{len(targets)}")
        result = download_repository(
            download_dir=str(download_dir),
            repo_id=repo_id,
            revision=revision,
        )
        if result != 0:
            failed_repositories += 1

    if failed_repositories:
        print(f"\n[ERROR] {failed_repositories} repository download(s) failed.")
        return 1

    print("\n[SUCCESS] All configured model repositories downloaded successfully.")
    return 0


def main(local_dir: Optional[str] = None) -> int:
    return download_configured_repositories(local_dir)


if __name__ == "__main__":
    raise SystemExit(main())
