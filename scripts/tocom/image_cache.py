"""Bounded, single-writer cache of original image bytes for an epoch.

Prepare only while all DataLoader workers are stopped. Readers use an immutable
membership set and fall back to the source on a miss. No transforms are cached.
"""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time


def remaining_indices(sampler, batch_size, start_micro=0):
    """Keep the official rank order/repeats, drop_last, and resume position."""
    indices = list(sampler)
    end = len(indices) // batch_size * batch_size
    if not 0 <= start_micro * batch_size <= end:
        raise ValueError('Resume microbatch is outside the epoch')
    return indices[start_micro * batch_size:end]


class CachedImageLoader:
    def __init__(self, source, cache, members, loader):
        self.source = str(Path(source).resolve())
        self.cache = Path(cache) / 'images'
        self.members = frozenset(members)
        self.loader = loader

    def __call__(self, path):
        relative = os.path.relpath(path, self.source)
        if relative in self.members:
            try:
                return self.loader(str(self.cache / relative))
            except (OSError, ValueError):
                # Missing/undecodable cache files never replace a source image.
                pass
        return self.loader(path)


class EpochImageCache:
    """Own one private directory; never delete or rewrite source images.

    The cap covers allocated image blocks plus SQLite/owner files. Reservations
    include in-flight copies. The free-space reserve is rechecked each batch.
    """
    def __init__(self, source, root, max_bytes, reserve_bytes, workers=8):
        self.source, self.root = Path(source).resolve(), Path(root).resolve()
        if (self.root == self.source or self.root.is_relative_to(self.source)
                or self.source.is_relative_to(self.root)):
            raise ValueError('Cache and source directories must be separate')
        if max_bytes <= 0 or reserve_bytes < 0 or workers < 1:
            raise ValueError('Invalid cache capacity/reserve/workers')
        self.max_bytes, self.reserve_bytes, self.workers = max_bytes, reserve_bytes, workers
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / '.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            identity = {'format': 'tocom-raw-image-cache-v1', 'source': str(self.source)}
            owner = self.root / 'owner.json'
            if owner.exists():
                if json.loads(owner.read_text()) != identity:
                    raise ValueError('Cache owner/source mismatch')
            else:
                if any(p.name != '.lock' for p in self.root.iterdir()):
                    raise ValueError('Refusing to adopt a nonempty directory as an image cache')
                owner.write_text(json.dumps(identity) + '\n')
            self.images = self.root / 'images'
            self.images.mkdir(exist_ok=True)
            self.db = sqlite3.connect(self.root / 'index.sqlite')
            self.db.execute('CREATE TABLE IF NOT EXISTS images '
                            '(path TEXT PRIMARY KEY, size INTEGER, mtime INTEGER, '
                            'allocated INTEGER, digest TEXT)')
            self.db.commit()
            self.block_size = os.statvfs(self.root).f_frsize
            self._recover()
        except BaseException:
            if hasattr(self, 'db'):
                self.db.close()
            self.lock.close()
            raise

    def close(self):
        self.db.close()
        self.lock.close()

    def _path(self, relative):
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Invalid cache index path')
        return self.images / path

    def _recover(self):
        # A killed copy can leave a temporary or an uncommitted final file.
        known = set(self.members())
        for directory, _, files in os.walk(self.images):
            for name in files:
                path = Path(directory) / name
                if str(path.relative_to(self.images)) not in known:
                    path.unlink()

    def members(self):
        return [row[0] for row in self.db.execute('SELECT path FROM images')]

    def _overhead(self):
        # Allow for the next SQLite transaction and directory metadata too.
        return sum(p.stat().st_blocks * 512 for p in self.root.iterdir() if p.is_file()) + 1024**2

    def _remove(self, relative):
        self._path(relative).unlink(missing_ok=True)
        self.db.execute('DELETE FROM images WHERE path = ?', (relative,))

    def _copy(self, item):
        relative, size, mtime, allocation = item
        source, destination = self.source / relative, self._path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + '.part')
        try:
            digest = hashlib.sha256()
            with source.open('rb') as src, temporary.open('wb') as dst:
                for data in iter(lambda: src.read(1024 * 1024), b''):
                    digest.update(data)
                    dst.write(data)
                dst.flush()
                os.fsync(dst.fileno())
            copied = hashlib.sha256()
            with temporary.open('rb') as stream:
                for data in iter(lambda: stream.read(1024 * 1024), b''):
                    copied.update(data)
            info = source.stat()
            if (info.st_size != size or info.st_mtime_ns != mtime
                    or temporary.stat().st_size != size or copied.digest() != digest.digest()):
                raise OSError('Source changed or cache copy failed byte verification')
            actual = temporary.stat().st_blocks * 512
            if actual > allocation:
                raise OSError('Cache allocation exceeds its reservation')
            os.replace(temporary, destination)
            return (relative, size, mtime, actual, digest.hexdigest()), None
        except OSError as error:
            return None, f'{relative}: {error}'
        finally:
            temporary.unlink(missing_ok=True)

    def prepare(self, source_paths, progress=None):
        started = time.monotonic()
        wanted = sorted({str(Path(p).relative_to(self.source)) for p in source_paths})
        records = {row[0]: row[1:] for row in self.db.execute('SELECT * FROM images')}
        wanted_set = set(wanted)
        evicted = 0
        for relative in records:
            if relative not in wanted_set:
                self._remove(relative)
                evicted += 1
        candidates, reused, used = [], 0, 0
        for relative in wanted:
            info = (self.source / relative).stat()
            old = records.get(relative)
            destination = self._path(relative)
            if old and old[:2] == (info.st_size, info.st_mtime_ns) and destination.is_file():
                cached = destination.stat()
                if cached.st_size == info.st_size:
                    reused += 1
                    used += cached.st_blocks * 512
                    continue
            if old:
                self._remove(relative)
            allocation = (info.st_size + self.block_size - 1) // self.block_size * self.block_size
            candidates.append((relative, info.st_size, info.st_mtime_ns, allocation))
        # A smaller cap or external disk usage on restart can require eviction.
        for relative in self.members():
            if (used + self._overhead() <= self.max_bytes
                    and shutil.disk_usage(self.root).free >= self.reserve_bytes):
                break
            allocation = self._path(relative).stat().st_blocks * 512
            self._remove(relative)
            used -= allocation
            reused -= 1
            evicted += 1
        self.db.commit()
        copied_count, copied_bytes, skipped = 0, 0, 0
        errors = []
        reported = time.monotonic()
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            # Bounded queue: do not create half a million futures at once.
            for offset in range(0, len(candidates), 64):
                free = max(0, shutil.disk_usage(self.root).free - self.reserve_bytes - self._overhead())
                capacity = max(0, self.max_bytes - used - self._overhead())
                batch, reserved = [], 0
                for item in candidates[offset:offset + 64]:
                    if reserved + item[3] <= min(free, capacity):
                        batch.append(item)
                        reserved += item[3]
                    else:
                        skipped += 1
                for record, error in pool.map(self._copy, batch):
                    if record is not None:
                        self.db.execute('INSERT OR REPLACE INTO images VALUES (?, ?, ?, ?, ?)', record)
                        used += record[3]
                        copied_count += 1
                        copied_bytes += record[1]
                    elif len(errors) < 10:
                        errors.append(error)
                self.db.commit()
                if progress and time.monotonic() - reported >= 30:
                    progress({'event': 'cache_fill', 'copied_images': copied_count,
                              'reused_images': reused, 'requested_images': len(wanted),
                              'allocated_bytes': used, 'seconds': time.monotonic() - started})
                    reported = time.monotonic()
        return {'event': 'cache_ready', 'requested_images': len(wanted),
                'cached_images': reused + copied_count, 'reused_images': reused,
                'copied_images': copied_count, 'copied_bytes': copied_bytes,
                'evicted_images': evicted, 'capacity_skips': skipped, 'errors': errors,
                'allocated_image_bytes': used, 'max_bytes': self.max_bytes,
                'free_bytes': shutil.disk_usage(self.root).free,
                'seconds': time.monotonic() - started}


def read_members(root):
    connection = sqlite3.connect(f'{Path(root).resolve().as_uri()}/index.sqlite?mode=ro', uri=True)
    try:
        return [row[0] for row in connection.execute('SELECT path FROM images')]
    finally:
        connection.close()
