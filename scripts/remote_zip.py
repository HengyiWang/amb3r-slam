"""Extract selected members of a remote zip over HTTP range requests.

Reads the archive's central directory remotely and transfers only the members whose path
matches ``--match``, in archive order so the reads stay sequential. Already extracted
members of the right size are skipped, so an interrupted run resumes.

    python scripts/remote_zip.py URL DEST --match 'dataset/sequences/0[0-9]/image_2/' \
        --strip dataset/ [--limit N]
"""

import argparse
import io
import os
import re
import time
import zipfile

import requests


class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests."""

    def __init__(self, url):
        self.url, self.s = url, requests.Session()
        self.s.headers['User-Agent'] = 'Mozilla/5.0'
        # A one-byte range request both proves range support and reports the size.
        h = self._get(0, 0, stream=True)
        h.close()
        self.url = h.url
        if h.status_code != 206 or '/' not in h.headers.get('Content-Range', ''):
            raise RuntimeError(f'{url}: server does not serve byte ranges')
        self.size = int(h.headers['Content-Range'].rsplit('/', 1)[1])
        self.pos = self.n_bytes = 0

    def _get(self, start, end, tries=10, stream=False):
        """GET bytes [start, end], waiting out rate limits and transient failures."""
        for attempt in range(tries):
            try:
                r = self.s.get(self.url, headers={'Range': f'bytes={start}-{end}'},
                               timeout=300, stream=stream)
                if r.status_code != 429 and r.status_code < 500:
                    r.raise_for_status()
                    return r
                after = r.headers.get('Retry-After', '')
                wait = int(after) if after.isdigit() else 30 * (attempt + 1)
                why = f'HTTP {r.status_code}'
            except requests.ConnectionError as e:
                wait, why = 30 * (attempt + 1), type(e).__name__
            if attempt == tries - 1:
                raise RuntimeError(f'{self.url}: {why} after {tries} attempts')
            print(f'  {why}, retrying in {wait} s', flush=True)
            time.sleep(wait)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        r = self._get(self.pos, min(self.pos + len(b), self.size) - 1)
        data = r.content
        b[:len(data)] = data
        self.pos += len(data)
        self.n_bytes += len(data)
        return len(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('url')
    ap.add_argument('dest')
    ap.add_argument('--match', required=True, help='regex searched in each member path')
    ap.add_argument('--strip', default='', help='prefix removed from member paths')
    ap.add_argument('--limit', type=int, default=0, help='first N matching members only')
    ap.add_argument('--buffer_mb', type=int, default=32)
    ap.add_argument('--dry_run', action='store_true', help='only report what matches')
    args = ap.parse_args()

    hf = HttpFile(args.url)
    pat = re.compile(args.match)
    with zipfile.ZipFile(io.BufferedReader(hf, buffer_size=args.buffer_mb << 20)) as z:
        infos = [i for i in z.infolist() if not i.is_dir() and pat.search(i.filename)]
        if args.limit:
            infos = sorted(infos, key=lambda i: i.filename)[:args.limit]
        infos.sort(key=lambda i: i.header_offset)
        total = sum(i.file_size for i in infos)
        print(f'{args.url}: {len(infos)} members, {total / 1e9:.2f} GB', flush=True)
        if args.dry_run:
            return
        done, t0 = 0, time.time()
        for k, i in enumerate(infos):
            rel = i.filename[len(args.strip):] if i.filename.startswith(args.strip) \
                else i.filename
            dst = os.path.join(args.dest, rel)
            done += i.file_size
            if os.path.exists(dst) and os.path.getsize(dst) == i.file_size:
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with z.open(i) as src, open(dst + '.part', 'wb') as fh:
                while True:
                    chunk = src.read(1 << 24)
                    if not chunk:
                        break
                    fh.write(chunk)
            os.replace(dst + '.part', dst)
            if k % 1000 == 0 or k == len(infos) - 1:
                rate = hf.n_bytes / max(time.time() - t0, 1e-9) / 1e6
                print(f'  {k + 1}/{len(infos)}  {done / 1e9:.2f}/{total / 1e9:.2f} GB  '
                      f'{rate:.1f} MB/s', flush=True)


if __name__ == '__main__':
    main()
