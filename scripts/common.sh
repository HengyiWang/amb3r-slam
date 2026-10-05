# Sourced by the download scripts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

say () { echo "[$(date +%H:%M:%S)] $*"; }

# fetch URL OUT: resumable; parallel ranges through aria2c when it is installed.
fetch () {
    local url=$1 out=$2
    mkdir -p "$(dirname "$out")"
    if command -v aria2c >/dev/null 2>&1; then
        aria2c -x16 -s16 -k 10M --continue=true --max-tries=5 --retry-wait=10 \
               --auto-file-renaming=false --allow-overwrite=true \
               --console-log-level=warn --summary-interval=0 \
               -d "$(dirname "$out")" -o "$(basename "$out")" "$url"
    else
        wget -c -q --show-progress -U "Mozilla/5.0" -O "$out" "$url"
    fi
}

# unpack ARCHIVE DEST: extract, then delete the archive.
unpack () {
    local archive=$1 dest=$2
    mkdir -p "$dest"
    case "$archive" in
        *.zip) unzip -q -o "$archive" -d "$dest" ;;
        *.tgz|*.tar.gz) tar -xzf "$archive" -C "$dest" ;;
        *) echo "unknown archive type: $archive" >&2; return 1 ;;
    esac
    rm -f "$archive"
}
