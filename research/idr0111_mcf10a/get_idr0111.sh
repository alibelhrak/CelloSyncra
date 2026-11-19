#!/bin/bash
# idr0111 (Lee et al. 2021, MCF10A + mutants, MCF10A vs MDA-MB-231): download the 198 time-lapse videos of
# experiment A only (no still tiles, no experiment B), from the public EBI server.
#
# Put this file, videos.txt and videos_annotation.csv in one folder, e.g. SI_Project/idr0111, then:
#   bash get_idr0111.sh size     # total download size (a few minutes), downloads nothing
#   bash get_idr0111.sh get      # download (resumes where it stopped if run again)
#   bash get_idr0111.sh check    # how many of the 198 videos are complete
#
# To keep it running after you log out:
#   nohup bash get_idr0111.sh get > download.log 2>&1 &
#   tail -n 5 download.log
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LIST="$HERE/videos.txt"
DEST="$HERE/data"
BASE="https://ftp.ebi.ac.uk/pub/databases/IDR"
[ -f "$LIST" ] || { echo "videos.txt not found next to this script"; exit 1; }
sed -i 's/\r$//' "$LIST"
N=$(grep -c . "$LIST")
log() { echo "[$(date '+%F %T')] $*"; }

size() {
  local tot=0 n=0 b
  while read -r p; do
    [ -z "$p" ] && continue
    b=$(curl -sIL "$BASE/$p" | tr -d '\r' | awk 'tolower($1)=="content-length:"{v=$2} END{print v+0}')
    tot=$((tot + b)); n=$((n + 1))
    [ $((n % 20)) -eq 0 ] && log "$n/$N files checked, $((tot / 1024 / 1024 / 1024)) GB so far"
  done < "$LIST"
  log "total: $N videos, $((tot / 1024 / 1024 / 1024)) GB"
  df -h "$HERE" | tail -n 1 | awk '{print "free space here: " $4}'
}

get() {
  mkdir -p "$DEST"
  local n=0 bad=0
  while read -r p; do
    [ -z "$p" ] && continue
    n=$((n + 1))
    out="$DEST/$(basename "$(dirname "$p")")/$(basename "$p")"
    mkdir -p "$(dirname "$out")"
    if [ -f "$out.done" ]; then continue; fi
    log "$n/$N $(basename "$p")"
    if wget -q -c --tries=5 --timeout=60 -O "$out" "$BASE/$p" \
       || wget -q -c --tries=5 --timeout=60 -O "$out" "ftp://ftp.ebi.ac.uk/pub/databases/IDR/$p"; then
      touch "$out.done"
    else
      log "  FAILED (run 'get' again later to resume)"; bad=$((bad + 1))
    fi
  done < "$LIST"
  log "finished: $((N - bad)) of $N videos complete"
  check
}

check() {
  local done; done=$(find "$DEST" -name "*.done" 2>/dev/null | wc -l)
  log "complete videos: $done of $N"
  du -sh "$DEST" 2>/dev/null
}

case "${1:-}" in
  size) size ;;
  get) get ;;
  check) check ;;
  *) echo "usage: bash get_idr0111.sh {size|get|check}"; exit 2 ;;
esac
