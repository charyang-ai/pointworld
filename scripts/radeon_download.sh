#!/bin/bash
# Host side. Resumable downloads of the rows of downloads.tsv whose "repo/path" matches
# FILTER, checked against the sha256 the Hugging Face API reports. Datasets go under
# $ROOT/data/hf, checkpoints under $ROOT/weights.
#
#   ROOT=/path/to/workdir bash scripts/radeon_download.sh scripts/downloads.tsv 'PointWorld-DROID/|small-droid/'
set -uo pipefail
ROOT=${ROOT:?set ROOT to the host workdir}
TSV=${1:?usage: radeon_download.sh TSV [FILTER]}
FILTER=${2:-.}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
fail=0
while IFS=$'\t' read -r repo path size sha; do
  [[ -z "$repo" ]] && continue
  [[ "$repo/$path" =~ $FILTER ]] || continue
  if [[ "$repo" == datasets/* ]]; then
    dest="$ROOT/data/hf/$repo/$path"
  else
    dest="$ROOT/weights/$path"
  fi
  mkdir -p "$(dirname "$dest")"
  url="$HF_ENDPOINT/$repo/resolve/main/$path"
  ok=0
  for attempt in $(seq 1 40); do
    have=$(stat -c %s "$dest" 2>/dev/null || echo 0)
    if [ "$have" -eq "$size" ]; then ok=1; break; fi
    if [ "$have" -gt "$size" ]; then rm -f "$dest"; have=0; fi
    echo "$(date -Is) GET $repo/$path attempt=$attempt have=$have want=$size"
    curl -L --fail -sS -C - --connect-timeout 30 --speed-limit 262144 --speed-time 60 \
      -o "$dest" "$url" < /dev/null || echo "$(date -Is) curl exit $? for $path"
  done
  if [ "$ok" -ne 1 ]; then
    echo "$(date -Is) FAILED $repo/$path"
    fail=1
    continue
  fi
  # Markers live outside the package dirs: recover_dataset_from_parts.sh globs *.part-*.
  marker="$ROOT/data/verified/${repo//\//_}__${path//\//_}.sha256"
  if [ "$sha" != "-" ] && [ ! -f "$marker" ]; then
    got=$(sha256sum "$dest" | cut -d' ' -f1)
    if [ "$got" != "$sha" ]; then
      echo "$(date -Is) SHA_MISMATCH $repo/$path got=$got"
      mv "$dest" "$ROOT/data/verified/$(basename "$dest").bad"
      fail=1
      continue
    fi
    mkdir -p "$(dirname "$marker")"
    echo "$sha" > "$marker"
  fi
  echo "$(date -Is) DONE $repo/$path $size"
done < "$TSV"
echo "$(date -Is) DOWNLOAD_EXIT $fail"
exit "$fail"
