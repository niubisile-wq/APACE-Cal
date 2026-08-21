#!/usr/bin/env bash
set -euo pipefail
name="$1"; total="$2"; expected_md5="$3"; parts="${4:-12}"
base="data/batterylife_zenodo"
dir="$base/$name.zip.chunks"
url="https://zenodo.org/records/21149533/files/$name.zip?download=1"
mkdir -p "$dir"
step=$(( (total + parts - 1) / parts ))
pids=()
for ((i=0;i<parts;i++)); do
  start=$((i*step)); end=$((start+step-1)); (( end >= total )) && end=$((total-1))
  need=$((end-start+1)); file="$dir/$i"
  if [[ -f "$file" ]] && [[ "$(stat -c %s "$file")" == "$need" ]]; then continue; fi
  env -u ALL_PROXY -u all_proxy curl -L --fail --retry 20 --retry-all-errors --retry-delay 3 \
    --connect-timeout 20 -r "$start-$end" -o "$file" "$url" >"$dir/$i.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || { echo "chunk download failed" >&2; exit 2; }
for ((i=0;i<parts;i++)); do
  start=$((i*step)); end=$((start+step-1)); (( end >= total )) && end=$((total-1)); need=$((end-start+1))
  got="$(stat -c %s "$dir/$i")"; [[ "$got" == "$need" ]] || { echo "bad chunk $i: $got != $need" >&2; exit 3; }
done
cp "$dir/0" "$base/$name.zip"
for ((i=1;i<parts;i++)); do dd if="$dir/$i" of="$base/$name.zip" oflag=append conv=notrunc status=none; done
actual="$(md5sum "$base/$name.zip" | awk '{print $1}')"
[[ "$actual" == "$expected_md5" ]] || { echo "MD5 mismatch: $actual" >&2; exit 4; }
unzip -t "$base/$name.zip" >/dev/null
echo "$name OK size=$total md5=$actual"
