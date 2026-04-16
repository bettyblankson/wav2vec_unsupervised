#!/bin/bash
# Download LibriSpeech splits + WikiText text corpus for wav2vec-u experiments.
# This script downloads archives only once and extracts WAV-ready audio.
#
# Usage:
#   bash scripts/download_librispeech_wiki.sh [TARGET_ROOT]
#
# Default TARGET_ROOT:
#   $HOME/datasets
#
# Outputs:
#   $TARGET_ROOT/librispeech/{train-clean-100,dev-clean,test-clean}
#   $TARGET_ROOT/librispeech/{train-clean-100-wav,dev-clean-wav,test-clean-wav}
#   $TARGET_ROOT/wiki/wikitext-103-v1/wiki.train.tokens

set -euo pipefail

TARGET_ROOT="${1:-$HOME/datasets}"
LIBRI_ROOT="$TARGET_ROOT/librispeech"
WIKI_ROOT="$TARGET_ROOT/wiki"
mkdir -p "$LIBRI_ROOT" "$WIKI_ROOT"

download_if_missing() {
  local url="$1"
  local out="$2"
  if [ ! -f "$out" ]; then
    echo "[download] $url"
    wget -O "$out" "$url"
  else
    echo "[skip] already exists: $out"
  fi
}

extract_if_missing() {
  local archive="$1"
  local expected_dir="$2"
  if [ ! -d "$expected_dir" ]; then
    echo "[extract] $archive"
    tar -xf "$archive" -C "$LIBRI_ROOT"
  else
    echo "[skip] already extracted: $expected_dir"
  fi
}

to_wav_copy() {
  local src_dir="$1"
  local out_dir="$2"
  mkdir -p "$out_dir"
  if [ -n "$(ls -A "$out_dir" 2>/dev/null)" ]; then
    echo "[skip] wav folder already populated: $out_dir"
    return 0
  fi
  echo "[convert] $src_dir -> $out_dir"
  find "$src_dir" -type f -name "*.flac" | while read -r flac; do
    rel="${flac#$src_dir/}"
    wav="$out_dir/${rel%.flac}.wav"
    mkdir -p "$(dirname "$wav")"
    ffmpeg -loglevel error -y -i "$flac" "$wav"
  done
}

# 1) LibriSpeech
download_if_missing "https://www.openslr.org/resources/12/train-clean-100.tar.gz" "$LIBRI_ROOT/train-clean-100.tar.gz"
download_if_missing "https://www.openslr.org/resources/12/dev-clean.tar.gz" "$LIBRI_ROOT/dev-clean.tar.gz"
download_if_missing "https://www.openslr.org/resources/12/test-clean.tar.gz" "$LIBRI_ROOT/test-clean.tar.gz"

extract_if_missing "$LIBRI_ROOT/train-clean-100.tar.gz" "$LIBRI_ROOT/LibriSpeech/train-clean-100"
extract_if_missing "$LIBRI_ROOT/dev-clean.tar.gz" "$LIBRI_ROOT/LibriSpeech/dev-clean"
extract_if_missing "$LIBRI_ROOT/test-clean.tar.gz" "$LIBRI_ROOT/LibriSpeech/test-clean"

to_wav_copy "$LIBRI_ROOT/LibriSpeech/train-clean-100" "$LIBRI_ROOT/train-clean-100-wav"
to_wav_copy "$LIBRI_ROOT/LibriSpeech/dev-clean" "$LIBRI_ROOT/dev-clean-wav"
to_wav_copy "$LIBRI_ROOT/LibriSpeech/test-clean" "$LIBRI_ROOT/test-clean-wav"

# 2) WikiText
download_if_missing "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip" "$WIKI_ROOT/wikitext-103-v1.zip"
if [ ! -d "$WIKI_ROOT/wikitext-103-v1" ]; then
  echo "[extract] $WIKI_ROOT/wikitext-103-v1.zip"
  unzip -o "$WIKI_ROOT/wikitext-103-v1.zip" -d "$WIKI_ROOT"
else
  echo "[skip] already extracted: $WIKI_ROOT/wikitext-103-v1"
fi

echo
echo "[done] dataset root: $TARGET_ROOT"
echo "train audio dir: $LIBRI_ROOT/train-clean-100-wav"
echo "val audio dir:   $LIBRI_ROOT/dev-clean-wav"
echo "test audio dir:  $LIBRI_ROOT/test-clean-wav"
echo "wiki text file:  $WIKI_ROOT/wikitext-103-v1/wiki.train.tokens"
