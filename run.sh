#!/usr/bin/env sh
# usage: ./run.sh <audio_folder> [output_folder]
exec python run.py "$1" --out "${2:-outputs}"
