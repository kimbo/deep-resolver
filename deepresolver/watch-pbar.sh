#!/bin/bash

# 4139955
set -e
clear
total="${total:-100}"
delay="${delay:-}"
outputfile="${1}"
if [ -z "$outputfile" ]; then
  echo "Usage: $0 <file>" >&2
  exit 1
fi
while test "1"; do
	n_lines="$(wc -l < $outputfile)"
	percent_done=$((n_lines * 100 / total))
	printf "\r%s" "$n_lines/$total ($percent_done%) |"
	for ((i=0; i<percent_done; i++)); do
	  printf "#"
	done
	for ((i=0; i<100-percent_done; i++)); do
	  printf " "
	done
	printf "%s" "|"
	[ ! -z "$delay" ] && sleep "$delay"
done
