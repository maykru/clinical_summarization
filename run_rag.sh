#!/usr/bin/env bash
if [ $# -ne 1 ]; then
    echo "Error: Please provide a generation round number as argument"
    exit 1
fi

n=$1

base_dir="rag_summaries/round$n"

mkdir -p "$base_dir"

python RAG_script.py --outputdir "$base_dir"
