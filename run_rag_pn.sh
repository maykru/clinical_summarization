#!/usr/bin/env bash
if [ $# -ne 1 ]; then
    echo "Error: Please provide a generation round number as argument"
    exit 1
fi

n=$1

base_dir="RAG_PN_output/round$n"

mkdir -p "$base_dir/method-1"
mkdir -p "$base_dir/method1"
mkdir -p "$base_dir/method2"

python RAG_script_pn.py --method -1 --outputdir "$base_dir/method-1"
echo "-----------------------method -1 done -------------------------------"

python RAG_script_pn.py --method 1 --outputdir "$base_dir/method1"
echo "-----------------------method 1 done --------------------------------"

python RAG_script_pn.py --method 2 --outputdir "$base_dir/method2"
echo "-----------------------method 2 done --------------------------------"

