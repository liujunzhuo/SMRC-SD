#!/usr/bin/env bash

set -euo pipefail

data=""

# Displays information on how to use script
helpFunction()
{
  echo "Usage: $0 [-d small|all]"
  echo -e "\t-d small|all - Specify whether to download entire dataset (all) or just 1000 (small)"
  exit 1 # Exit script after printing help
}

# Get values of command line flags
while getopts d: flag
do
  case "${flag}" in
    d) data=${OPTARG};;
  esac
done

if [ -z "$data" ]; then
  echo "[ERROR]: Missing -d flag"
  helpFunction
fi

# Install Python and native dependencies.  FAISS 1.10 currently expects MKL's
# .so.2 ABI, so do not let conda solve this to MKL 2026 (.so.3).
python -m pip install -r requirements.txt
conda install -y -c conda-forge "mkl=2025.3.1" faiss-cpu openjdk=11

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  jvm_path="${CONDA_PREFIX}/lib/jvm/lib/server/libjvm.so"
  conda env config vars set -p "${CONDA_PREFIX}" JAVA_HOME="${CONDA_PREFIX}" JVM_PATH="${jvm_path}"
fi

# Download dataset into `data` folder via `gdown` command
mkdir -p data;
cd data;
download_if_missing() {
  local url="$1"
  local output="$2"
  if [[ ! -s "${output}" ]]; then
    gdown "${url}" -O "${output}"
  fi
}
if [ "$data" == "small" ]; then
  download_if_missing "https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib" items_shuffle_1000.json
  download_if_missing "https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu" items_ins_v2_1000.json
elif [ "$data" == "all" ]; then
  download_if_missing "https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib" items_shuffle_1000.json
  download_if_missing "https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu" items_ins_v2_1000.json
  download_if_missing "https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB" items_shuffle.json
  download_if_missing "https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi" items_ins_v2.json
else
  echo "[ERROR]: argument for `-d` flag not recognized"
  helpFunction
fi
download_if_missing "https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O" items_human_ins.json
cd ..

# WebShop runtime loads only en_core_web_sm.  The explicit wheel avoids an old
# spaCy downloader bug that can construct a version-less GitHub URL.
python -m pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"

# Build search engine index
cd search_engine
mkdir -p resources resources_100 resources_1k resources_100k
python convert_product_file_format.py # convert items.json => required doc format
if [[ ! -f indexes/segments_1 ]]; then
  mkdir -p indexes indexes_100 indexes_1k indexes_100k
  ./run_indexing.sh
else
  echo "Existing Lucene indexes found; skipping rebuild."
fi
cd ..

# Create logging folder + samples of log data
# get_human_trajs () {
#   PYCMD=$(cat <<EOF
# import gdown
# url="https://drive.google.com/drive/u/1/folders/16H7LZe2otq4qGnKw_Ic1dkt-o3U9Zsto"
# gdown.download_folder(url, quiet=True, remaining_ok=True)
# EOF
#   )
#   python -c "$PYCMD"
# }
# mkdir -p user_session_logs/
# cd user_session_logs/
# echo "Downloading 50 example human trajectories..."
# get_human_trajs
# echo "Downloading example trajectories complete"
# cd ..
