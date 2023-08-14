#!/bin/bash
# source current file to export PYTHONPATH
echo "source current file to export PYTHONPATH"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH=$PYTHONPATH:${SCRIPT_DIR}/