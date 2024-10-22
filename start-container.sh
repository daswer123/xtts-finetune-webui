#!/bin/bash

# Enable resolution of libcudnn_ops_infer.so.8
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/app/xtts-finetune-webui/.local/lib/python3.11/site-packages/torch/lib:/app/xtts-finetune-webui/.local/lib/python3.11/site-packages/nvidia/cudnn/lib"

python3 xtts_demo.py
