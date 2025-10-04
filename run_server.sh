#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/model/yolov5:${PROJECT_ROOT}:$PYTHONPATH"

python3 main_fedml_object_detection.py --cf config/fedml_config.yaml --run_id yolov5 --rank 0 --role server
