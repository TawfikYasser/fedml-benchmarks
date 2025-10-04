#!/usr/bin/env bash
WORKER_NUM=$1
PROCESS_NUM=`expr $WORKER_NUM + 1`

# ensure project root and yolov5 are on PYTHONPATH
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
# if run_simulation.sh lives in project root adjust path accordingly; above assumes script in root
export PYTHONPATH="${PROJECT_ROOT}/model/yolov5:${PROJECT_ROOT}:$PYTHONPATH"

echo "process_num=$PROCESS_NUM"
hostname > mpi_host_file

# explicit export for mpirun (openmpi) — ensures child procs receive PYTHONPATH
mpirun -x PYTHONPATH -np $PROCESS_NUM python main_fedml_object_detection.py --cf config/simulation/fedml_config.yaml
