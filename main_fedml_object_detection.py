import fedml
from fedml import FedMLRunner
from model.init_yolo import init_yolo
from trainer.yolo_aggregator import YOLOAggregator

from pathlib import Path
import csv
from filelock import FileLock

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()
    # p = Path(args.log_file_name)
    # lock = FileLock(str(p) + ".lock")

    # with lock:
    #     with p.open("a", newline="", encoding="utf-8") as f:
    #         writer = csv.writer(f)
    #         writer.writerow(["run_id", "rank", "role",
    #                             "comm_round", "epochs", "batch_size", "lr", "worker_num",
    #                             "process_id"])
    #         writer.writerow([args.run_id, args.rank, args.role,
    #                          args.comm_round, args.epochs, args.batch_size, args.lr, args.worker_num,
    #                          args.process_id])
            
    # init device
    device = fedml.device.get_device(args)

    # init yolo
    model, dataset, trainer, args = init_yolo(args=args, device=device)
    aggregator = YOLOAggregator(model, args)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model, trainer, aggregator)
    fedml_runner.run()
