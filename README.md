# Federated YOLOv5 Object Detection with Performance Benchmarking using FedML

A comprehensive federated learning implementation for object detection using YOLOv5 and the FedML framework, enhanced with detailed performance tracking and benchmarking capabilities for research and production evaluation.

---

## 🎯 Overview

This project enables **privacy-preserving collaborative training** of YOLOv5 object detection models across multiple clients without sharing raw data. Built on the FedML framework, it provides a complete solution for federated learning experiments with extensive performance monitoring and benchmarking tools.

### Key Highlights

- **Federated Learning**: Horizontal federated learning simulation with multiple clients
- **Privacy-Preserving**: Train models collaboratively without sharing raw image data
- **Production-Ready**: Comprehensive logging, checkpointing, and monitoring
- **Performance Benchmarking**: Detailed metrics tracking for communication overhead, training efficiency, and model performance
- **Flexible Architecture**: Easy to extend and customize for different datasets and scenarios

---

## ✨ Features

### Core Functionality
- ✅ **FedML Integration**: Seamless federated learning orchestration with MPI backend
- ✅ **YOLOv5 Support**: Full YOLOv5 object detection capabilities in federated setting
- ✅ **Custom Dataset Support**: Default COCO128 dataset with easy configuration for custom data
- ✅ **Multi-Client Simulation**: Simulate distributed training with configurable number of clients
- ✅ **Flexible Aggregation**: FedAvg aggregation with support for custom strategies

### Training & Optimization
- ✅ **Configurable Optimizers**: Support for Adam and SGD optimizers
- ✅ **Learning Rate Scheduling**: Cosine annealing scheduler for stable convergence
- ✅ **Hyperparameter Tuning**: Comprehensive hyperparameter configuration via YAML
- ✅ **Batch Processing**: Efficient batch training with configurable batch sizes
- ✅ **Model Checkpointing**: Automatic model saving at configurable intervals

### Advanced Benchmarking Features

#### 📊 Performance Metrics
- **Training Metrics**: Loss tracking (box, objectness, classification losses)
- **Validation Metrics**: mAP, mAP@0.5, precision, recall per round
- **Communication Overhead**: Model size tracking for sent/received updates
- **Time Analysis**: Round duration, batch processing speed, epoch timing
- **Resource Monitoring**: GPU memory usage, CPU utilization

#### 📈 Comprehensive Logging
- **CSV-Based Logging**: Thread-safe logging with file locking mechanism
- **Real-Time Monitoring**: Per-round and per-epoch metrics tracking
- **Client-Specific Logs**: Individual client performance tracking
- **Server Aggregation Logs**: Global model performance after aggregation

#### 🔬 Research-Ready Outputs
- **Training Logs**: Box loss, objectness loss, classification loss per round
- **Validation Logs**: mAP, precision, recall, model size tracking
- **Communication Logs**: Model transmission sizes (MB) for bandwidth analysis
- **Timing Logs**: Round duration, learning rate, number of examples processed

---

## 📦 Installation

### Prerequisites
- Conda or Miniconda
- Python 3.10
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/fedml-yolov5-benchmarks.git
cd fedml-yolov5-benchmarks
```

### Step 2: Clone YOLOv5 (Required)
⚠️ **Important**: The `model/` folder is ignored in git, so you need to clone YOLOv5 manually.

```bash
# Clone YOLOv5 v6.2 inside the project folder
git clone -b v6.2 https://github.com/ultralytics/yolov5.git model/yolov5
```

### Step 3: Create Conda Environment
```bash
conda env create -f environment.yml
conda activate fedml-benchmarks
```

### Step 4: Bootstrap Dependencies and Dataset
```bash
bash bootstrap.sh
```

This script will:
- Install additional Python dependencies (OpenCV, Pandas, Matplotlib, Seaborn, Addict)
- Create the `~/fedcv_data` directory
- Download the COCO128 dataset automatically

---

## 🚀 Usage

### Quick Start: Run Federated Simulation

Run a federated learning simulation with a specified number of clients:

```bash
bash run_simulation.sh 2  # Replace 2 with desired number of clients
```

This will:
- Launch 1 server process + N client processes
- Train the model for the configured number of communication rounds
- Save checkpoints and logs automatically
- Generate comprehensive performance metrics

### Advanced Usage

#### Run Server Independently
```bash
bash run_server.sh
```

#### Run Client Independently
```bash
bash run_client.sh
```

#### Custom Configuration
Modify `config/simulation/fedml_config.yaml` to customize:
- Number of clients and communication rounds
- Batch size, learning rate, optimizer
- Model architecture and hyperparameters
- Checkpoint intervals and validation frequency
- Logging and tracking settings

---

## ⚙️ Configuration

### Main Configuration File: `fedml_config.yaml`

```yaml
# Key Parameters
train_args:
  client_num_in_total: 1        # Total number of clients
  client_num_per_round: 1       # Clients participating per round
  comm_round: 3                 # Number of communication rounds
  epochs: 1                     # Local epochs per round
  batch_size: 16                # Training batch size
  lr: 0.01                      # Learning rate
  client_optimizer: adam        # Optimizer (adam/sgd)
  federated_optimizer: FedAvg   # Aggregation strategy

model_args:
  yolo_cfg: ./model/yolov5/models/yolov5s.yaml  # YOLOv5 architecture
  weights: none                 # Pretrained weights (optional)
  
data_args:
  data_conf: ./data/coco128.yaml  # Dataset configuration
  img_size: [640, 640]          # Input image size
```

### Benchmarking Configuration

```yaml
common_args:
  log_file_name: fedml-bench-logging-exp1.csv  # CSV log file for benchmarks

train_args:
  checkpoint_interval: 5              # Save client checkpoints every N epochs
  server_checkpoint_interval: 5       # Save server checkpoints every N rounds

validation_args:
  frequency_of_the_test: 1            # Run validation every N epochs
```

---

## 📊 Benchmarking & Performance Tracking

### Enhanced Logging System

The project includes a sophisticated benchmarking system that tracks:

#### Training Logs (Per Round)
- **Type**: `train`
- **Metrics**: Client ID, round ID, round loss, round duration
- **Performance**: Learning rate, number of examples, worker count
- **Communication**: Received model size (MB), sent model size (MB)

#### Validation Logs (Per Round)
- **Type**: `val`
- **Metrics**: Client ID, round ID, mAP, precision, recall
- **Model Info**: Current model size (MB)

### Sample Log Output

```csv
type,client_id,role,round_id,comm_round,round_loss,round_duration,lr,num_examples,worker_num,recv_model_size,sent_model_size
train,1,client,0,3,2.345,45.67,0.01,128,2,14.23,14.23
val,1,client,0,3,0.456,0.789,0.812,14.23
```

### Key Performance Indicators

1. **Communication Efficiency**
   - Model size tracking (MB) for bandwidth analysis
   - Round trip time measurement
   
2. **Training Efficiency**
   - Loss convergence per round
   - Training speed (seconds per batch)
   - Epoch duration

3. **Model Performance**
   - Mean Average Precision (mAP)
   - Precision and Recall metrics
   - Per-class AP (optional verbose mode)

4. **Resource Utilization**
   - GPU memory usage
   - Number of training examples
   - Worker distribution

---

## 📁 Project Structure

```
fedml-yolov5-benchmarks/
│
├── config/                           # Configuration files
│   ├── simulation/
│   │   └── fedml_config.yaml        # Main federated learning config
│   └── hyps/
│       └── hyp.scratch.yaml         # YOLOv5 hyperparameters
│
├── data/                            # Dataset directory
│   └── coco128.yaml                 # COCO128 dataset config
│
├── model/                           # YOLOv5 model code (clone manually)
│   └── yolov5/                      # YOLOv5 repository (v6.2)
│
├── trainer/                         # Federated learning components
│   ├── yolo_aggregator.py          # Server-side aggregation logic
│   └── yolov5_trainer.py           # Client-side training logic
│
├── logs/                            # Training logs and outputs
│   └── *.csv                        # Benchmark CSV logs
│
├── runs/                            # Training artifacts
│   └── train/                       # Checkpoints and visualizations
│       └── weights/                 # Saved model weights
│
├── main_fedml_object_detection.py  # Entry point for federated training
├── environment.yml                  # Conda environment specification
├── bootstrap.sh                     # Dependency installation script
├── run_simulation.sh                # Simulation launcher script
└── README.md                        # This file
```

---

## 🔬 Benchmarking Advantages

### 1. **Comprehensive Performance Analysis**
- Track communication overhead and bandwidth usage
- Measure training efficiency across federated rounds
- Monitor model convergence and performance metrics

### 2. **Research-Ready Outputs**
- CSV format compatible with Pandas, Excel, and visualization tools
- Thread-safe logging with file locking for concurrent writes
- Structured data for easy analysis and plotting

### 3. **Production Monitoring**
- Real-time tracking of federated training progress
- Model size monitoring for deployment planning
- Client-specific performance insights

### 4. **Easy Integration with Analysis Tools**
- Direct compatibility with Pandas DataFrames
- Ready for matplotlib/seaborn visualizations
- Compatible with WandB for experiment tracking

---

## 🛠️ Advanced Features

### Model Checkpointing
- **Client Checkpoints**: Saved every `checkpoint_interval` epochs
- **Server Checkpoints**: Saved every `server_checkpoint_interval` rounds
- **Location**: `runs/train/weights/`
- **Naming**: `model_client_{id}_epoch_{epoch}.pt` or `model_{round}.pt`

### Validation Integration
- Built-in validation using YOLOv5's validation pipeline
- Configurable validation frequency
- Full metrics reporting (mAP, mAP@0.5, precision, recall)

### Thread-Safe Logging
- FileLock mechanism ensures safe concurrent writes
- No log corruption in multi-process MPI environment
- Atomic CSV operations

---

## 📋 Requirements

### Core Dependencies (via Conda)
- Python 3.10
- PyTorch (CPU/GPU)
- OpenMPI & mpi4py
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn, SciPy

### Additional Dependencies (via pip)
- FedML framework
- Ultralytics (YOLOv5 utilities)
- OpenCV (headless)
- Albumentations
- TensorBoard, WandB (optional)
- THOP, pycocotools

See `environment.yml` for complete dependency list.

---

## 🎓 Use Cases

### Research Applications
- Federated learning algorithm development
- Communication efficiency studies
- Privacy-preserving computer vision research
- Distributed training performance analysis

### Industrial Applications
- Cross-organization model training
- Edge device deployment scenarios
- Privacy-compliant ML pipelines
- Bandwidth-constrained environments

---

## 📊 Example Workflow

1. **Configure Experiment**: Edit `config/simulation/fedml_config.yaml`
2. **Run Training**: Execute `bash run_simulation.sh 4` (4 clients)
3. **Monitor Progress**: Check logs in real-time
4. **Analyze Results**: Load CSV logs for performance analysis
5. **Deploy Model**: Use checkpoints from `runs/train/weights/`

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: YOLOv5 import errors
- **Solution**: Ensure `model/yolov5` is cloned correctly and YOLOv5 v6.2 is used

**Issue**: MPI process failures
- **Solution**: Check MPI installation with `mpirun --version`

**Issue**: CUDA out of memory
- **Solution**: Reduce `batch_size` in config or use CPU (`using_gpu: false`)

**Issue**: CSV log corruption
- **Solution**: FileLock should prevent this; check file permissions

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional aggregation algorithms (FedProx, FedOpt)
- Support for more YOLOv5 variants (v8, v9)
- Enhanced visualization tools
- Additional benchmark metrics
- Cross-silo federated learning support

---

## 📜 License

This project combines multiple open-source components:
- YOLOv5: GPL-3.0 License
- FedML: Apache-2.0 License

Please ensure compliance with respective licenses when using this project.

---

## 🙏 Acknowledgments

- **FedML Team**: For the excellent federated learning framework
- **Ultralytics**: For YOLOv5 object detection models
- **PyTorch Community**: For the deep learning foundation

---

## 📧 Contact & Support

For questions, issues, or collaboration:
- Open an issue on GitHub
- Check FedML documentation: https://docs.fedml.ai
- Check YOLOv5 documentation: https://docs.ultralytics.com

---

**Star ⭐ this repository if you find it useful for your research or projects!**
