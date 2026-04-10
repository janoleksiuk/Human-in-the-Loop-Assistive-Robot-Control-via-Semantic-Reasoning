# Human-in-the-Loop Assistive Robot Control via Semantic Reasoning

<p align="left">
  <strong>Authors:</strong>
  Jan Oleksiuk<sup>1</sup>, Vibekananda Dutta<sup>1</sup>, Teresa Zielińska<sup>2</sup>, 
  Quan Fu<sup>2</sup>, Listane E. Awong<sup>2</sup>, and Zuyang Fan<sup>2</sup>
</p>

<p align="left">
  <sup>1</sup> <em>Warsaw University of Technology, Faculty of Mechatronics, 02-525 Warsaw, Poland</em><br>
  <sup>2</sup> <em>Warsaw University of Technology, Faculty of Power and Aeronautics Engineering, 00-665 Warsaw, Poland</em>
</p>

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-installed-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-installed-150458?style=for-the-badge&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-installed-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-installed-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-installed-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Torchvision](https://img.shields.io/badge/Torchvision-installed-5C3EE8?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-installed-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-installed-111111?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-installed-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-installed-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![PyZED](https://img.shields.io/badge/PyZED-installed-00A6D6?style=for-the-badge)
![SpotSDK](https://img.shields.io/badge/SpotSDK-installed-222222?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-FCC624?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Research--Ready-2EA043?style=for-the-badge)

## 📚 Table of contents

- [Introduction](#-introduction)
- [Listed key features](#-listed-key-features)
- [Description how the system works with module components](#-description-how-the-system-works-with-module-components)
- [System requirements](#-system-requirements)
- [Installation guide](#-installation-guide)
- [Data preparation](#-data-preparation)
- [Configuration and list of parameters](#-configuration-and-list-of-parameters)
- [Execution](#-execution)
- [Limitations](#-limitations)
- [License](#-license-mit)

## 🧭 Introduction

This repository presents a unified research codebase for an assistive robotics framework that combines human action recognition, ontology-driven semantic reasoning, and robotic task execution. The original implementation was distributed across separate internal projects focused on symbolic reasoning, machine learning models, and hardware-oriented robot control. In this consolidated version, those components are reorganized into a single publication-ready repository intended to support paper submission, code inspection, and follow-up research.

At a system level, the repository models a perception-to-action pipeline for assistive environments. Human motion is observed from body-tracking data, represented either as pose sequences for learned recognition or as symbolic pose events for reasoning. The learned HAR subsystem contains dataset utilities, multiple model definitions, pretrained checkpoints, and a real-time GRU-based inference script for ZED camera input. In parallel, the ontology reasoning subsystem implements an event-driven architecture inspired by Arianna+, where perception updates are converted into symbolic statements, accumulated in memory, interpreted as human actions, and linked to robot tasks or preparatory actions. The robotic execution subsystem contains ZED-based body tracking, a Probabilistic Neural Network pose decoder, simple action-sequence detection logic, and Boston Dynamics Spot routines for task execution with object localization and grasping.

The repository is organized to make these roles explicit. Source code is grouped by subsystem under `src/`, data and checkpoints are placed in dedicated top-level folders, and root-level scripts expose the principal workflows. The codebase intentionally preserves the original implementations where possible, while improving naming, navigation, and documentation. Where behavior could not be inferred reliably, this README documents assumptions conservatively and marks unresolved details as placeholders instead of introducing unsupported claims. The result is a transparent reference implementation of an assistive robotics stack that spans perception, recognition, reasoning, and robot action.

## ✨ Listed key features

- 🤖 Unified repository spanning HAR, ontology-based reasoning, and robotic task execution.
- 🧠 Multiple HAR model implementations collected in one module, including GRU, LSTM, CNN, MLP, SVM, KNN, SOM, and XGBoost wrappers.
- 🎥 Real-time ZED-camera inference workflow for pose-based action recognition.
- 🕸️ Event-driven symbolic reasoning pipeline with ontology-inspired state updates, action families, and pre-task dispatch.
- 🐕 Boston Dynamics Spot control routines for approach, grasp, delivery, and return behaviors.
- 📦 Included sample dataset splits, pretrained checkpoints, reference classifier data, and example outputs for reproducibility.

## 🧩 Description how the system works with module components

The repository contains three main technical modules:

```text
.
├── assets/                     # Example diagrams and trace outputs
├── checkpoints/                # Pretrained HAR and detection weights
├── data/                       # HAR dataset assets and Spot classifier reference data
├── docs/                       # Navigation notes and retained notebook
├── examples/                   # Spot SDK examples
├── scripts/                    # Top-level entry points
└── src/
    ├── human_action_recognition/
    ├── ontology_reasoning/
    └── robotic_task_execution/
```

The end-to-end logic is organized as follows:

- `src/human_action_recognition` implements data preprocessing, train/validation/test splitting, reusable model classes, and a real-time GRU inference script (`predict.py`). The repository keeps pretrained checkpoints in `checkpoints/har/` and the sample dataset in `data/har_dataset/`.
- `src/ontology_reasoning` implements a symbolic pipeline driven by events. `main.py` wires the scheduler, ontology-inspired state objects, action definitions, action-family early intent logic, and a simulated Spot interface. This subsystem produces execution traces and architecture diagrams in `runs/`.
- `src/robotic_task_execution` implements the hardware-facing execution stack. `body_tracker/body_tracking.py` acquires ZED body keypoints, `pose_classifier/pnn.py` decodes short pose windows, `pose_classifier/detect_human_action.py` maps pose sequences to action codes, and `spot_control/action_control.py` executes Spot tasks when an action is detected.

Operationally, the framework can be read in three layers:

- 👁️ Perception layer: ZED body tracking yields pose measurements or pose windows.
- 🧠 Interpretation layer: motion is interpreted either by learned HAR models or by symbolic action reasoning.
- 🤖 Action layer: a task controller dispatches assistive behaviors such as search, approach, grasp, delivery, or return-to-start.

The learned and symbolic pipelines are complementary in this repository rather than fully merged into one executable. The ontology module currently uses simulated pose input, while the Spot execution module uses its own PNN-based pose/action decoding chain.

It implements the following workflow regarding the assistivie robot control based on the reasoning and network of ontologies reasoning:

<img width="5891" height="4191" alt="fig2" src="https://github.com/user-attachments/assets/b963a17a-75bb-4be9-a1da-5986b9acac29" />

<img width="5550" height="3166" alt="Posture I (3)" src="https://github.com/user-attachments/assets/fcd1503a-7317-4156-88e7-5693d9be0a22" />

## ⚙️ System requirements

The following requirements can be inferred directly from the repository:

- `Python`: Python 3.10+ is recommended. The original Spot and ZED code was documented for Windows, and some scripts were clearly developed on Windows.
- `Operating system`: Windows 10/11 is the safest assumption for the hardware-facing Spot/ZED workflows. The ontology-reasoning subsystem is standard Python and should be portable.
- `Camera`: ZED 2 or another ZED device supported by `pyzed` is required for the real-time HAR and body-tracking scripts.
- `Robot`: Boston Dynamics Spot with Spot Arm is required for the robotic execution scripts under `src/robotic_task_execution/spot_control/`.
- `GPU`: CUDA-capable acceleration is optional but likely beneficial for PyTorch inference and YOLO-based object detection.
- `External software`: Graphviz is optional if SVG architecture rendering is desired from the ontology subsystem.

Main Python dependencies inferred from imports:

- `numpy`, `pandas`, `scipy`, `scikit-learn`
- `torch`, `torchvision`
- `opencv-python`, `pyzed`
- `rdflib`, `streamlit`, `plotly`
- `ultralytics`
- `bosdyn-api`, `bosdyn-client`, `bosdyn-core`

If you only want the symbolic reasoning subsystem, the Spot SDK and ZED SDK are not required.

## 🛠️ Installation guide

1. Clone the repository and enter it.

```bash
git clone <repository-url>
cd Human-in-the-Loop-Assistive-Robot-Control-via-Semantic-Reasoning
```

2. Create and activate a Python environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install Python dependencies.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. Install vendor SDKs if you plan to run hardware-dependent modules.

- 🎥 Install the ZED SDK before using `pyzed`-based scripts.
- 🐕 Install and configure the Boston Dynamics Spot SDK before using Spot control scripts.
- 🧭 Install Graphviz if you want `runs/diagrams/*.svg` to be generated automatically by the ontology pipeline.

5. Verify the repository runners that expose the main workflows:

```bash
python scripts/run_ontology_reasoning.py
python scripts/run_realtime_har.py
python scripts/run_spot_pipeline.py
python scripts/run_spot_control.py --hostname <SPOT_IP>
```

The last two commands require the corresponding hardware and SDKs.

## 🗂️ Data preparation

The repository already includes a small sample pose dataset under `data/har_dataset/` and a reference CSV for the Spot PNN classifier under `data/reference/spot_pose_classifier/reference_data.csv`.

Expected HAR dataset structure:

- `data/har_dataset/raw/<class_name>/*.csv`: raw pose recordings grouped by action class.
- `data/har_dataset/processed/*.npz`: per-class processed sequence datasets.
- `data/har_dataset/split/{train,val,test}.npz`: merged dataset splits for model development.

Observed dataset assumptions from the code:

- 📏 Raw inputs are CSV files derived from ZED body tracking.
- 🧍 The learned HAR scripts expect 34 body keypoints with 3D coordinates.
- ⏱️ Sequence windows are built with configurable `seq_len` and `step` parameters.
- 🏷️ Class definitions are controlled by `src/human_action_recognition/data/classes.py`.

To preprocess one class from raw CSV files:

```bash
python scripts/data_preparation/preprocess_pose_data.py ^
  --input_dir data/har_dataset/raw/standing ^
  --output_path data/har_dataset/processed/standing.npz ^
  --class_name standing ^
  --smooth sg ^
  --seq_len 30 ^
  --step 5
```

To merge processed classes into train/validation/test splits:

```bash
python scripts/data_preparation/split_pose_dataset.py ^
  --processed_dir data/har_dataset/processed ^
  --output_dir data/har_dataset/split
```

For the Spot execution pipeline, the body tracker writes short pose windows into shared memory, which are then consumed by the PNN classifier. The reference file `data/reference/spot_pose_classifier/reference_data.csv` contains labeled examples for the default PNN workflow. If a new label set is introduced, the corresponding pose mappings and action-sequence logic must also be updated.

## 🔧 Configuration and list of parameters

The most important configurable parameters are distributed across subsystem-specific files:

- `src/ontology_reasoning/config.py`
- `src/robotic_task_execution/config/config.py`
- `src/human_action_recognition/predict.py`
- `src/human_action_recognition/data/data_preprocessing.py`
- `src/human_action_recognition/data/data_split.py`
- `src/robotic_task_execution/spot_control/robot_task.py`

Key ontology-reasoning parameters:

- `POSE_TICK_SECONDS`: simulated perception tick interval.
- `MAX_POSE_BUFFER_LEN`: memory length for retained pose segments.
- `TRACE`, `TRACE_JSONL`, `TRACE_JSONL_PATH`: trace capture settings.
- `ARCH_DIAGRAM_ENABLE`, `ARCH_DIAGRAM_OUTPUT_DIR`, `ARCH_DIAGRAM_RENDER_SVG`: architecture export settings.
- `POSE_DETECTOR_MODE`: simulated pose generation mode (`random` or `action_sequence`).
- `ACTION_DEFINITIONS`: symbolic action templates.
- `ACTION_TO_TASK`: mapping from recognized action to robot task.
- `TASK_DEFINITIONS` and `PRETASK_DEFINITIONS`: task and pre-task behavior sequences.
- `FAMILY_PRETASK_BY_PREFIX`: early-intent family preparation mapping.

Key robotic execution parameters:

- `BODY_IDX`, `CONFIDENCE_THR`, `SAVE_SESSION_DATA`, `POSES_DICT` in `src/robotic_task_execution/config/config.py`.
- Shared-memory channel names in both `src/robotic_task_execution/config/config.py` and `src/robotic_task_execution/spot_control/utils/shared_memory.py`.
- `MODEL_PATH` in `src/robotic_task_execution/spot_control/action_control.py` for the YOLO checkpoint.
- `FIRST_TARGET`, `SECOND_TARGET`, `GRAB_OBJECT`, and `ROT_VEL` in `src/robotic_task_execution/spot_control/robot_task.py`.

Key learned HAR parameters:

- `INPUT_DIM`, `SEQ_LEN`, `HIDDEN_DIM`, `NUM_LAYERS`, and `MODEL_PATH` in `src/human_action_recognition/predict.py`.
- `seq_len`, `step`, `augment`, and `smooth` in `src/human_action_recognition/data/data_preprocessing.py`.
- `train_ratio`, `val_ratio`, `test_ratio`, and `random_state` in `src/human_action_recognition/data/data_split.py`.

Several of these parameters are still edited directly in Python files rather than through a single centralized configuration system.

## ▶️ Execution

Practical entry points from the repository root:

1. Run the symbolic ontology-based reasoning pipeline.

```bash
python scripts/run_ontology_reasoning.py
```

This produces a live event-driven simulation and, by default, writes outputs to `runs/trace.jsonl` and `runs/diagrams/`.

2. Launch the Streamlit trace dashboard for ontology execution logs.

```bash
streamlit run src/ontology_reasoning/visualization/dashboard/app.py
```

3. Run real-time learned HAR inference from a ZED camera.

```bash
python scripts/run_realtime_har.py
```

4. Launch the Spot-side perception pipeline.

```bash
python scripts/run_spot_pipeline.py
```

This starts the body tracker, PNN pose classifier, and action detector as coordinated subprocesses with shared memory.

5. Run Spot robot task execution after the Spot-side detector is active.

```bash
python scripts/run_spot_control.py --hostname <SPOT_IP>
```

6. Inspect legacy Spot examples retained for reference.

```bash
python examples/spot_control/example_01.py --hostname <SPOT_IP>
python examples/spot_control/example_02.py --hostname <SPOT_IP>
```

Because these examples were preserved from the original project, they may require minor path adjustments or execution from a prepared Spot SDK environment.

## ⚠️ Limitations

- The repository contains two parallel recognition pathways: a learned HAR stack and a symbolic ontology stack. They are documented together as one framework, but they are not yet exposed through a single merged runtime entry point.
- The ontology subsystem currently uses simulated pose input rather than directly consuming the learned HAR output or the Spot shared-memory pipeline.
- Training code for the learned HAR models is only partially packaged. The reusable model definitions and data utilities are included, but the main training workflow appears to rely on the retained notebook in `docs/notebooks/TrainingNotebook.ipynb`.
- Hardware-dependent execution was inferred from the code and original folder contents. Exact device firmware, SDK versions, and network setup details are not fully specified in the source and should be treated as environment-specific.
- Some scripts still rely on direct file editing for configuration rather than a unified CLI or YAML configuration layer.
- The Spot action detector currently implements only a small number of hard-coded action sequences; extending the behavior set requires manual updates in `src/robotic_task_execution/pose_classifier/detect_human_action.py`.

## 📄 License (MIT)

This repository is released under the MIT License. See `LICENSE` for the full text.
