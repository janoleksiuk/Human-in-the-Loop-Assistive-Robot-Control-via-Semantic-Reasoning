# Repository Structure

This repository is organized around three research subsystems:

- `src/human_action_recognition`: learned HAR models, dataset utilities, and real-time inference.
- `src/robotic_task_execution`: ZED-based pose capture, PNN pose/action decoding, and Boston Dynamics Spot control.
- `src/ontology_reasoning`: symbolic event-driven reasoning, ontology-inspired state updates, and simulated task dispatch.

Supporting assets are placed at the repository root:

- `data/har_dataset`: raw, processed, and split pose datasets.
- `data/reference/spot_pose_classifier`: reference CSV used by the Spot PNN classifier.
- `checkpoints/har`: pretrained HAR model weights.
- `checkpoints/detection`: YOLO checkpoint for Spot-side object detection.
- `examples/spot_control`: legacy Spot SDK examples kept for reference.
- `assets/diagrams` and `assets/example_outputs`: example outputs retained from the original projects.
- `scripts/`: root-level entry points that expose the main workflows without requiring users to inspect internal folders.
