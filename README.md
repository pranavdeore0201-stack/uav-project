# Fixed-Wing UAV Ground Control Station & Autonomous Mapping Simulator

## What it is
This project is an advanced, physics-heavy Ground Control Station (GCS) simulation built for fixed-wing autonomous mapping and search-and-rescue sorties. It integrates a Boustrophedon coverage path planner with analytic Dubins turn kinematics, a rigorous endurance-aware energy model tracking LiPo electrochemistry, voltage sag, and $I^2R$ Joule heating penalties, alongside a pinhole camera model for real-time target geolocation and YOLO-based aerial person detection.

## How to run
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the main 3D GCS simulation and physics engine:
   ```bash
   python pranav2.py
   ```
3. Run the offline computer vision and pinhole geolocation demo:
   ```bash
   python vision_geolocation_demo.py
   ```

## Screenshots
* **3D GCS Flight Simulation:** Displays live telemetry, dynamic power spikes during banked turns, and real-time sensor footprint projection.
* **Offline Vision Demo:** Outputs YOLO bounding box inferences and back-projected local ground coordinates.

## Limitations
* **YOLO Detection:** The vision script utilizes a standard COCO-pretrained YOLOv8 model on a stand-in aerial photograph rather than a domain-specific fine-tuned aerial dataset (e.g., VisDrone), resulting in lower recall on small sub-10-pixel targets.
* **Geolocation Model:** Assumes a flat-earth approximation and a strictly nadir-pointing camera, omitting terrain elevation models (DEM) and aircraft roll/pitch attitude perturbations.

## Image Credit
* Sample aerial photograph provided for the vision demo courtesy of public-domain aerial dataset repositories (VisDrone / Open-Source Mapping Imagery).
