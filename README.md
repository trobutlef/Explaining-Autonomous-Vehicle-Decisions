# Explaining Autonomous Vehicle Decisions: XAI for Perception and Action

This project implements Explainable AI (XAI) techniques to interpret the decisions of an autonomous driving agent in the MetaDrive environment. It covers both perception (visual attention) and action (decision-making) explainability.

## Project Structure

This project unifies Perception and Action explainability within the MetaDrive simulation environment. Instead of separate phases, we apply both visual and feature-based XAI techniques to driving agents to understand their behavior holistically.

- **`notebooks/`**: Contains the main executable Jupyter notebooks.
  - `01_metadrive_general_xai.ipynb`: **General Simulation XAI (Perception + Action).** Demonstrates the full XAI pipeline on a general MetaDrive agent (IDM Policy).
    - **Perception:** Uses **Grad-CAM** on the 3D driver's view to show visual attention.
    - **Action:** Uses **SHAP & LIME** to explain driving decisions based on state features (Lidar, Speed, Steering).
  - `02_metadrive_pretrained_agent_xai.ipynb`: **Trained Agent XAI (Perception + Action).** Applies the same XAI pipeline to your **pretrained DQN agent**.
    - **Perception:** Visualizes the agent's context using **Grad-CAM** on the simulation frames.
    - **Action:** Explains the specific Q-network decisions using **SHAP (DeepExplainer)**.
  - `bdd100k_gradcam.ipynb`: **Supplementary: Real-World Perception.** Applies Grad-CAM to real-world driving images from the BDD100K dataset for comparison.
- **`src/`**: Source code for agents, environments, and XAI utilities.
- **`outputs/`**: Generated results, including videos and models.
  - `simulation_general.mp4`: Raw video of the general simulation.
  - `gradcam_general.mp4`: Grad-CAM overlay video for the general simulation.
  - `simulation_agent.mp4`: Raw video of the pretrained agent.
  - `gradcam_agent.mp4`: Grad-CAM overlay video for the pretrained agent.
- **`configs/`**: Configuration files (YAML).
- **`environment.yml`**: List of Python dependencies.

## Installation

1.  **Prerequisites:** Python 3.8+
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    # OR using conda/mamba
    conda env create -f environment.yml
    conda activate xai_autonomous_vehicle
    ```
3.  **MetaDrive:** Ensure MetaDrive is installed. If using the specific version from this project:
    ```bash
    pip install git+https://github.com/metadriverse/metadrive.git
    ```

## Dataset Usage

- **MetaDrive:** The simulation environment generates synthetic data (images and vector states) on the fly. No external download is required for the main simulation notebooks.
- **BDD100K (Optional for `bdd100k_gradcam.ipynb`):** This notebook uses a subset of the BDD100K dataset.
  - **Status:** A small subset is included in `data/bdd100k_subset` (if available).
  - **Manual Download:** If you wish to run this on the full dataset, please download it from [BDD100K Website](https://bdd-data.berkeley.edu/) and update the path in `configs/config.yaml`.

## How to Run

1.  **Navigate to the project root.**
2.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
3.  **Open and Run:**
    - `notebooks/01_metadrive_general_xai.ipynb`
    - `notebooks/02_metadrive_pretrained_agent_xai.ipynb`

## Results for Presentation

- **Videos:** Check the `outputs/` directory for MP4 videos of the driving agents with and without Grad-CAM overlays.
- **Plots:** SHAP and LIME feature importance plots are generated inline within the notebooks. You can save these images directly from the notebook (Right-click -> Save Image).

## Code Explanations

- **Grad-CAM:** Visualizes which parts of the camera input the model focuses on. Implemented using `pytorch-grad-cam` on a proxy ResNet model.
- **SHAP (SHapley Additive exPlanations):** Assigns importance values to each state feature (e.g., speed, steering, lidar points) to explain the agent's action.
- **LIME (Local Interpretable Model-agnostic Explanations):** Approximates the complex model locally with a simple linear model to explain individual predictions.
