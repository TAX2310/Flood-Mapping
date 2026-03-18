# Flood-Mapping

This repository contains the notebooks and configuration for a flood-mapping project based on the STURM-Flood dataset.

---

## Repository Structure

```text 
Flood-Mapping/
├── config.py
├── 01_setup.ipynb
├── S1/
|   ├──s1_config.py
|   └──02_train_s1.ipynb
├── S2/
|   ├──s2_config.py
|   └──03_train_s3.ipynb
├── Fusion/
|   ├──fusion_config.py
|   └──04_train_fusion.ipynb
```

---

## Getting Started

Clone the repository:

git clone https://github.com/TAX2310/Flood-Mapping.git
cd Flood-Mapping


## Google Drive Mounting

All notebooks in this project include a form at the top that allows you to optionally mount a Google Drive and or clone the git repository.

- Set `mount_drive = True` to mount your Google Drive and persist data (datasets, outputs, checkpoints)  
- Specify the `root_path` to define where the project will live within your Drive  
- Set `mount_drive = False` to run entirely within the temporary Colab / hosted VM environment  

- Set `clone_repo = True` to clone the Flood-Mapping git repo  
- Set `clone_repo = False` to assume the repo already exists at the specified location  

---

- If `mount_drive = True` and `clone_repo = True`  
    - Google Drive will be mounted  
    - The repo will be cloned into the specified `root_path`  
    - The final working directory will be: `root_path/Flood-Mapping`  
    - This is recommended for **initial setup with persistent storage**  

- If `mount_drive = True` and `clone_repo = False`  
    - Google Drive will be mounted  
    - The codebase is assumed to already exist at `root_path`  
    - No cloning will occur  
    - This is recommended for **continuing work from a previous session**  

- If `mount_drive = False` and `clone_repo = True`  
    - The repo will be cloned into the temporary runtime (e.g. `/content/`)  
    - No data or code will persist after the session ends  
    - This is recommended for **Colab / hosted VM / pod environments** where a fresh environment is needed  

- If `mount_drive = False` and `clone_repo = False`  
    - No mounting or cloning will occur  
    - The codebase is assumed to already exist locally  
    - This is recommended for **local development environments** where the repo has already been cloned  

---

- In summary:  
    - `mount + clone` → first-time setup with persistence  
    - `mount only` → reuse existing persistent setup  
    - `clone only` → temporary execution in hosted environments  
    - `neither` → local execution with existing codebase 

## Workflow
### 1. 01_setup.ipynb

This notebook should be run first.

downloads the 01_setup.ipynb dataset

extracts the dataset

stores it in the root project directory (Flood-Mapping/Dataset)