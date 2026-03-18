# Flood-Mapping

This repository contains the notebooks and configuration for a flood-mapping project based on the STURM-Flood dataset.

---

## Repository Structure

```text 
Flood-Mapping/
├── config.py
├── 01_setup.ipynb
├── S1/
├── S2/
├── Fusion/
```

---

## Getting Started

Clone the repository:

git clone https://github.com/TAX2310/Flood-Mapping.git
cd Flood-Mapping


## Google Drive Mounting

All notebooks in this project include a form at the top that allows you to optionally mount Google Drive.

- Set `mount_drive = True` to mount your Google Drive and persist data (datasets, outputs, checkpoints)
- Specify the `root_path` to define where the project will live within your Drive
- Set `mount_drive = False` to run entirely within the temporary Colab environment

## Workflow
### 1. 01_setup.ipynb

This notebook should be run first.

downloads the STURM-Flood dataset

extracts the dataset

stores it in the root project directory (Flood-Mapping/Dataset)