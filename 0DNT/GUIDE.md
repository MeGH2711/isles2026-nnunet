# ISLES-2026 nnU-Net Complete Pipeline & Operations Guide

A comprehensive, end-to-end operational guide for setting up the environment, configuring paths, training baseline & custom instance-loss models, synchronizing weights to Google Drive, and downloading evaluation results.

---

## Table of Contents
1. [Project & Python Environment Setup](#1-project--python-environment-setup)
   - [Python 3.11 Installation](#python-311-installation)
   - [Virtual Environment Creation](#virtual-environment-creation)
   - [Prerequisites & Dependencies](#prerequisites--dependencies)
   - [PyTorch & CUDA 12.1 Installation](#pytorch--cuda-121-installation)
   - [PyTorch GPU Verification](#pytorch-gpu-verification)
2. [nnU-Net Environment Variables Setup](#2-nnu-net-environment-variables-setup)
   - [Home System Path Setup](#home-system-path-setup)
   - [College System Path Setup](#college-system-path-setup)
   - [Permanent Environment Variables (PowerShell)](#permanent-environment-variables-powershell)
   - [Verification Commands](#verification-commands)
3. [RClone Installation & Google Drive Configuration](#3-rclone-installation--google-drive-configuration)
   - [Installation & PATH Setup](#installation--path-setup)
   - [Interactive Google Drive Setup (`rclone config`)](#interactive-google-drive-setup-rclone-config)
   - [Remote Connection Verification](#remote-connection-verification)
4. [Automated Cloud Weight Synchronization (`sync_weights.py`)](#4-automated-cloud-weight-synchronization-sync_weightspy)
   - [Sync Script Generator](#sync-script-generator)
   - [Script Source Code](#script-source-code)
   - [Running the Auto-Sync](#running-the-auto-sync)
5. [Model Training](#5-model-training)
   - [Baseline 5-Fold Training](#baseline-5-fold-training)
   - [Instance Loss Training with Pretrained Weights](#instance-loss-training-with-pretrained-weights)
6. [Best Configuration & Ensemble](#6-best-configuration--ensemble)
7. [Downloading Results from Google Drive](#7-downloading-results-from-google-drive)

---

## 1. Project & Python Environment Setup

### Python 3.11 Installation
Install Python 3.11 on Windows via Windows Package Manager (`winget`):

```powershell
winget install Python.Python.3.11
```

### Virtual Environment Creation
Create and activate an isolated virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

### Prerequisites & Dependencies
1. **Visual C++ Redistributable (Required for compilation & dependencies)**:
   - Download: [VC Redist x64](https://aka.ms/vs/17/release/vc_redist.x64.exe)

2. **Python Libraries**:
   Install via `requirements.txt`:
   ```powershell
   pip install -r requirements.txt
   ```
   *Or* install dependencies directly:
   ```powershell
   pip install nnunetv2 monai SimpleITK nibabel scikit-image scikit-learn matplotlib pandas
   ```

### PyTorch & CUDA 12.1 Installation
Install PyTorch with CUDA 12.1 compute wheel:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### PyTorch GPU Verification
Verify that CUDA is recognized and active:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

---

## 2. nnU-Net Environment Variables Setup

nnU-Net v2 requires three core directory environment variables:
- `nnUNet_raw`: Directory containing original formatted raw datasets.
- `nnUNet_preprocessed`: Directory for preprocessed images, plans, and dataset fingerprints.
- `nnUNet_results`: Output directory for checkpoints, progress logs, and validation results.

### Home System Path Setup

#### Command Prompt (CMD)
```cmd
set nnUNet_raw=E:\isles26\data\nnunet_raw
set nnUNet_preprocessed=E:\isles26\data\nnunet_preprocessed
set nnUNet_results=E:\isles26\data\nnunet_results

:: Verify
echo %nnUNet_raw%
echo %nnUNet_preprocessed%
echo %nnUNet_results%
```
*(Reference from `Environment Setup nnUnet.txt`: verify mapping ensures `nnUNet_raw` points to `nnunet_raw` and `nnUNet_results` points to `nnunet_results`)*

#### PowerShell (Session-Level)
```powershell
$env:nnUNet_raw = "E:\isles26\data\nnunet_raw"
$env:nnUNet_preprocessed = "E:\isles26\data\nnunet_preprocessed"
$env:nnUNet_results = "E:\isles26\data\nnunet_results"

## Verify
echo $env:nnUNet_raw
echo $env:nnUNet_preprocessed
echo $env:nnUNet_results
```

---

### College System Path Setup

#### Command Prompt (CMD)
```cmd
set nnUNet_raw=C:\Users\Admin\Desktop\meghpatel\isles26\data\nnunet_raw
set nnUNet_preprocessed=C:\Users\Admin\Desktop\meghpatel\isles26\data\nnunet_preprocessed
set nnUNet_results=C:\Users\Admin\Desktop\meghpatel\isles26\data\nnunet_results

:: Verify
echo %nnUNet_raw%
echo %nnUNet_preprocessed%
echo %nnUNet_results%
```

#### PowerShell (Session-Level)
```powershell
$env:nnUNet_raw = "C:\Users\Admin\Desktop\meghpatel\isles26\data\nnunet_raw"
$env:nnUNet_preprocessed = "C:\Users\Admin\Desktop\meghpatel\isles26\data\nnunet_preprocessed"
$env:nnUNet_results = "C:\Users\Admin\Desktop\meghpatel\isles26\data\nnunet_results"

## Verify
echo $env:nnUNet_raw
echo $env:nnUNet_preprocessed
echo $env:nnUNet_results
```

---

### Permanent Environment Variables (PowerShell)
To set environment variables permanently for your user account on Windows:

```powershell
[Environment]::SetEnvironmentVariable("nnUNet_results", "C:\Users\Admin\Desktop\meghpatel\isles26\data\nnunet_results", "User")
[Environment]::SetEnvironmentVariable("nnUNet_preprocessed", "C:\Users\Admin\Desktop\meghpatel\isles26\data\nnunet_preprocessed", "User")
[Environment]::SetEnvironmentVariable("nnUNet_raw", "C:\Users\Admin\Desktop\meghpatel\isles26\data\nnunet_raw", "User")
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\rclone", "User")
```

Verify in a new terminal:
```powershell
echo $env:nnUNet_results
```

---

## 3. RClone Installation & Google Drive Configuration

### Installation & PATH Setup
Run the following commands in PowerShell to download, extract, install to `C:\rclone`, and register RClone into `User` PATH:

```powershell
# Download rclone zip
Invoke-WebRequest -Uri "https://downloads.rclone.org/rclone-current-windows-amd64.zip" -OutFile "$env:TEMP\rclone.zip"

# Extract it
Expand-Archive -Path "$env:TEMP\rclone.zip" -DestinationPath "$env:TEMP\rclone_extracted" -Force

# Copy rclone.exe to a permanent location
New-Item -ItemType Directory -Force -Path "C:\rclone"
Copy-Item "$env:TEMP\rclone_extracted\rclone-*-windows-amd64\rclone.exe" "C:\rclone\rclone.exe"

# Add to PATH permanently
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\rclone", "User")
```

> [!IMPORTANT]
> Restart your PowerShell/terminal after setting the PATH variable.

Verify installation:
```powershell
rclone --version
```

---

### Interactive Google Drive Setup (`rclone config`)
Run the configuration wizard:
```powershell
C:\rclone\rclone.exe config
```

Follow the prompts step-by-step:

| Prompt | Input / Action | Explanation |
|---|---|---|
| `No remotes found, make a new one?` | Type `n`, press Enter | Create new remote |
| `name>` | Type `gdrive`, press Enter | Remote name identifier |
| `Storage>` | Type `drive`, press Enter | Select Google Drive |
| `client_id>` | Leave blank, press Enter | Default Google OAuth client |
| `client_secret>` | Leave blank, press Enter | Default Google OAuth secret |
| `scope>` | Type `1`, press Enter | Full access to files |
| `root_folder_id>` | Leave blank, press Enter | Default root directory |
| `service_account_file>` | Leave blank, press Enter | Not using service account |
| `Edit advanced config?` | Type `n`, press Enter | Skip advanced settings |
| `Use auto config?` | Type `y`, press Enter | Web browser login authorization |
| `Configure this as a Shared Drive?` | Type `n`, press Enter | Standard personal / institution Drive |
| `y) Yes this is OK` | Type `y`, press Enter | Confirm configuration |
| `q) Quit config` | Type `q`, press Enter | Exit configuration wizard |

---

### Remote Connection Verification
Test the connection by listing directories in the Google Drive remote:
```powershell
C:\rclone\rclone.exe lsd gdrive:
```

---

## 4. Automated Cloud Weight Synchronization (`sync_weights.py`)

During multi-day training runs, checkpoints must be safely backed up to Google Drive at regular intervals without risking corruption from copying active checkpoints.

### Sync Script Generator
Generate `C:\Users\Admin\Desktop\meghpatel\isles26\sync_weights.py` using PowerShell:

```powershell
@"
import subprocess
import time
import shutil
import os
from datetime import datetime

RCLONE_EXE     = r"C:\rclone\rclone.exe"
LOCAL_WEIGHTS  = r"C:\Users\Admin\Desktop\meghpatel\isles26\data\nnunet_results"
TEMP_DIR       = r"C:\Users\Admin\Desktop\meghpatel\isles26\sync_temp"
GDRIVE_WEIGHTS = "gdrive:isles26-2f/nnunet_results"
SYNC_INTERVAL_MINUTES = 10

def make_snapshot(src, temp):
    """Copy entire nnunet_results to a temp folder first"""
    if os.path.exists(temp):
        shutil.rmtree(temp)
    print(f"  Taking local snapshot...", flush=True)
    shutil.copytree(src, temp)
    print(f"  Snapshot ready!", flush=True)

def sync(local, remote):
    process = subprocess.Popen(
        [RCLONE_EXE, "copy", local, remote,
         "--checksum",
         "--verbose",
         "--progress",
         "--stats", "5s",
         "--stats-one-line"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    process.wait()
    return process.returncode == 0

def run():
    print("Auto-sync started. Press Ctrl+C to stop.\n")
    while True:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*50}")
        print(f"  Sync started at {ts}")
        print(f"{'='*50}\n")

        try:
            # Step 1: snapshot live files to temp folder
            make_snapshot(LOCAL_WEIGHTS, TEMP_DIR)

            # Step 2: upload from temp (safe, no active writes)
            ok = sync(TEMP_DIR, GDRIVE_WEIGHTS)

            # Step 3: cleanup temp
            shutil.rmtree(TEMP_DIR)

            ts_end = datetime.now().strftime("%H:%M:%S")
            if ok:
                print(f"\n[{ts_end}] Sync complete! Next sync in {SYNC_INTERVAL_MINUTES} min...")
            else:
                print(f"\n[{ts_end}] Sync FAILED. Retrying in {SYNC_INTERVAL_MINUTES} min...")

        except Exception as e:
            print(f"\n[ERROR] {e}")

        time.sleep(SYNC_INTERVAL_MINUTES * 60)

if __name__ == "__main__":
    run()
"@ | Out-File -FilePath "C:\Users\Admin\Desktop\meghpatel\isles26\sync_weights.py" -Encoding UTF8
```

### Running the Auto-Sync
Start the auto-sync daemon in a dedicated terminal window:

```powershell
python C:\Users\Admin\Desktop\meghpatel\isles26\sync_weights.py
```

---

## 5. Model Training

### Baseline 5-Fold Training
Standard nnU-Net 3D full resolution model training on Dataset `1` (`Dataset001_ISLES26`) with `--npz` export (needed for ensembling) and `--c` (continue training checkpoint flag):

```powershell
# Fold 0
nnUNetv2_train 1 3d_fullres 0 --npz --c

# Fold 1
nnUNetv2_train 1 3d_fullres 1 --npz --c

# Fold 2
nnUNetv2_train 1 3d_fullres 2 --npz --c

# Fold 3
nnUNetv2_train 1 3d_fullres 3 --npz --c

# Fold 4
nnUNetv2_train 1 3d_fullres 4 --npz --c
```

---

### Instance Loss Training with Pretrained Weights
Fine-tuning / training with custom trainer `nnUNetTrainerInstanceLoss` using checkpoints from the baseline training run as pretrained weights:

```powershell
# Fold 0
nnUNetv2_train Dataset001_ISLES26 3d_fullres 0 -tr nnUNetTrainerInstanceLoss -p nnUNetPlans --npz -pretrained_weights "C:\Users\Admin\Desktop\meghpatel\isles26\data\nnunet_results\Dataset001_ISLES26\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_0\checkpoint_final.pth"

# Fold 1
nnUNetv2_train Dataset001_ISLES26 3d_fullres 1 -tr nnUNetTrainerInstanceLoss -p nnUNetPlans --npz -pretrained_weights "C:\Users\Admin\Desktop\meghpatel\isles26\data\nnunet_results\Dataset001_ISLES26\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_1\checkpoint_final.pth"

# Fold 2
nnUNetv2_train Dataset001_ISLES26 3d_fullres 2 -tr nnUNetTrainerInstanceLoss -p nnUNetPlans --npz -pretrained_weights "C:\Users\Admin\Desktop\meghpatel\isles26\data\nnunet_results\Dataset001_ISLES26\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_2\checkpoint_final.pth"

# Fold 3
nnUNetv2_train Dataset001_ISLES26 3d_fullres 3 -tr nnUNetTrainerInstanceLoss -p nnUNetPlans --npz -pretrained_weights "C:\Users\Admin\Desktop\meghpatel\isles26\data\nnunet_results\Dataset001_ISLES26\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_3\checkpoint_final.pth"

# Fold 4
nnUNetv2_train Dataset001_ISLES26 3d_fullres 4 -tr nnUNetTrainerInstanceLoss -p nnUNetPlans --npz -pretrained_weights "C:\Users\Admin\Desktop\meghpatel\isles26\data\nnunet_results\Dataset001_ISLES26\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_4\checkpoint_final.pth"
```

---

## 6. Best Configuration & Ensemble

Once the cross-validation folds are complete, execute configuration identification and post-processing determination:

```powershell
nnUNetv2_find_best_configuration Dataset001_ISLES26 -c 3d_fullres
```

---

## 7. Downloading Results from Google Drive

To download trained fold weights or cross-validation results from Google Drive using a shared folder ID:

```powershell
rclone copy --drive-root-folder-id 1n8MAT1apo8GDUVKmKay31-g1xCw080u6 gdrive: C:\Users\Admin\Desktop\meghpatel\isles26\data\nnunet_results\Dataset001_ISLES26\nnUNetTrainerInstanceLoss__nnUNetPlans__3d_fullres\crossval_results_folds_0_1_2_3_4 --progress
```
