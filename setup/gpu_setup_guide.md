# Windows + VSCode GPU Setup Guide

This guide describes how to configure your Windows environment to enable GPU acceleration in TensorFlow 2.15.0 for the Fashion-MNIST GAN project.

---

## 1. Verify NVIDIA Driver (`nvidia-smi`)

First, ensure you have a compatible NVIDIA graphics card and the latest drivers installed. Open PowerShell or Command Prompt and run:

```bash
nvidia-smi
```

You should see details about your GPU model, driver version, and CUDA version support (e.g., CUDA 12.x supported by driver).

---

## 2. Install CUDA Toolkit 11.8

TensorFlow 2.15.0 is compatible with CUDA 11.8. 

1. Download the installer from the official archive:
   👉 **[CUDA Toolkit 11.8.0 Downloads](https://developer.nvidia.com/cuda-11-8-0-download-archive)**
2. Select **Windows**, your architecture (**x86_64**), version, and installer type (**exe (local)**).
3. Run the installer and choose the **Express Installation** option.

---

## 3. Install cuDNN 8.6

1. Go to the **[NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)** (requires a free NVIDIA Developer account).
2. Download **cuDNN v8.6.0 (September 21st, 2022), for CUDA 11.x**.
3. Extract the downloaded zip file.
4. Copy the contents of the extracted folders into your CUDA installation directory (typically `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`):
   * Copy files in `bin/` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`
   * Copy files in `include/` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include`
   * Copy files in `lib/x64/` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64`
5. Verify that your system Environment Variables include the paths to CUDA's `bin` and `libnvvp` directories.

---

## 4. Create Python Virtual Environment

Create and activate a clean virtual environment in the project directory:

```powershell
# Create venv
python -m venv venv

# Activate venv (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate venv (Command Prompt)
.\venv\Scripts\activate.bat
```

---

## 5. Install Dependencies

Install the specific TensorFlow version and project requirements:

```bash
pip install --upgrade pip
pip install tensorflow==2.15.0
# Or install all requirements
pip install -r requirements.txt
```

---

## 6. Select VSCode Interpreter

Configure VSCode to use the virtual environment interpreter:

1. Open your project folder in VSCode.
2. Open the Command Palette (`Ctrl + Shift + P`).
3. Type and select: **`Python: Select Interpreter`**.
4. Choose the interpreter associated with the virtual environment you just created (`.\venv\Scripts\python.exe`).

---

## 7. Troubleshooting & Out Of Memory (OOM) Fixes

If you experience Out Of Memory (OOM) errors during training (common on cards with lower VRAM like the RTX 2050):
* Open [setup/train_config.py](file:///c:/Users/neera/OneDrive/Documents/projects/Assignment Project/setup/train_config.py).
* Reduce the `BATCH_SIZE` from `128` to `64` or `32`.
