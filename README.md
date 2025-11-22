# Blood-group from fingerprint — README

This repository contains code and model artifacts for a project that attempts to predict a person's blood group from fingerprint images using a fine-tuned Qwen2-VL model (adapter-style). It includes a small FastAPI backend for serving predictions, example datasets, saved model artifacts, and checkpoints.

## Contents

- `backend/` — FastAPI server, database helpers, auth, and tests.
- `dataset_blood_group/` — example dataset structure (class subfolders for A+/-, B+/-, AB+/-, O+/-, etc.).
- `blood_group_qwen2vl_model/` and `saved_model_qwen2vl/` — saved checkpoints and final model artifacts.
- `blood_group_vlm_finetune.ipynb` — notebook with training/finetuning experiments.

## Prerequisites

- Python 3.10+ recommended.
- A machine with a GPU and CUDA for training/inference using float16 (optional but strongly recommended for speed and memory).
- Git installed (optional, for repo management).

Recommended Python packages (some are in `backend/requirements.txt`):

- fastapi, uvicorn, sqlmodel, pillow, python-multipart, requests, pytest, httpx, qwen-vl-utils
- transformers, accelerate, peft, safetensors, torch (GPU build if available)

## Setup (Windows PowerShell)

1. Create a virtual environment and install dependencies:

```powershell
# from repository root
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install backend requirements
pip install -r .\backend\requirements.txt

# install training / hf dependencies (adjust versions as needed)
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118  # or cpu wheel if you don't have CUDA
pip install transformers accelerate peft safetensors
```

2. (Optional) If you will use the notebook, install jupyter and open it:

```powershell
pip install notebook jupyterlab
jupyter lab blood_group_vlm_finetune.ipynb
```

## How to run the FastAPI backend (serve predictions)

The backend is in `backend/`. It provides endpoints for registration, token login, `/predict` for image uploads, and admin endpoints to inspect feedback.

Run the API server (PowerShell):

```powershell
# activate venv if not active
.\.venv\Scripts\Activate.ps1

# run uvicorn from repository root; backend.main is the FastAPI app
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Default endpoints:

- POST /register — register a user (body: username, password, role)
- POST /token — login (form fields `username` and `password`), returns a simple access token (username used as token in this simple setup)
- POST /predict — accepts multipart file upload (`file`) and a bearer token; returns `{ "blood_group": "A+", "confidence": 0.95 }` (mock if model not present)
- POST /feedback — upload image + form fields `predicted_label` and `correct_label`

Notes:

- The server attempts to load a model from `../saved_model_qwen2vl` relative to `backend/` — update `model_path` in `backend/main.py` if you move the model.
- The simple auth uses the username as the access token for convenience; do not use this in production.

## How to run tests

From repository root (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r .\backend\requirements.txt
pytest -q
```

## How to train (overview)

There are two recommended ways to reproduce or continue training:

1. Use the provided notebook `blood_group_vlm_finetune.ipynb` — this includes dataset-loading, preprocessing, and training steps adapted to Qwen2-VL and adapter-style fine-tuning. It is the most direct, reproducible method, and it shows hyperparameters and trainer loops used during experiments.

2. Use a script-based training flow with `transformers` + `accelerate` + `peft`.

Below are general steps and an example command you can adapt if you have a `train.py` training script. Because training code and scripts vary by project, the repository includes the notebook as the canonical recipe.

### Dataset layout

The repo contains `dataset_blood_group/` with subfolders for each label, e.g.:

- dataset_blood_group/A+/
- dataset_blood_group/A-/
- dataset_blood_group/B+/
- ...

This is a standard image-classification layout. The notebook shows how images are loaded and converted into the input format expected by Qwen2-VL (images + simple text prompts). If you write a script, make sure to mirror the dataset transforms in the notebook.

### Example training using a `train.py` script (placeholder)

Create or adapt a training script that:

- Loads a base Qwen2-VL model and processor (from `transformers` / project-specific `Qwen2VLForConditionalGeneration` and `AutoProcessor`).
- Applies adapter/PEFT-style weights (the repo uses adapter checkpoints and safetensors).
- Builds a Dataset that yields messages of the same structure used by the processor (image + question prompt).
- Uses `Trainer` or a custom loop to fine-tune the model.

Example `accelerate` command (edit paths and script flags to match your script):

```powershell
.\.venv\Scripts\Activate.ps1
accelerate launch train.py `
	--dataset_dir .\dataset_blood_group `
	--output_dir .\blood_group_qwen2vl_model `
	--per_device_train_batch_size 8 `
	--gradient_accumulation_steps 2 `
	--max_steps 2000 `
	--learning_rate 5e-5 `
	--checkpointing_steps 600
```

Notes on checkpoints and formats:

- Checkpoints in this repository include `checkpoint-600/`, `checkpoint-1200/`, `checkpoint-1800/` under `blood_group_qwen2vl_model/`. They contain files such as `adapter_model.safetensors`, `optimizer.pt`, and `trainer_state.json`.
- The final or deployable model is in `saved_model_qwen2vl/` and includes tokenizer files and `adapter_model.safetensors`.

### Training tips

- Use a GPU-enabled environment. Training Qwen2-VL without GPUs will be very slow.
- Use mixed precision (float16) with a GPU to save memory and speed up training; set torch dtype accordingly and use `accelerate` with `--mixed_precision fp16`.
- If you only have a small dataset, prefer adapter-style fine-tuning (small number of parameters) rather than full fine-tuning.
- Monitor validation performance and keep a separate held-out set.

## Where artifacts are stored

- `saved_model_qwen2vl/` — final model artifacts for loading with `Qwen2VLForConditionalGeneration.from_pretrained()` and `AutoProcessor.from_pretrained()`.
- `blood_group_qwen2vl_model/checkpoint-*` — intermediate checkpoints.
- `incorrect_predictions/` — images saved when feedback is submitted.

## Stopping tracking of large files already in git

If you have already committed large model files, adding `.gitignore` prevents future commits but doesn't remove existing tracked files. To stop tracking files that should be ignored (example: a previously committed `saved_model_qwen2vl/`), run:

```powershell
git rm -r --cached saved_model_qwen2vl
git commit -m "Remove saved model from tracking; add to .gitignore"
```

## Troubleshooting

- If the server prints "Warning: Could not load model", confirm `saved_model_qwen2vl/` exists and contains model files (`adapter_model.safetensors`, tokenizer files, etc.).
- If you hit CUDA / torch errors, make sure your torch version matches your CUDA drivers and GPU. Alternatively install a CPU-only wheel for quick testing.

## License & credits

Check individual files and notebooks for dataset and model license information. The repository reuses community model code (Qwen2-VL-related classes) and standard Python libraries.

---

If you'd like, I can:

- Add a small `train.py` example that mirrors what the notebook does (scripted training with `transformers` + `accelerate`).
- Run a quick `git status` so we can identify large tracked files to remove from git history.
