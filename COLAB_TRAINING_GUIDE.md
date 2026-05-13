# Google Colab T4 Training Guide

## Before You Open Colab

1. Keep the default `Qwen/Qwen3-4B` unless you have a reason to switch
2. Copy your Hugging Face token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Clone or upload `physics-chatbot-finetune`
4. If you switch to gated models such as Gemma or Llama, accept their licenses on Hugging Face first

## Colab Setup

1. Open [colab.research.google.com](https://colab.research.google.com)
2. Click `Runtime > Change runtime type > GPU`
3. Save and wait for the session to reconnect
4. Run these cells one by one

## Cell 1 - Clone The Repo

```python
%cd /content
!rm -rf Sara-Phy-Chatbot
!git clone https://github.com/harkarshaurya-eng/Sara-Phy-Chatbot.git
%cd /content/Sara-Phy-Chatbot
!ls
```

## Cell 2 - Install Packages

```python
!python -m pip install --upgrade pip -q
!pip install -r requirements.txt
```

## Cell 3 - Verify GPU

```python
!nvidia-smi
```

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
```

If this does not show a CUDA GPU, stop and switch the runtime again before training.

## Cell 4 - Login To Hugging Face

```python
from huggingface_hub import login
login()
```

## Cell 5 - Check The Environment

```python
!python scripts/00_check_env.py --config config.yaml
```

## Cell 6 - Download Datasets

```python
!python scripts/01_download_datasets.py --config config.yaml
```

## Cell 7 - Prepare Training Data

```python
!python scripts/02_prepare_dataset.py --config config.yaml
```

## Cell 8 - Start Training

```python
!python scripts/03_train_tpu.py --config config.yaml --num-train-epochs 30
```

The script name is kept for compatibility, but the default config now targets Colab T4 with QLoRA.

## Cell 9 - Save Outputs To Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
import os
import shutil

dst = "/content/drive/MyDrive/physics-chatbot-exports"
os.makedirs(dst, exist_ok=True)
shutil.copytree("outputs/adapters/final", f"{dst}/adapter_final", dirs_exist_ok=True)
for name in ["train_metrics.json", "train_summary.md", "data_report.md"]:
    src = f"outputs/logs/{name}"
    if os.path.exists(src):
        shutil.copy2(src, f"{dst}/{name}")
print(f"Saved artifacts to {dst}")
```

## Cell 10 - Quick Smoke Test

```python
import sys
sys.path.insert(0, ".")

from src.inference import load_chat_model, generate_chat_response
from src.train_utils import load_config, get_system_prompt

config = load_config("config.yaml")
model, tokenizer, runtime = load_chat_model(
    base_model_name=config["base_model"],
    adapter_path="outputs/adapters/final",
    trust_remote_code=True,
    load_in_4bit=bool(config.get("load_in_4bit", False)),
)
result = generate_chat_response(
    model=model,
    tokenizer=tokenizer,
    model_name=config["base_model"],
    messages=[{"role": "user", "content": "Explain Newton's second law."}],
    temperature=0.7,
    max_new_tokens=256,
    system_prompt=get_system_prompt(config),
)
print(result["text"])
```

## Common Fixes

- If `CUDA available` is `False`, you are not on a GPU runtime yet
- If `bitsandbytes` fails to import, reinstall `requirements.txt` and restart the runtime
- If training is too slow, lower `max_samples_per_dataset` or `num_train_epochs` for the first run
- If you run out of memory, reduce `max_seq_length` or keep the default `Qwen/Qwen3-4B`
