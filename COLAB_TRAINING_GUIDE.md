# Google Colab TPU v5e-1 Training Guide

## BEFORE YOU OPEN COLAB

1. If you keep the default `Qwen/Qwen3-4B`, no extra model approval is required
2. Copy your HuggingFace token from https://huggingface.co/settings/tokens
3. Upload the `physics-chatbot-finetune` folder to your Google Drive root

Default model note:
- The repo now defaults to `Qwen/Qwen3-4B`, so step 1 is only needed if you manually switch back to Gemma or to gated Llama checkpoints.

---

## COLAB SETUP

1. Open https://colab.research.google.com
2. Click **Runtime > Change runtime type > TPU** (pick v5e-1)
3. Click Save
4. Paste and run these cells one by one:

---

## CELL 1 — Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## CELL 2 — Go to project folder

```python
import os
os.chdir('/content/drive/MyDrive/physics-chatbot-finetune')
!ls
```

## CELL 3 — Install packages

```python
!pip install --upgrade pip -q
!pip install -r requirements_tpu.txt -q 2>&1 | tail -3
```

## CELL 4 — Check TPU works

```python
import os
os.environ['PJRT_DEVICE'] = 'TPU'

import torch
import torch_xla.core.xla_model as xm

device = xm.xla_device()
print(f"TPU OK: {device} / {xm.xla_device_hw(device)}")
```

## CELL 5 — Login to HuggingFace

```python
from huggingface_hub import login
login()
```

Paste your token when asked.

## CELL 6 — Download datasets

```python
!python scripts/01_download_datasets.py --config config.yaml
```

## CELL 7 — Prepare training data

```python
!python scripts/02_prepare_dataset.py --config config.yaml
```

## CELL 8 — START TRAINING

```python
import os
os.environ['PJRT_DEVICE'] = 'TPU'
!python scripts/03_train_tpu.py --config config.yaml
```

Training takes 30 min to 2 hours depending on dataset size.
If Colab disconnects, re-run from Cell 1 — it auto-resumes.

## CELL 9 — Save to Drive

```python
import shutil, os
dst = '/content/drive/MyDrive/physics-chatbot-exports'
os.makedirs(dst, exist_ok=True)
shutil.copytree('outputs/adapters/final', f'{dst}/adapter_final', dirs_exist_ok=True)
print(f"SAVED to {dst}/adapter_final")
```

## CELL 10 — Test it

```python
import sys; sys.path.insert(0, '.')
from src.inference import load_chat_model, generate_chat_response
from src.train_utils import load_config, get_system_prompt

config = load_config('config.yaml')
model, tokenizer, runtime = load_chat_model(
    base_model_name=config['base_model'],
    adapter_path='outputs/adapters/final',
    trust_remote_code=True,
)
result = generate_chat_response(
    model=model, tokenizer=tokenizer,
    model_name=config['base_model'],
    messages=[{'role': 'user', 'content': "Explain Newton's second law."}],
    temperature=0.7, max_new_tokens=256,
    system_prompt=get_system_prompt(config),
)
print(result['text'])
```

---

## AFTER TRAINING — On your local machine

Chat with the model:
```bash
python scripts/05_chat_cli.py --config config.yaml --adapter outputs/adapters/final
```

Serve as API:
```bash
python scripts/06_serve_openai_api.py --config config.yaml --adapter outputs/adapters/final
```

Merge adapter (optional):
```bash
python scripts/04_merge_adapter.py --config config.yaml --adapter outputs/adapters/final --output-dir outputs/merged_model
```

Run evaluation:
```bash
python scripts/07_eval_physics.py --config config.yaml --adapter outputs/adapters/final
```
