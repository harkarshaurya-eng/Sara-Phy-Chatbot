# physics-chatbot-finetune

`physics-chatbot-finetune` is a complete starter repository for building a physics-focused chat model from open datasets, fine-tuning it with LoRA on Google Colab TPU v5e-1, and serving it through a local OpenAI-compatible API for tools like ComfyUI.

The default base model is `Qwen/Qwen3-4B`, and you can switch to `Qwen/Qwen3.5-4B`, `google/gemma-3-4b-it`, or `meta-llama/Llama-3.2-3B-Instruct` by editing `config.yaml`.

## What This Project Does

This repository helps you:

1. Download openly licensed physics datasets, conversational SFT datasets, and related science datasets with a license-aware manifest.
2. Clean, normalize, deduplicate, and convert records into instruction/chat JSONL.
3. Fine-tune a 4B to 5B model with LoRA by default.
4. Run on Google Colab TPU v5e-1 through PyTorch/XLA, with GPU/CPU fallback if TPU is unavailable.
5. Save LoRA adapters and optionally merge them into a standalone model.
6. Chat locally through a terminal CLI.
7. Serve a local OpenAI-compatible `POST /v1/chat/completions` API for ComfyUI or other local clients.

## Why LoRA Instead Of Full Fine-Tuning

LoRA updates a small set of trainable adapter weights instead of rewriting the full base model. That matters here because:

- It is much cheaper in memory than full fine-tuning.
- It is more realistic on Colab TPU and consumer GPU setups.
- It is easier to export, version, and swap between different adapters.
- It reduces the chance of destroying the base model's general instruction-following behavior.

This repo does **not** attempt full fine-tuning unless you explicitly enable it in `config.yaml`.

## Hardware Guidance

Recommended:

- Google Colab TPU v5e-1 for training with standard LoRA
- A CUDA GPU for local inference if you want better response speed

Supported fallback:

- GPU without TPU
- CPU only, for smoke tests and API validation

Important:

- True `bitsandbytes` QLoRA is a CUDA-only path.
- On TPU, this repo automatically falls back to standard LoRA.
- Some base models are gated or require license acceptance on Hugging Face before download.

## Repository Layout

```text
physics-chatbot-finetune/
  README.md
  requirements.txt
  requirements_tpu.txt
  config.yaml
  notebooks/
    colab_tpu_train.ipynb
  scripts/
    00_check_env.py
    01_download_datasets.py
    02_prepare_dataset.py
    03_train_tpu.py
    04_merge_adapter.py
    05_chat_cli.py
    06_serve_openai_api.py
    07_eval_physics.py
  src/
    data_sources.py
    data_cleaning.py
    formatting.py
    train_utils.py
    inference.py
    rag_optional.py
    api_server.py
  data/
    raw/
    processed/
    final/
    custom/
  outputs/
    adapters/
    merged_model/
    logs/
  comfyui/
    physics_chatbot_workflow.json
    README_COMFYUI.md
```

## Supported Dataset Types

The pipeline supports:

- Physics QA datasets from Hugging Face
- Conversational SFT datasets from Hugging Face
- General science QA datasets filtered down to physics
- arXiv physics abstracts, but only when explicit reuse licenses are present
- OpenStax physics textbooks
- Public-domain physics books
- Your own files in `data/custom/*.jsonl`

The default config already wires in a larger mixed curriculum:

- Physics and science: `convaiinnovations/physics-reasoning-dataset`, `allenai/sciq`, `derek-thomas/ScienceQA`, `UGPhysics/ugphysics`, `mhla/gpt1900-physics-clm`
- Conversational alignment: `OpenAssistant/oasst1`, `HuggingFaceH4/ultrachat_200k`, `Open-Orca/SlimOrca-Dedup`, `databricks/databricks-dolly-15k`

Raw corpora are downloaded on demand into `data/raw/`; they are not committed into the repository because they are large and license-sensitive.

Every configured source has:

- a source name
- a URL or Hugging Face dataset ID
- a license field
- a citation field
- an enabled flag

The downloader writes a manifest to `data/raw/dataset_manifest.json` and will not silently pull unknown-license data.

## Base Model Options

Default:

- `Qwen/Qwen3-4B`

Alternatives:

- `Qwen/Qwen3.5-4B`
- `google/gemma-3-4b-it`
- `meta-llama/Llama-3.2-3B-Instruct`

Change models by editing `base_model:` in `config.yaml`.

Why this default:

- `Qwen/Qwen3-4B` is a text-only causal LM, which keeps SFT simpler on TPU than multimodal checkpoints.
- Its official model card explicitly highlights multi-turn dialogue, instruction following, and chat use cases.
- `google/gemma-3-4b-it` is still supported, but its official Hugging Face packaging is `image-text-to-text`, which adds avoidable complexity for this repo's pure chatbot focus.

## Quick Start

### 1. Install Dependencies

Local CPU/GPU:

```bash
pip install -r requirements.txt
```

Colab TPU:

```bash
pip install -r requirements_tpu.txt
```

The TPU requirements file is pinned to the current PyTorch/XLA 2.8 line for Colab Python 3.12 compatibility.

### 2. Check Your Environment

```bash
python scripts/00_check_env.py --config config.yaml
```

### 3. Download Datasets

```bash
python scripts/01_download_datasets.py --config config.yaml
```

To download only one source while testing:

```bash
python scripts/01_download_datasets.py --config config.yaml --only-source physics_reasoning_hf
```

To inspect configured sources first:

```bash
python scripts/01_download_datasets.py --config config.yaml --list-sources
```

### 4. Prepare The Final SFT Dataset

```bash
python scripts/02_prepare_dataset.py --config config.yaml
```

This creates:

- `data/final/physics_sft.jsonl`
- `data/final/train.jsonl`
- `data/final/validation.jsonl`
- `data/final/test.jsonl`
- `outputs/logs/data_report.md`

### 5. Train On TPU Or Fallback Hardware

```bash
python scripts/03_train_tpu.py --config config.yaml
```

The default config is now set to `30` training epochs. You can override that from the command line too:

```bash
python scripts/03_train_tpu.py --config config.yaml --num-train-epochs 30
```

Note:

- `early_stopping_patience` is still enabled in `config.yaml`, so training may stop before epoch 30 if validation loss stops improving.

This saves the final adapter to:

```text
outputs/adapters/final
```

### 6. Optional: Merge The Adapter

```bash
python scripts/04_merge_adapter.py --config config.yaml --adapter outputs/adapters/final --output-dir outputs/merged_model
```

### 7. Chat Locally

With base model plus adapter:

```bash
python scripts/05_chat_cli.py --config config.yaml --adapter outputs/adapters/final
```

With a merged model:

```bash
python scripts/05_chat_cli.py --config config.yaml --merged-model outputs/merged_model
```

### 8. Serve The Local API

```bash
python scripts/06_serve_openai_api.py --config config.yaml --adapter outputs/adapters/final
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Example OpenAI-style request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "physics-chatbot",
    "messages": [
      {"role": "user", "content": "Explain Newton'\''s second law."}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

## Running On Google Colab TPU v5e-1

The included notebook is:

- `notebooks/colab_tpu_train.ipynb`

It contains cells for:

1. mounting Google Drive
2. installing TPU requirements
3. checking TPU availability
4. logging into Hugging Face
5. downloading datasets
6. preparing the dataset
7. training the model
8. copying the final adapter to Drive
9. optionally merging the adapter
10. running a chat smoke test

If you switch to gated checkpoints such as Gemma or Llama, accept the license on Hugging Face before running the notebook. The default Qwen model does not require that extra approval step in the same way.

## Adding Custom Physics Data

Put your files into:

```text
data/custom/*.jsonl
```

Supported custom row patterns:

1. Final chat format

```json
{
  "messages": [
    {"role": "system", "content": "You are PhysicsGPT..."},
    {"role": "user", "content": "Explain Newton's second law."},
    {"role": "assistant", "content": "Newton's second law states that..."}
  ],
  "source": "my_notes",
  "topic": "mechanics",
  "difficulty": "beginner",
  "license": "user-owned"
}
```

2. Simple QA format

```json
{
  "question": "What is kinetic energy?",
  "answer": "Kinetic energy is the energy of motion. For mass m and speed v, KE = 1/2 mv^2.",
  "topic": "mechanics",
  "license": "user-owned"
}
```

3. Text corpus format

```json
{
  "title": "Electromagnetic Waves",
  "text": "Electromagnetic waves are oscillations of electric and magnetic fields...",
  "topic": "electromagnetism",
  "license": "user-owned"
}
```

## Evaluation

Run:

```bash
python scripts/07_eval_physics.py --config config.yaml --adapter outputs/adapters/final
```

This writes:

```text
outputs/logs/eval_report.md
```

The evaluation script does three simple things:

- scores held-out dataset responses with a lightweight token-overlap F1
- checks built-in probe prompts across physics domains
- computes RMSE and RAE on the numeric-answer subset of held-out examples
- saves a qualitative grading template for optional manual review

## Safety And Legal Notes

- This repo is strict about dataset licensing by design.
- arXiv entries are only kept when explicit allowed licenses are present.
- Public-domain and OpenStax text sources are supported, but you should still review the generated manifest before training.
- Do not place private, proprietary, classroom-restricted, or personally identifiable data into `data/custom/` unless you understand the consequences of training on it.
- No paid APIs are required.
- No private data is uploaded anywhere by default.
- Weights & Biases is not required.

## Troubleshooting

### TPU not detected

- Re-run `python scripts/00_check_env.py --config config.yaml`
- In Colab, make sure the runtime hardware is set to TPU
- Reinstall `requirements_tpu.txt`

### QLoRA fails on TPU

- Expected behavior. Use standard LoRA on TPU.
- If you want QLoRA, switch to a CUDA runtime and set `training_mode: "qlora"`

### Gated model download fails

- Log in with `huggingface_hub.login()`
- Accept the model license on the Hugging Face model page
- Retry the script

### Dataset preparation produced zero rows

- Check `data/raw/dataset_manifest.json`
- Inspect `outputs/logs/data_report.md`
- Loosen `min_answer_chars` or enable more sources in `config.yaml`
- Add your own `data/custom/*.jsonl`

### Colab TPU install fails on `torch_xla`

- Use the current `requirements_tpu.txt` from this repo
- If you uploaded an older copy, replace it before installing
- On Colab, restart the runtime after changing TPU packages

### Local API starts but generations are slow

- Use a CUDA GPU for local inference if possible
- Merge the adapter for slightly simpler deployment
- Lower `max_new_tokens`

## Reference Links

- Gemma 3 4B IT: [https://huggingface.co/google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)
- Qwen3 4B: [https://huggingface.co/Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
- Qwen3.5 4B: [https://huggingface.co/Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- Llama 3.2 3B Instruct: [https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- OpenStax licensing: [https://openstax.org/licensing](https://openstax.org/licensing)
- PyTorch/XLA TPU docs: [https://docs.pytorch.org/xla/master/](https://docs.pytorch.org/xla/master/)
