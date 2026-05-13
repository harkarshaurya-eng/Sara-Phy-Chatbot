# physics-chatbot-finetune

`physics-chatbot-finetune` is a complete starter repository for building a physics-focused chat model from open datasets, fine-tuning it with QLoRA on a Google Colab T4 GPU, and serving it through a local OpenAI-compatible API for tools like ComfyUI.

The default base model is `Qwen/Qwen3-4B`, and you can switch to `Qwen/Qwen3.5-4B`, `google/gemma-3-4b-it`, or `meta-llama/Llama-3.2-3B-Instruct` by editing `config.yaml`.

## What This Project Does

This repository helps you:

1. Download openly licensed physics datasets, conversational SFT datasets, and related science datasets with a license-aware manifest.
2. Clean, normalize, deduplicate, and convert records into instruction/chat JSONL.
3. Fine-tune a 4B to 5B model with QLoRA by default, or standard LoRA when CUDA quantization is unavailable.
4. Run on a Google Colab T4 GPU by default, with CPU fallback if CUDA is unavailable.
5. Save LoRA adapters and optionally merge them into a standalone model.
6. Chat locally through a terminal CLI.
7. Serve a local OpenAI-compatible `POST /v1/chat/completions` API for ComfyUI or other local clients.

## Why LoRA Instead Of Full Fine-Tuning

LoRA updates a small set of trainable adapter weights instead of rewriting the full base model. That matters here because:

- It is much cheaper in memory than full fine-tuning.
- It is more realistic on a Colab T4 and other consumer GPU setups.
- It is easier to export, version, and swap between different adapters.
- It reduces the chance of destroying the base model's general instruction-following behavior.

This repo does **not** attempt full fine-tuning unless you explicitly enable it in `config.yaml`.

## Hardware Guidance

Recommended:

- Google Colab T4 GPU for training with QLoRA
- A CUDA GPU for local inference if you want better response speed

Supported fallback:

- Other CUDA GPUs
- CPU only, for smoke tests and API validation

Important:

- `bitsandbytes` QLoRA is a CUDA-only path, which is why the default Colab target is now T4.
- If CUDA is unavailable, the training script automatically falls back to standard LoRA.
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

- Physics and science: `convaiinnovations/physics-reasoning-dataset`, `allenai/sciq`, `derek-thomas/ScienceQA`, `allenai/ai2_arc`, `IUTVanguard/PhysicsEval`, `UGPhysics/ugphysics`, `mhla/gpt1900-physics-clm`
- Conversational alignment: `OpenAssistant/oasst1`, `OpenAssistant/oasst2`, `HuggingFaceH4/ultrachat_200k`, `Open-Orca/SlimOrca-Dedup`, `databricks/databricks-dolly-15k`

Raw corpora are downloaded on demand into `data/raw/`; they are not committed into the repository because they are large and license-sensitive.

Some attractive physics corpora are still intentionally left out by default when they are image-only, missing a clear machine-readable license on Hugging Face, or would require a separate multimodal pipeline. This repo favors sources that can be safely downloaded, attributed, and converted into text/chat supervision.

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

- `Qwen/Qwen3-4B` is a text-only causal LM, which keeps SFT and QLoRA straightforward on a T4 GPU.
- Its official model card explicitly highlights multi-turn dialogue, instruction following, and chat use cases.
- `google/gemma-3-4b-it` is still supported, but its official Hugging Face packaging is `image-text-to-text`, which adds avoidable complexity for this repo's pure chatbot focus.

## Quick Start

### 1. Install Dependencies

Local CPU/GPU or Colab T4:

```bash
pip install -r requirements.txt
```

Legacy TPU only:

```bash
pip install -r requirements_tpu.txt
```

`requirements.txt` is the default path for this repo now because it includes the CUDA-side QLoRA dependency `bitsandbytes`. `requirements_tpu.txt` is kept only for legacy TPU/XLA experiments.

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

### 5. Train On A T4 GPU Or Fallback Hardware

```bash
python scripts/03_train_tpu.py --config config.yaml
```

The default config is tuned for a Colab T4:

- `training_mode: "qlora"`
- `load_in_4bit: true`
- `fp16: true`
- `bf16: false`

The default config is also set to `30` training epochs. You can override that from the command line too:

```bash
python scripts/03_train_tpu.py --config config.yaml --num-train-epochs 30
```

Note:

- `early_stopping_patience` is still enabled in `config.yaml`, so training may stop before epoch 30 if validation loss stops improving.
- With the full mixed dataset, `30` epochs on a T4 can take a long time. For a first smoke test, temporarily lower `max_samples_per_dataset` or `num_train_epochs`.

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

## Running On Google Colab T4 GPU

The included notebook is:

- `notebooks/colab_tpu_train.ipynb`

It contains cells for:

1. cloning the repo into Colab
2. installing GPU requirements
3. checking CUDA availability
4. logging into Hugging Face
5. downloading datasets
6. preparing the dataset
7. training the model
8. saving the final adapter to Drive
9. running a chat smoke test

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

### GPU not detected

- Re-run `python scripts/00_check_env.py --config config.yaml`
- In Colab, make sure the runtime hardware is set to `GPU`
- Run `nvidia-smi` and confirm the notebook reports a T4 or other CUDA device

### QLoRA falls back to standard LoRA

- This happens when CUDA is unavailable or the selected base model is not a text-only causal LM
- Switch to a CUDA runtime and keep the default `Qwen/Qwen3-4B`

### Gated model download fails

- Log in with `huggingface_hub.login()`
- Accept the model license on the Hugging Face model page
- Retry the script

### Dataset preparation produced zero rows

- Check `data/raw/dataset_manifest.json`
- Inspect `outputs/logs/data_report.md`
- Loosen `min_answer_chars` or enable more sources in `config.yaml`
- Add your own `data/custom/*.jsonl`

### `bitsandbytes` install or import fails

- Make sure the runtime is actually `GPU`, not CPU
- Re-run `pip install -r requirements.txt`
- Restart the runtime after changing CUDA-side packages if imports still fail

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
- Hugging Face bitsandbytes docs: [https://huggingface.co/docs/transformers/quantization/bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
