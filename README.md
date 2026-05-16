# physics-gpt-from-scratch

`physics-gpt-from-scratch` is a beginner-friendly project that trains a **small decoder-only GPT-style language model from scratch** on physics-oriented text and QA data, then serves it as a chatbot in Google Colab or locally with Gradio.

This repository does **not** fine-tune a pretrained model. It:

- trains a tokenizer from scratch
- builds a small GPT architecture from scratch in PyTorch
- trains the model with a custom next-token prediction loop
- generates answers and runs a simple chatbot UI

## Important realism

Training a large ChatGPT-like model from scratch on Google Colab is **not realistic**. Colab does not have the compute, memory, time budget, or dataset scale for that.

This project is intentionally scoped to a **small educational model**, roughly in the **10M to 100M parameter range depending on config**, so you can understand the full pipeline end to end:

`download data -> build corpus -> train tokenizer -> tokenize dataset -> train GPT -> generate text -> chat`

## What this project does

- Downloads public physics and science datasets where available
- Supports fallback local text files under `data/raw/local/`
- Builds a plain text corpus using physics tutor conversation tags
- Trains a BPE tokenizer from scratch
- Trains a decoder-only Transformer language model from scratch in PyTorch
- Saves checkpoints for resume and inference
- Generates text and runs a terminal or Gradio chatbot

## Difference between pretraining and fine-tuning

Pretraining from scratch:

- starts from random weights
- requires a tokenizer
- requires a training corpus
- learns language modeling from zero

Fine-tuning:

- starts from a pretrained model
- adapts it to a narrower task
- usually needs much less compute

This project is a **small-scale pretraining-from-scratch educational project**, not a fine-tuning project.

## Project structure

```text
physics-gpt-from-scratch/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE_NOTES.md
│
├── configs/
│   ├── tiny_gpt.yaml
│   ├── small_gpt.yaml
│   └── train_config.yaml
│
├── notebooks/
│   └── colab_train_from_scratch.ipynb
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── tokenizer/
│   └── sample/
│
├── checkpoints/
│
├── src/
│   ├── __init__.py
│   ├── download_datasets.py
│   ├── prepare_text_corpus.py
│   ├── train_tokenizer.py
│   ├── tokenize_dataset.py
│   ├── model.py
│   ├── train.py
│   ├── generate.py
│   ├── chat.py
│   ├── evaluate.py
│   └── utils.py
│
├── app/
│   └── gradio_chatbot.py
│
└── scripts/
    ├── setup_colab.sh
    ├── download_data.sh
    ├── prepare_data.sh
    ├── train_tokenizer.sh
    ├── train_model.sh
    └── run_chatbot.sh
```

## Requirements

Install:

```bash
pip install -r requirements.txt
```

## Exact commands

```bash
pip install -r requirements.txt
```

```bash
python src/download_datasets.py
```

```bash
python src/prepare_text_corpus.py
```

```bash
python src/train_tokenizer.py --config configs/tiny_gpt.yaml
```

```bash
python src/tokenize_dataset.py --config configs/tiny_gpt.yaml
```

```bash
python src/train.py --model_config configs/tiny_gpt.yaml --train_config configs/train_config.yaml
```

```bash
python src/generate.py --checkpoint checkpoints/final_model.pt --prompt "Explain Newton's second law."
```

```bash
python app/gradio_chatbot.py
```

## Google Colab workflow

The notebook [notebooks/colab_train_from_scratch.ipynb](C:/Users/Admin/Desktop/SaraLLM/physics-gpt-from-scratch/notebooks/colab_train_from_scratch.ipynb) walks through:

1. checking GPU or TPU status
2. installing requirements
3. mounting Google Drive
4. downloading datasets
5. preparing the training corpus
6. training the tokenizer from scratch
7. tokenizing the dataset
8. training the tiny GPT model
9. resuming training if needed
10. generating a sample answer
11. launching the Gradio chatbot
12. saving checkpoints to Drive

Recommended Colab setting:

- `Runtime > Change runtime type > GPU`

This project is built for GPU. TPU is not the intended path here.

## Dataset instructions

The downloader uses a registry in [src/download_datasets.py](C:/Users/Admin/Desktop/SaraLLM/physics-gpt-from-scratch/src/download_datasets.py) and tries public sources such as:

- `camel-ai/physics`
- `allenai/sciq`
- `allenai/ai2_arc` physics-filtered samples
- `tasksource/mmlu` physics-related subsets
- `OpenAssistant/oasst1`
- `HuggingFaceH4/ultrachat_200k`
- `databricks/databricks-dolly-15k`
- `Open-Orca/OpenOrca`
- local `.txt`, `.md`, `.pdf`, `.csv`, `.tsv`, `.json`, and `.jsonl` files from `data/raw/local/`

Not every internet dataset is enabled by default. This project keeps the registry **curated and Colab-friendly**:

- big public datasets are sampled with caps so tokenization and training stay manageable on Colab GPU
- optional textbook sources stay disabled until you opt in
- public but license-ambiguous sources are skipped unless you explicitly include them

The downloader:

- continues when one dataset fails
- logs failures instead of crashing the full run
- creates a tiny fallback sample dataset so the rest of the pipeline can still run
- skips sources that still need manual license review unless you pass `--include-review-sources`

Useful downloader commands:

```bash
python src/download_datasets.py --list-sources
```

```bash
python src/download_datasets.py --max-samples-per-dataset 3000
```

```bash
python src/download_datasets.py --only-group physics --only-group conversation
```

```bash
python src/download_datasets.py --include-review-sources
```

The default global sample cap is set in [configs/train_config.yaml](C:/Users/Admin/Desktop/SaraLLM/physics-gpt-from-scratch/configs/train_config.yaml):

- `data.max_samples_per_dataset: 5000`

That gives you a much larger corpus than the original smoke-test setup without making the Colab pipeline explode in size.

## Corpus format

The corpus builder writes training text in this style:

```text
<|system|>
You are PhysicsGPT, a helpful physics tutor. Explain physics clearly and solve problems step by step.
<|user|>
QUESTION_TEXT
<|assistant|>
ANSWER_TEXT
<|endoftext|>
```

Saved files:

- `data/processed/corpus.txt`
- `data/processed/train.txt`
- `data/processed/val.txt`
- `data/processed/test.txt`

## Tokenizer training

Tokenizer special tokens:

- `<|pad|>`
- `<|unk|>`
- `<|bos|>`
- `<|eos|>`
- `<|system|>`
- `<|user|>`
- `<|assistant|>`
- `<|endoftext|>`

Default vocab sizes:

- tiny model: `16000`
- small model: `32000`

Tokenizer files are saved to:

- `data/tokenizer/`

## Model sizes

### Tiny model

Use [configs/tiny_gpt.yaml](C:/Users/Admin/Desktop/SaraLLM/physics-gpt-from-scratch/configs/tiny_gpt.yaml)

Good for:

- smoke tests
- quick Colab debugging
- making sure the full pipeline works

### Small model

Use [configs/small_gpt.yaml](C:/Users/Admin/Desktop/SaraLLM/physics-gpt-from-scratch/configs/small_gpt.yaml)

Good for:

- better generations than the tiny model
- slower but still educational Colab runs

## Training configuration

The optimizer and training loop settings live in [configs/train_config.yaml](C:/Users/Admin/Desktop/SaraLLM/physics-gpt-from-scratch/configs/train_config.yaml).

Default values:

- batch size `16`
- gradient accumulation `4`
- max steps `5000`
- learning rate `3e-4`
- min learning rate `3e-5`
- warmup steps `200`
- eval interval `250`
- save interval `500`
- max grad norm `1.0`

The training loop supports:

- cross entropy next-token loss
- AdamW
- warmup plus cosine decay
- mixed precision with `torch.cuda.amp` on CUDA
- checkpoint saving
- checkpoint resume
- validation loss and perplexity reporting

## How to train

Tiny config:

```bash
python src/train_tokenizer.py --config configs/tiny_gpt.yaml
python src/tokenize_dataset.py --config configs/tiny_gpt.yaml
python src/train.py --model_config configs/tiny_gpt.yaml --train_config configs/train_config.yaml
```

Small config:

```bash
python src/train_tokenizer.py --config configs/small_gpt.yaml
python src/tokenize_dataset.py --config configs/small_gpt.yaml
python src/train.py --model_config configs/small_gpt.yaml --train_config configs/train_config.yaml
```

## How to chat with the model

Terminal chat:

```bash
python src/chat.py --checkpoint checkpoints/final_model.pt
```

One-shot generation:

```bash
python src/generate.py --checkpoint checkpoints/final_model.pt --prompt "What is Gauss's law?"
```

Gradio UI:

```bash
python app/gradio_chatbot.py
```

## How to improve quality

- train longer
- use the small config instead of the tiny config
- add more open physics data
- add better local files under `data/raw/local/`
- improve deduplication and cleanup
- increase block size if GPU memory allows
- reduce repetition with better sampling settings

## Where to add your own data

Drop your own files into [data/raw/local](C:/Users/Admin/Desktop/SaraLLM/physics-gpt-from-scratch/data/raw/local).

Supported local formats:

- `.txt`
- `.md`
- `.pdf`
- `.csv`
- `.tsv`
- `.json`
- `.jsonl`

For CSV and TSV files, the loader looks for common columns such as:

- question-like: `question`, `prompt`, `instruction`, `query`
- answer-like: `answer`, `response`, `output`, `completion`
- text-like: `text`, `content`, `body`
- optional topic-like: `topic`, `subject`, `category`

## Common errors and fixes

### CUDA out of memory

- lower `batch_size`
- lower `block_size`
- use `configs/tiny_gpt.yaml`
- reduce `gradient_accumulation_steps`
- close other GPU-heavy processes in Colab

### Colab disconnected

- save checkpoints often
- keep checkpoints in Google Drive
- resume with `--resume checkpoints/checkpoint_step_x.pt`

### tokenizer file not found

- run `python src/train_tokenizer.py --config configs/tiny_gpt.yaml` first
- verify files exist under `data/tokenizer/`

### bad or empty generations

- make sure the model trained for enough steps
- lower temperature
- increase prompt clarity
- check that tokenizer and checkpoint belong to the same run

### loss not decreasing

- verify the corpus is not empty
- inspect `data/processed/train.txt`
- lower learning rate slightly
- try the tiny config first to confirm the code path works

### model repeats itself

- reduce temperature
- use top-k sampling
- train longer
- make sure `<|endoftext|>` is present in the corpus

### model gives wrong physics answers

- expected at small scale
- add higher-quality physics text
- train longer
- use the model as an educational toy, not an authoritative solver

## License and data safety

See [LICENSE_NOTES.md](C:/Users/Admin/Desktop/SaraLLM/physics-gpt-from-scratch/LICENSE_NOTES.md).

Do not scrape copyrighted textbooks illegally. Only use public datasets and open or self-provided local text that you have the right to train on.
