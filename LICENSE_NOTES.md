# License Notes

This repository downloads and processes public physics or science datasets where possible, but **you are responsible for verifying the license of every dataset before training, distributing, or commercializing a model**.

## Important reminders

- Dataset licenses can change over time
- Hugging Face metadata is helpful, but you should still read the upstream dataset card
- Local files under `data/raw/local/` are entirely your responsibility
- Do not scrape copyrighted textbooks illegally

## Safe rule

Only use:

- public datasets with clear reuse terms
- open educational resources
- your own text files
- locally added content that you are legally allowed to process

## What this project does

- stores raw downloaded text or normalized records under `data/raw/`
- creates a sample fallback dataset so the pipeline stays runnable
- keeps the downloader resilient when one dataset is missing
- skips registry entries that require manual license review unless you explicitly opt in
- lets you add your own `.txt`, `.md`, `.pdf`, `.csv`, `.tsv`, `.json`, or `.jsonl` files under `data/raw/local/`

## What this project does not do

- it does not provide legal advice
- it does not guarantee that redistribution is permitted
- it does not validate the license of user-added local files
