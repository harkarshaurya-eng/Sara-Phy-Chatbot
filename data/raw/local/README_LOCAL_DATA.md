# Local Data Folder

Put your own local files here when you want to extend the corpus without editing Python code.

Supported formats:

- `.txt`
- `.md`
- `.pdf`
- `.csv`
- `.tsv`
- `.json`
- `.jsonl`

Recommended tabular column names:

- question-like: `question`, `prompt`, `instruction`, `query`
- answer-like: `answer`, `response`, `output`, `completion`
- text-like: `text`, `content`, `body`
- optional topic-like: `topic`, `subject`, `category`

Example CSV:

```csv
question,answer,topic
Explain Newton's second law.,Force equals mass times acceleration.,mechanics
What is entropy?,Entropy measures how spread out energy is.,thermodynamics
```

Everything in this folder is treated as user-provided data, so you are responsible for making sure you have the right to use it.
