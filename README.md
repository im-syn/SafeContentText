
# SafeContentText

A Python toolkit and web API for detecting “bad” content (profanity, hate speech, gore, etc.) in text using a zero‑shot AI classifier.

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage](#usage)

   * [CLI: `bad_text_detector.py`](#cli-bad_text_detectorpy)
   * [Web API: `SafeTextContentApi.py`](#web-api-safetextcontentapipy)
   * [Test Script: `test_api.py`](#test-scripttest_apipy)
6. [Example API Responses](#example-api-responses)
7. [Project Structure & Line‑by‑Line File Walkthrough](#project-structure--line-by-line-file-walkthrough)

   * [`SafeTextContentApi.py`](#safetextcontentapipy)
   * [`bad_text_detector.py`](#bad_text_detectorpy)
   * [`test_api.py`](#test_apipy)
8. [Contributing](#contributing)
9. [License](#license)

---

## Features

* **Zero‑shot classification** with Hugging Face’s `facebook/bart-large-mnli`.
* **CLI tool** (`bad_text_detector.py`) to scan single strings, files, or directories.
* **FastAPI web service** (`SafeTextContentApi.py`) with GET/POST `/detect` and POST `/detect/file`.
* **Customizable** labels, thresholds, cache directories, CORS, and “is\_safe” flag.
* **Test script** (`test_api.py`) with colorful console output.

---

## Prerequisites

* Python 3.8+
* `pip`

---

## Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/im-syn/SafeContentText.git
   cd SafeContentText
   ```
2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```
3. **Install dependencies**

   ```bash
   pip install fastapi uvicorn transformers torch requests colorama
   ```

---

## Configuration

The following environment variables let you override defaults:

| Variable         | Default                   | Description                                    |
| ---------------- | ------------------------- | ---------------------------------------------- |
| `STC_API_HOST`   | `127.0.0.1`               | Host for the FastAPI server                    |
| `STC_API_PORT`   | `8989`                    | Port for the FastAPI server                    |
| `STC_API_RELOAD` | `True`                    | Whether `uvicorn` runs with `--reload`         |
| `HF_CACHE_DIR`   | `<project‑root>/hf_cache` | Where HuggingFace models are downloaded/cached |

To change, e.g.:

```bash
export HF_CACHE_DIR=/data/models/zero_shot
export STC_API_HOST=0.0.0.0
export STC_API_PORT=8080
```

---

## Usage

### CLI: `bad_text_detector.py`

```bash
# Single text
python bad_text_detector.py --text "I hate you"

# File
python bad_text_detector.py --file comments.txt

# Directory of .txt files
python bad_text_detector.py --dir ./logs/

# Custom labels & threshold
python bad_text_detector.py -t "Example" -l "profanity,insult" -T 0.6

# Save JSON output
python bad_text_detector.py -t "Test" -o results.json

# Verbose logging
python bad_text_detector.py -t "Test" -v
```

### Web API: `SafeTextContentApi.py`

1. **Start the server**

   ```bash
   python SafeTextContentApi.py
   # or
   uvicorn SafeTextContentApi:app --host localhost --port 8989 --reload
   ```
2. **Endpoints**

   * **GET** `/detect`

     * Query params:

       * `texts=first&texts=second`
       * *or* `text=single` (if `ENABLE_TEXT_PARAM=True`)
       * `labels=insult&labels=profanity`
       * `threshold=0.5`
   * **POST** `/detect`

     * JSON body:

       ```json
       {
         "texts": ["one", "two"],
         "labels": ["hate speech","insult"],
         "threshold": 0.6
       }
       ```
   * **POST** `/detect/file`

     * Multipart form:

       * `file`: text/plain (.txt)
       * `labels`: as form fields
       * `threshold`: as form field

### Test Script: `test_api.py`

Run:

```bash
python test_api.py
```

It will exercise all three endpoints and print colored results.

---

## Example API Responses

**GET** `http://localhost:8989/detect?text=sex`

```json
{
  "results": [
    {
      "text": "sex",
      "scores": {
        "sexual content": 0.999351501464844,
        "profanity": 0.937217891216278,
        "graphic violence": 0.639289915561676,
        "insult": 0.146385952830315,
        "hate speech": 0.00694961939007044,
        "self-harm": 0.00394586473703384,
        "terrorism": 0.000741451571229845
      },
      "flagged_labels": {
        "sexual content": 0.999351501464844,
        "profanity": 0.937217891216278,
        "graphic violence": 0.639289915561676
      },
      "is_safe": false
    }
  ]
}
```

**GET** `http://localhost:8989/detect?text=i%20love%20cats`

```json
{
  "results": [
    {
      "text": "i love cats",
      "scores": {
        "sexual content": 0.00827411003410816,
        "profanity": 0.00117546890396625,
        "graphic violence": 0.00117279822006822,
        "insult": 0.000300033862004056,
        "self-harm": 0.000142173827043734,
        "hate speech": 0.000105988452560268,
        "terrorism": 0.000031683866836829
      },
      "flagged_labels": {},
      "is_safe": true
    }
  ]
}
```

---

## Project Structure & Line‑by‑Line File Walkthrough

### `SafeTextContentApi.py`

| Line    | Code                                                                  | Explanation                                                                    |
| ------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| 1       | `#!/usr/bin/env python3`                                              | Shebang: run with system’s Python 3 interpreter.                               |
| 2–9     | `""" … """`                                                           | Module docstring describing purpose & endpoints.                               |
| 11–14   | `import …`                                                            | Import standard libraries and FastAPI, pydantic, transformers.                 |
| 17–23   | Configurable options (`ALLOW_GET`, `HOST`, `CACHE_DIR`, …)            | Toggle GET/POST/file endpoints, CORS, host/port, reload, cache directory.      |
| 26–34   | `DEFAULT_LABELS = [...]`                                              | Default categories to detect.                                                  |
| 37–43   | `logging.basicConfig(...)`                                            | Configure global logging format & level.                                       |
| 46      | `os.makedirs(CACHE_DIR, exist_ok=True)`                               | Ensure cache directory exists.                                                 |
| 49–54   | `classifier = pipeline(...)`                                          | Load zero‑shot model into `CACHE_DIR` once at startup.                         |
| 57–68   | `class DetectRequest(BaseModel): ...`                                 | Pydantic model for JSON POST requests.                                         |
| 70–78   | `class DetectResult(BaseModel): ...`                                  | Pydantic model for each classification result (includes `is_safe` if enabled). |
| 80–83   | `class DetectResponse(BaseModel): ...`                                | Pydantic model wrapping `List[DetectResult]`.                                  |
| 86–93   | `app = FastAPI(...)`                                                  | Create FastAPI instance with metadata.                                         |
| 96–103  | `if ENABLE_CORS: app.add_middleware(CORSMiddleware,…)`                | Conditionally enable CORS.                                                     |
| 106–124 | `def classify_texts(...):`                                            | Helper to run zero‑shot, compute `flagged_labels`, and `is_safe`.              |
| 127–136 | Error handlers for validation, HTTPException, general exceptions.     | Return consistent JSON error responses.                                        |
| 139–147 | `@app.post("/detect")`                                                | POST `/detect` endpoint, reads JSON body, calls `classify_texts`.              |
| 149–166 | `@app.get("/detect")`                                                 | GET `/detect`, supports `texts` list or single `text` param.                   |
| 168–180 | `@app.post("/detect/file")`                                           | File‐upload endpoint, reads `.txt`, splits lines into separate texts.          |
| 183–190 | Uvicorn runner block (`if __name__ == "__main__": uvicorn.run(...)`). | Allows `python SafeTextContentApi.py` to start the server.                     |

---

### `bad_text_detector.py`

| Line   | Code                                                                   | Explanation                                     |
| ------ | ---------------------------------------------------------------------- | ----------------------------------------------- |
| 1      | `#!/usr/bin/env python3`                                               | Shebang for CLI usage.                          |
| 2–9    | Module docstring describing CLI functionality.                         |                                                 |
| 11–18  | `import …`                                                             | Standard libs + `transformers.pipeline`.        |
| 21–28  | `DEFAULT_LABELS`                                                       | Categories to detect by default.                |
| 30–38  | `configure_logging(...)`                                               | Sets up timestamped, leveled console logs.      |
| 40–51  | `load_texts_from_dir(directory)`                                       | Recursively collects `.txt` files’ contents.    |
| 53–62  | `detect_bad_content(classifier, texts, labels)`                        | Runs zero‑shot and normalizes output to a list. |
| 64–106 | `main()` function:                                                     |                                                 |
| 66–75  | ‑ Parse CLI arguments (`--text`, `--file`, `--dir`, `--labels`, etc.). |                                                 |
| 77–91  | ‑ Build `inputs` dict from string, file, or directory.                 |                                                 |
| 93     | ‑ Load zero‑shot model: `pipeline("zero-shot-classification", …)`.     |                                                 |
| 95     | ‑ Call `detect_bad_content(...)`.                                      |                                                 |
| 97–104 | ‑ Print flagged vs safe texts, optionally write `--output` JSON file.  |                                                 |
| 108    | `if __name__ == "__main__": main()`                                    | CLI entry point.                                |

---

### `test_api.py`

| Line  | Code                                                                                        | Explanation           |
| ----- | ------------------------------------------------------------------------------------------- | --------------------- |
| 1     | `#!/usr/bin/env python3`                                                                    | Shebang for CLI.      |
| 2–9   | Module docstring explaining it tests all endpoints.                                         |                       |
| 11–13 | `import …`: `requests`, `colorama` for colored output.                                      |                       |
| 16    | `BASE_URL = "http://localhost:8989"`                                                        | Base address for API. |
| 18–22 | `pretty_print(title, color)`: prints a colored separator & title.                           |                       |
| 24–41 | `test_get()`: calls GET `/detect?`, prints JSON or error in color.                          |                       |
| 43–60 | `test_post_json()`: POST `/detect` with JSON, prints result.                                |                       |
| 62–83 | `test_post_file()`: POST `/detect/file` with a temp `.txt` file, prints result & cleans up. |                       |
| 85–89 | `if __name__ == "__main__":` runs all three tests.                                          |                       |

---

## Contributing

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/XYZ`)
3. Commit your changes (`git commit -m "Add XYZ"`)
4. Push (`git push origin feature/XYZ`)
5. Open a Pull Request

---

## License

MIT © SYN

---
## ☕ Like It?

If this helped you, consider giving the repo a 🌟 or forking it to your toolkit.
Thank you for using **SafeContentText**! Feel free to open issues or PRs for improvements.
