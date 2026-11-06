# Layout Translator [work in progress]

A fully offline tool for in-line text translation in PDFs or image files, using an OpenAI-compatible LLM endpoint. Outputs the translated file, similar to capabilities offered by DeepL or Google Translate.

## Quick Start

1. Download the appropriate installer or portable version of the app from the Releases/Nightly builds.
2. Install or extract the zip file, and run `Layout Translator.exe`
3. Launch an OpenAI-compatible LLM server endpoint. See [here](#example-server-setup) for an example.
4. Ensure that the server is reachable with the `Test connection` button.
5. Drop in your image file/PDF.
6. (Optional) Adjust the text detection boxes if needed.
7. `Run all` to initiate the translation.
8. `Save file`

## Features

- Adjustable text recognition boxes to demarcate areas to translate, or fix text detection issues.
- Choice between direct text selection, OCR, or hybrid text detection.
- Inpaints the area behind the translated text to make it look less distracting.
- Possibilty of editing individual translated texts prior to saving the file.
- Option to add additional context or glossary before translation, to tune the tone or setting of the translated text.
- Direct comparison of original and translated document with a slider.

## Example server setup

If you are new to running LLMs locally, here are simplified steps on Windows 11 to give you an idea:
1. Install [llama.cpp](https://github.com/ggml-org/llama.cpp): [(Installation instructions)](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md)
    - On Windows 11, you can install it via the terminal with
    ```bash 
    winget install llama.cpp
    ``` 
2. Launch OpenAI-compatible API server with `llama-server`
    - You can manually download a model, or directly use `-hf` to download from Hugging Face.
    - The following command does a one-time download of a Gemma4 E4B model from Hugging Face and launches the server. This particular command is well suited for GPUs with 8 GB VRAM or above.

WIth a GPU with 8 GB VRAM or more:

```bash
llama-server -hf unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL -ngl 99 -c 65536 -ctk q8_0 -ctv q8_0 --spec-type draft-mtp,ngram-mod --spec-draft-n-max 4 -fit on --flash-attn on --n-cpu-moe 0 --parallel 4 --temp 1.0 --top-p 0.95 --top-k 64 --load-mode none
```
If no GPU is available, then run on the CPU with:

```bash
llama-server -hf unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL -ngl 0 -c 65536 -ctk q8_0 -ctv q8_0 --spec-type draft-mtp,ngram-mod --spec-draft-n-max 2 -fit on --flash-attn on --parallel 1 --temp 1.0 --top-p 0.95 --top-k 64 --load-mode none
```

## Languages

Text detection and recognition use [PaddleOCR PP-OCRv5](https://github.com/PaddlePaddle/PaddleOCR) models in ONNX format. Download the ready-to-use files from the [oar-ocr v0.3.0 release](https://github.com/GreatV/oar-ocr/releases/tag/v0.3.0). The English model ships in `models/en`.

Each language is one folder with exactly three files:

| File | Download from that release | Purpose |
|---|---|---|
| `det.onnx` | `pp-ocrv5_mobile_det.onnx` | Finds the text boxes. The same file works for every language. |
| `rec.onnx` | `<name>_pp-ocrv5_mobile_rec.onnx` | Reads the text. This file is language-specific. |
| `dict.txt` | `ppocrv5_<name>_dict.txt` | Character list that matches `rec.onnx`. |

Recognisers in the release: `latin` (French, German, Spanish, Italian, Portuguese, ...), `en`, `cyrillic`, `eslav`, `el` (Greek), `korean`, `arabic`, `devanagari`, `ta`, `te`, `th`. For Chinese, use `pp-ocrv5_mobile_rec.onnx` with `ppocrv5_dict.txt`.

Example for French: create a folder `fr` and put these three files in it:

models/fr/
  det.onnx    <- pp-ocrv5_mobile_det.onnx
  rec.onnx    <- latin_pp-ocrv5_mobile_rec.onnx
  dict.txt    <- ppocrv5_latin_dict.txt

Rename the files to exactly `det.onnx`, `rec.onnx` and `dict.txt`; the language is the folder name.

To install them:

1. Click **Open models folder** in the app (section **1. Input**).
2. Create the language folder there, for example `fr`.
3. Put the three files in it.
4. Restart the app. The language then appears in the **Text language for OCR** list.

For the portable build the folder is next to `Layout Translator.exe`. For the installed app it is the app's resource folder; if that folder is read-only, use `%APPDATA%\com.example.layouttranslator\models` instead. A folder is only listed when all three files exist and are not empty.

Developers can fetch the bundled English model, fonts and PDFium with `npm run models`.
