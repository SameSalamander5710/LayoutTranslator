# Layout Translator [work in progress]

A fully offline tool for in-line text translation in PDFs or image files, using an OpenAI-compatible LLM endpoint. Outputs the translated file, similar to capabilities offered by DeepL or Google Translate.

## Quick Start

1. Download the appropriate installer or portable version of the app from the Releases/Nightly builds.
2. Install or extract the zip file, and run `Layout Translator.exe`
3. Launch an OpenAI-compatible LLM server endpoint. See *here* for an example.
4. Ensure that the server is reachable with the `Test connection` button.
5. Drop in your image file/PDF.
6. (Optional) Adjust the text detection boxes if needed.
7. `Run all` to initiate the translation.
8. `Save file`

## Languages
The English model ships in `models/en` next to the app. To add one, create `models/<name>/` there with `det.onnx`, `rec.onnx`, `dict.txt`.

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
    - You can manually download a model, or directly use `-hf` to download from HuggingFace.
    - 
    - The following command does a one-time download of a Gemma4 E4B model from HuggingFace and launches the server. This particular command is well suited for GPUs with 8 GB VRAM or above.

WIth a GPU with 8 GB VRAM or more:

```bash
llama-server -hf unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL -ngl 99 -c 65536 -ctk q8_0 -ctv q8_0 --spec-type draft-mtp,ngram-mod --spec-draft-n-max 4 -fit on --flash-attn on --n-cpu-moe 0 --parallel 4 --temp 1.0 --top-p 0.95 --top-k 64 --load-mode none
```
If no GPU is available, then run on the CPU with:

```bash
llama-server -hf unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL -ngl 0 -c 65536 -ctk q8_0 -ctv q8_0 --spec-type draft-mtp,ngram-mod --spec-draft-n-max 2 -fit on --flash-attn on --parallel 1 --temp 1.0 --top-p 0.95 --top-k 64 --load-mode none
```


