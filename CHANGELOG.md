# Changelog

What a user can see, newest first. Add a line here with every user-visible change (see `AGENTS.md`).
Dates are not recorded yet: nothing has been released.

## Unreleased (0.1.0)

### Detect and edit
- Detection starts by itself when a file is opened, a page is shown for the first time or the method changes. **Run all** keeps the boxes you edited.
- Boxes: add, delete, move and resize; undo and redo of the last 10 changes per page; a button that resets the detection of a page. A box that covers other text reads that text.
- Detected boxes are shrunk from the top and the bottom to the rows of their text, and a box that is almost inside another one is merged into it. Boxes you draw stay as you drew them.
- The text colour is found from the page colour right behind each letter, so white letters on a red stripe are white, not red.

### Look of the result
- The original text is erased with a classical fill (no model): thin strokes are filled from their surroundings, busy backgrounds become a soft blur instead of a flat block. Boxes that overlap are erased together, so no seam shows.
- The translation is typeset in the largest size that fits the box (Noto Sans, with fonts for Chinese, Japanese and Korean). Compare view with a divider.

### Input and output
- Open an image or a PDF by dragging it in, with **Browse**, or paste an image from the clipboard (**Ctrl+V**, **Cmd+V** or the paste button). PDFs are read at 200 DPI; selectable text is used where it exists. The result can be saved as PNG or PDF.
- Languages are folders in `models/`. The English model is included.

### Translation
- Any local OpenAI-compatible server (for example llama.cpp `llama-server`). **Test connection** checks it. **Launch server** runs a command you set, in a terminal window. A **Stop** button ends a long run.

### Safety and builds
- Strict content security policy for the app window (checked by `npm run check:csp`). A crash writes `last-error.log` in the app data folder.
- Windows (x64) and macOS (Apple Silicon): installer and portable version, built by the `release` workflow. The builds are not signed.
