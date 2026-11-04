Layout Translator - portable version (Linux, x86_64)
====================================================

1. Unpack this folder anywhere:  tar -xzf LayoutTranslator_*_linux-x64_portable.tar.gz
2. Run  ./bin/layout-translator   Nothing is installed.

Keep the folders bin and lib as they are: the app looks for its models, fonts and pdfium
in "lib/Layout Translator" next to "bin". To add an OCR language, create
"lib/Layout Translator/models/<name>/" with det.onnx, rec.onnx and dict.txt and restart.

Requirements
- A desktop Linux with glibc 2.38 or newer, GTK 3 and WebKitGTK 4.1 (package libwebkit2gtk-4.1-0
  on Debian and Ubuntu, webkit2gtk4.1 on Fedora): Ubuntu 24.04+, Debian 13+ or Fedora 39+.
  The ONNX Runtime library in this build needs the newer glibc, so older distributions cannot run it.
- A translation server (llama.cpp llama-server): start it yourself, or with the Launch server button (below).
- Pasting an image (Ctrl+V or the paste button) works under X11 and XWayland.

The .AppImage from the same release is a single file that needs nothing unpacked: make it
executable (chmod +x) and run it. It needs libfuse2; without it run it with
--appimage-extract-and-run.

Launch server (optional)
The "Launch server" button next to "Test connection" runs a command that you set with the
pencil icon, for example your llama-server command, in a terminal window. The command is
saved in "lib/Layout Translator/server-command.txt" (any text editor can change it). The
terminal is the program in the TERMINAL environment variable, or the first of
x-terminal-emulator, gnome-terminal, konsole, xfce4-terminal, mate-terminal, kitty,
alacritty, lxterminal and xterm that is installed.

What it still writes to your user profile
- ~/.local/share/com.example.layouttranslator : WebKit data and settings (for example the
  endpoint address) and last-error.log (only after a crash)
- ~/.cache/com.example.layouttranslator : temporary page images (removed when the app closes)

This build is not signed.
