Layout Translator - portable version
====================================

1. Unzip this folder anywhere (a normal folder, not "Program Files" if you want to add languages).
2. Run "Layout Translator.exe". Nothing is installed.

Keep the folders models, fonts and pdfium next to the .exe. To add an OCR language,
create models\<name>\ with det.onnx, rec.onnx and dict.txt next to the .exe and restart.

Requirements
- Windows 10 or 11 with the Microsoft Edge WebView2 Runtime. Windows 11 and updated
  Windows 10 have it. If the window stays empty or the app does not start, install the
  "Evergreen Standalone Installer" from https://developer.microsoft.com/microsoft-edge/webview2/
- A translation server (llama.cpp llama-server) that you start yourself; see the project README.

What it still writes to your user profile
- %LOCALAPPDATA%\com.example.layouttranslator : WebView2 data and settings (for example the
  endpoint address) and temporary page images (removed when the app closes)
- %APPDATA%\com.example.layouttranslator\last-error.log : only after a crash

This build is not signed. Windows SmartScreen may say "Windows protected your PC":
choose "More info", then "Run anyway".
