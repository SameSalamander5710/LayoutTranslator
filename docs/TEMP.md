#TODO: These are text snippets moved away from README.md, but is relevant to be included in the architecture details.

## Start the server from the app
Next to **Test connection** is the **Launch server** button. It runs a command that you set, for example the `llama-server` command from step 4 above.
1. Click the pencil icon next to **Launch server**.
2. Paste your command and click **Save**.
3. Click **Launch server**. A terminal window opens and runs the command.
4. Click **Test connection** when the model has loaded.

Close the terminal window to stop the server.

The command is a plain text file named `server-command.txt`. Any text editor can change it, and the **Show file** button opens its folder. The file is next to the program when that folder is writable (the Windows installer and the portable zip). Otherwise it is in the per-user app data folder. On macOS it is always there, because the app folder is inside the signed app.

The terminal depends on the system:
- Windows: `cmd.exe`. The command runs as a batch file, so write a percent sign as `%%`.
- macOS: Terminal.
- Linux: the program in the `TERMINAL` environment variable, or the first of `x-terminal-emulator`, `gnome-terminal`, `konsole`, `xfce4-terminal`, `mate-terminal`, `kitty`, `alacritty`, `lxterminal` and `xterm` that is installed.

The command runs with your user rights, exactly as you wrote it. Do not use a command that you do not trust.



## If the app closes by itself
The release build has no console window. A crash writes its message and a backtrace to `last-error.log` in the app data folder (on Windows `%APPDATA%\com.example.layouttranslator\last-error.log`, on macOS `~/Library/Application Support/com.example.layouttranslator/last-error.log`, on Linux `~/.local/share/com.example.layouttranslator/last-error.log`). Attach that file to a bug report. It is only written when the app crashes. Look through it before sharing it: a panic message can quote text the app was working on.
