(build-from-source)=

# Build the ImageTool Manager application

If you want to build the standalone application from source due to platform compatibility or other reasons, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/kmnhan/erlabpy.git
   cd erlabpy
   ```

2. Install dependencies (requires `uv`):

   ```bash
   uv sync --all-extras --group pyinstaller --group pyqt6
   ```

3. Build the application:

   ```bash
   uv run pyinstaller manager.spec
   ```

4. The resulting app will be in `dist/ImageTool Manager`.

5. *(Optional, Windows only)* Install [Inno Setup](https://jrsoftware.org/isinfo.php) and add to your system PATH. Then run `iscc manager.iss` to create an installer file.
