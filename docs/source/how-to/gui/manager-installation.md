# Manager installation and updates

Use these guides to install or update the standalone ImageTool Manager application.

(how-to-gui-install-manager)=

(imagetool-manager-standalone)=

## Installing the standalone Manager

Standalone bundles for Windows and macOS let you run the manager without managing a
Python environment.

Download the latest release from the [project’s releases page](https://github.com/kmnhan/erlabpy/releases), then follow the platform-specific steps below. For other platforms, or if you prefer full control, build from source via {ref}`build-from-source`.

### Windows

1. Download the latest Windows build `.zip` file from the [releases page](https://github.com/kmnhan/erlabpy/releases).

2. Extract it and double-click the included `.exe` installer, then follow the prompts.

### macOS

1. Download the latest `.zip` archive that matches your architecture from the [releases page](https://github.com/kmnhan/erlabpy/releases).

2. Extract it to obtain `ImageTool Manager.app`.

3. Move the app into `/Applications` (or any folder you prefer) and launch it like any other macOS application.

### Linux and source-built bundles

Official standalone release bundles are currently only provided for Windows and macOS.
Linux users can build from source (see {ref}`build-from-source`), and the resulting app
can be launched directly from the build folder.

(update-manager)=

## Updating an existing installation

1. Select {guilabel}`Check for Updates` from {guilabel}`Help` on Windows. On macOS,
   select it from {guilabel}`ImageTool Manager` next to {fab}`apple`.
2. Follow the prompts to install the update.
3. Close all Manager windows after the installation finishes.
4. Restart ImageTool Manager.
5. Open {guilabel}`About` and confirm that it shows the expected ERLabPy version.

If the update fails, close ImageTool Manager. Download the current bundle from the
[project releases page](https://github.com/kmnhan/erlabpy/releases). Install it with the
platform steps in {ref}`imagetool-manager-standalone`. Restart the application and check
the version again.
