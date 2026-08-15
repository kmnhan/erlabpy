# Settings

The interactive tools in {mod}`erlab.interactive` provide several options to customize
their behavior and appearance, mainly controlled in the Settings window. It can be
opened from the menu bar:

- macOS:

  {menuselection}`Preferences…` in the application menu next to {fab}`apple`

- Windows/Linux:

  {menuselection}`File --> Settings…`

Changes are saved immediately. The bottom of the window reports the save status, and
{guilabel}`Revert Changes` restores all user and workspace changes made since the
Settings window was opened. Closing the window does not discard changes that have
already been saved.

Use the sidebar to switch between setting groups:

- {guilabel}`Visualization` controls default colormap, gamma, cursor colors, and
  related display defaults.
- {guilabel}`I/O` controls the default loader and the default folder used by
  manager file dialogs and new Data Explorer windows.
- {guilabel}`ktool` controls defaults for newly opened momentum-conversion tools.
- {guilabel}`Figure Composer` controls default Matplotlib stylesheets, figure DPI, and
  export settings.
- {guilabel}`Security` controls which workspace locations can run stored executable
  content without review.

Rows in the {guilabel}`User` scope include a reset action that restores the application
default for that setting. Broadly resetting all user settings still asks for
confirmation.

When Settings is opened from ImageTool Manager, a {guilabel}`Workspace` scope is also
available. Workspace settings are local overrides saved inside the manager's `.itws`
workspace file. Turn on the checkbox to store a value.

(options-trusted-workspace-folders)=

## Trusting a workspace folder

A trusted workspace folder is an explicit choice to let every `.itws` file below that
folder execute code stored in the file. Choose a dedicated analysis folder that only
you and trusted collaborators can modify. Do not trust a download folder, a temporary
folder, or a shared folder that untrusted users or processes can write to.

To add a folder:

1. Open the {guilabel}`Security` page in the {guilabel}`User` scope.
2. Select {guilabel}`Add Folder…` beside {guilabel}`Trusted workspace folders`.
3. Select the analysis folder that you control.
