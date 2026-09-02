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

## Security

The {guilabel}`Security` page is available in the {guilabel}`User` scope.

(options-trusted-workspace-folders)=

### Trusted workspace folders

Every `.itws` file in a trusted folder and its subfolders can run stored executable
content without per-file review. Use {guilabel}`Add Folder…` to add a folder and
{guilabel}`Remove` to remove the selected folder.

Use a dedicated analysis folder that only you or trusted collaborators can modify. Do
not add a downloads folder, temporary folder, or shared folder that untrusted users or
programs can modify.

### Saved workspace approvals

{guilabel}`Reset Saved Trust…` removes approvals that ERLab stored for previously saved
workspaces. It does not remove entries from {guilabel}`Trusted workspace folders`.

See {ref}`imagetool-manager-code-trust` for the workspace status values and
{ref}`how-to-gui-review-workspace-code` for per-workspace review.
