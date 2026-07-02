# Configuration

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
- {guilabel}`Figure Composer` controls default Matplotlib stylesheets and optional
  default DPI for newly created figures.

```{versionadded} 3.25.0
The I/O settings include a default folder for ImageTool Manager file dialogs and
new Data Explorer windows.
```

Rows in the {guilabel}`User` scope include a reset action that restores the application
default for that setting. Broadly resetting all user settings still asks for
confirmation.

When Settings is opened from ImageTool Manager, a {guilabel}`Workspace` scope is also
available. Workspace settings are sparse overrides saved inside the manager's `.itws`
workspace file, so they travel with that workspace. Turn on {guilabel}`Override` for a
row to store a workspace value; turn it off, or use the row action, to inherit the user
setting again. Clearing all workspace overrides asks for confirmation. Workspace
overrides affect newly opened tools and new Figure Composer defaults, but do not mutate
tools that are already open.
