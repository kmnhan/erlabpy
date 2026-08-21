(how-to-gui-manager-extensions)=

# Using Manager extensions

Use these procedures after you obtain or write a valid extension script. Review an
extension as executable Python code before you approve it.

For extension types, signature rules, and workspace states, see the
{ref}`Manager extension reference <imagetool-manager-extensions>`.

(how-to-gui-register-extension)=

## Registering a script

1. Start ImageTool Manager.
2. Select {menuselection}`Extensions --> Add Script…`.
3. Select the `.py` file.
4. Review the complete source.
5. Select {guilabel}`OK` to approve and register it.

Each registered script must have a unique file name. Manager compares file names
without case differences. For example, you cannot register both `gaussian_tools.py`
and `GAUSSIAN_TOOLS.py`.

## Running an analysis routine

1. Select one ImageTool in Manager.
2. Select the routine from the {menuselection}`Extensions` menu or the selected row's
   {menuselection}`Extensions` submenu.
3. Enter the routine parameters.
4. Select {guilabel}`OK`.

Manager opens the result in a new ImageTool and records the extension operation in its
provenance.

## Loading a file

1. Open a Manager file dialog or Data Explorer.
2. Select the file filter supplied by the extension loader.
3. Select the file and enter any loader parameters.
4. Open the file.

Use the normal {ref}`Manager file-loading procedure <how-to-gui-manager-open>` for
drag-and-drop, batch loading, and Data Explorer workflows.

## Approving a script update

After you edit a registered script, Manager stops running that changed source until you
approve it.

1. Select {menuselection}`Extensions --> Manage Extensions`.
2. Select the script with the {guilabel}`Approval required` state.
3. Select {guilabel}`Review Update…`.
4. Review the complete source and select {guilabel}`OK`.

If validation fails, select {guilabel}`Show Error Details`. Correct the source file,
then review the update again.

## Locating a moved script

If Manager reports that a registered script is missing:

1. Select the script in the {guilabel}`Extension Scripts Not Found` dialog.
2. Select {guilabel}`Locate Script…`.
3. Select the script at its new location.

The selected file must have the same file name and the same contents as the approved
script. To use changed contents, restore the approved file first, then follow the
script update procedure.

## Selecting workspace embedding

1. Select {menuselection}`Extensions --> Manage Extensions`.
2. Select the script.
3. Set {guilabel}`Workspace embedding` to one of these values:

   - {guilabel}`Embed when referenced` stores the script when a saved operation uses
     it.
   - {guilabel}`Always embed` stores the script even when no saved operation uses it.
   - {guilabel}`Never embed` omits the source from the workspace.

Use {guilabel}`Never embed` only when another recovery method preserves the exact
approved source.

(how-to-gui-recover-extension)=

## Recovering a script from a workspace

If a workspace contains an embedded copy of an unavailable extension:

1. Select {menuselection}`Extensions --> Workspace Requirements`.
2. Select the unavailable extension.
3. Select {guilabel}`Save and Register Script…`.
4. Review the embedded source.
5. Save it as a local `.py` file.

Manager registers the saved file and updates the workspace requirement. It never runs
the embedded source directly.

See {doc}`workspaces and provenance
<../../explanation/workspaces-and-provenance>` for the source-trust and replay model.
