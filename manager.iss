#define MyAppName "ImageTool Manager"
#define MyAppPublisher "kmnhan"
#define MyAppURL "https://github.com/kmnhan/erlabpy"
#define MyAppExeName "ImageTool Manager.exe"
#define MyAppAssocName "ImageTool Workspace File"
#define MyAppOutputDir ".\\dist"
#define MyAppDistDir (MyAppOutputDir + "\\ImageTool Manager")
#define MyAppVersion GetVersionNumbersString(MyAppDistDir + "\\" + MyAppExeName)
#define MyAppAssocPrefix "dev.kmnhan.erlabpy"

[Setup]
AppId={{9E766090-E7A4-4D16-ACB4-C670DF6B2148}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
;AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
Compression=lzma2/max
DefaultDirName={autopf}\{#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}
; "ArchitecturesAllowed=x64compatible" specifies that Setup cannot run
; on anything but x64 and Windows 11 on Arm.
ArchitecturesAllowed=x64compatible
; "ArchitecturesInstallIn64BitMode=x64compatible" requests that the
; install be done in "64-bit mode" on x64 or Windows 11 on Arm,
; meaning it should use the native 64-bit Program Files directory and
; the 64-bit view of the registry.
ArchitecturesInstallIn64BitMode=x64compatible
ChangesAssociations=yes
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
PrivilegesRequired=lowest
OutputDir={#MyAppOutputDir}
OutputBaseFilename=Setup
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#MyAppDistDir}\\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Registry]
; ImageTool workspace (.itws)
; Make our ProgID the default handler and also populate OpenWith
Root: HKA; Subkey: "Software\Classes\.itws"; ValueType: string; ValueName: ""; ValueData: "{#MyAppAssocPrefix}.itws"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\.itws\OpenWithProgids"; ValueType: string; ValueName: "{#MyAppAssocPrefix}.itws"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocPrefix}.itws"; ValueType: string; ValueName: ""; ValueData: "ImageTool Workspace File"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocPrefix}.itws\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName},0"
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocPrefix}.itws\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""

; HDF5 (.h5)
Root: HKA; Subkey: "Software\Classes\.h5\OpenWithProgids"; ValueType: string; ValueName: "{#MyAppAssocPrefix}.h5"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocPrefix}.h5"; ValueType: string; ValueName: ""; ValueData: "HDF5 Data"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocPrefix}.h5\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName},0"
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocPrefix}.h5\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""

; NetCDF (.nc)
Root: HKA; Subkey: "Software\Classes\.nc\OpenWithProgids"; ValueType: string; ValueName: "{#MyAppAssocPrefix}.nc"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocPrefix}.nc"; ValueType: string; ValueName: ""; ValueData: "NetCDF Data"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocPrefix}.nc\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName},0"
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocPrefix}.nc\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""

; NeXus (.nxs)
Root: HKA; Subkey: "Software\Classes\.nxs\OpenWithProgids"; ValueType: string; ValueName: "{#MyAppAssocPrefix}.nxs"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocPrefix}.nxs"; ValueType: string; ValueName: ""; ValueData: "NeXus Data"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocPrefix}.nxs\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName},0"
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocPrefix}.nxs\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""

; FITS (.fits)
Root: HKA; Subkey: "Software\Classes\.fits\OpenWithProgids"; ValueType: string; ValueName: "{#MyAppAssocPrefix}.fits"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocPrefix}.fits"; ValueType: string; ValueName: ""; ValueData: "FITS Data"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocPrefix}.fits\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName},0"
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocPrefix}.fits\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""

; Igor Packed Stationery (.pxt/.PXT)
Root: HKA; Subkey: "Software\Classes\.pxt\OpenWithProgids"; ValueType: string; ValueName: "{#MyAppAssocPrefix}.pxt"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\.PXT\OpenWithProgids"; ValueType: string; ValueName: "{#MyAppAssocPrefix}.pxt"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocPrefix}.pxt"; ValueType: string; ValueName: ""; ValueData: "Igor Pro Packed Stationery"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocPrefix}.pxt\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName},0"
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocPrefix}.pxt\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""

; Igor Binary Wave (.ibw/.bwav)
Root: HKA; Subkey: "Software\Classes\.ibw\OpenWithProgids"; ValueType: string; ValueName: "{#MyAppAssocPrefix}.ibw"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\.bwav\OpenWithProgids"; ValueType: string; ValueName: "{#MyAppAssocPrefix}.ibw"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocPrefix}.ibw"; ValueType: string; ValueName: ""; ValueData: "Igor Pro Binary Wave"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocPrefix}.ibw\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName},0"
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocPrefix}.ibw\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""

; Igor Text Data (.itx/.ITX/.awav/.AWAV)
Root: HKA; Subkey: "Software\Classes\.itx\OpenWithProgids"; ValueType: string; ValueName: "{#MyAppAssocPrefix}.itx"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\.ITX\OpenWithProgids"; ValueType: string; ValueName: "{#MyAppAssocPrefix}.itx"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\.awav\OpenWithProgids"; ValueType: string; ValueName: "{#MyAppAssocPrefix}.itx"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\.AWAV\OpenWithProgids"; ValueType: string; ValueName: "{#MyAppAssocPrefix}.itx"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocPrefix}.itx"; ValueType: string; ValueName: ""; ValueData: "Igor Pro Text Data"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocPrefix}.itx\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName},0"
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocPrefix}.itx\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""

; ZIP archives
Root: HKA; Subkey: "Software\Classes\.zip\OpenWithProgids"; ValueType: string; ValueName: "{#MyAppAssocPrefix}.zip"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocPrefix}.zip"; ValueType: string; ValueName: ""; ValueData: "ZIP Archive"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocPrefix}.zip\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName},0"
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocPrefix}.zip\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
