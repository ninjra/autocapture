#ifndef MyAppName
#define MyAppName "Autocapture"
#endif

#ifndef MyAppVersion
#define MyAppVersion "0.0.0"
#endif

#ifndef MyAppPublisher
#define MyAppPublisher "Ninjra"
#endif

#ifndef MyAppExeName
#define MyAppExeName "autocapture.exe"
#endif

[Setup]
AppId={{C65E8F5F-6B3C-4F65-9E6C-8C28A2A5E9F1}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={pf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=dist-installer
OutputBaseFilename=Autocapture-{#MyAppVersion}-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64
UninstallDisplayIcon={app}\{#MyAppExeName}

[Tasks]
Name: "startup"; Description: "Run Autocapture at startup"; Flags: unchecked

[Files]
Source: "dist\autocapture\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Autocapture"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall Autocapture"; Filename: "{uninstallexe}"

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "Autocapture"; ValueData: "\"{app}\{#MyAppExeName}\""; Tasks: startup

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch Autocapture"; Flags: nowait postinstall skipifsilent
