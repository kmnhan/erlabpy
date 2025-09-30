import os
import pathlib
import sys

import pyinstaller_versionfile
from PyInstaller.utils.hooks import collect_all

import erlab

build_number = 0  # Increment this when making changes to the spec file

if sys.platform == "win32":
    pyinstaller_versionfile.create_versionfile(
        output_file="versionfile.txt",
        version=f"{erlab.__version__}.{build_number}",
        company_name="kmnhan",
        file_description="ImageTool Manager",
        internal_name="ImageTool Manager",
        legal_copyright="© Kimoon Han. All rights reserved.",
        original_filename="ImageTool Manager.exe",
        product_name="ImageTool Manager",
        translations=[1033, 1200],  # English - Unicode
    )

manager_dir = pathlib.Path(erlab.interactive.imagetool.manager.__file__).parent

datas = []
binaries = []
hiddenimports = ["PyQt6", "dask", "distributed"]

for module_name in (
    "erlab",
    "debugpy",  # Needed by qtconsole
    "numbagg",  # This requires special handling
    "cmasher",  # Colormap library
    "cmocean",  # Colormap library
    "colorcet",  # Colormap library
    "cmcrameri",  # Colormap library
    "seaborn",  # Colormap library
    "xarray",  # Full xarray for repr, etc.
):
    tmp_ret = collect_all(module_name)
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]

if sys.platform == "darwin":
    icon_path = str(manager_dir / "icon.icns")
    datas += [("./resources/Assets.car", ".")]  # Liquid glass icon for macOS 26+
else:
    icon_path = str(manager_dir / "icon.png")

a = Analysis(
    [str(manager_dir / "__main__.py")],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PySide6", "PySide2", "PyQt5", "numbagg.test", "xarray.tests"],
    noarchive=False,
    optimize=0,
    module_collection_mode={"erlab": "pyz+py"},
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ImageTool Manager",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=sys.platform == "win32",
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[icon_path],
    version="versionfile.txt" if sys.platform == "win32" else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ImageTool Manager",
)
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="ImageTool Manager.app",
        icon=icon_path,
        bundle_identifier="dev.kmnhan.erlabpy.imagetoolmanager",
        version=str(erlab.__version__),
        info_plist={
            "CFBundleIconName": "icon",  # macOS 26+ liquid glass icon
            "CFBundleVersion": f"{erlab.__version__}.{build_number}",
            "CFBundleDocumentTypes": [
                {  # HDF5 (.h5)
                    "CFBundleTypeName": "HDF5 Data",
                    "LSItemContentTypes": ["dev.kmnhan.erlabpy.hdf5"],
                    "CFBundleTypeRole": "Viewer",  # or "Editor"
                    "LSHandlerRank": "Default",
                },
                {  # NetCDF (.nc)
                    "CFBundleTypeName": "NetCDF Data",
                    "LSItemContentTypes": ["dev.kmnhan.erlabpy.netcdf"],
                    "CFBundleTypeRole": "Viewer",
                    "LSHandlerRank": "Default",
                },
                {  # NeXus (.nxs) — HDF5-based container
                    "CFBundleTypeName": "NeXus Data",
                    "LSItemContentTypes": ["dev.kmnhan.erlabpy.nexus"],
                    "CFBundleTypeRole": "Viewer",
                    "LSHandlerRank": "Alternate",
                },
                {  # FITS (.fits)
                    "CFBundleTypeName": "FITS Data",
                    "LSItemContentTypes": ["dev.kmnhan.erlabpy.fits"],
                    "CFBundleTypeRole": "Viewer",
                    "LSHandlerRank": "Alternate",
                },
                {  # Igor Packed Experiment Template (.pxt)
                    "CFBundleTypeName": "Igor Pro Packed Stationery",
                    "LSItemContentTypes": [
                        "com.wavemetrics.igorpromach.packed-stationery"
                    ],
                    "CFBundleTypeRole": "Viewer",
                    "LSHandlerRank": "Alternate",
                },
                {  # Igor Binary Wave (.ibw)
                    "CFBundleTypeName": "Igor Pro Binary Wave",
                    "LSItemContentTypes": ["com.wavemetrics.igorpromach.binary-wave"],
                    "CFBundleTypeRole": "Viewer",
                    "LSHandlerRank": "Alternate",
                },
                {  # Igor Text Data (.itx)
                    "CFBundleTypeName": "Igor Pro Text Data",
                    "LSItemContentTypes": ["com.wavemetrics.igorpromach.text-data"],
                    "CFBundleTypeRole": "Viewer",
                    "LSHandlerRank": "Alternate",
                },
                {  # Scienta DA30 deflector scans (.zip)
                    "CFBundleTypeName": "ZIP archive",
                    "LSItemContentTypes": [
                        "public.zip-archive",
                        "com.pkware.zip-archive",
                    ],
                    "CFBundleTypeRole": "Viewer",
                    "LSHandlerRank": "Alternate",
                },
            ],
            # Import UTIs for Igor Pro formats
            "UTImportedTypeDeclarations": [
                {  # HDF5
                    "UTTypeIdentifier": "dev.kmnhan.erlabpy.hdf5",
                    "UTTypeDescription": "HDF5 Data File",
                    "UTTypeConformsTo": ["public.data"],
                    "UTTypeTagSpecification": {
                        "public.filename-extension": ["h5"],
                        "public.mime-type": ["application/x-hdf5", "application/x-hdf"],
                    },
                },
                {  # NetCDF
                    "UTTypeIdentifier": "dev.kmnhan.erlabpy.netcdf",
                    "UTTypeDescription": "NetCDF Data File",
                    "UTTypeConformsTo": ["public.data"],
                    "UTTypeTagSpecification": {
                        "public.filename-extension": ["nc"],
                        "public.mime-type": [
                            "application/netcdf",
                            "application/x-netcdf",
                        ],
                    },
                },
                {  # NeXus (.nxs)
                    "UTTypeIdentifier": "dev.kmnhan.erlabpy.nexus",
                    "UTTypeDescription": "NeXus File",
                    "UTTypeConformsTo": ["public.data", "dev.kmnhan.erlabpy.hdf5"],
                    "UTTypeTagSpecification": {
                        "public.filename-extension": ["nxs"],
                        "public.mime-type": ["application/x-hdf5", "application/x-hdf"],
                    },
                },
                {  # FITS
                    "UTTypeIdentifier": "dev.kmnhan.erlabpy.fits",
                    "UTTypeDescription": "Flexible Image Transport System (FITS)",
                    "UTTypeConformsTo": ["public.data"],
                    "UTTypeTagSpecification": {
                        "public.filename-extension": ["fits"],
                        # RFC 4047 standard MIME type
                        "public.mime-type": ["application/fits", "image/fits"],
                    },
                },
                {  # Igor Packed Experiment Template (.pxt)
                    "UTTypeIdentifier": "com.wavemetrics.igorpromach.packed-stationery",
                    "UTTypeDescription": "Igor Pro Packed Stationery",
                    "UTTypeConformsTo": ["public.data"],
                    "UTTypeTagSpecification": {
                        "public.filename-extension": ["pxt", "PXT"],
                        "com.apple.ostype": ["IGsS", "sGsU", "sGsS"],
                    },
                },
                {  # Igor Binary Wave (.ibw)
                    "UTTypeIdentifier": "com.wavemetrics.igorpromach.binary-wave",
                    "UTTypeDescription": "Igor Pro Binary Wave",
                    "UTTypeConformsTo": ["public.data"],
                    "UTTypeTagSpecification": {
                        "public.filename-extension": ["ibw", "bwav"],
                        "com.apple.ostype": ["IGBW"],
                    },
                },
                {  # Igor Text Data (.itx)
                    "UTTypeIdentifier": "com.wavemetrics.igorpromach.text-data",
                    "UTTypeDescription": "Igor Pro Text Data",
                    "UTTypeConformsTo": ["public.plain-text"],
                    "UTTypeTagSpecification": {
                        "public.filename-extension": ["itx", "ITX", "awav", "AWAV"],
                        "com.apple.ostype": ["IGTX"],
                    },
                },
            ],
        },
    )

if sys.platform == "win32":
    # Cleanup versionfile.txt after the build
    os.remove("versionfile.txt")
