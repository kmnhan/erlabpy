## v2.11.2 (2024-10-14)

### Bug Fixes

- **io.dataloader:** fix `coordinate_attrs` not being propagated ([278675b](https://github.com/kmnhan/erlabpy/commit/278675b54d2e12471ce8629fbd6d249aa7184c0e))

## v2.11.1 (2024-10-14)

### Code Refactor

- add app icon for imagetool manager ([e1cbcd2](https://github.com/kmnhan/erlabpy/commit/e1cbcd29b045cb2d586baa3c6272fd60cfd05979))

## v2.11.0 (2024-10-13)

### Features

- **io.dataloader:** add new argument that can control combining ([bdec5ff](https://github.com/kmnhan/erlabpy/commit/bdec5ff24e02e82597cef10d225997599efaa257))

  Adds a new parameter `combine` to `io.load`. If `False`, returns a list of post-processed files without attempting to concatenate or merge the data into a single object. If `True`, retains the current default behavior.

### Bug Fixes

- **imagetool:** allow coords of any dtype coercible to float64 ([4342ebc](https://github.com/kmnhan/erlabpy/commit/4342ebc1bc4be01fcc9c7883ecfbaef0f5857e5d))
- **io.dataloader:** properly handle combining multi-axis scans ([2cd22c7](https://github.com/kmnhan/erlabpy/commit/2cd22c7998cd22d399e59a86131e4c5712127b23))

### Code Refactor

- **io.plugins:** update type hints ([54d0c5d](https://github.com/kmnhan/erlabpy/commit/54d0c5d7e55cc7e6af6b4d83feeb9d6c863e52f6))
- remove unused imports ([f1e35de](https://github.com/kmnhan/erlabpy/commit/f1e35ded993f1fb2be04c549aa241809c7d68a4d))
- **interactive:** add informative error message for missing pyqt ([1347a02](https://github.com/kmnhan/erlabpy/commit/1347a0231698f8104be71116823f797caeccc9a6))
- **io.plugins:** add warning when plugin load fails ([ed5b184](https://github.com/kmnhan/erlabpy/commit/ed5b184538cde60c1fa2ba2421ff30c27acb1eed))

## v2.10.0 (2024-10-08)

### Features

- **io.plugins:** add loader for beamline ID21 ESM at NSLS-II ([c07e490](https://github.com/kmnhan/erlabpy/commit/c07e490c37b52706aa9407d6adb5aa7787e2c1b0))

  This commit adds a new data loader for beamline ID21 ESM at NSLS-II, Brookhaven National Laboratory.
- **io.dataloader:** add formatters ([2ee9a4a](https://github.com/kmnhan/erlabpy/commit/2ee9a4a2361b91727c4e964adc00a26398863f2f))

  A new attribute named `formatters` and a new method `get_formatted_attr_or_coord` has been added to loaders. This allows custom per-attribute pretty-printing behavior.
- **io:** add parallel argument to `load` ([88cd924](https://github.com/kmnhan/erlabpy/commit/88cd924efd9375ebe89417b504c80e55f7071404))
- **io:** add xarray backend for igor files ([1fe5ca5](https://github.com/kmnhan/erlabpy/commit/1fe5ca514777f4356f1b67251dc1a6f21b320d48))

  `.pxt`, `.pxp`, and `.ibw` files can now be opened with xarray methods such as `xr.open_dataset` and `xr.open_dataarray`. See the updated user guide for more information.

### Bug Fixes

- **io.dataloader:** properly reorder coordinates ([3ebfb0f](https://github.com/kmnhan/erlabpy/commit/3ebfb0fdbc21e690e51c185da92ed9c88921d3b1))

  Coordinate order was broken for loaders which assign coordinates in inherited `post_process`. This is now fixed, and returned data will be consistently ordered with respect to the mapping, with the dimension coordinates coming first.
- **erlab.io.plugins.maestro:** temporary fix for xarray issue ([c2d04a3](https://github.com/kmnhan/erlabpy/commit/c2d04a31ee98cbbcc403dafce6cf8dd6e37a09e5))
- **io.plugins.da30:** properly handle output types ([6297aba](https://github.com/kmnhan/erlabpy/commit/6297aba2474ce4a28e865f48320d37e47c4ba1f7))

  The DA30 loader now tries to return a dataset from `.zip` files only when there are no coordinate conflicts. In the case of conflicts, the loader will return a `DataTree`.

  Also, single region DA30 `.pxt` files will now return a `DataArray` consistent with the equivalent `.ibw` file.
- **interactive.fermiedge:** allow transposed input to fermi edge fitting tool ([dcae75e](https://github.com/kmnhan/erlabpy/commit/dcae75e718dd6b0e2cc773e43407e3a6a2cd97f0))
- **interactive.imagetool:** retain attrs when exporting slice ([1bed572](https://github.com/kmnhan/erlabpy/commit/1bed572dc7309d723152237902b021d27d0dbb40))

  When accessing the data of a single slice from the right-click menu of `ImageTool`, the attributes of the original data are now kept. This allows saved slices or data opened in other tools to retain their attributes.
- resolve gui script not working on windows with conda ([62253d0](https://github.com/kmnhan/erlabpy/commit/62253d0d64d9cf458ee3e58e20fc890914c13c18))
- **io.dataloader:** clear plot before loading in interactive summaries ([21d6dea](https://github.com/kmnhan/erlabpy/commit/21d6dea9876066c64b9a9d2dc62dd9d41ba3ec9b))
- **plotting.general:** fix `plot_slices` compatibility with slice object as argument ([5948a7b](https://github.com/kmnhan/erlabpy/commit/5948a7b64fb415cfe86668bb17f3c656df1466a6))
- **io:** disable memmapping when loading data ([c39da1b](https://github.com/kmnhan/erlabpy/commit/c39da1b64ad28447260b9131f38206131bd6c0cb))

  Memmapping seemed to interfere loading multiple files when called through ipywidgets.

### Code Refactor

- **io.utils:** use pathlib in `get_files` ([b7a0f5b](https://github.com/kmnhan/erlabpy/commit/b7a0f5b33e9a1f7a6840413f431fd60a10bb754e))
- **io.plugins.merlin:** combine ImageTool file menu into single entry ([6e28ac2](https://github.com/kmnhan/erlabpy/commit/6e28ac21d7b534f5243f25a40710910eefb3a2b8))
- **io:** add postprocessing and validation for `DataTree` objects ([3fb3ff5](https://github.com/kmnhan/erlabpy/commit/3fb3ff5ec3755a40d6f2a66e9f242e6be9529816))
- **io:** add warning when file is ambiguous ([8daabb8](https://github.com/kmnhan/erlabpy/commit/8daabb85eb4b4313fc05199fa0eb8ed4de6c80f6))
- **io:** remove renaming steps from `load_single` in multi-file loaders ([542f4f2](https://github.com/kmnhan/erlabpy/commit/542f4f2874f5255e71a53ecc58c0b87c68c84aaa))

  Combining before renaming coords should be more straightforward
- **io:** allow missing alpha coord ([c9deed4](https://github.com/kmnhan/erlabpy/commit/c9deed43a0ecce20a70145d9fe6a21b2a1ea1693))

  Validation checks will not warn about missing detector angle, allowing XPS measurements.
- **io.dataloader:** only allow real file or folder names as input ([b9a59cc](https://github.com/kmnhan/erlabpy/commit/b9a59cc4c4cd16c44c93237b24b519ab270aff79))

  The previous behavior allowed passing `f_001.pxt` to load `f_001_S001.pxt`, `f_001_S002.pxt`... but this was confusing since there is no file named `f_001.pxt`. This commit disallows such input.
- cleanup erplot namespace ([007eedb](https://github.com/kmnhan/erlabpy/commit/007eedb8862fcdcd8a54bb170c5ec0272db3f194))
- **io:** implement metaclass ([e787b1c](https://github.com/kmnhan/erlabpy/commit/e787b1c63b1cf786797ddcd2c360438619574f80))

  Whenever the `identify()` method failed to find any files, subclasses had to explicitly raise `FileNotFoundError`. This resulted in a lot of boilerplate code and ambiguous error messages. Now, all subclasses can just return `None` in `identify()` when no files are found. The appropriate error is automatically raised.
- **io.igor:** raise OSError on load fail ([6c7a4c4](https://github.com/kmnhan/erlabpy/commit/6c7a4c471874c8fb92c267f6c551e332e35d0bc6))
- move dataloader cell formatting implementation to utils module ([0f2cb1c](https://github.com/kmnhan/erlabpy/commit/0f2cb1c70b2c7dbadd5ab936770290486b50200a))
- **io:** deprecate calling igor functions from top level namespace ([bb8af7c](https://github.com/kmnhan/erlabpy/commit/bb8af7c76b1bf31f481de2a43f104f8c8a638225))

  Calling `erlab.io.load_wave` and `erlab.io.load_experiment` is deprecated. When writing new code, use `xarray.load_dataarray` and `xarray.load_dataset` instead.

### Performance

- **interactive.imagetool:** improve manager speed ([891c4ee](https://github.com/kmnhan/erlabpy/commit/891c4eed0921fdb5b4ebd2cb17e6c932afa79ccc))

## v2.9.1 (2024-09-03)

### Bug Fixes

- ui file compatibility with Qt5 ([66c776d](https://github.com/kmnhan/erlabpy/commit/66c776d912e5cf3ba1be819bd46b972f22cbb560))

## v2.9.0 (2024-08-30)

### Features

- **interactive.imagetool:** add rotation ([fdeb8a9](https://github.com/kmnhan/erlabpy/commit/fdeb8a96be515b75d0536c48f2e3b042e9eccea5))

  A rotation dialog has been added to the Edit menu. Rotation guidelines can be overlaid on the main image.
- **interactive.utils:** add rotatable lines that can be rotated by dragging ([31b55e5](https://github.com/kmnhan/erlabpy/commit/31b55e5a937d3018f1b6e5ac85a3dacb46cb4839))
- **analysis.transform:** add `rotate` function ([83a2ad8](https://github.com/kmnhan/erlabpy/commit/83a2ad8ea52ab27635e1acff6d9317fe13110c97))

  Added a new function that can rotate DataArray values using spline interpolation. Previous simple implementations are marked as deprecated.
- **utils.array:** add new function`trim_na` ([c628b5b](https://github.com/kmnhan/erlabpy/commit/c628b5b092ecc3a78bda2d66f5611e41e4d80402))

  This function trims the edges of DataArrays where all values are NaN.
- **accessors.kspace:** add method argument ([204073e](https://github.com/kmnhan/erlabpy/commit/204073e9b748bf8c86fd0cd5b6aa98acfa86d3aa))

  Momentum conversion through the `convert()` method of the kspace accessor now supports an additional keyword argument `method` that can be used to choose different interpolation methods supported by `scipy.interpolate.RegularGridInterpolator`. Note that methods other than `'linear'` will take much longer to execute.
- **analysis.interpolate:** add solver args ([24be3b0](https://github.com/kmnhan/erlabpy/commit/24be3b0d86e4418010a5a69abf354a86987c0712))

  `FastInterpolator` now supports `solver` and `solver_args` keyword arguments introduced in scipy 1.13.0.
- **interactive.colors:** implement `BetterColorBarItem` limit editor ([7dd1477](https://github.com/kmnhan/erlabpy/commit/7dd1477d694b883326a0dfc8f8e552f2b905da06))

  A new context menu in `BetterColorBarItem`'s viewbox enables manually editing color limits.
- **analysis.interpolate:** implement slicing along a vector ([cba8567](https://github.com/kmnhan/erlabpy/commit/cba85675ea68925c39ed184f062f81aaded4d37b))

  A new function `slice_along_vector` has been added which enables interpolating through a line defined by a vector and a point.
- **interactive.imagetool:** add goldtool and dtool to menu ([33d5e35](https://github.com/kmnhan/erlabpy/commit/33d5e35dc7deb5c730da21cade55fd814a24268d))

  The interactive tools goldtool and dtool are now directly accessible from the right-click menu of 2D images in ImageTool.
- **accessors.general:** add option to `qsel.around` to disable averaging ([5aaed85](https://github.com/kmnhan/erlabpy/commit/5aaed856a1d3d71771a2bbf95d8c912c5b119e99))
- **plotting.general:** add `NonUniformImage` functionality to `plot_array` ([86d8c1a](https://github.com/kmnhan/erlabpy/commit/86d8c1a45510a41d7d7c07a1cf417a0c45efbdb9))

  `plot_array` can now plot data with unevenly spaced coordinates. It uses `matplotlib.image.NonUniformImage` with the default interpolation option set to 'nearest'. The resulting plot may be different from `xarray.DataArray.plot` which uses `pcolormesh` to generate image plots.
- **interactive.imagetool:** add copy limits to colorbar menu ([29c37c4](https://github.com/kmnhan/erlabpy/commit/29c37c45f9de516b3764d4b4ed80282955ea1f3a))

  Right-clicking on the colorbar will now show a menu which contains a button that copies the current color limits to the clipboard. This is useful when manually adjusting color limits.

### Bug Fixes

- **interactive.imagetool:** properly disconnect signals ([dce236f](https://github.com/kmnhan/erlabpy/commit/dce236f1da1aae44bc3210aac1ff2eb710d71f41))
- **interactive.imagetool:** fix autoscale when loading data ([2c12f59](https://github.com/kmnhan/erlabpy/commit/2c12f592e3d1d57de649aabae742eee230545387))
- **interactive.imagetool:** scale spinbox decimals relative to coordinate step size ([9a801a5](https://github.com/kmnhan/erlabpy/commit/9a801a5391cd35e1fdeaf9377fd8b220420d4829))
- **interactive.utils:** update `BetterSpinBox` width on changing decimals ([0a70884](https://github.com/kmnhan/erlabpy/commit/0a70884eb7742cea53962beb80ba42cad723fe4e))
- **interactive:** fix compatibility issue with PySide6 ([da5f4af](https://github.com/kmnhan/erlabpy/commit/da5f4af139988e38f1c9b0534a5957644e01b9aa))
- **interactive.imagetool:** do not copy code when unnecessary ([9131029](https://github.com/kmnhan/erlabpy/commit/91310295fb9ffab994b0b681c84a4646680583b0))
- **accessors.general:** qshow now triggers hvplot properly for 1D data ([8a84813](https://github.com/kmnhan/erlabpy/commit/8a84813e561a52eaf5ef5fc7118b992e7537b1f6))
- **interactive.imagetool:** make manager socket use default backlog ([0ac7f0b](https://github.com/kmnhan/erlabpy/commit/0ac7f0b35b3288a62708c1c5e2c54a483e563bd7))
- **interactive.imagetool:** ensure proper socket termination in manager ([2cceb27](https://github.com/kmnhan/erlabpy/commit/2cceb27a7d0e9db80fc39aa65340df6612587206))

### Code Refactor

- **interactive.utils:** improve code generation ([78c403f](https://github.com/kmnhan/erlabpy/commit/78c403fed2220a8033acc71ba8280f1446509bc1))
- **analysis:** move `shift` to `transform` ([08baf05](https://github.com/kmnhan/erlabpy/commit/08baf0556650167bd495628e65e5f2e415380712))

  The `shift` function has been moved from `utils` to `transform`. Calling from the `utils` module is now deprecated.
- **analysis:** cleanup namespace ([e3e641d](https://github.com/kmnhan/erlabpy/commit/e3e641d4e176e042430b51f74d8ca79d34270e24))

  Three functions that were directly accesible from the `erlab.analysis` namespace are now deprecated. Import them from their respective modules.
- remove deprecated module `analysis.utilities` ([8b79ab5](https://github.com/kmnhan/erlabpy/commit/8b79ab5d978f14bb12123d283c74235d4e829094))
- **analysis.image:** add check for NaN in input ([095554f](https://github.com/kmnhan/erlabpy/commit/095554fcde5b1e2a8540e3a2626b7b7da3e8f181))

  Derivative functions now check for NaNs in input data and raise a warning.

  The interactive derivative tool automatically fills NaNs in the input data with zeros and shows a warning message.
- remove unpacking inside `np.r_` ([6c27864](https://github.com/kmnhan/erlabpy/commit/6c278648eb46378227e49fc8ed01dd822ed52217))
- improve initial import time ([f720973](https://github.com/kmnhan/erlabpy/commit/f7209735840355c9fa6ab799b578aecf65008d31))

## v2.8.5 (2024-07-31)

### Bug Fixes

- **plotting.annotations:** properly pass keyword arguments in `mark_points_outside` ([2136939](https://github.com/kmnhan/erlabpy/commit/2136939e09656f921aed7204ca11cc6615605b7f))
- **plotting.annotations:** expose property label generation to public api ([545781d](https://github.com/kmnhan/erlabpy/commit/545781d1aa5d04b5dd3bf6d0498821d104f837ac))

  A new `property_labels` function can be used to generate strings that are used by `label_subplot_properties` so that the labels can be used as titles easily through `eplt.set_titles`. Also, label generation now recognizes time, given as 't' with default unit seconds.

## v2.8.4 (2024-07-26)

### Bug Fixes

- **erlab.plotting.general:** improve `plot_array` keyword versatility ([1dc41cd](https://github.com/kmnhan/erlabpy/commit/1dc41cd52f8d7f879cfe54f2adf3a512b78ac007))

  Enables additional kwargs with valid data dimensions as the key to be passed onto `qsel`.
- **erlab.analysis.gold:** fix `quick_fit` attribute detection ([3797f93](https://github.com/kmnhan/erlabpy/commit/3797f93e1a578e356ce21e7b7c933341099ab156))
- **interactive.imagetool:** retain window title upon archiving ([b5d8aa4](https://github.com/kmnhan/erlabpy/commit/b5d8aa4884562ba4b53351baf58d598e27a1e757))

### Code Refactor

- **plotting.general:** remove `LabeledCursor` ([912b4fb](https://github.com/kmnhan/erlabpy/commit/912b4fb73f88e3a529d1e3880a2253e0cb26e7ae))

  We skip the deprecation step since nobody is likely to be using it anyway.
- **accessors:** split submodule ([6ed5c03](https://github.com/kmnhan/erlabpy/commit/6ed5c039889624d3589d9ce71a75ed6047f4406f))

  Accessors in `utils.py` has been moved to `general.py`, so that `utils.py` only contains utilities for creating accessors.
- improve type annotations ([b242f44](https://github.com/kmnhan/erlabpy/commit/b242f44d49239e51b4bd9e4b1ae7fd952c59e2c2))

## v2.8.3 (2024-07-08)

### Bug Fixes

- **interactive.imagetool:** various fixes related to manager ([3d3f55e](https://github.com/kmnhan/erlabpy/commit/3d3f55e84c2837dc86592bc2f5aa68282ca44fa5))

  This fix incorporates many changes to the ImageTool and ImageTool Manager.

  First, the archiving function of the manager now works properly, and tries to clear memory eagerly.

  When opening data from a file using the GUI, the name of the file will now be displayed in the title bar of the ImageTool. This file name is also propagated to the name displayed in the manager.

  Furthermore, the archiving and show/hide functionality of the manager has been updated to restore the window geometry automatically. When the user shows or unarchives a hidden or archived window, the previous position of the window  is restored.

  Some icons and the layout of the manager has been modified, and tooltips has been added to the buttons.

  Also, some unexpected behavior regarding linking has been resolved.
- **plotting.plot3d:** temporarily disable broken monkey patch ([220f23f](https://github.com/kmnhan/erlabpy/commit/220f23fd078a4563f0eb33371af66d5b486d34cd))
- replace broken signature for dynamic functions ([39a3954](https://github.com/kmnhan/erlabpy/commit/39a39549b074055bafb93238492dc2dd3ba3c834))
- **interactive.imagetool:** fix broken binning controls on loading fron GUI ([0ca5437](https://github.com/kmnhan/erlabpy/commit/0ca5437481e4b7c269acde8bb1badec1070752e7))

### Code Refactor

- satisfy type checker ([042a7b1](https://github.com/kmnhan/erlabpy/commit/042a7b1f72a9a29b93736fe1eea61f18cc8ea49d))
- **interactive.imagetool:** add batch close button to manager ([efc6089](https://github.com/kmnhan/erlabpy/commit/efc6089669d73ec5ba39acbbeb08720f0543fe3e))

## v2.8.2 (2024-07-01)

### Bug Fixes

- **interactive.imagetool:** fix crash while linking more than 3 tools ([d5f8a30](https://github.com/kmnhan/erlabpy/commit/d5f8a30224f72d7159216fa5638056569521f75f))
- update resistance loader ([6fcf2ab](https://github.com/kmnhan/erlabpy/commit/6fcf2abe797313ee3c21fd3cd2f4daebf412225f))

### Code Refactor

- **interactive.imagetool:** show error message in GUI when opening file ([287a7e8](https://github.com/kmnhan/erlabpy/commit/287a7e84e5110ac08e17d9a852b0d2b0da830e42))

## v2.8.1 (2024-06-21)

### Bug Fixes

- **interactive.imagetool:** properly implement caching and linking from GUI ([ffacdce](https://github.com/kmnhan/erlabpy/commit/ffacdce93d1ff89e1be823317a6d59a400a6dee2))
- **plotting.general:** pass DataArray to `func` argument to `plot_array` ([ed76e64](https://github.com/kmnhan/erlabpy/commit/ed76e64e45eb3ea93fba61380bc0d63864446fd3))

### Performance

- **interactive.imagetool:** speedup file loading and saving ([a6c869b](https://github.com/kmnhan/erlabpy/commit/a6c869b7d6ce0419d84a46086004d451845c23e3))

  Use pickle to save and load files instead of `erlab.io.load_hdf5` and `erlab.io.save_as_hdf5`.

## v2.8.0 (2024-06-17)

### Features

- **erlab.io.plugins.ssrl52:** changes to loader ([512a89b](https://github.com/kmnhan/erlabpy/commit/512a89b051911c88bafd59bdc9bd993ec727321a))

  The loader now promotes all attributes that varies during the scan to coordinates. Also, if the energy axis is given in kinetic energy and the work function is inferrable from the data attributes, the energy values are automatically converted to binding energy. This may require changes to existing code. This commit also includes a fix for hv-dependent swept cuts.
- **erlab.io.dataloader:** reorder output coordinates ([178edd2](https://github.com/kmnhan/erlabpy/commit/178edd27f3e58387b12b7a7928a26e87766fa9be))

  Coordinates on the loaded data will now respect the order given in `name_map` and `additional_coords`, improving readability.
- **interactive.imagetool:** add ImageTool window manager ([b52d249](https://github.com/kmnhan/erlabpy/commit/b52d2490ec61053b7b933e274a68a163761827ce))

  Start the manager with the cli command `itool-manager`. While running, all calls to `erlab.interactive.imagetool.itool` will make the ImageTool open in a separate process. The behavior can be controlled with a new keyword argument, `use_manager`.
- **interactive.imagetool:** add undo and redo ([e7e8213](https://github.com/kmnhan/erlabpy/commit/e7e8213964c9739468b65e6a56dcc1a0d9d648e4))

  Adjustments made in ImageTool can now be undone with Ctrl+Z. Virtually all actions except window size change and splitter position change should be undoable. Up to 1000 recent actions are stored in memory.
- **interactive.imagetool:** remember last used loader for each tool ([eb0cd2f](https://github.com/kmnhan/erlabpy/commit/eb0cd2f41992845988f5e500416ed98f5d078c14))

### Bug Fixes

- **interactive.imagetool:** fix code generation behaviour for non-uniform coordinates ([3652a21](https://github.com/kmnhan/erlabpy/commit/3652a21cf126ebcde015d5b7373bf5d5a675b177))

### Code Refactor

- **interactive.imagetool:** preparation for saving and loading state ([eca8262](https://github.com/kmnhan/erlabpy/commit/eca8262defe8d135168ca7da115d947bda3c1040))

## v2.7.2 (2024-06-14)

### Bug Fixes

- **erlab.io:** regression in handling getattr of dataloader ([dd0a568](https://github.com/kmnhan/erlabpy/commit/dd0a5680c6aed6e3b7ab391a10fbeb5c3cdc9c14))

## v2.7.1 (2024-06-14)

### Bug Fixes

- **interactive.imagetool:** Integrate data loaders to imagetool ([7e7ea25](https://github.com/kmnhan/erlabpy/commit/7e7ea25a8fbe3a43222fbc7baedaa04c6522e24d))

  A new property called `file_dialog_methods` can be set in each loader which determines the method and name that is used in the file chooser window in imagetool.
- **accessors.kspace:** `hv_to_kz` now accepts iterables ([36770d7](https://github.com/kmnhan/erlabpy/commit/36770d723b1e3592bf83750f7559603026059bb1))

## v2.7.0 (2024-06-09)

### Features

- **analysis.gold:** add function for quick resolution fitting ([2fae1c3](https://github.com/kmnhan/erlabpy/commit/2fae1c351f29b2fb1ceef39a69706b3f198e4659))
- **analysis.fit:** Add background option to `MultiPeakModel` and `MultiPeakFunction` ([2ccd8ad](https://github.com/kmnhan/erlabpy/commit/2ccd8ad835cbc8de9764d2f8bbadda425babddb1))

### Bug Fixes

- **erlab.io.plugins:** fix for hv-dependent data ([d52152f](https://github.com/kmnhan/erlabpy/commit/d52152f24807b9334ad5ffcc22c45a4af7a8d9ec))

## v2.6.3 (2024-06-07)

### Bug Fixes

- **erlab.io.plugins:** support SSRL hv dependent data ([1529b6a](https://github.com/kmnhan/erlabpy/commit/1529b6a0af43f09c51691ad8bebf9208d421940a))

### Code Refactor

- cleanup namespace ([847fbbe](https://github.com/kmnhan/erlabpy/commit/847fbbe4975b507905dc85ca5ae75fe16f5f887e))

## v2.6.2 (2024-06-03)

### Bug Fixes

- **interactive.imagetool:** fix regression with nonuniform data ([67df972](https://github.com/kmnhan/erlabpy/commit/67df9720193611816e2a562ce71d080fccbbec5e))

## v2.6.1 (2024-05-30)

### Bug Fixes

- re-trigger due to CI failure ([b6d69b5](https://github.com/kmnhan/erlabpy/commit/b6d69b556e3d0dbe6d8d71596e32d9d7cfdc5267))

## v2.6.0 (2024-05-30)

### Features

- **interactive.imagetool:** add bin amount label to binning controls ([7a7a692](https://github.com/kmnhan/erlabpy/commit/7a7a692b881e4cc1bd49342f31f3fe50407d72b5))
- add accessor for selecting around a point ([aa24457](https://github.com/kmnhan/erlabpy/commit/aa244576fcfa17f71be0a765be8f270a6ae28080))
- **accessors.fit:** add support for background models ([550be2d](https://github.com/kmnhan/erlabpy/commit/550be2deebf54fab77bef591ccbe059b5b219937))

  If one coordinate is given but there are two independent variables are present in the model,  the second one will be treated as the data. This makes the accessor compatible with y-dependent background models, such as the Shirley background provided in `lmfitxps`.
- **io:** make the dataloader behavior more customizable ([4824127](https://github.com/kmnhan/erlabpy/commit/4824127181b4383788f6dbe5cbeae4b2060f1f4f))

  Now, a new `average_attrs` class attribute exists for attributes that would be averaged over multiple file scans. The current default just takes the attributes from the first file. This also works when you wish to demote a coordinate to an attribute while averaging over its values.

  For more fine-grained control of the resulting data attributes, a new method `combine_attrs` can be overridden to take care of attributes for scans over multiple files. The default behavior is to just use the attributes from the first file.

### Bug Fixes

- **plotting:** make `gradient_fill` keep axis scaling ([51507dd](https://github.com/kmnhan/erlabpy/commit/51507dd966a0ce2db4aabff2aac8222bee184cf8))

### Code Refactor

- **analysis.image:** add check for 2D and uniform inputs ([22bb02d](https://github.com/kmnhan/erlabpy/commit/22bb02dd8dfbd5eb6b5d577abe9138a769a079b3))
- try to fix synced itool garbage collection ([932cc5a](https://github.com/kmnhan/erlabpy/commit/932cc5a690dcebc92c65ea3f17081ac9f9c3ef8f))

  This only happens in GH actions, and it doesn't happen every time so it's hard to debug.
- create utils subpackage to host internal methods ([3fa2873](https://github.com/kmnhan/erlabpy/commit/3fa287386fc0e94e8a558e2f0e5520be869acb43))

  The parallel module is now part of utils, without a compatibiliity layer or deprecation warning since nobody is using the functions from parallel anyway.
- add deprecation warnings for utilities ([5d375b8](https://github.com/kmnhan/erlabpy/commit/5d375b8fe0766ea3f2c5fe2421937ce7309e3da5))

  All submodules named `utilities.py` have been renamed to `utils.py` for consistency. The old call to `utilities.py` will still work but will raise a warning. The modules will be removed on 3.0 release.
- rename `erlab.interactive.utilities` to `erlab.interactive.utils` ([d9f1fb0](https://github.com/kmnhan/erlabpy/commit/d9f1fb081be8d2e8710ec08421780f927341b71a))
- rename `erlab.analysis.utilities` to `erlab.analysis.utils` ([ed81b62](https://github.com/kmnhan/erlabpy/commit/ed81b6234bd2960da785875e0aaaf2e9e5e48f15))
- rename `erlab.io.utilities` to `erlab.io.utils` ([6e0813d](https://github.com/kmnhan/erlabpy/commit/6e0813d3873b09593ec9d539d72c7512fac77c70))
- **io.plugins.merlin:** regard temperature as coordinate ([2fda047](https://github.com/kmnhan/erlabpy/commit/2fda04781961f2384c711a3b1c3c00ddaecaa617))

## v2.5.4 (2024-05-23)

### Bug Fixes

- **io.plugins.maestro:** load correct beta for non deflector scans ([5324c36](https://github.com/kmnhan/erlabpy/commit/5324c362d7bdd1dcf0c21303f370fd2e77f12448))

## v2.5.3 (2024-05-22)

### Bug Fixes

- **io.utilities:** `get_files` now only list files, not directories ([60f9230](https://github.com/kmnhan/erlabpy/commit/60f92307f94484361e0ba11b10a52be4c4cc05a1))
- **accessors.fit:** add `make_params` call before determining param names, closes [#38](https://github.com/kmnhan/erlabpy/issues/38) ([f1d161d](https://github.com/kmnhan/erlabpy/commit/f1d161de089b93e16b2947b126ac075764d98f75))
- **analysis.fit:** make some models more robust to DataArray input ([afe5ddd](https://github.com/kmnhan/erlabpy/commit/afe5ddd9d1e6796ba0261a147c2733d607916d81))

### Code Refactor

- add loader for ALS BL7 MAESTRO `.h5` files ([4f33402](https://github.com/kmnhan/erlabpy/commit/4f3340228ae2e1cbd8baf57d5d426043f5e28688))
- **interactive:** add informative error message for missing Qt bindings ([560615b](https://github.com/kmnhan/erlabpy/commit/560615bb89d2646965d1a2a967133f0df08e3f6e))
- **io:** rename some internal variables and reorder ([76fe284](https://github.com/kmnhan/erlabpy/commit/76fe284b4bc9f1e0c3cb94857a65599b07ee04df))

  Also added a check for astropy in FITS file related utility.

## v2.5.2 (2024-05-16)

### Bug Fixes

- make mathtext copy default to svg ([2f6e0e5](https://github.com/kmnhan/erlabpy/commit/2f6e0e558f251c846bc3dec39cd150391802460d))
- resolve MemoryError in prominent color estimation ([3bdcd03](https://github.com/kmnhan/erlabpy/commit/3bdcd0341c41b424ebbcb565b7cda0db839e4cb8))

  Due to numpy/numpy/#11879 changed the auto method to sqrt. This should also improve memory usage and speed, with little to no impact on the end result.

## v2.5.1 (2024-05-15)

### Bug Fixes

- **plotting:** fixes [#35](https://github.com/kmnhan/erlabpy/issues/35) ([a67be68](https://github.com/kmnhan/erlabpy/commit/a67be6869c2d25780f8a56794aad0386379202dd))

  Gradient fill disappears upon adding labels
- **fit.models:** wrong StepEdgeModel guess with DataArray input ([6778c8d](https://github.com/kmnhan/erlabpy/commit/6778c8dd2c048b0cab67c6d3668b25b3f79a71da))

### Code Refactor

- **plotting:** code cleanup ([aef10e4](https://github.com/kmnhan/erlabpy/commit/aef10e472a3ebc935711253e91124cfd87beb9cc))

## v2.5.0 (2024-05-13)

### Features

- extended interactive accessor ([f6f19ab](https://github.com/kmnhan/erlabpy/commit/f6f19abd8edfb33585b5e19040a2ebaff39b2b70))

  The `qshow` accessor has been updated so that it calls `hvplot` (if installed) for data not supported by ImageTool.

  Also, the `qshow` accessor has been introduced to Datasets. For valid fit result datasets produced by the `modelfit` accessor, calling `qshow` will now show an `hvplot`-based interactive visualization of the fit result.
- **itool:** make itool accept Datasets ([f77b699](https://github.com/kmnhan/erlabpy/commit/f77b699abdf312a23832611052d67e8c4c8d4930))

  When a Dataset is passed to `itool`, each data variable will be shown in a separate ImageTool window.
- **analysis.image:** add multidimensional Savitzky-Golay filter ([131b32d](https://github.com/kmnhan/erlabpy/commit/131b32d9e562693e98a2f9e45cf6db4635405b44))

### Bug Fixes

- **itool:** add input data dimension check ([984f2db](https://github.com/kmnhan/erlabpy/commit/984f2db0f69db2b5b99211728840447d9617f8bf))
- **analysis.image:** correct argument order parsing in some filters ([6043413](https://github.com/kmnhan/erlabpy/commit/60434136224c0875ed8fba41d24e32fc6868127c))
- **interactive:** improve formatting for code copied to clipboard ([d8b6d91](https://github.com/kmnhan/erlabpy/commit/d8b6d91a4d2688486886f2464426935fdf8cabc2))

### Code Refactor

- **plotting:** update `clean_labels` to use `Axes.label_outer` ([0c64756](https://github.com/kmnhan/erlabpy/commit/0c647564c6027f5b60f9ff288f13019e0e5933b6))

## v2.4.2 (2024-05-07)

### Bug Fixes

- **ktool:** resolve ktool initialization problem, closes [#32](https://github.com/kmnhan/erlabpy/issues/32) ([e88a58e](https://github.com/kmnhan/erlabpy/commit/e88a58e6aaed326af1a68aa33322d6ea9f0e800d))
- **itool:** disable flag checking for non-numpy arrays ([da6eb1d](https://github.com/kmnhan/erlabpy/commit/da6eb1db9e81d51b52d4b361de938bcf7ba45e68))

## v2.4.1 (2024-05-03)

### Bug Fixes

- **plotting:** fix wrong regex in `scale_units` ([d7826d0](https://github.com/kmnhan/erlabpy/commit/d7826d0269214dfd822a4d0293e42a9840015ce8))
- fix bug in `modelfit` parameter concatenation ([edaa556](https://github.com/kmnhan/erlabpy/commit/edaa5566c6e3817e1d9220f7a96e8e731cf7eede))
- **itool:** ensure DataArray is readable on load ([5a0ff00](https://github.com/kmnhan/erlabpy/commit/5a0ff002802cdf5bd3ceb34f9cddc53c9674e7bd))

## v2.4.0 (2024-05-02)

### Features

- **imagetool:** add method to update only the values ([ca40fe4](https://github.com/kmnhan/erlabpy/commit/ca40fe41a0320fd7843c86f95b68f8b6e19adca8))
- add interpolation along a path ([7366ec4](https://github.com/kmnhan/erlabpy/commit/7366ec4db600617e585c724d05aafea387456cf2))

  The `slice_along_path` function has been added to `analysis.interpolate`, which enables easy interpolation along a evenly spaced path that is specified by its vertices and step size. The path can have an arbitrary number of dimensions and points.

### Bug Fixes

- **io:** remove direct display call in interactive summary ([d44b3a5](https://github.com/kmnhan/erlabpy/commit/d44b3a56aecfb054a38d944c5c8b7f45d362cf3b))

  This was causing duplicated plots.
- **plotting:** add some validation checks to `plot_array` ([2e0753c](https://github.com/kmnhan/erlabpy/commit/2e0753c90ffbe6fdd05af210ac6a4dbfa9aba899))

  The functions `plot_array` and `plot_array_2d` now checks if the input array coordinates are uniformly spaced. If they are not, a warning is issued and the user is informed that the plot may not be accurate.
- **plotting:** increase default colorbar size ([3208399](https://github.com/kmnhan/erlabpy/commit/32083990e9e77df6e94b2b0836bc1f9764cfaaf7))

  The default `width` argument to `nice_colorbar` is changed to 8 points. This ensures visibility in subplots, especially when constrained layout is used.
- delay interactive imports until called ([ad15910](https://github.com/kmnhan/erlabpy/commit/ad15910f921cb5ffffc388e7a5e02832935f8547))

### Code Refactor

- various cleanup ([2b38397](https://github.com/kmnhan/erlabpy/commit/2b383970b602507b6efedbf396f14d470db60d8f))

  Improve docstring formatting and tweak linter settings

## v2.3.2 (2024-04-25)

### Bug Fixes

- **io:** make summary caching togglable ([99b8e22](https://github.com/kmnhan/erlabpy/commit/99b8e221e75db73382bf599170c58d8a68ca049e))

  Also fixes a bug where interactive summary plots were duplicated
- **io:** data loader related fixes ([da08e90](https://github.com/kmnhan/erlabpy/commit/da08e9076e59895b35c393c8e2556c3592adf4a5))

  DA30 dataloader now preserves case for attribute names from zip files.

  Post processing for datasets now works properly

## v2.3.1 (2024-04-25)

### Bug Fixes

- **interactive:** keep pointer for imagetool, fix typing issues ([c98c38e](https://github.com/kmnhan/erlabpy/commit/c98c38ea11bce50ed9bfd8d374064bb2b1659d0c))

### Code Refactor

- move `characterization` to `io` ([9c30f1b](https://github.com/kmnhan/erlabpy/commit/9c30f1b7df51460f502dcbf999e3fac34be1cf99))

## v2.3.0 (2024-04-22)

### Features

- **kspace:** rewrite conversion with `xarray.apply_ufunc` ([156cef8](https://github.com/kmnhan/erlabpy/commit/156cef830582e01dc378a7437a0c85f4c7efc077))

  Momentum conversion now relies on xarray broadcasting for all computations, and objects with extra dimensions such as temperature can be automatically broadcasted.

  Dask arrays can also be converted.
- **exampledata:** enable specifying seed for noise rng ([aa542e8](https://github.com/kmnhan/erlabpy/commit/aa542e8c288ff1ca64820960f469b2c244ca5c95))
- **interpolate:** enable fast interpolation for 1D arrays ([ff333a0](https://github.com/kmnhan/erlabpy/commit/ff333a05803d7079034e36f2e1dc3d22d0b686f7))
- make both arguments optional for loader_context ([6780197](https://github.com/kmnhan/erlabpy/commit/6780197f68abfe7a9edbda951d804a9bc5ba56e9))
- **kspace:** automatically detect kinetic energy axis and convert to binding ([bbde447](https://github.com/kmnhan/erlabpy/commit/bbde44717155d1dd9ffefbc286da32b0bfac2180))
- add more output and parallelization to fit accessor ([59b35f5](https://github.com/kmnhan/erlabpy/commit/59b35f53f3ef7f518aec92e05854dba42ddba56f))

  Allows dictionary of `DataArray`s as parameter to fit accessor.

  Now, the return `Dataset` contains the data and the best fit array. Relevant tests have been added.
- add callable fit accessor using apply_ufunc ([11e3546](https://github.com/kmnhan/erlabpy/commit/11e35466fec158e40d0e8e738dd81ed10225d83c))

  Add a `Dataset.modelfit` and `DataArray.modelfit` accessor with similar syntax and output as `Dataset.curvefit`. Closes [#22](https://github.com/kmnhan/erlabpy/issues/22)
- add option to plot_array_2d so that users can pass non-normalized color array ([74cf961](https://github.com/kmnhan/erlabpy/commit/74cf961532a50d9c324189318460a9f840291a85))
- **analysis.gold:** add option to normalize energy axis in fitting ([3dffad6](https://github.com/kmnhan/erlabpy/commit/3dffad65993520c4b9a9a3afd6be85671bac9d3a))

  This improves performance and results when eV is large like ~100eV.

### Bug Fixes

- **kspace:** allow explicit coordinate kwargs ([fe47efc](https://github.com/kmnhan/erlabpy/commit/fe47efcde941767c02b582ce8b29d4b3678fd843))
- **exampledata:** change noise generation parameters ([b213f11](https://github.com/kmnhan/erlabpy/commit/b213f1151ed2555fc80374e9ebe3fc0856a13948))
- **fit:** make FermiEdge2dModel compatible with flattened meshgrid-like input arrays ([c0dba26](https://github.com/kmnhan/erlabpy/commit/c0dba261670774862f2dfae62c770bbab81aac2f))
- fix progress bar for parallel objects that return generators ([23d41b3](https://github.com/kmnhan/erlabpy/commit/23d41b31a3f3ee6c7343d471f7cec34dc374bafa))

  Tqdm imports are also simplified. We no longer handle `is_notebook` ourselves, but just import from `tqdm.auto`
- **plotting:** fix 2d colormaps ([8299576](https://github.com/kmnhan/erlabpy/commit/8299576ce3cbcbaec106bef952c6df148bb7ca18))

  Allow images including nan to be plotted with gen_2d_colormap, also handle plot_array_2d colorbar aspect

### Code Refactor

- make zip strict (ruff B905) ([78bf5f5](https://github.com/kmnhan/erlabpy/commit/78bf5f5a2db52c14ccf5bfd3c83659ca53c4a408))
- fix some type hints ([2dfa5e1](https://github.com/kmnhan/erlabpy/commit/2dfa5e1b4582e00d0631376ee32aa7d0b1b945b6))
- **example:** move exampledata from interactive to io ([1fc7e6c](https://github.com/kmnhan/erlabpy/commit/1fc7e6c22ce477fe7ebbd8b0c6844d1a85df3fcf))

  Also add sample data generation for fermi edge
- refactor accessors as submodule ([9fc37bd](https://github.com/kmnhan/erlabpy/commit/9fc37bd4825de519e4c4b6e30e9e32bf9392ed2d))
- rewrite either_dict_or_kwargs with public api ([34953d1](https://github.com/kmnhan/erlabpy/commit/34953d10b6fd67720b1c29dbed1ab7a24e4d3060))
- move correct_with_edge from era.utilities to era.gold ([08a906f](https://github.com/kmnhan/erlabpy/commit/08a906ff61a74febc0f47ed08ac24e7a4cd0977f))

  Calling from utilities will now raise a DeprecationWarning.

  The erlab.analysis namespace is unchanged, so the affect will be minimal.
- qsel now raises a warning upon scalar indexing outside coordinate bounds ([d6ed628](https://github.com/kmnhan/erlabpy/commit/d6ed628111be8ac594d3a1b83cc2785a31e3f06e))

## v2.2.2 (2024-04-15)

### Bug Fixes

- **io:** unify call signature for summarize ([e2782c8](https://github.com/kmnhan/erlabpy/commit/e2782c898d5aaaa1443b2bc82bb61fb40a28d232))
- resolve failing tests due to changes in sample data generation ([80f0045](https://github.com/kmnhan/erlabpy/commit/80f004574950834e42dbfa7677031d0f9f113bda))
- **interactive.exampledata:** properly generate 2D data ([825260c](https://github.com/kmnhan/erlabpy/commit/825260c8ceb0a79b8c071750003529b91cda3573))

### Code Refactor

- **io:** allow for more complex setups ([f67b2e4](https://github.com/kmnhan/erlabpy/commit/f67b2e4c7b092b7ca2db00ce02a23647879c514b))

  LoaderBase.infer_index now returns a second argument, which is a dictionary containing optional keyword arguments to load.
- **io:** provide rich interactive summary ([b075a9e](https://github.com/kmnhan/erlabpy/commit/b075a9ee59b61892462fc475e78b036a54408099))
- **io:** include "Path" column in ssrl loader summary ([ae1d8ae](https://github.com/kmnhan/erlabpy/commit/ae1d8aee051aa71563f6a6009ce9672e56edfae7))
- **io:** improve array formatting in summary ([1718529](https://github.com/kmnhan/erlabpy/commit/171852957db7fe53ff6a5c5c5f843530078d4b46))

### Performance

- **io:** speedup merlin summary generation by excluding duplicates ([d6b4253](https://github.com/kmnhan/erlabpy/commit/d6b42537ce48232b5112daef8f31e5cf86ea921a))

## v2.2.1 (2024-04-14)

### Bug Fixes

- **fit:** add sigma and amplitude expressions to MultiPeakModel parameters ([3f6ba5e](https://github.com/kmnhan/erlabpy/commit/3f6ba5e84922129296183e02255506df73da0276))
- **fit.minuit:** properly handle parameters constrained with expressions ([d03f012](https://github.com/kmnhan/erlabpy/commit/d03f012b4fde92f445a24657dca1fb5b3600fa45))

### Code Refactor

- set informative model name for MultiPeakModel ([d14ee9d](https://github.com/kmnhan/erlabpy/commit/d14ee9d6ac7962207700de50039a5b7a858fea6a))
- add gaussian and lorentzian for consistency ([07c0dfb](https://github.com/kmnhan/erlabpy/commit/07c0dfb9ecfb882e4f5f0ccfe942c1a835b613b2))

## v2.2.0 (2024-04-12)

### Features

- enable component evaluation for MultiPeakModel ([8875b74](https://github.com/kmnhan/erlabpy/commit/8875b7443d26313156fcdcc43586d40af4ff4f00))
- **analysis.fit:** add BCS gap equation and Dynes formula ([f862aa4](https://github.com/kmnhan/erlabpy/commit/f862aa4af4d2ba470f1ea074fc90442d9b18b336))

### Bug Fixes

- curvefittingtool errors ([9abb99c](https://github.com/kmnhan/erlabpy/commit/9abb99c35633bc722469276d4837a2372c132042))

### Code Refactor

- cleanup fit namespace ([906aa99](https://github.com/kmnhan/erlabpy/commit/906aa99193f78577e705218b2d6c22378611f84b))
- rename ExtendedAffineBroadenedFD to FermiEdgeModel ([a98aa82](https://github.com/kmnhan/erlabpy/commit/a98aa82bcbdf22ff8a156d800e336653f9afba07))
- **interactive:** exclude bad colormaps ([877c915](https://github.com/kmnhan/erlabpy/commit/877c915def6eb3dddb3862d6ac64c8c70f456ad3))

## v2.1.3 (2024-04-11)

### Bug Fixes

- **interactive:** update data load functions used in imagetool ([c3abe35](https://github.com/kmnhan/erlabpy/commit/c3abe3517046ed603a9221de38b22257322d3a51))

## v2.1.2 (2024-04-11)

### Bug Fixes

- **io:** prevent specifying invalid data_dir ([701b011](https://github.com/kmnhan/erlabpy/commit/701b011339ecba657a0f4a14e2fef19adeb4bf2b))
- **io:** fixes merlin summary data type resolving ([a91ad3d](https://github.com/kmnhan/erlabpy/commit/a91ad3d4387a23d25ac1b208cba8217e67efbec0))
- **io:** fix summary loading ([a5dd84a](https://github.com/kmnhan/erlabpy/commit/a5dd84af9eec0f835b3116bc7c470e57ef3f3e02))

## v2.1.1 (2024-04-10)

### Bug Fixes

- **io:** enable specifying data_dir in loader context manager ([37913b8](https://github.com/kmnhan/erlabpy/commit/37913b80a1d7c6313a5b6cc4a3ab614565274c81))
- **io:** allow loader_class aliases to be None ([7eae2eb](https://github.com/kmnhan/erlabpy/commit/7eae2ebf13f972d368ddb9922a71fd3bbed014e5))

### Code Refactor

- remove igor2 import checking ([b64d8f7](https://github.com/kmnhan/erlabpy/commit/b64d8f7fe22ebc1c4818e26f93f864fd402bbd05))
- **io:** default to always_single=True ([007bb3b](https://github.com/kmnhan/erlabpy/commit/007bb3b2703a647856c0a85e89075cf6572d263a))

## v2.1.0 (2024-04-09)

### Features

- **interactive:** overhaul dtool ([8e5ec38](https://github.com/kmnhan/erlabpy/commit/8e5ec3827dd2bd52475d454d5c5ef8aef7d665aa))

  Now supports interpolation, copying code, opening in imagetool, and 2D laplacian method.
- **interactive:** improve code generation ([7cbe857](https://github.com/kmnhan/erlabpy/commit/7cbe8572272f6c84a486599a990098ce8e3ff754))

  Automatically shortens code and allows literals in kwargs
- **interactive:** extend xImageItem, add right-click menu to open imagetool ([2b5bb2d](https://github.com/kmnhan/erlabpy/commit/2b5bb2dfc3d4173d950135306b3b30a018c6d389))

### Bug Fixes

- sign error in minimum gradient ([c45be0c](https://github.com/kmnhan/erlabpy/commit/c45be0cf1a025c67e8af959ff83a9339cddbaaaa))
- **analysis.image:** normalize data for mingrad output for numerical stability ([0fc3711](https://github.com/kmnhan/erlabpy/commit/0fc3711a521ffb0cbb4f5206c06d923eced1200c))

### Code Refactor

- **io:** validation now defaults to warning instead of raising an error ([8867a07](https://github.com/kmnhan/erlabpy/commit/8867a07304129beda749fa82d3909bf920fdb975))

## v2.0.0 (2024-04-08)

### BREAKING CHANGE

- `PolyFunc` is now `PolynomialFunction`, and `FermiEdge2dFunc` is now `FermiEdge2dFunction`. The corresponding model names are unchanged. ([20d784c](https://github.com/kmnhan/erlabpy/commit/20d784c1d8fdcd786ab73b3ae03d3e331dc04df5))

  BREAKING CHANGE: `PolyFunc` is now `PolynomialFunction`, and `FermiEdge2dFunc` is now `FermiEdge2dFunction`. The corresponding model names are unchanged.
- This change disables the use of guess_fit. All fitting must be performed in the syntax recommended by lmfit. Addition of a accessor or a convenience function for coordinate-aware fitting is planned in the next release. ([59163d5](https://github.com/kmnhan/erlabpy/commit/59163d5f0e000d65aa53690a51b6db82df1ce5f1))

  BREAKING CHANGE: This change disables the use of guess_fit. All fitting must be performed in the syntax recommended by lmfit. Addition of a accessor or a convenience function for coordinate-aware fitting is planned in the next release.

### Features

- **itool:** add copy code to PlotItem vb menu ([7b4f30a](https://github.com/kmnhan/erlabpy/commit/7b4f30ada21c5accc1d3824ad3d0f8097f9a99c1))

  For each plot in imagetool, a new 'copy selection code' button has been added to the right-click menu that copies the code that can slice the data to recreate the data shown in the plot.
- add 2D curvature, finally closes [#14](https://github.com/kmnhan/erlabpy/issues/14) ([7fe95ff](https://github.com/kmnhan/erlabpy/commit/7fe95ffcdf0531e456cfc97ae605467e4ae433c0))
- **plotting:** add N argument to plot_array_2d ([2cd79f7](https://github.com/kmnhan/erlabpy/commit/2cd79f7ee007058da09aff244cd75748698444ee))
- add scaled laplace ([079e1d2](https://github.com/kmnhan/erlabpy/commit/079e1d21201c7523877b06a0f04f7640027b0614))
- add gaussian filter and laplacian ([8628d33](https://github.com/kmnhan/erlabpy/commit/8628d336ff5b4219e4fd382293736e4cbf026d56))
- add derivative module with minimum gradient implementation ([e0eabde](https://github.com/kmnhan/erlabpy/commit/e0eabde60e6860c3827959b45be6d4f491918363))
- **fit:** directly base models on lmfit.Model ([59163d5](https://github.com/kmnhan/erlabpy/commit/59163d5f0e000d65aa53690a51b6db82df1ce5f1))

  BREAKING CHANGE: This change disables the use of guess_fit. All fitting must be performed in the syntax recommended by lmfit. Addition of a accessor or a convenience function for coordinate-aware fitting is planned in the next release.

### Bug Fixes

- **dynamic:** properly broadcast xarray input ([2f6672f](https://github.com/kmnhan/erlabpy/commit/2f6672f3b003792ecd98b4fbc99fb11fcc0efb8b))
- **fit.functions:** polynomial function now works for xarray input ([3eb80de](https://github.com/kmnhan/erlabpy/commit/3eb80dea31b6414fa9a694049b92b7334a4e10f5))
- **analysis.image:** remove critical typo ([fb7de0f](https://github.com/kmnhan/erlabpy/commit/fb7de0fc3ba9049c488a90bef8ee3c4feb935341))
- **analysis.image:** dtype safety of cfunc ([b4f9b17](https://github.com/kmnhan/erlabpy/commit/b4f9b17656c64be4cff876843ed0f3491d8310d4))
- set autodownsample off for colorbar ([256bf2d](https://github.com/kmnhan/erlabpy/commit/256bf2dc8c368d093a3578d7f9279b1ee4653534))
- disable itool downsample ([e626bba](https://github.com/kmnhan/erlabpy/commit/e626bba9fcd4fd31387ca3a07a9a33b7690f3645))

### Code Refactor

- **fit:** unify dynamic function names ([20d784c](https://github.com/kmnhan/erlabpy/commit/20d784c1d8fdcd786ab73b3ae03d3e331dc04df5))

  BREAKING CHANGE: `PolyFunc` is now `PolynomialFunction`, and `FermiEdge2dFunc` is now `FermiEdge2dFunction`. The corresponding model names are unchanged.
- update dtool to use new functions ([a6e46bb](https://github.com/kmnhan/erlabpy/commit/a6e46bb8b19512e438291afbbd5e0e9a4eb4fe87))
- **analysis.image:** add documentation and reorder functions ([340665d](https://github.com/kmnhan/erlabpy/commit/340665dc507a99acc7d56c46a2a2326fbb56b1e3))
- rename module to image and add citation ([b74a654](https://github.com/kmnhan/erlabpy/commit/b74a654e07d9f4522cee2db0b897f1ffcdb86e94))
- **dtool:** cleanup unused code ([f4abd34](https://github.com/kmnhan/erlabpy/commit/f4abd34bbf3130c0ec0fd2f9c830c8da43849f13))

### Performance

- **itool:** add explicit signatures to fastbinning ([62e1d51](https://github.com/kmnhan/erlabpy/commit/62e1d516f0260f661fe9cd8f1fae9cb81afbcabe))

  Speedup initial binning by providing explicit signatures.

## v1.6.5 (2024-04-03)

### Bug Fixes

- make imports work without optional pip dependencies ([b8ac11d](https://github.com/kmnhan/erlabpy/commit/b8ac11d8fb4379f70a39c817332382c352391a64))

## v1.6.4 (2024-04-03)

### Bug Fixes

- load colormaps only when igor2 is  available ([7927c7d](https://github.com/kmnhan/erlabpy/commit/7927c7db264bedb1a27b980d820d352f779b64c9))

## v1.6.3 (2024-04-03)

### Bug Fixes

- leave out type annotation for passing tests ([eb25008](https://github.com/kmnhan/erlabpy/commit/eb2500838820172529ee751b5d8a624c950f66d2))

## v1.6.2 (2024-04-03)

### Bug Fixes

- igor2 does not have to be installed on import time ([186727a](https://github.com/kmnhan/erlabpy/commit/186727ac8d50b662efeba8bee567cf1013ca936a))

## v1.6.1 (2024-04-03)

### Bug Fixes

- remove all pypi dependencies from pyproject.toml ([1b2fd55](https://github.com/kmnhan/erlabpy/commit/1b2fd5594f00bba8367419cd00919eba45cde5a7))

### Code Refactor

- remove ktool_old ([18ea072](https://github.com/kmnhan/erlabpy/commit/18ea0723fdf538bdbf2789ca73b2b962839ca3e5))

## v1.6.0 (2024-04-02)

### Features

- add mdctool ([a4976f9](https://github.com/kmnhan/erlabpy/commit/a4976f93cde51a41d667321a93dc2a90f23bddc3))

### Code Refactor

- remove deprecated function and dependencies ([4b9c7b1](https://github.com/kmnhan/erlabpy/commit/4b9c7b1629d99fbf0108ca33791d1bfd59632199))

## v1.5.2 (2024-04-01)

### Bug Fixes

- set values after setting bounds ([ab6d682](https://github.com/kmnhan/erlabpy/commit/ab6d682d0afafefcaec4c1ab6d673a39a75f40a6))
- proper patch all interpolator selection functions ([b91834e](https://github.com/kmnhan/erlabpy/commit/b91834e1b0be200bafb86ed3581f08cf1a5d42ef))
- make bz voronoi robust ([8259760](https://github.com/kmnhan/erlabpy/commit/8259760249be45892cd32f143b1b83aefe166c49))

### Code Refactor

- remove debug print statement in FastInterpolator class ([712bd2c](https://github.com/kmnhan/erlabpy/commit/712bd2ce90ad3534212d8a63c3fe10d780e243f5))
- add edge correction ([87adcef](https://github.com/kmnhan/erlabpy/commit/87adceffda2364f404de0860bfe8bf36b4cc1394))
- change variable name ([b68949e](https://github.com/kmnhan/erlabpy/commit/b68949ec59fd6bd7d7dad4ff9cc232b0e1ce4fba))
- make rotation transformations try fast interpolator first ([e0a7908](https://github.com/kmnhan/erlabpy/commit/e0a790833025f0c7e952ad17d120f46de3100555))
- update warning message ([af67c1a](https://github.com/kmnhan/erlabpy/commit/af67c1a507be35348b58862b6b51b92fac52781b))
- add several new accessors ([664e92a](https://github.com/kmnhan/erlabpy/commit/664e92a3e171512be26ea957df945e84134c880a))
- use new accessors and attrs ([8e1dee2](https://github.com/kmnhan/erlabpy/commit/8e1dee22d9d716f7e9bce29a1be3e68311494aa1))
- add qplot accessor ([cb9aa01](https://github.com/kmnhan/erlabpy/commit/cb9aa017bebd2ee6661f4eb87b988509d28a37a5))
- remove annotate_cuts ([004ee80](https://github.com/kmnhan/erlabpy/commit/004ee808dab13073cb3d2021d331767f6c28388a))
- dataloader cleanup ([fd97780](https://github.com/kmnhan/erlabpy/commit/fd977800a504256afd6018e9991b2d1e996277df))

## v1.5.1 (2024-03-28)

### Bug Fixes

- restore argname detection that was broken with namespace changes ([863b702](https://github.com/kmnhan/erlabpy/commit/863b702b6373f9a219a1e770aa49c71145371681))
- namespace collision ([10edcdc](https://github.com/kmnhan/erlabpy/commit/10edcdc8b06425c380ca6caa2d3f5f2be5c13733))
- followup namespace change ([4c5222c](https://github.com/kmnhan/erlabpy/commit/4c5222cc93196f0b6a75a0101107a37e73748eeb))

### Code Refactor

- allow offsetview upate chaining ([8d5ca4f](https://github.com/kmnhan/erlabpy/commit/8d5ca4f5b12c7d7060ea444773a9851f23db9850))

  This also means that _repr_html_ is automatically displayed when update or reset is called.
- improve consistency in accessors ([9596fd7](https://github.com/kmnhan/erlabpy/commit/9596fd723206f3e992fe00990f73364a61604cd6))

  Added setter method for configuration too.
- make prints consistent ([0021302](https://github.com/kmnhan/erlabpy/commit/002130224e3efc01615948a6443516e29d333cf5))
- change module names to prevent conflict with function names ([493a5aa](https://github.com/kmnhan/erlabpy/commit/493a5aab19c0d66851ca068e286a6aec92131e33))

  Cleanup erplot namespace and move tools to interactive.
- follow class naming conventions ([efb9610](https://github.com/kmnhan/erlabpy/commit/efb9610a864ef637f424c2f1b2871add7324b090))

## v1.5.0 (2024-03-27)

### Features

- add interactive tool to kspace accessor ([fb91cdb](https://github.com/kmnhan/erlabpy/commit/fb91cdb50229154c070df8dfaa80cddc8520ae6d))

### Code Refactor

- accessors are now registered upon package import ([d79fee2](https://github.com/kmnhan/erlabpy/commit/d79fee2a28dd5ee59bfc6bd1ce224a44c5f40a24))

## v1.4.1 (2024-03-26)

### Bug Fixes

- update package metadata ([ecfb88f](https://github.com/kmnhan/erlabpy/commit/ecfb88f2c23a7681e12d6f2dedcc316a28aa22c7))

  This should be classified as chore, but commiting as a fix to trigger CI

## v1.4.0 (2024-03-26)

### Features

- calculate kz in MomentumAccessor ([46979f9](https://github.com/kmnhan/erlabpy/commit/46979f907b120e5a4a88fdacd7d74a4b9dd41d6d))

  Add method that calculates kz array from given photon energy float
- make momentum conversion functions xarray compatible ([a7aa34b](https://github.com/kmnhan/erlabpy/commit/a7aa34ba983d3159c555ed66579d46eaf9e993aa))

## v1.3.1 (2024-03-25)

### Bug Fixes

- fixes [#12](https://github.com/kmnhan/erlabpy/issues/12) ([02b49a1](https://github.com/kmnhan/erlabpy/commit/02b49a1da7550ae2b07819e6ccde3dcf750fc527))

## v1.3.0 (2024-03-25)

### Features

- **io:** add new data loader plugin for DA30 + SES ([7a27a2f](https://github.com/kmnhan/erlabpy/commit/7a27a2f27d9658f1091aaa48bcc78dea562898d8))

### Bug Fixes

- **io:** properly handle registry getattr ([499526f](https://github.com/kmnhan/erlabpy/commit/499526fc1705bfbfbf8d3b80d50d65450dec7eae))

  This fixes an issue where _repr_html_ will fallback to __repr__.

  Additionally, `get` will now raise a KeyError instead of a ValueError.
