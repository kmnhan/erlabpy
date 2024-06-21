# CHANGELOG



## v2.8.1 (2024-06-21)

### Ci

* (**pre-commit**) pre-commit autoupdate ([`856aa0f`](https://github.com/kmnhan/erlabpy/commit/856aa0fe825eb94424a546c855ee8d8e941897a2))

  updates: - [github.com/astral-sh/ruff-pre-commit: v0.4.8 → v0.4.9](https://github.com/astral-sh/ruff-pre-commit/compare/v0.4.8...v0.4.9)

### Fix

* (**interactive.imagetool**) properly implement caching and linking from GUI ([`ffacdce`](https://github.com/kmnhan/erlabpy/commit/ffacdce93d1ff89e1be823317a6d59a400a6dee2))

* (**plotting.general**) pass DataArray to `func` argument to `plot_array` ([`ed76e64`](https://github.com/kmnhan/erlabpy/commit/ed76e64e45eb3ea93fba61380bc0d63864446fd3))

### Performance

* (**interactive.imagetool**) speedup file loading and saving ([`a6c869b`](https://github.com/kmnhan/erlabpy/commit/a6c869b7d6ce0419d84a46086004d451845c23e3))

  Use pickle to save and load files instead of `erlab.io.load_hdf5` and `erlab.io.save_as_hdf5`.

### Test

* add coverage for QThread ([`ed74df0`](https://github.com/kmnhan/erlabpy/commit/ed74df0be007c8d0caeff7d9b4d44072184ff3ee))


## v2.8.0 (2024-06-17)

### Documentation

* update user guide with ImageTool manager ([`21a2c09`](https://github.com/kmnhan/erlabpy/commit/21a2c09cba58479bdbc8c22bc83b6ab994b44ec6))

### Feature

* (**erlab.io.plugins.ssrl52**) changes to loader ([`512a89b`](https://github.com/kmnhan/erlabpy/commit/512a89b051911c88bafd59bdc9bd993ec727321a))

  The loader now promotes all attributes that varies during the scan to coordinates. Also, if the energy axis is given in kinetic energy and the work function is inferrable from the data attributes, the energy values are automatically converted to binding energy. This may require changes to existing code. This commit also includes a fix for hv-dependent swept cuts.

* (**erlab.io.dataloader**) reorder output coordinates ([`178edd2`](https://github.com/kmnhan/erlabpy/commit/178edd27f3e58387b12b7a7928a26e87766fa9be))

  Coordinates on the loaded data will now respect the order given in `name_map` and `additional_coords`, improving readability.

* (**interactive.imagetool**) add ImageTool window manager ([`b52d249`](https://github.com/kmnhan/erlabpy/commit/b52d2490ec61053b7b933e274a68a163761827ce))

  Start the manager with the cli command `itool-manager`. While running, all calls to `erlab.interactive.imagetool.itool` will make the ImageTool open in a separate process. The behavior can be controlled with a new keyword argument, `use_manager`.

* (**interactive.imagetool**) add undo and redo ([`e7e8213`](https://github.com/kmnhan/erlabpy/commit/e7e8213964c9739468b65e6a56dcc1a0d9d648e4))

  Adjustments made in ImageTool can now be undone with Ctrl+Z. Virtually all actions except window size change and splitter position change should be undoable. Up to 1000 recent actions are stored in memory.

* (**interactive.imagetool**) remember last used loader for each tool ([`eb0cd2f`](https://github.com/kmnhan/erlabpy/commit/eb0cd2f41992845988f5e500416ed98f5d078c14))

### Fix

* (**interactive.imagetool**) fix code generation behaviour for non-uniform coordinates ([`3652a21`](https://github.com/kmnhan/erlabpy/commit/3652a21cf126ebcde015d5b7373bf5d5a675b177))

### Refactor

* (**interactive.imagetool**) preparation for saving and loading state ([`eca8262`](https://github.com/kmnhan/erlabpy/commit/eca8262defe8d135168ca7da115d947bda3c1040))

### Test

* change coverage configuration ([`dca143f`](https://github.com/kmnhan/erlabpy/commit/dca143f72147a0a0c094b8f31c62379c04872363))

* add conftest.py ([`5d573be`](https://github.com/kmnhan/erlabpy/commit/5d573be103f20c1d288b9fdb2ee9398298a02877))


## v2.7.2 (2024-06-14)

### Fix

* (**erlab.io**) regression in handling getattr of dataloader ([`dd0a568`](https://github.com/kmnhan/erlabpy/commit/dd0a5680c6aed6e3b7ab391a10fbeb5c3cdc9c14))


## v2.7.1 (2024-06-14)

### Ci

* (**github**) schedule test runs ([`b5fa6c8`](https://github.com/kmnhan/erlabpy/commit/b5fa6c83b229c846e1fff39dbf38fc281f031bd1))

* (**pre-commit**) pre-commit autoupdate (#42) ([`a5a2de7`](https://github.com/kmnhan/erlabpy/commit/a5a2de7eae25071aaefaeddbabe7ff7c8bd33ac0))

  updates: - [github.com/astral-sh/ruff-pre-commit: v0.4.7 → v0.4.8](https://github.com/astral-sh/ruff-pre-commit/compare/v0.4.7...v0.4.8)
  Co-authored-by: pre-commit-ci[bot] &lt;66853113+pre-commit-ci[bot]@users.noreply.github.com&gt;

### Fix

* (**interactive.imagetool**) Integrate data loaders to imagetool ([`7e7ea25`](https://github.com/kmnhan/erlabpy/commit/7e7ea25a8fbe3a43222fbc7baedaa04c6522e24d))

  A new property called `file_dialog_methods` can be set in each loader which determines the method and name that is used in the file chooser window in imagetool.

* (**accessors.kspace**) `hv_to_kz` now accepts iterables ([`36770d7`](https://github.com/kmnhan/erlabpy/commit/36770d723b1e3592bf83750f7559603026059bb1))


## v2.7.0 (2024-06-09)

### Documentation

* improve getting started guide ([`fed2108`](https://github.com/kmnhan/erlabpy/commit/fed2108b77be5f83860e5967f487951fd1e77b67))

### Feature

* (**analysis.gold**) add function for quick resolution fitting ([`2fae1c3`](https://github.com/kmnhan/erlabpy/commit/2fae1c351f29b2fb1ceef39a69706b3f198e4659))

* (**analysis.fit**) Add background option to `MultiPeakModel` and `MultiPeakFunction` ([`2ccd8ad`](https://github.com/kmnhan/erlabpy/commit/2ccd8ad835cbc8de9764d2f8bbadda425babddb1))

### Fix

* (**erlab.io.plugins**) fix for hv-dependent data ([`d52152f`](https://github.com/kmnhan/erlabpy/commit/d52152f24807b9334ad5ffcc22c45a4af7a8d9ec))


## v2.6.3 (2024-06-07)

### Ci

* (**pre-commit**) pre-commit autoupdate (#41) ([`997e83d`](https://github.com/kmnhan/erlabpy/commit/997e83d1cd8c9b42d4072aa147ea55fcee7cb6ae))

  updates: - [github.com/astral-sh/ruff-pre-commit: v0.4.5 → v0.4.7](https://github.com/astral-sh/ruff-pre-commit/compare/v0.4.5...v0.4.7)
  Co-authored-by: pre-commit-ci[bot] &lt;66853113+pre-commit-ci[bot]@users.noreply.github.com&gt;

### Fix

* (**erlab.io.plugins**) support SSRL hv dependent data ([`1529b6a`](https://github.com/kmnhan/erlabpy/commit/1529b6a0af43f09c51691ad8bebf9208d421940a))

### Refactor

* cleanup namespace ([`847fbbe`](https://github.com/kmnhan/erlabpy/commit/847fbbe4975b507905dc85ca5ae75fe16f5f887e))


## v2.6.2 (2024-06-03)

### Fix

* (**interactive.imagetool**) fix regression with nonuniform data ([`67df972`](https://github.com/kmnhan/erlabpy/commit/67df9720193611816e2a562ce71d080fccbbec5e))


## v2.6.1 (2024-05-30)

### Fix

* re-trigger due to CI failure ([`b6d69b5`](https://github.com/kmnhan/erlabpy/commit/b6d69b556e3d0dbe6d8d71596e32d9d7cfdc5267))


## v2.6.0 (2024-05-30)

### Ci

* (**github**) disable parallel testing ([`fd024f5`](https://github.com/kmnhan/erlabpy/commit/fd024f5f1d6870ff6c30ee32ee8c3a708245a958))

* (**github**) enable color output ([`44071db`](https://github.com/kmnhan/erlabpy/commit/44071db669e66a39d025d23d63962399c59d9e1b))

* (**pre-commit**) pre-commit autoupdate ([`acb9c1c`](https://github.com/kmnhan/erlabpy/commit/acb9c1c31cfa8595512f30820b713c6bd1205983))

  updates: - [github.com/astral-sh/ruff-pre-commit: v0.4.4 → v0.4.5](https://github.com/astral-sh/ruff-pre-commit/compare/v0.4.4...v0.4.5)

### Documentation

* add `qsel.around` to indexing guide ([`1f95659`](https://github.com/kmnhan/erlabpy/commit/1f95659ae30a432d2fd91ec8c669c39a33b41f15))

### Feature

* (**interactive.imagetool**) add bin amount label to binning controls ([`7a7a692`](https://github.com/kmnhan/erlabpy/commit/7a7a692b881e4cc1bd49342f31f3fe50407d72b5))

* add accessor for selecting around a point ([`aa24457`](https://github.com/kmnhan/erlabpy/commit/aa244576fcfa17f71be0a765be8f270a6ae28080))

* (**accessors.fit**) add support for background models ([`550be2d`](https://github.com/kmnhan/erlabpy/commit/550be2deebf54fab77bef591ccbe059b5b219937))

  If one coordinate is given but there are two independent variables are present in the model,  the second one will be treated as the data. This makes the accessor compatible with y-dependent background models, such as the Shirley background provided in `lmfitxps`.

* (**io**) make the dataloader behavior more customizable ([`4824127`](https://github.com/kmnhan/erlabpy/commit/4824127181b4383788f6dbe5cbeae4b2060f1f4f))

  Now, a new `average_attrs` class attribute exists for attributes that would be averaged over multiple file scans. The current default just takes the attributes from the first file. This also works when you wish to demote a coordinate to an attribute while averaging over its values.
  For more fine-grained control of the resulting data attributes, a new method `combine_attrs` can be overridden to take care of attributes for scans over multiple files. The default behavior is to just use the attributes from the first file.

### Fix

* (**plotting**) make `gradient_fill` keep axis scaling ([`51507dd`](https://github.com/kmnhan/erlabpy/commit/51507dd966a0ce2db4aabff2aac8222bee184cf8))

### Refactor

* (**analysis.image**) add check for 2D and uniform inputs ([`22bb02d`](https://github.com/kmnhan/erlabpy/commit/22bb02dd8dfbd5eb6b5d577abe9138a769a079b3))

* try to fix synced itool garbage collection ([`932cc5a`](https://github.com/kmnhan/erlabpy/commit/932cc5a690dcebc92c65ea3f17081ac9f9c3ef8f))

  This only happens in GH actions, and it doesn&#39;t happen every time so it&#39;s hard to debug.

* create utils subpackage to host internal methods ([`3fa2873`](https://github.com/kmnhan/erlabpy/commit/3fa287386fc0e94e8a558e2f0e5520be869acb43))

  The parallel module is now part of utils, without a compatibiliity layer or deprecation warning since nobody is using the functions from parallel anyway.

* add deprecation warnings for utilities ([`5d375b8`](https://github.com/kmnhan/erlabpy/commit/5d375b8fe0766ea3f2c5fe2421937ce7309e3da5))

  All submodules named `utilities.py` have been renamed to `utils.py` for consistency. The old call to `utilities.py` will still work but will raise a warning. The modules will be removed on 3.0 release.

* rename `erlab.interactive.utilities` to `erlab.interactive.utils` ([`d9f1fb0`](https://github.com/kmnhan/erlabpy/commit/d9f1fb081be8d2e8710ec08421780f927341b71a))

* rename `erlab.analysis.utilities` to `erlab.analysis.utils` ([`ed81b62`](https://github.com/kmnhan/erlabpy/commit/ed81b6234bd2960da785875e0aaaf2e9e5e48f15))

* rename `erlab.io.utilities` to `erlab.io.utils` ([`6e0813d`](https://github.com/kmnhan/erlabpy/commit/6e0813d3873b09593ec9d539d72c7512fac77c70))

* (**io.plugins.merlin**) regard temperature as coordinate ([`2fda047`](https://github.com/kmnhan/erlabpy/commit/2fda04781961f2384c711a3b1c3c00ddaecaa617))


## v2.5.4 (2024-05-23)

### Fix

* (**io.plugins.maestro**) load correct beta for non deflector scans ([`5324c36`](https://github.com/kmnhan/erlabpy/commit/5324c362d7bdd1dcf0c21303f370fd2e77f12448))


## v2.5.3 (2024-05-22)

### Fix

* (**io.utilities**) `get_files` now only list files, not directories ([`60f9230`](https://github.com/kmnhan/erlabpy/commit/60f92307f94484361e0ba11b10a52be4c4cc05a1))

* (**accessors.fit**) add `make_params` call before determining param names, closes #38 ([`f1d161d`](https://github.com/kmnhan/erlabpy/commit/f1d161de089b93e16b2947b126ac075764d98f75))

* (**analysis.fit**) make some models more robust to DataArray input ([`afe5ddd`](https://github.com/kmnhan/erlabpy/commit/afe5ddd9d1e6796ba0261a147c2733d607916d81))

### Refactor

* add loader for ALS BL7 MAESTRO `.h5` files ([`4f33402`](https://github.com/kmnhan/erlabpy/commit/4f3340228ae2e1cbd8baf57d5d426043f5e28688))

* (**interactive**) add informative error message for missing Qt bindings ([`560615b`](https://github.com/kmnhan/erlabpy/commit/560615bb89d2646965d1a2a967133f0df08e3f6e))

* (**io**) rename some internal variables and reorder ([`76fe284`](https://github.com/kmnhan/erlabpy/commit/76fe284b4bc9f1e0c3cb94857a65599b07ee04df))

  Also added a check for astropy in FITS file related utility.


## v2.5.2 (2024-05-16)

### Ci

* (**github**) re-enable parallel tests and tweak coverage ([`0fd910f`](https://github.com/kmnhan/erlabpy/commit/0fd910f63d576942fbf8d66d71c468b00157ca19))

* (**github**) disable `.pyc` generation ([`54c7dd1`](https://github.com/kmnhan/erlabpy/commit/54c7dd1262db8fc3744cc154fe434672d2a7313b))

* (**codecov**) update config path ([`47a833d`](https://github.com/kmnhan/erlabpy/commit/47a833dc90aa9e9b03f3155c15aecab9149f1ec5))

### Documentation

* update README ([`d90441a`](https://github.com/kmnhan/erlabpy/commit/d90441ac0d397439d63698c35677c5afe333ca09))

### Fix

* make mathtext copy default to svg ([`2f6e0e5`](https://github.com/kmnhan/erlabpy/commit/2f6e0e558f251c846bc3dec39cd150391802460d))

* resolve MemoryError in prominent color estimation ([`3bdcd03`](https://github.com/kmnhan/erlabpy/commit/3bdcd0341c41b424ebbcb565b7cda0db839e4cb8))

  Due to numpy/numpy/#11879 changed the auto method to sqrt. This should also improve memory usage and speed, with little to no impact on the end result.

### Test

* fix clipboard and interactive tests ([`08d7448`](https://github.com/kmnhan/erlabpy/commit/08d7448f80eb5a00debec89e6a9e949fab1fdeba))

* add tests across multiple modules ([`cc35955`](https://github.com/kmnhan/erlabpy/commit/cc35955e2868d71fd522024c18d7e9cb168f98e1))

* add tests interactive tools and `plotting.atoms` ([`a1b5154`](https://github.com/kmnhan/erlabpy/commit/a1b5154c54e6ae6278038ec11c8305fc09193481))

* add tests for `goldtool` and `dtool` ([`99bac03`](https://github.com/kmnhan/erlabpy/commit/99bac03a6df882baa625aa5a065fd0be571d36a9))

* increase test speed ([`031d2e3`](https://github.com/kmnhan/erlabpy/commit/031d2e346636d64976ea96386ac8eba9e638546c))


## v2.5.1 (2024-05-15)

### Chore

* exclude merge commits from semantic release ([`9179cab`](https://github.com/kmnhan/erlabpy/commit/9179cab62a299b65992ed2621d9618bf7a9f35ee))

* (**deps**) add dask to optional dependencies ([`cdc90dd`](https://github.com/kmnhan/erlabpy/commit/cdc90dd785f8f7d32443e521e89b415a7575303a))

* add coverage configuration ([`9c1e6a5`](https://github.com/kmnhan/erlabpy/commit/9c1e6a5365922303e0119e9f07604ba6d89b539c))

### Ci

* (**github**) ci changes for testing and coverage ([`67550d6`](https://github.com/kmnhan/erlabpy/commit/67550d6ebbf6423052b22bc5224ad331d824d502))

  Parallelized tests for `gold.poly`, setup Qt testing, and added `.codecov.yml` config file.

* (**github**) update codecov script ([`5cab279`](https://github.com/kmnhan/erlabpy/commit/5cab2796d588bc40fc74f8cd130a1efad3b76b37))

* (**github**) update test script ([`7dbb384`](https://github.com/kmnhan/erlabpy/commit/7dbb384caa461992690893017cacb4bd72273149))

* (**pre-commit**) pre-commit autoupdate (#34) ([`acf3dc5`](https://github.com/kmnhan/erlabpy/commit/acf3dc5f4e7987619d8a9be53ad7717f17d12fec))

### Fix

* (**plotting**) fixes #35 ([`a67be68`](https://github.com/kmnhan/erlabpy/commit/a67be6869c2d25780f8a56794aad0386379202dd))

  Gradient fill disappears upon adding labels

* (**fit.models**) wrong StepEdgeModel guess with DataArray input ([`6778c8d`](https://github.com/kmnhan/erlabpy/commit/6778c8dd2c048b0cab67c6d3668b25b3f79a71da))

### Refactor

* (**plotting**) code cleanup ([`aef10e4`](https://github.com/kmnhan/erlabpy/commit/aef10e472a3ebc935711253e91124cfd87beb9cc))

### Test

* fix tests for interactive ([`57403f7`](https://github.com/kmnhan/erlabpy/commit/57403f70f61500e248d57584962c1a0df3a2defc))

* add tests for analysis.gold ([`61d4d36`](https://github.com/kmnhan/erlabpy/commit/61d4d36fb3f858ae9d096f2c8b3f9d090d8905ff))


## v2.5.0 (2024-05-13)

### Chore

* (**deps**) unpin PyQt6 ([`55a8ce5`](https://github.com/kmnhan/erlabpy/commit/55a8ce5e0c74c38c22a44f4f385c5ee87ee5fdcb))

### Ci

* (**pre-commit**) update commit messages ([`0321ec1`](https://github.com/kmnhan/erlabpy/commit/0321ec131b61e8bf6881f1f72fdc550937d96959))

### Documentation

* make view button point to github ([`f968c37`](https://github.com/kmnhan/erlabpy/commit/f968c3777ed796011338f7ff14b0561d9fada05e))

### Feature

* extended interactive accessor ([`f6f19ab`](https://github.com/kmnhan/erlabpy/commit/f6f19abd8edfb33585b5e19040a2ebaff39b2b70))

  The `qshow` accessor has been updated so that it calls `hvplot` (if installed) for data not supported by ImageTool.
  Also, the `qshow` accessor has been introduced to Datasets. For valid fit result datasets produced by the `modelfit` accessor, calling `qshow` will now show an `hvplot`-based interactive visualization of the fit result.

* (**itool**) make itool accept Datasets ([`f77b699`](https://github.com/kmnhan/erlabpy/commit/f77b699abdf312a23832611052d67e8c4c8d4930))

  When a Dataset is passed to `itool`, each data variable will be shown in a separate ImageTool window.

* (**analysis.image**) add multidimensional Savitzky-Golay filter ([`131b32d`](https://github.com/kmnhan/erlabpy/commit/131b32d9e562693e98a2f9e45cf6db4635405b44))

### Fix

* (**itool**) add input data dimension check ([`984f2db`](https://github.com/kmnhan/erlabpy/commit/984f2db0f69db2b5b99211728840447d9617f8bf))

* (**analysis.image**) correct argument order parsing in some filters ([`6043413`](https://github.com/kmnhan/erlabpy/commit/60434136224c0875ed8fba41d24e32fc6868127c))

* (**interactive**) improve formatting for code copied to clipboard ([`d8b6d91`](https://github.com/kmnhan/erlabpy/commit/d8b6d91a4d2688486886f2464426935fdf8cabc2))

### Refactor

* (**plotting**) update `clean_labels` to use `Axes.label_outer` ([`0c64756`](https://github.com/kmnhan/erlabpy/commit/0c647564c6027f5b60f9ff288f13019e0e5933b6))


## v2.4.2 (2024-05-07)

### Fix

* (**ktool**) resolve ktool initialization problem, closes #32 ([`e88a58e`](https://github.com/kmnhan/erlabpy/commit/e88a58e6aaed326af1a68aa33322d6ea9f0e800d))

* (**itool**) disable flag checking for non-numpy arrays ([`da6eb1d`](https://github.com/kmnhan/erlabpy/commit/da6eb1db9e81d51b52d4b361de938bcf7ba45e68))

### Unknown

* [pre-commit.ci] pre-commit autoupdate ([`ec62bea`](https://github.com/kmnhan/erlabpy/commit/ec62bea6af2b4074f77ef11a2ddf82b7b7a4db33))

  updates:
  - [github.com/astral-sh/ruff-pre-commit: v0.4.2 → v0.4.3](https://github.com/astral-sh/ruff-pre-commit/compare/v0.4.2...v0.4.3)


## v2.4.1 (2024-05-03)

### Fix

* (**plotting**) fix wrong regex in `scale_units` ([`d7826d0`](https://github.com/kmnhan/erlabpy/commit/d7826d0269214dfd822a4d0293e42a9840015ce8))

* fix bug in `modelfit` parameter concatenation ([`edaa556`](https://github.com/kmnhan/erlabpy/commit/edaa5566c6e3817e1d9220f7a96e8e731cf7eede))

* (**itool**) ensure DataArray is readable on load ([`5a0ff00`](https://github.com/kmnhan/erlabpy/commit/5a0ff002802cdf5bd3ceb34f9cddc53c9674e7bd))


## v2.4.0 (2024-05-02)

### Chore

* (**deps**) update dependencies ([`37c1b4b`](https://github.com/kmnhan/erlabpy/commit/37c1b4bf838eeaabeda9ee255fa22902f6ce955b))

  Numbagg is now an optional dependency.

### Documentation

* improve documentation ([`8f23f99`](https://github.com/kmnhan/erlabpy/commit/8f23f9974672a8432ff1e6bd3fdc6ed01a82c937))

* improve io documentation ([`4369d23`](https://github.com/kmnhan/erlabpy/commit/4369d23310c0ad91b41952d9ead4b87458fe107e))

* fix PyQt version pinning to resolve build failures on Qt6.7 ([`433ee9e`](https://github.com/kmnhan/erlabpy/commit/433ee9ed13ea254c6d1e1c83531c9c89f9ff9698))

### Feature

* (**imagetool**) add method to update only the values ([`ca40fe4`](https://github.com/kmnhan/erlabpy/commit/ca40fe41a0320fd7843c86f95b68f8b6e19adca8))

* add interpolation along a path ([`7366ec4`](https://github.com/kmnhan/erlabpy/commit/7366ec4db600617e585c724d05aafea387456cf2))

  The `slice_along_path` function has been added to `analysis.interpolate`, which enables easy interpolation along a evenly spaced path that is specified by its vertices and step size. The path can have an arbitrary number of dimensions and points.

### Fix

* (**io**) remove direct display call in interactive summary ([`d44b3a5`](https://github.com/kmnhan/erlabpy/commit/d44b3a56aecfb054a38d944c5c8b7f45d362cf3b))

  This was causing duplicated plots.

* (**plotting**) add some validation checks to `plot_array` ([`2e0753c`](https://github.com/kmnhan/erlabpy/commit/2e0753c90ffbe6fdd05af210ac6a4dbfa9aba899))

  The functions `plot_array` and `plot_array_2d` now checks if the input array coordinates are uniformly spaced. If they are not, a warning is issued and the user is informed that the plot may not be accurate.

* (**plotting**) increase default colorbar size ([`3208399`](https://github.com/kmnhan/erlabpy/commit/32083990e9e77df6e94b2b0836bc1f9764cfaaf7))

  The default `width` argument to `nice_colorbar` is changed to 8 points. This ensures visibility in subplots, especially when constrained layout is used.

* delay interactive imports until called ([`ad15910`](https://github.com/kmnhan/erlabpy/commit/ad15910f921cb5ffffc388e7a5e02832935f8547))

### Refactor

* various cleanup ([`2b38397`](https://github.com/kmnhan/erlabpy/commit/2b383970b602507b6efedbf396f14d470db60d8f))

  Improve docstring formatting and tweak linter settings

### Style

* remove % formatting ([`ae18a34`](https://github.com/kmnhan/erlabpy/commit/ae18a341f36542c2f39d72b0dd975dbe640c7e24))

### Unknown

* [pre-commit.ci] pre-commit autoupdate ([`3c351cc`](https://github.com/kmnhan/erlabpy/commit/3c351cc255dc31010b2c5fab2d134531f00a4dac))

  updates:
  - [github.com/astral-sh/ruff-pre-commit: v0.4.1 → v0.4.2](https://github.com/astral-sh/ruff-pre-commit/compare/v0.4.1...v0.4.2)


## v2.3.2 (2024-04-25)

### Chore

* pin PyQt due to tests failing on Qt6.7 ([`9dee08c`](https://github.com/kmnhan/erlabpy/commit/9dee08cd8a9ac2ac16b9b9a1cd96ce537b7104cb))

### Fix

* (**io**) make summary caching togglable ([`99b8e22`](https://github.com/kmnhan/erlabpy/commit/99b8e221e75db73382bf599170c58d8a68ca049e))

  Also fixes a bug where interactive summary plots were duplicated

* (**io**) data loader related fixes ([`da08e90`](https://github.com/kmnhan/erlabpy/commit/da08e9076e59895b35c393c8e2556c3592adf4a5))

  DA30 dataloader now preserves case for attribute names from zip files. Post processing for datasets now works properly


## v2.3.1 (2024-04-25)

### Chore

* (**deps**) make `iminuit` and `superqt` optional ([`1bbcc24`](https://github.com/kmnhan/erlabpy/commit/1bbcc24268312f8c285df0774e1e5d5c8c775650))

* (**deps**) reduce dependencies ([`6a03518`](https://github.com/kmnhan/erlabpy/commit/6a0351859ace99dbfb4f251ccbb78e581d6f7218))

* (**github**) update issue templates ([`6a2dd50`](https://github.com/kmnhan/erlabpy/commit/6a2dd504ac05c5c47a39499dab991d44daee57f9))

* (**deps**) update lmfit dependencies to include &gt;1.3.0 ([`942a810`](https://github.com/kmnhan/erlabpy/commit/942a810783fb574cc36a446333be09a82b1d22ae))

### Fix

* (**interactive**) keep pointer for imagetool, fix typing issues ([`c98c38e`](https://github.com/kmnhan/erlabpy/commit/c98c38ea11bce50ed9bfd8d374064bb2b1659d0c))

* (**kspace**) allow explicit coordinate kwargs ([`fe47efc`](https://github.com/kmnhan/erlabpy/commit/fe47efcde941767c02b582ce8b29d4b3678fd843))

### Refactor

* move `characterization` to `io` ([`9c30f1b`](https://github.com/kmnhan/erlabpy/commit/9c30f1b7df51460f502dcbf999e3fac34be1cf99))

* make zip strict (ruff B905) ([`78bf5f5`](https://github.com/kmnhan/erlabpy/commit/78bf5f5a2db52c14ccf5bfd3c83659ca53c4a408))

### Style

* add mypy compatible type hints ([`c97724d`](https://github.com/kmnhan/erlabpy/commit/c97724dcd9095a3cdc1842e5afb1f29b3c472c45))

### Unknown

* Merge branch &#39;main&#39; into dev ([`184afb0`](https://github.com/kmnhan/erlabpy/commit/184afb023dc704b4ab6ffe8ef5c098c19ca19084))

* [pre-commit.ci] pre-commit autoupdate ([`43bfbab`](https://github.com/kmnhan/erlabpy/commit/43bfbabc93720a3afb232a75de05d43aa2567ace))

  updates:
  - [github.com/astral-sh/ruff-pre-commit: v0.3.7 → v0.4.1](https://github.com/astral-sh/ruff-pre-commit/compare/v0.3.7...v0.4.1)


## v2.3.0 (2024-04-22)

### Chore

* cleanup directory ([`40e1f8d`](https://github.com/kmnhan/erlabpy/commit/40e1f8dcce66d53f9de6fd5905cef3c690633c98))

* customize issue template chooser ([`0aa7617`](https://github.com/kmnhan/erlabpy/commit/0aa7617e7794449756df0abfc3260c6e545f97a3))

* update issue templates ([`5dfb250`](https://github.com/kmnhan/erlabpy/commit/5dfb25098a6843f378303486499a89489e00ee76))

### Ci

* (**github**) update test requirements ([`21080cc`](https://github.com/kmnhan/erlabpy/commit/21080ccb5553e3c753db29d8fa372f6000ea59a7))

* (**readthedocs**) rollback to 3.11 due to bug in python 3.12.0 and 3.12.1 ([`5eb6152`](https://github.com/kmnhan/erlabpy/commit/5eb615266ed78473583ca53a2e38f11ee58fc13f))

### Documentation

* update documentation ([`ab32920`](https://github.com/kmnhan/erlabpy/commit/ab3292032bcfa6e6e39e3b367090909a09ea813b))

  Updated requirements, contrib page, fitting guide. Source button now links to github.

* fix typo in getting-started.rst ([`3304a18`](https://github.com/kmnhan/erlabpy/commit/3304a182a7e46b7799977af93966bae9c5d3a95c))

* fix typos and add compat notes to getting-started ([`3bb2ac5`](https://github.com/kmnhan/erlabpy/commit/3bb2ac576ad029cacaf2d2043a3bf57a184ace24))

* add masking section, WIP ([`0844f6a`](https://github.com/kmnhan/erlabpy/commit/0844f6a58b971684a5cc02867dd36f61263e8c6c))

* add imagetool to user guide ([`84d3586`](https://github.com/kmnhan/erlabpy/commit/84d35866f9c221f82af7a4dfdcfbd871a88e4eaf))

* update io and plotting documentation and cleanup contributing ([`b4c4f7c`](https://github.com/kmnhan/erlabpy/commit/b4c4f7ce028cc03e87d4a54fefc72c8a610ce0ac))

* update badges ([`5e373f4`](https://github.com/kmnhan/erlabpy/commit/5e373f4f6a4f45ba90600f18a76dfd4830bb075e))

* enable latexpdf generation ([`7469271`](https://github.com/kmnhan/erlabpy/commit/746927115faba6c76a791c94aee3ef026fd11bf5))

### Feature

* (**kspace**) rewrite conversion with `xarray.apply_ufunc` ([`156cef8`](https://github.com/kmnhan/erlabpy/commit/156cef830582e01dc378a7437a0c85f4c7efc077))

  Momentum conversion now relies on xarray broadcasting for all computations, and objects with extra dimensions such as temperature can be automatically broadcasted. Dask arrays can also be converted.

* (**exampledata**) enable specifying seed for noise rng ([`aa542e8`](https://github.com/kmnhan/erlabpy/commit/aa542e8c288ff1ca64820960f469b2c244ca5c95))

* (**interpolate**) enable fast interpolation for 1D arrays ([`ff333a0`](https://github.com/kmnhan/erlabpy/commit/ff333a05803d7079034e36f2e1dc3d22d0b686f7))

* make both arguments optional for loader_context ([`6780197`](https://github.com/kmnhan/erlabpy/commit/6780197f68abfe7a9edbda951d804a9bc5ba56e9))

* (**kspace**) automatically detect kinetic energy axis and convert to binding ([`bbde447`](https://github.com/kmnhan/erlabpy/commit/bbde44717155d1dd9ffefbc286da32b0bfac2180))

* add more output and parallelization to fit accessor ([`59b35f5`](https://github.com/kmnhan/erlabpy/commit/59b35f53f3ef7f518aec92e05854dba42ddba56f))

  Allows dictionary of `DataArray`s as parameter to fit accessor. Now, the return `Dataset` contains the data and the best fit array. Relevant tests have been added.

* add callable fit accessor using apply_ufunc ([`11e3546`](https://github.com/kmnhan/erlabpy/commit/11e35466fec158e40d0e8e738dd81ed10225d83c))

  Add a `Dataset.modelfit` and `DataArray.modelfit` accessor with similar syntax and output as `Dataset.curvefit`. Closes #22

* add option to plot_array_2d so that users can pass non-normalized color array ([`74cf961`](https://github.com/kmnhan/erlabpy/commit/74cf961532a50d9c324189318460a9f840291a85))

* (**analysis.gold**) add option to normalize energy axis in fitting ([`3dffad6`](https://github.com/kmnhan/erlabpy/commit/3dffad65993520c4b9a9a3afd6be85671bac9d3a))

  This improves performance and results when eV is large like ~100eV.

### Fix

* (**exampledata**) change noise generation parameters ([`b213f11`](https://github.com/kmnhan/erlabpy/commit/b213f1151ed2555fc80374e9ebe3fc0856a13948))

* (**fit**) make FermiEdge2dModel compatible with flattened meshgrid-like input arrays ([`c0dba26`](https://github.com/kmnhan/erlabpy/commit/c0dba261670774862f2dfae62c770bbab81aac2f))

* fix progress bar for parallel objects that return generators ([`23d41b3`](https://github.com/kmnhan/erlabpy/commit/23d41b31a3f3ee6c7343d471f7cec34dc374bafa))

  Tqdm imports are also simplified. We no longer handle `is_notebook` ourselves, but just import from `tqdm.auto`

* (**plotting**) fix 2d colormaps ([`8299576`](https://github.com/kmnhan/erlabpy/commit/8299576ce3cbcbaec106bef952c6df148bb7ca18))

  Allow images including nan to be plotted with gen_2d_colormap, also handle plot_array_2d colorbar aspect

### Refactor

* fix some type hints ([`2dfa5e1`](https://github.com/kmnhan/erlabpy/commit/2dfa5e1b4582e00d0631376ee32aa7d0b1b945b6))

* (**example**) move exampledata from interactive to io ([`1fc7e6c`](https://github.com/kmnhan/erlabpy/commit/1fc7e6c22ce477fe7ebbd8b0c6844d1a85df3fcf))

  Also add sample data generation for fermi edge

* refactor accessors as submodule ([`9fc37bd`](https://github.com/kmnhan/erlabpy/commit/9fc37bd4825de519e4c4b6e30e9e32bf9392ed2d))

* rewrite either_dict_or_kwargs with public api ([`34953d1`](https://github.com/kmnhan/erlabpy/commit/34953d10b6fd67720b1c29dbed1ab7a24e4d3060))

* move correct_with_edge from era.utilities to era.gold ([`08a906f`](https://github.com/kmnhan/erlabpy/commit/08a906ff61a74febc0f47ed08ac24e7a4cd0977f))

  Calling from utilities will now raise a DeprecationWarning. The erlab.analysis namespace is unchanged, so the affect will be minimal.

* qsel now raises a warning upon scalar indexing outside coordinate bounds ([`d6ed628`](https://github.com/kmnhan/erlabpy/commit/d6ed628111be8ac594d3a1b83cc2785a31e3f06e))

### Unknown

* Merge remote-tracking branch &#39;origin/main&#39; into dev ([`fd8e1ad`](https://github.com/kmnhan/erlabpy/commit/fd8e1ad14b345664289810c4c8a605df6b299c3a))

* [pre-commit.ci] pre-commit autoupdate ([`7e3a89e`](https://github.com/kmnhan/erlabpy/commit/7e3a89e22bb5491f7d4a2bcb21f5a7baeead8773))

  updates:
  - [github.com/astral-sh/ruff-pre-commit: v0.3.5 → v0.3.7](https://github.com/astral-sh/ruff-pre-commit/compare/v0.3.5...v0.3.7)


## v2.2.2 (2024-04-15)

### Chore

* cleanup pyproject.toml ([`0331132`](https://github.com/kmnhan/erlabpy/commit/033113247ea5d8fa8bc4afb9513b349b95080bed))

* (**deps**) add ipywidgets to optional dependency group viz ([`0062966`](https://github.com/kmnhan/erlabpy/commit/00629663742ada02ade556cb19ca0b14bd864fec))

### Ci

* (**readthdocs**) update build python version and add zip format ([`b2cc6fc`](https://github.com/kmnhan/erlabpy/commit/b2cc6fc1f732bab8904f90e10e194d4dffee5d57))

### Documentation

* (**io**) add tutorial for writing advanced plugins ([`11f289e`](https://github.com/kmnhan/erlabpy/commit/11f289edd451e10773d99ae1c9fc47cde22b06dc))

* add ipywidgets to intersphinx mapping ([`0ee46f8`](https://github.com/kmnhan/erlabpy/commit/0ee46f8c6d783f2ee63cad807abf5e8582cfaa31))

### Fix

* (**io**) unify call signature for summarize ([`e2782c8`](https://github.com/kmnhan/erlabpy/commit/e2782c898d5aaaa1443b2bc82bb61fb40a28d232))

* resolve failing tests due to changes in sample data generation ([`80f0045`](https://github.com/kmnhan/erlabpy/commit/80f004574950834e42dbfa7677031d0f9f113bda))

* (**interactive.exampledata**) properly generate 2D data ([`825260c`](https://github.com/kmnhan/erlabpy/commit/825260c8ceb0a79b8c071750003529b91cda3573))

### Performance

* (**io**) speedup merlin summary generation by excluding duplicates ([`d6b4253`](https://github.com/kmnhan/erlabpy/commit/d6b42537ce48232b5112daef8f31e5cf86ea921a))

### Refactor

* (**io**) allow for more complex setups ([`f67b2e4`](https://github.com/kmnhan/erlabpy/commit/f67b2e4c7b092b7ca2db00ce02a23647879c514b))

  LoaderBase.infer_index now returns a second argument, which is a dictionary containing optional keyword arguments to load.

* (**io**) provide rich interactive summary ([`b075a9e`](https://github.com/kmnhan/erlabpy/commit/b075a9ee59b61892462fc475e78b036a54408099))

* (**io**) include &#34;Path&#34; column in ssrl loader summary ([`ae1d8ae`](https://github.com/kmnhan/erlabpy/commit/ae1d8aee051aa71563f6a6009ce9672e56edfae7))

* (**io**) improve array formatting in summary ([`1718529`](https://github.com/kmnhan/erlabpy/commit/171852957db7fe53ff6a5c5c5f843530078d4b46))

### Test

* add test for dataloader ([`64cde09`](https://github.com/kmnhan/erlabpy/commit/64cde099dbb13d8b148e67d1fe23a8849041dae4))


## v2.2.1 (2024-04-14)

### Chore

* (**deps**) pin lmfit&lt;1.3.0 ([`915fc60`](https://github.com/kmnhan/erlabpy/commit/915fc60e8e7e8a2dfc9a56bbc1afd1c737bcc3d5))

### Documentation

* rephrase some docstrings ([`e67597c`](https://github.com/kmnhan/erlabpy/commit/e67597c90e009748d1bd39c43c03a8cc1b439840))

* add link to changelog ([`fbb6d32`](https://github.com/kmnhan/erlabpy/commit/fbb6d3254ae254278dec74b2ce9965e20a4dc88d))

* add ipywidgets as requirement ([`41024eb`](https://github.com/kmnhan/erlabpy/commit/41024ebbef02609d9a2fc70c4630fec06aa96012))

* temporarily pin lmfit&lt;1.3.0 to build docs ([`6b86ac2`](https://github.com/kmnhan/erlabpy/commit/6b86ac2f89c2a822753f3fbe106eb5dfaa2cb22c))

### Fix

* (**fit**) add sigma and amplitude expressions to MultiPeakModel parameters ([`3f6ba5e`](https://github.com/kmnhan/erlabpy/commit/3f6ba5e84922129296183e02255506df73da0276))

* (**fit.minuit**) properly handle parameters constrained with expressions ([`d03f012`](https://github.com/kmnhan/erlabpy/commit/d03f012b4fde92f445a24657dca1fb5b3600fa45))

### Refactor

* set informative model name for MultiPeakModel ([`d14ee9d`](https://github.com/kmnhan/erlabpy/commit/d14ee9d6ac7962207700de50039a5b7a858fea6a))

* add gaussian and lorentzian for consistency ([`07c0dfb`](https://github.com/kmnhan/erlabpy/commit/07c0dfb9ecfb882e4f5f0ccfe942c1a835b613b2))

### Test

* add tests for fit models ([`3f9125c`](https://github.com/kmnhan/erlabpy/commit/3f9125ce19a4a30dd31b9d039d6614a8cae19966))


## v2.2.0 (2024-04-12)

### Documentation

* improve fitting documentation ([`9e0a106`](https://github.com/kmnhan/erlabpy/commit/9e0a10611a32ac75798e68f864cff55b5661330f))

* add curve fitting guide ([`ff9743c`](https://github.com/kmnhan/erlabpy/commit/ff9743c2203eb773af6bdb8d88426907f4300924))

* add docstrings to plotting.colors ([`1a15a70`](https://github.com/kmnhan/erlabpy/commit/1a15a706aa2fd591a18401ea53f950005391c88f))

### Feature

* enable component evaluation for MultiPeakModel ([`8875b74`](https://github.com/kmnhan/erlabpy/commit/8875b7443d26313156fcdcc43586d40af4ff4f00))

* (**analysis.fit**) add BCS gap equation and Dynes formula ([`f862aa4`](https://github.com/kmnhan/erlabpy/commit/f862aa4af4d2ba470f1ea074fc90442d9b18b336))

### Fix

* curvefittingtool errors ([`9abb99c`](https://github.com/kmnhan/erlabpy/commit/9abb99c35633bc722469276d4837a2372c132042))

### Refactor

* cleanup fit namespace ([`906aa99`](https://github.com/kmnhan/erlabpy/commit/906aa99193f78577e705218b2d6c22378611f84b))

* rename ExtendedAffineBroadenedFD to FermiEdgeModel ([`a98aa82`](https://github.com/kmnhan/erlabpy/commit/a98aa82bcbdf22ff8a156d800e336653f9afba07))

* (**interactive**) exclude bad colormaps ([`877c915`](https://github.com/kmnhan/erlabpy/commit/877c915def6eb3dddb3862d6ac64c8c70f456ad3))


## v2.1.3 (2024-04-11)

### Fix

* (**interactive**) update data load functions used in imagetool ([`c3abe35`](https://github.com/kmnhan/erlabpy/commit/c3abe3517046ed603a9221de38b22257322d3a51))


## v2.1.2 (2024-04-11)

### Documentation

* update syntax ([`2b72991`](https://github.com/kmnhan/erlabpy/commit/2b7299150a50af38e6e05d5f9690558cbeb7a9ad))

* improve intro pages ([`ec2a4f8`](https://github.com/kmnhan/erlabpy/commit/ec2a4f816e1ad0ed1ad154d6544e91bea1b5d9c5))

* (**io**) add complete data loading examples ([`63e88c4`](https://github.com/kmnhan/erlabpy/commit/63e88c40b14584214cd45bd0258f8ef7a32d716c))

* (**io**) simplify flowchart ([`355d023`](https://github.com/kmnhan/erlabpy/commit/355d02374e11f7a78b87f8d3159f288a1d15d22d))

### Fix

* (**io**) prevent specifying invalid data_dir ([`701b011`](https://github.com/kmnhan/erlabpy/commit/701b011339ecba657a0f4a14e2fef19adeb4bf2b))

* (**io**) fixes merlin summary data type resolving ([`a91ad3d`](https://github.com/kmnhan/erlabpy/commit/a91ad3d4387a23d25ac1b208cba8217e67efbec0))

* (**io**) fix summary loading ([`a5dd84a`](https://github.com/kmnhan/erlabpy/commit/a5dd84af9eec0f835b3116bc7c470e57ef3f3e02))


## v2.1.1 (2024-04-10)

### Documentation

* (**io**) improve docstrings and user guide ([`8e69abb`](https://github.com/kmnhan/erlabpy/commit/8e69abb37a99818081bf2e03453d64e1b48b16ab))

* update io documentation ([`b0d2d01`](https://github.com/kmnhan/erlabpy/commit/b0d2d01d0dec2ec8180cb5d7da2034900d0d0aba))

* change reference format ([`44e159a`](https://github.com/kmnhan/erlabpy/commit/44e159af6102f42182e4f705c18bacff1add7972))

* add missing type annotations and docstrings ([`b8c7471`](https://github.com/kmnhan/erlabpy/commit/b8c747111663c07b441a3cbf0e11652c2f5cac49))

### Fix

* (**io**) enable specifying data_dir in loader context manager ([`37913b8`](https://github.com/kmnhan/erlabpy/commit/37913b80a1d7c6313a5b6cc4a3ab614565274c81))

* (**io**) allow loader_class aliases to be None ([`7eae2eb`](https://github.com/kmnhan/erlabpy/commit/7eae2ebf13f972d368ddb9922a71fd3bbed014e5))

### Refactor

* remove igor2 import checking ([`b64d8f7`](https://github.com/kmnhan/erlabpy/commit/b64d8f7fe22ebc1c4818e26f93f864fd402bbd05))

* (**io**) default to always_single=True ([`007bb3b`](https://github.com/kmnhan/erlabpy/commit/007bb3b2703a647856c0a85e89075cf6572d263a))

### Style

* sort __all__ and change linter configuration ([`c07262e`](https://github.com/kmnhan/erlabpy/commit/c07262eb647f17638eec77829a12a223f88b09d5))

* apply perf lint and more ([`9cb4242`](https://github.com/kmnhan/erlabpy/commit/9cb424222e75360cf7240ce3325a63169ea67911))

### Test

* refactor directory structure ([`895ea0d`](https://github.com/kmnhan/erlabpy/commit/895ea0da46b4ed1ddcb81ff5dbff15ed20c7377b))


## v2.1.0 (2024-04-09)

### Chore

* update changelog template ([`46a79e5`](https://github.com/kmnhan/erlabpy/commit/46a79e53c3bd6ce358d0fbf1a632d947671444c1))

### Ci

* (**pre-commit**) merge pull request #18 from kmnhan/pre-commit-ci-update-config ([`7018fd3`](https://github.com/kmnhan/erlabpy/commit/7018fd3cc9775db47478fe0782aa1fafd30f83a1))

  [pre-commit.ci] pre-commit autoupdate

### Documentation

* improve io guide ([`28a2961`](https://github.com/kmnhan/erlabpy/commit/28a296131752e1df07d80c778e324adc2ef3746c))

* add docstring for undocumented io functions ([`3583aad`](https://github.com/kmnhan/erlabpy/commit/3583aadbb1c707f11842ec4c154e8bcb99723056))

* change directory structure, rename contributing guide ([`0b3d734`](https://github.com/kmnhan/erlabpy/commit/0b3d734b4f92cb9497bdfa7f133cbc66e6d99fb1))

* update development documentation ([`38efae6`](https://github.com/kmnhan/erlabpy/commit/38efae6c591aa90dc6e1d280565df5c1dd1a004c))

* update installation instructions to include conda-forge ([`c0ca81d`](https://github.com/kmnhan/erlabpy/commit/c0ca81d4c17fd7d97d1d684f4be8e5a4d49cc271))

### Feature

* (**interactive**) overhaul dtool ([`8e5ec38`](https://github.com/kmnhan/erlabpy/commit/8e5ec3827dd2bd52475d454d5c5ef8aef7d665aa))

  Now supports interpolation, copying code, opening in imagetool, and 2D laplacian method.

* (**interactive**) improve code generation ([`7cbe857`](https://github.com/kmnhan/erlabpy/commit/7cbe8572272f6c84a486599a990098ce8e3ff754))

  Automatically shortens code and allows literals in kwargs

* (**interactive**) extend xImageItem, add right-click menu to open imagetool ([`2b5bb2d`](https://github.com/kmnhan/erlabpy/commit/2b5bb2dfc3d4173d950135306b3b30a018c6d389))

### Fix

* sign error in minimum gradient ([`c45be0c`](https://github.com/kmnhan/erlabpy/commit/c45be0cf1a025c67e8af959ff83a9339cddbaaaa))

* (**analysis.image**) normalize data for mingrad output for numerical stability ([`0fc3711`](https://github.com/kmnhan/erlabpy/commit/0fc3711a521ffb0cbb4f5206c06d923eced1200c))

### Refactor

* (**io**) validation now defaults to warning instead of raising an error ([`8867a07`](https://github.com/kmnhan/erlabpy/commit/8867a07304129beda749fa82d3909bf920fdb975))

### Style

* sort imports with ruff ([`81efec9`](https://github.com/kmnhan/erlabpy/commit/81efec9d3937d3ef9be1287d6275246682de296d))

* avoid trailing whitespace in changelog ([`aafc441`](https://github.com/kmnhan/erlabpy/commit/aafc441003cb679624ee4521f97da8c490ebdcf4))

### Test

* fix tests according to minimum gradient behaviour change ([`41290f2`](https://github.com/kmnhan/erlabpy/commit/41290f210557ce48e029c9c4926f7a61f09bdd97))

### Unknown

* [pre-commit.ci] pre-commit autoupdate ([`1dc0de8`](https://github.com/kmnhan/erlabpy/commit/1dc0de892920d3487e6c85de700d9b8400930dd3))

  updates:
  - [github.com/pre-commit/pre-commit-hooks: v4.5.0 → v4.6.0](https://github.com/pre-commit/pre-commit-hooks/compare/v4.5.0...v4.6.0)


## v2.0.0 (2024-04-08)

### Breaking

* (**fit**) unify dynamic function names ([`20d784c`](https://github.com/kmnhan/erlabpy/commit/20d784c1d8fdcd786ab73b3ae03d3e331dc04df5))

  BREAKING CHANGE: `PolyFunc` is now `PolynomialFunction`, and `FermiEdge2dFunc` is now `FermiEdge2dFunction`. The corresponding model names are unchanged.

* (**fit**) directly base models on lmfit.Model ([`59163d5`](https://github.com/kmnhan/erlabpy/commit/59163d5f0e000d65aa53690a51b6db82df1ce5f1))

  BREAKING CHANGE: This change disables the use of guess_fit. All fitting must be performed in the syntax recommended by lmfit. Addition of a accessor or a convenience function for coordinate-aware fitting is planned in the next release.

### Build

* add templates for changelog and release notes ([`be72b24`](https://github.com/kmnhan/erlabpy/commit/be72b245f1194d1bd894bd12c845f68eedbb8f3b))

### Chore

* add setuptools_scm configuration ([`506faa6`](https://github.com/kmnhan/erlabpy/commit/506faa6bcb63b03bbce4add7fd5a7b5a4f761320))

* (**deps**) update dependency to use igor2&gt;=0.5.6 now on conda-forge ([`b59fc5a`](https://github.com/kmnhan/erlabpy/commit/b59fc5a75071191e199c20424df6b76860d69029))

* (**deps**) remove igor2 direct dependency from requirements.txt ([`bfb5518`](https://github.com/kmnhan/erlabpy/commit/bfb551893d7858ed1816a376b25ef6aba5004b3b))

* (**deps**) remove importlib metadata ([`b5718e7`](https://github.com/kmnhan/erlabpy/commit/b5718e7aeb988b1cfe8e5026358640d49481db61))

* (**deps**) update minimum versions and env configurations ([`18a3d67`](https://github.com/kmnhan/erlabpy/commit/18a3d67becacbfe2b907c206490917a85541df2c))

* (**deps**) update dependencies ([`b3e2494`](https://github.com/kmnhan/erlabpy/commit/b3e249460d99f429c403cdaad01d7f3ea16084e9))

* (**deps**) remove importlib ([`8a6b818`](https://github.com/kmnhan/erlabpy/commit/8a6b818a1c38e87e648bdc80a82148d31083bf0d))

### Ci

* (**github**) swap pip installation order ([`afa4722`](https://github.com/kmnhan/erlabpy/commit/afa472259a564019c3d8e78fee158277eed9c923))

### Documentation

* fix typo ([`dc8204a`](https://github.com/kmnhan/erlabpy/commit/dc8204aa67488e6efc29d50ccf73f8b0babc0f9c))

* update conf.py ([`3f092c4`](https://github.com/kmnhan/erlabpy/commit/3f092c472ae5312dee18ab9be98d080124f02edc))

* update bibliography ([`8e32515`](https://github.com/kmnhan/erlabpy/commit/8e325159b7f8f4e631dece557028f94527481cfd))

* improve readability ([`bd8049d`](https://github.com/kmnhan/erlabpy/commit/bd8049d1060791f5df22ef6b610fe26a700c14cc))

* update minimum gradient documentation ([`a8df0f3`](https://github.com/kmnhan/erlabpy/commit/a8df0f3ba04e979f0a8d6fb9b2598c65b28767e1))

### Feature

* (**itool**) add copy code to PlotItem vb menu ([`7b4f30a`](https://github.com/kmnhan/erlabpy/commit/7b4f30ada21c5accc1d3824ad3d0f8097f9a99c1))

  For each plot in imagetool, a new &#39;copy selection code&#39; button has been added to the right-click menu that copies the code that can slice the data to recreate the data shown in the plot.

* add 2D curvature, finally closes #14 ([`7fe95ff`](https://github.com/kmnhan/erlabpy/commit/7fe95ffcdf0531e456cfc97ae605467e4ae433c0))

* (**plotting**) add N argument to plot_array_2d ([`2cd79f7`](https://github.com/kmnhan/erlabpy/commit/2cd79f7ee007058da09aff244cd75748698444ee))

* add scaled laplace ([`079e1d2`](https://github.com/kmnhan/erlabpy/commit/079e1d21201c7523877b06a0f04f7640027b0614))

* add gaussian filter and laplacian ([`8628d33`](https://github.com/kmnhan/erlabpy/commit/8628d336ff5b4219e4fd382293736e4cbf026d56))

* add derivative module with minimum gradient implementation ([`e0eabde`](https://github.com/kmnhan/erlabpy/commit/e0eabde60e6860c3827959b45be6d4f491918363))

### Fix

* (**dynamic**) properly broadcast xarray input ([`2f6672f`](https://github.com/kmnhan/erlabpy/commit/2f6672f3b003792ecd98b4fbc99fb11fcc0efb8b))

* (**fit.functions**) polynomial function now works for xarray input ([`3eb80de`](https://github.com/kmnhan/erlabpy/commit/3eb80dea31b6414fa9a694049b92b7334a4e10f5))

* (**analysis.image**) remove critical typo ([`fb7de0f`](https://github.com/kmnhan/erlabpy/commit/fb7de0fc3ba9049c488a90bef8ee3c4feb935341))

* (**analysis.image**) dtype safety of cfunc ([`b4f9b17`](https://github.com/kmnhan/erlabpy/commit/b4f9b17656c64be4cff876843ed0f3491d8310d4))

* set autodownsample off for colorbar ([`256bf2d`](https://github.com/kmnhan/erlabpy/commit/256bf2dc8c368d093a3578d7f9279b1ee4653534))

* disable itool downsample ([`e626bba`](https://github.com/kmnhan/erlabpy/commit/e626bba9fcd4fd31387ca3a07a9a33b7690f3645))

### Performance

* (**itool**) add explicit signatures to fastbinning ([`62e1d51`](https://github.com/kmnhan/erlabpy/commit/62e1d516f0260f661fe9cd8f1fae9cb81afbcabe))

  Speedup initial binning by providing explicit signatures.

### Refactor

* update dtool to use new functions ([`a6e46bb`](https://github.com/kmnhan/erlabpy/commit/a6e46bb8b19512e438291afbbd5e0e9a4eb4fe87))

* (**analysis.image**) add documentation and reorder functions ([`340665d`](https://github.com/kmnhan/erlabpy/commit/340665dc507a99acc7d56c46a2a2326fbb56b1e3))

* rename module to image and add citation ([`b74a654`](https://github.com/kmnhan/erlabpy/commit/b74a654e07d9f4522cee2db0b897f1ffcdb86e94))

* (**dtool**) cleanup unused code ([`f4abd34`](https://github.com/kmnhan/erlabpy/commit/f4abd34bbf3130c0ec0fd2f9c830c8da43849f13))

### Tests

* reduce test time by specifying explicit path ([`60fb0d0`](https://github.com/kmnhan/erlabpy/commit/60fb0d0cedd9f0aaeca7101dddf0848f8872ccc3))

  This will not trigger directory recursion, so tests will run a bit faster

* add tests for fitting functions ([`4992251`](https://github.com/kmnhan/erlabpy/commit/499225149346e970d00b60dcb5ca39af5e5ddb47))

* add tests for image and shift ([`7e4daeb`](https://github.com/kmnhan/erlabpy/commit/7e4daeb5aea9689aadfe3eedb561d313e217684c))


## v1.6.5 (2024-04-03)

### Fix

* make imports work without optional pip dependencies ([`b8ac11d`](https://github.com/kmnhan/erlabpy/commit/b8ac11d8fb4379f70a39c817332382c352391a64))


## v1.6.4 (2024-04-03)

### Fix

* load colormaps only when igor2 is  available ([`7927c7d`](https://github.com/kmnhan/erlabpy/commit/7927c7db264bedb1a27b980d820d352f779b64c9))


## v1.6.3 (2024-04-03)

### Fix

* leave out type annotation for passing tests ([`eb25008`](https://github.com/kmnhan/erlabpy/commit/eb2500838820172529ee751b5d8a624c950f66d2))


## v1.6.2 (2024-04-03)

### Fix

* igor2 does not have to be installed on import time ([`186727a`](https://github.com/kmnhan/erlabpy/commit/186727ac8d50b662efeba8bee567cf1013ca936a))


## v1.6.1 (2024-04-03)

### Chore

* (**deps**) add pre-commit to dev dependency ([`3a2fccd`](https://github.com/kmnhan/erlabpy/commit/3a2fccd978d23d806d2088ebd9ef60c7a2b20902))

* make csaps optional ([`db31b06`](https://github.com/kmnhan/erlabpy/commit/db31b064c1f46edef7743fdd1c3ab7984e170b3c))

* update issue templates ([`dfc2ab0`](https://github.com/kmnhan/erlabpy/commit/dfc2ab0fdfcf1fd5ab83dac2c9d6473b4d2cb7e1))

### Ci

* (**github**) remove linting, let pre-commit handle it ([`b209ecb`](https://github.com/kmnhan/erlabpy/commit/b209ecbb3c0a35d2bbeba8155bea3da9ffa58fe1))

* (**pre-commit**) add hooks ([`9b401c3`](https://github.com/kmnhan/erlabpy/commit/9b401c328bb3ff18dddcce40b935afa2b6e2624a))

### Documentation

* rephrase kconv guide ([`dd2c022`](https://github.com/kmnhan/erlabpy/commit/dd2c022e42e692c2af640a1fc8d21c3e429781b2))

* add ipykernel dependency to resolve failing builds ([`e5774a5`](https://github.com/kmnhan/erlabpy/commit/e5774a51c14ef6df190eb9f6198c274d2061cdd5))

* add hvplot example ([`6997020`](https://github.com/kmnhan/erlabpy/commit/69970208ba6658f15e900ee6b9367177fcd86d29))

### Fix

* remove all pypi dependencies from pyproject.toml ([`1b2fd55`](https://github.com/kmnhan/erlabpy/commit/1b2fd5594f00bba8367419cd00919eba45cde5a7))

### Refactor

* remove ktool_old ([`18ea072`](https://github.com/kmnhan/erlabpy/commit/18ea0723fdf538bdbf2789ca73b2b962839ca3e5))

### Style

* apply ruff to deprecated imagetools ([`b2c7596`](https://github.com/kmnhan/erlabpy/commit/b2c7596ed12d89edaa2be3fe2923388014c68007))

* apply pre-commit fixes ([`12b6441`](https://github.com/kmnhan/erlabpy/commit/12b6441419ed6c4ff4da921790c57a599032dba7))


## v1.6.0 (2024-04-02)

### Ci

* speedup tests ([`618851e`](https://github.com/kmnhan/erlabpy/commit/618851e74d94301ec4f85a46facd46d3b6272571))

* parallelize tests ([`232301a`](https://github.com/kmnhan/erlabpy/commit/232301a0ab26c9c32a355af11b5458395a1cd832))

* migrate from pylint to ruff ([`2acd5e3`](https://github.com/kmnhan/erlabpy/commit/2acd5e3177f97f196d94644d75e3566a2714bf40))

* add pre-commit configuration ([`063067d`](https://github.com/kmnhan/erlabpy/commit/063067dfdedefefc47e55096d310a4df54a5b999))

### Documentation

* add pre-commit ci status badge ([`ae39d3d`](https://github.com/kmnhan/erlabpy/commit/ae39d3dbb0a058b59493b97507f88576f6b1737a))

* add pre-commit badges ([`1b6702b`](https://github.com/kmnhan/erlabpy/commit/1b6702b9615c9881afb86883466f3e8846a2db12))

* replace black with ruff ([`cb1a4b5`](https://github.com/kmnhan/erlabpy/commit/cb1a4b56a1b11b6d4630e5a36307befc48270294))

### Feature

* add mdctool ([`a4976f9`](https://github.com/kmnhan/erlabpy/commit/a4976f93cde51a41d667321a93dc2a90f23bddc3))

### Refactor

* remove deprecated function and dependencies ([`4b9c7b1`](https://github.com/kmnhan/erlabpy/commit/4b9c7b1629d99fbf0108ca33791d1bfd59632199))

### Style

* remove unnecessary dict call ([`ea0e0e8`](https://github.com/kmnhan/erlabpy/commit/ea0e0e822f8487ec5238b651f3d72aafac5c6bcb))

* apply formatting ([`12e3a16`](https://github.com/kmnhan/erlabpy/commit/12e3a1649ce03792f79df8220f70572ff0ecc97a))

* remove implicit optionals and apply more linter suggestions ([`798508c`](https://github.com/kmnhan/erlabpy/commit/798508c6a65ac439be70f9b7cc32c801ae8632cb))

* reduce indentation ([`274a330`](https://github.com/kmnhan/erlabpy/commit/274a33037b0155b82d8f9eb5ec542568c54da1db))

* move imports to type-checking block ([`e1f4005`](https://github.com/kmnhan/erlabpy/commit/e1f400516dcbc220979346f25a7dcfe4018df906))

* cleanup kwargs and unnecessary pass statements ([`7867623`](https://github.com/kmnhan/erlabpy/commit/7867623e779636531cdf1e0675846d22d0045249))

* make collections literal ([`74a8878`](https://github.com/kmnhan/erlabpy/commit/74a887853c2e84f315d45e52844a9c0fa7b46e28))

* rewrite unnecessary dict calls as literal ([`10637f6`](https://github.com/kmnhan/erlabpy/commit/10637f622b29703a02b4666c5712e8cf03a96066))

* format with ruff ([`64f3fed`](https://github.com/kmnhan/erlabpy/commit/64f3fed42e4766c1fe70d6a9488b75179a905314))

* fix flake8-bugbear violations ([`4aade97`](https://github.com/kmnhan/erlabpy/commit/4aade97013cea63e20895fb39b43c04953a67984))

* apply ruff unsafe fixes ([`a1a7d9a`](https://github.com/kmnhan/erlabpy/commit/a1a7d9ae79d3afa88cffe7423bb942aca29bfd09))

* lint with pyupgrade and ruff ([`244e053`](https://github.com/kmnhan/erlabpy/commit/244e05305ce2e0b72c54e3eb7c96befb97762f87))

* apply linter suggestions ([`7295cbc`](https://github.com/kmnhan/erlabpy/commit/7295cbc5b08065d75447f80ab1d84eb1c15255f3))

### Unknown

* [pre-commit.ci] auto fixes from pre-commit.com hooks ([`b86c995`](https://github.com/kmnhan/erlabpy/commit/b86c9952be94b4b7f5e5918ed28cbf39b750ef09))

  for more information, see https://pre-commit.ci


## v1.5.2 (2024-04-01)

### Documentation

* update user guide notebooks ([`80ab771`](https://github.com/kmnhan/erlabpy/commit/80ab7717539e95c2cfe4a15f0713f259dfe04da3))

* update docstring ([`b262765`](https://github.com/kmnhan/erlabpy/commit/b2627651648066dc8b98f023c5028c11f2929426))

* update documentation ([`9051ed8`](https://github.com/kmnhan/erlabpy/commit/9051ed8d406c06ae4a037b65ed648a16843a0655))

### Fix

* set values after setting bounds ([`ab6d682`](https://github.com/kmnhan/erlabpy/commit/ab6d682d0afafefcaec4c1ab6d673a39a75f40a6))

* proper patch all interpolator selection functions ([`b91834e`](https://github.com/kmnhan/erlabpy/commit/b91834e1b0be200bafb86ed3581f08cf1a5d42ef))

* make bz voronoi robust ([`8259760`](https://github.com/kmnhan/erlabpy/commit/8259760249be45892cd32f143b1b83aefe166c49))

### Refactor

* remove debug print statement in FastInterpolator class ([`712bd2c`](https://github.com/kmnhan/erlabpy/commit/712bd2ce90ad3534212d8a63c3fe10d780e243f5))

* add edge correction ([`87adcef`](https://github.com/kmnhan/erlabpy/commit/87adceffda2364f404de0860bfe8bf36b4cc1394))

* change variable name ([`b68949e`](https://github.com/kmnhan/erlabpy/commit/b68949ec59fd6bd7d7dad4ff9cc232b0e1ce4fba))

* make rotation transformations try fast interpolator first ([`e0a7908`](https://github.com/kmnhan/erlabpy/commit/e0a790833025f0c7e952ad17d120f46de3100555))

* update warning message ([`af67c1a`](https://github.com/kmnhan/erlabpy/commit/af67c1a507be35348b58862b6b51b92fac52781b))

* add several new accessors ([`664e92a`](https://github.com/kmnhan/erlabpy/commit/664e92a3e171512be26ea957df945e84134c880a))

* use new accessors and attrs ([`8e1dee2`](https://github.com/kmnhan/erlabpy/commit/8e1dee22d9d716f7e9bce29a1be3e68311494aa1))

* add qplot accessor ([`cb9aa01`](https://github.com/kmnhan/erlabpy/commit/cb9aa017bebd2ee6661f4eb87b988509d28a37a5))

* remove annotate_cuts ([`004ee80`](https://github.com/kmnhan/erlabpy/commit/004ee808dab13073cb3d2021d331767f6c28388a))

* dataloader cleanup ([`fd97780`](https://github.com/kmnhan/erlabpy/commit/fd977800a504256afd6018e9991b2d1e996277df))


## v1.5.1 (2024-03-28)

### Documentation

* update README screenshots ([`04d6b44`](https://github.com/kmnhan/erlabpy/commit/04d6b443dc077cbf056dae9b2bf9630284e707ee))

* use svg plots ([`aaa4842`](https://github.com/kmnhan/erlabpy/commit/aaa48420f69c71eb08180934ef2051819df92c03))

* improve momentum conversion documentation ([`c315a1a`](https://github.com/kmnhan/erlabpy/commit/c315a1a6e4d6365a6cc02e861dae84daf9e0cc14))

* update dev docs ([`7406308`](https://github.com/kmnhan/erlabpy/commit/740630899108d562bcc542bd6ae9d147b893c27d))

### Fix

* restore argname detection that was broken with namespace changes ([`863b702`](https://github.com/kmnhan/erlabpy/commit/863b702b6373f9a219a1e770aa49c71145371681))

* namespace collision ([`10edcdc`](https://github.com/kmnhan/erlabpy/commit/10edcdc8b06425c380ca6caa2d3f5f2be5c13733))

* followup namespace change ([`4c5222c`](https://github.com/kmnhan/erlabpy/commit/4c5222cc93196f0b6a75a0101107a37e73748eeb))

### Refactor

* allow offsetview upate chaining ([`8d5ca4f`](https://github.com/kmnhan/erlabpy/commit/8d5ca4f5b12c7d7060ea444773a9851f23db9850))

  This also means that _repr_html_ is automatically displayed when update or reset is called.

* improve consistency in accessors ([`9596fd7`](https://github.com/kmnhan/erlabpy/commit/9596fd723206f3e992fe00990f73364a61604cd6))

  Added setter method for configuration too.

* make prints consistent ([`0021302`](https://github.com/kmnhan/erlabpy/commit/002130224e3efc01615948a6443516e29d333cf5))

* change module names to prevent conflict with function names ([`493a5aa`](https://github.com/kmnhan/erlabpy/commit/493a5aab19c0d66851ca068e286a6aec92131e33))

  Cleanup erplot namespace and move tools to interactive.

* follow class naming conventions ([`efb9610`](https://github.com/kmnhan/erlabpy/commit/efb9610a864ef637f424c2f1b2871add7324b090))


## v1.5.0 (2024-03-27)

### Chore

* remove unnecessary dependency on colorcet, cmasher, cmocean and seaborn ([`5fd2d61`](https://github.com/kmnhan/erlabpy/commit/5fd2d614f97e8bba4f34a9277c70835214a95be7))

* add isort profile to project configuration ([`df269a9`](https://github.com/kmnhan/erlabpy/commit/df269a990e642135c76a60bfd19e0a6767974a40))

* update dependencies and environment files ([`6ec32dd`](https://github.com/kmnhan/erlabpy/commit/6ec32ddedb342d0556aacec0625c889b01f18b62))

  Fix python version and remove editable installs

* change pyclip dependency to pyperclip ([`db78f8e`](https://github.com/kmnhan/erlabpy/commit/db78f8e5a8be47ca4f23aa560e8aef88efb58c5b))

  Although pyclip supports copying bytes, it&#39;s not on conda-forge. Using pyperclip instead.

### Documentation

* add momentum conversion documentation draft ([`5410763`](https://github.com/kmnhan/erlabpy/commit/54107632edd5a7a911a1c8d06c663fc48d5217a0))

* add installation and contribution information ([`93a4e7c`](https://github.com/kmnhan/erlabpy/commit/93a4e7c4f43a8133f3f2149eb638261a9d56cfe6))

* fix typo in README ([`2b5e2cf`](https://github.com/kmnhan/erlabpy/commit/2b5e2cf3d5dd9e93d34da578e5689f14d490405b))

### Feature

* add interactive tool to kspace accessor ([`fb91cdb`](https://github.com/kmnhan/erlabpy/commit/fb91cdb50229154c070df8dfaa80cddc8520ae6d))

### Refactor

* accessors are now registered upon package import ([`d79fee2`](https://github.com/kmnhan/erlabpy/commit/d79fee2a28dd5ee59bfc6bd1ce224a44c5f40a24))

### Style

* apply linter suggestions ([`fe35da9`](https://github.com/kmnhan/erlabpy/commit/fe35da9a3494af28420ead2d8d40c5339788ac80))


## v1.4.1 (2024-03-26)

### Fix

* update package metadata ([`ecfb88f`](https://github.com/kmnhan/erlabpy/commit/ecfb88f2c23a7681e12d6f2dedcc316a28aa22c7))

  This should be classified as chore, but commiting as a fix to trigger CI


## v1.4.0 (2024-03-26)

### Chore

* update workflow triggers ([`fb158f3`](https://github.com/kmnhan/erlabpy/commit/fb158f3a6b6ded4ed2d573f4d33f85fbd36809b5))

* update build command ([`a22b8e5`](https://github.com/kmnhan/erlabpy/commit/a22b8e58bb744d02c2e0214af6185da8c66cbe29))

* update CI/CD badge urls ([`db61b29`](https://github.com/kmnhan/erlabpy/commit/db61b29fa0d92f54f7134ce5bd1c021aacfae647))

* make pyproject.toml compatible ([`959f687`](https://github.com/kmnhan/erlabpy/commit/959f6874f421ddd7bdf816f96c78d1533081b24d))

  README file link fixed, and remove direct dependencies. Add build command for automatic building

* update workflows to upload to pypi ([`2902b68`](https://github.com/kmnhan/erlabpy/commit/2902b683051ce651be6d5e38c6bdf6e55a9681f1))

### Documentation

* update docstring and apply linter suggestions ([`de3ee01`](https://github.com/kmnhan/erlabpy/commit/de3ee01dd35973186d69125f24d9527cfa8abd94))

* update README ([`8bd239f`](https://github.com/kmnhan/erlabpy/commit/8bd239f562d2d2345178c339a455ec23a5aa8082))

### Feature

* calculate kz in MomentumAccessor ([`46979f9`](https://github.com/kmnhan/erlabpy/commit/46979f907b120e5a4a88fdacd7d74a4b9dd41d6d))

  Add method that calculates kz array from given photon energy float

* make momentum conversion functions xarray compatible ([`a7aa34b`](https://github.com/kmnhan/erlabpy/commit/a7aa34ba983d3159c555ed66579d46eaf9e993aa))


## v1.3.1 (2024-03-25)

### Documentation

* update documentation ([`69a02fa`](https://github.com/kmnhan/erlabpy/commit/69a02fa3591720cf79b01289fd9dfb9cf55c26db))

  - Move rst README contents to docs, replace with newly written markdown. - Add screenshot images to  documentation and README

* update README ([`15f61bf`](https://github.com/kmnhan/erlabpy/commit/15f61bfe7a1734cece17479064c6d7946e2701f9))

### Fix

* fixes #12 ([`02b49a1`](https://github.com/kmnhan/erlabpy/commit/02b49a1da7550ae2b07819e6ccde3dcf750fc527))


## v1.3.0 (2024-03-25)

### Chore

* fix wrong branch name in release workflow ([`76a51b8`](https://github.com/kmnhan/erlabpy/commit/76a51b87180065631b6f5ca0678a87dfaa7e267e))

* configure semantic release ([`3ebdecb`](https://github.com/kmnhan/erlabpy/commit/3ebdecb45b510ed5e45e25fbc10d58ebc0b4ce20))

* bump version to 1.2.1 ([`30ec306`](https://github.com/kmnhan/erlabpy/commit/30ec3065234b6f727ed8f74daa1a866b82b0abc7))

### Documentation

* update README ([`79ba5b4`](https://github.com/kmnhan/erlabpy/commit/79ba5b42f5089d9fd81ccfc69dadda21914b42a7))

### Feature

* (**io**) add new data loader plugin for DA30 + SES ([`7a27a2f`](https://github.com/kmnhan/erlabpy/commit/7a27a2f27d9658f1091aaa48bcc78dea562898d8))

### Fix

* (**io**) properly handle registry getattr ([`499526f`](https://github.com/kmnhan/erlabpy/commit/499526fc1705bfbfbf8d3b80d50d65450dec7eae))

  This fixes an issue where _repr_html_ will fallback to __repr__. Additionally, `get` will now raise a KeyError instead of a ValueError.

### Style

* adjust loader registry repr ([`1fc31af`](https://github.com/kmnhan/erlabpy/commit/1fc31af083654a6c093bf343a881fcab37f9fbe2))

* remove incorrect type annotation ([`69dbf8a`](https://github.com/kmnhan/erlabpy/commit/69dbf8a1041ab22ea5d928623adae497b5ecd919))


## v1.2.1 (2024-03-25)

### Build

* drop python 3.10 support ([`183769f`](https://github.com/kmnhan/erlabpy/commit/183769f9af371f5e3a910976356ac2ac384c9ebb))

* update pyproject.toml to properly include dependencies ([`d39e69e`](https://github.com/kmnhan/erlabpy/commit/d39e69e7f5a6cf1b9b6088689f1f7756e25edc4f))

* update requirements.txt ([`91cac05`](https://github.com/kmnhan/erlabpy/commit/91cac05fdade8c5ce21a9e28d311159f57351ac9))

* update requirements.txt ([`bf2e534`](https://github.com/kmnhan/erlabpy/commit/bf2e5346d4c3030a4ef8061e5f463491725e9f9c))

* bump version ([`a68a6ea`](https://github.com/kmnhan/erlabpy/commit/a68a6ea1e061ffbb6c4ae966a6b23c6710175744))

* bump version to 1.1.0 ([`791af02`](https://github.com/kmnhan/erlabpy/commit/791af027a4e3ebf3cbad4bcb9af490986a2be2c0))

* bump setuptools minver ([`b55da17`](https://github.com/kmnhan/erlabpy/commit/b55da17c30ec94e897cf2ee8abf151796f3f78b7))

* try automatic discovery ([`05040e3`](https://github.com/kmnhan/erlabpy/commit/05040e3f98b6094258d4a7be7a33feeefa1fd44b))

* modify to updated directory structure ([`625ffcf`](https://github.com/kmnhan/erlabpy/commit/625ffcf2eb5a2914024efd815de40a910b1ae040))

* add setuptools-scm as build dependency ([`4ff4791`](https://github.com/kmnhan/erlabpy/commit/4ff47910022571d0d37ddc923755705dfa8a549e))

* fix typo in requirements.txt ([`29571de`](https://github.com/kmnhan/erlabpy/commit/29571de3bec9a9bbbcdff850b34e6caec167a109))

* update README, remove setup.cfg and update pyproject.toml ([`3736f46`](https://github.com/kmnhan/erlabpy/commit/3736f4629621d036ec071f27b0660bf7e99f8e86))

* add reqquirements.txt ([`dacd5be`](https://github.com/kmnhan/erlabpy/commit/dacd5be9985e28d1c67ac46d2badd24f148d4062))

* add yml file for intel ([`4844091`](https://github.com/kmnhan/erlabpy/commit/48440915782e1e8b918c88e4d9dce38b9703e43c))

* update readme type ([`40cf807`](https://github.com/kmnhan/erlabpy/commit/40cf807e522a59e9aad8ff6196788fe58846584b))

* update dependencies ([`e1bb13a`](https://github.com/kmnhan/erlabpy/commit/e1bb13ae6a350adf51e1b08ebe83f2732d1797f8))

* update instructions, cleanup dependencies ([`66d0e25`](https://github.com/kmnhan/erlabpy/commit/66d0e25d07393dc4b16a4d5e5994ea4fa822dd33))

* add yml for intel ([`8db3b26`](https://github.com/kmnhan/erlabpy/commit/8db3b269544c2f086f98fa302beb594f10573f65))

* update dependencies ([`21e8fab`](https://github.com/kmnhan/erlabpy/commit/21e8fabc2896d2b1fac5d2567eecf2df63047fe4))

* update dependencies ([`88a0170`](https://github.com/kmnhan/erlabpy/commit/88a01706b4f49d3b56f017badfd69af63df897f9))

* update dependencies ([`832e1ed`](https://github.com/kmnhan/erlabpy/commit/832e1ed25c299f7838dee1bc3d862a43ffe0bbb7))

* update requirements ([`fe26fc2`](https://github.com/kmnhan/erlabpy/commit/fe26fc2d17e4c107ef9b9a2773ce6c6318e18b2e))

* refactor requirements ([`81d3038`](https://github.com/kmnhan/erlabpy/commit/81d3038b03a717cf18fff655dba4224a7e67570a))

* add environment.yml for conda env ([`ea7bbf4`](https://github.com/kmnhan/erlabpy/commit/ea7bbf4309efeecd0ed1dcbd1791e482ab4d3ced))

* fix dependencies ([`e344f99`](https://github.com/kmnhan/erlabpy/commit/e344f990e55e7ddcc7e6ef5c87999d2c777b44a5))

* fix dependencies ([`5a6e38c`](https://github.com/kmnhan/erlabpy/commit/5a6e38c8de1f554819aa74d9e9e5a4f4524a2f6f))

### Chore

* update .gitignore ([`a4b2cbc`](https://github.com/kmnhan/erlabpy/commit/a4b2cbc322a4e031d75db87e87e7d532c585709f))

* update .gitignore ([`221b623`](https://github.com/kmnhan/erlabpy/commit/221b6232a6dc39b42ac1a3fd1a5abd0e2f1441d4))

* add some type hints remove deprecated ([`79d7349`](https://github.com/kmnhan/erlabpy/commit/79d7349953a1747e177796d14da44dbed5b02874))

### Ci

* update flake8 args ([`d720ee2`](https://github.com/kmnhan/erlabpy/commit/d720ee2b27df60fb42eca954ae7691641669828d))

* install qt runtime dependency ([`9e18d14`](https://github.com/kmnhan/erlabpy/commit/9e18d1499cddf987f8d267d8f358646b2a23ae32))

* create test.yml ([`c788eef`](https://github.com/kmnhan/erlabpy/commit/c788eef6025382a744d944e9e64c38b935324158))

### Documentation

* update docstring ([`f7bf9cb`](https://github.com/kmnhan/erlabpy/commit/f7bf9cb2f51fc8a57aa9a84b2c690b06cd029ec8))

  Changed configuration so that type annotations appear in both the signature and the description.

* update plotting examples with display_expand_data=False ([`79117c4`](https://github.com/kmnhan/erlabpy/commit/79117c4814d66dd5dc3d7cfc85af8f9b662651bf))

* update dosctring ([`2eccbf3`](https://github.com/kmnhan/erlabpy/commit/2eccbf3b18b7c52250b52c958cf5899fd05ea9a4))

* update docstring ([`4e8be19`](https://github.com/kmnhan/erlabpy/commit/4e8be19c1153d816e06d41d446bef7d057a4eb9b))

* add link to api reference in guide ([`e750519`](https://github.com/kmnhan/erlabpy/commit/e750519f3936ad4b7d4d532daebbe72c1b35f3d3))

* add update instructions ([`71bb0c6`](https://github.com/kmnhan/erlabpy/commit/71bb0c66815eef6acf1222c1f6e1081b753db493))

* update installation instructions ([`64f9a3b`](https://github.com/kmnhan/erlabpy/commit/64f9a3bf39a4c0ef49fd23ead37e34bf7a94f620))

* update README ([`428ea67`](https://github.com/kmnhan/erlabpy/commit/428ea671748f7062872533603b954ed27933a72a))

* cleanup top-level headers ([`39e9fcf`](https://github.com/kmnhan/erlabpy/commit/39e9fcf8962f0282902a90595d52a7d576fe35dd))

* update documentation ([`32ca369`](https://github.com/kmnhan/erlabpy/commit/32ca369ee6238696792229b63a29c27aa4060ddc))

  Cleanup header styles and add cards to index page

* update docstrings for some functions ([`d6a8e7d`](https://github.com/kmnhan/erlabpy/commit/d6a8e7d408d795586bd6b6bc6fc6d4f48443a853))

* cleanup conf.py ([`eab5e60`](https://github.com/kmnhan/erlabpy/commit/eab5e6078850de71c447cc1aa42b266bc10caf3b))

* update documentation ([`e03d118`](https://github.com/kmnhan/erlabpy/commit/e03d1182f294ef909b03ff04ea3d805349304bc1))

* update to use bibtex ([`c571b11`](https://github.com/kmnhan/erlabpy/commit/c571b11774c2dc2c152ccace4953976ad29600ef))

* update accessor docstring ([`d32f352`](https://github.com/kmnhan/erlabpy/commit/d32f35283208b6fbc72264a8f966beef4c39e0c9))

* update documentation ([`ec7a47e`](https://github.com/kmnhan/erlabpy/commit/ec7a47e28eee265ec01a66419fd32412ed565ba2))

* update documentation ([`0390eb3`](https://github.com/kmnhan/erlabpy/commit/0390eb37d758459a83cbb20b9d567d6f4937619d))

* update documentation ([`a7bfea2`](https://github.com/kmnhan/erlabpy/commit/a7bfea28f2f9cef1a32d591d3ebbf32c0d968450))

* update docstring to use PEP annotation ([`fc496df`](https://github.com/kmnhan/erlabpy/commit/fc496df1a9efe0ca345a183decdddaf54cdb0cc6))

* add copybutton ([`5e14247`](https://github.com/kmnhan/erlabpy/commit/5e142478568d17e4160f7ea52b99fa4b15a75b0d))

* update docstring ([`9b9eaff`](https://github.com/kmnhan/erlabpy/commit/9b9eaff437f39e4b684a2e1ef1f0d91a2762507b))

* use default font for docs plot generation ([`d36ce1d`](https://github.com/kmnhan/erlabpy/commit/d36ce1d5687168958da9cd33c9e225a52443c6ef))

* revert to pip due to slow build time, keep at py311 until 3.12.2 is available ([`954f357`](https://github.com/kmnhan/erlabpy/commit/954f3573e578da9a73a3256011b9016df01f6f30))

* build with conda ([`d2b8c40`](https://github.com/kmnhan/erlabpy/commit/d2b8c408d75c5d1dcedef6f7cfbdd12d93a57d34))

* retry build with py312 ([`bb12503`](https://github.com/kmnhan/erlabpy/commit/bb12503c5c6d67166a78de986df70778902c65cf))

* update requirements.txt ([`0b2eaad`](https://github.com/kmnhan/erlabpy/commit/0b2eaad96ff270d688f7b0549f57f8a753730b38))

* Add varname to requirements.txt ([`90df11e`](https://github.com/kmnhan/erlabpy/commit/90df11e1dd07ac8c3ea009b0b6c98d42014910d7))

* Add h5netcdf to requirements.txt ([`6be918c`](https://github.com/kmnhan/erlabpy/commit/6be918c905f463dc171c9df63065f0e87bd1811f))

* try build with py311 ([`233553d`](https://github.com/kmnhan/erlabpy/commit/233553d56d7d83692c4b9a9e7a35848554e21846))

* comment out code for latex generation and add new dependencies ([`ad9b974`](https://github.com/kmnhan/erlabpy/commit/ad9b97470ea45c7f1abc3e3ccaeccc5f761e5fc1))

* update requirements ([`fde56f2`](https://github.com/kmnhan/erlabpy/commit/fde56f2c7ac7c146dac0a28e76decfc250d400dc))

* Add Sphinx documentation dependencies ([`11d200e`](https://github.com/kmnhan/erlabpy/commit/11d200e113a54c047acaccd430f90da6e9c19f1f))

* Add .readthedocs.yaml and docs/requirements.txt files ([`9296c71`](https://github.com/kmnhan/erlabpy/commit/9296c710fb082442bc4226b82f383e4e6fddf45c))

* update documentation ([`fd5fc1a`](https://github.com/kmnhan/erlabpy/commit/fd5fc1a616e4b76b6cd37e3065f58e49e94b53b5))

* Add figmpl_directive to extensions in conf.py ([`3ca728f`](https://github.com/kmnhan/erlabpy/commit/3ca728f0aa2d6904c8bf31791275992b27ce1ebe))

* update documentation ([`fa312c0`](https://github.com/kmnhan/erlabpy/commit/fa312c05cc9dabb858d9268808b3efd2bead1a3e))

* update documentation ([`2e4e573`](https://github.com/kmnhan/erlabpy/commit/2e4e5736cec670f51e55e81db603c82b98a9e78b))

* add some comments ([`e66bc31`](https://github.com/kmnhan/erlabpy/commit/e66bc31cd9f9905a16a94f3b491170556b29adbc))

* (**itool**) improve tooltip ([`97ad19b`](https://github.com/kmnhan/erlabpy/commit/97ad19b1581613b6eb7012d4e04d79209f12075d))

* update docstring ([`700a9a3`](https://github.com/kmnhan/erlabpy/commit/700a9a3982ddcd16756efacb53bd5be861f7ed18))

* update documentation ([`e12b9af`](https://github.com/kmnhan/erlabpy/commit/e12b9afc1750823db4edb8034f4b225e51460cca))

* update docstring and formatting ([`dc8e544`](https://github.com/kmnhan/erlabpy/commit/dc8e54415fd26090782075847fd78fa942f7efca))

* update docstring ([`9d7139c`](https://github.com/kmnhan/erlabpy/commit/9d7139ceb90676189e300479d7775c26a4c107ed))

* update README ([`8ca11fe`](https://github.com/kmnhan/erlabpy/commit/8ca11fec492fd1b29737cec5df3dc8535e23b9bb))

* update documentation ([`dc857f7`](https://github.com/kmnhan/erlabpy/commit/dc857f73931413a40eb1b14cced16dda9fe5f4c8))

* update documentation ([`dec0b39`](https://github.com/kmnhan/erlabpy/commit/dec0b39a4d6182af688f7030f88e747cab7b255d))

* update documentation ([`01d0cb4`](https://github.com/kmnhan/erlabpy/commit/01d0cb4a868076e4e60851a58606df6d5b00ba4e))

* update documentation ([`cd914c6`](https://github.com/kmnhan/erlabpy/commit/cd914c650b651ad97ce6f4083c999c5251be02fc))

* update documentation ([`1944109`](https://github.com/kmnhan/erlabpy/commit/194410948df009ea56c91b4fbacafd5dd07f6537))

* update documentation ([`83615f6`](https://github.com/kmnhan/erlabpy/commit/83615f67bc27ee8d1bf9dcf2a2fd3a9e62b1ef51))

* update documentation ([`c28e6c4`](https://github.com/kmnhan/erlabpy/commit/c28e6c4e52817ed2cdec7f7902529bb7f99b829a))

* update documentation ([`0577134`](https://github.com/kmnhan/erlabpy/commit/05771342c12225536328c5bb959ff4bc1898ee34))

* update README ([`ccd3c05`](https://github.com/kmnhan/erlabpy/commit/ccd3c05c9a46110b3b740ea0ee061cf8ba595661))

* update documentation ([`e4df36f`](https://github.com/kmnhan/erlabpy/commit/e4df36f98c0031cd5a25738fa4698fd77836e79b))

* update documentation ([`5388383`](https://github.com/kmnhan/erlabpy/commit/5388383fd27be899d9e749e3cf3423c3e9a07e0c))

* update docstring ([`793f1de`](https://github.com/kmnhan/erlabpy/commit/793f1de1b1be734343f7c67a2d59f93a955adeac))

* update documentation ([`1ff5855`](https://github.com/kmnhan/erlabpy/commit/1ff5855f70b33e47034c03347e6d531f5553b431))

* update documentation ([`d01edcb`](https://github.com/kmnhan/erlabpy/commit/d01edcbffeb38d0e9473e865053f52d646bfda55))

* update docstring ([`d9df4f6`](https://github.com/kmnhan/erlabpy/commit/d9df4f67d596cde8efed7bb5a07451be2eb8924b))

* update documentation ([`00a9472`](https://github.com/kmnhan/erlabpy/commit/00a947298a42c857574a0f1fa0c8a10c1e02dd27))

* update README ([`7f62e74`](https://github.com/kmnhan/erlabpy/commit/7f62e7416ad2514372379f0d8a0a33c2946848af))

* update README ([`f774da7`](https://github.com/kmnhan/erlabpy/commit/f774da73f7bf11cc2b2c6740552fb6f53acb2bb9))

* update README ([`e1aa11d`](https://github.com/kmnhan/erlabpy/commit/e1aa11dedba93636431b761160d883e36189050e))

* update docstring ([`ee31e1f`](https://github.com/kmnhan/erlabpy/commit/ee31e1fe997c245981057c59a602fe1b7a253501))

* update docstring ([`b32e92f`](https://github.com/kmnhan/erlabpy/commit/b32e92f0cee5c5aa2b7793a17d95c21ff49ce94d))

### Feature

* (**gold**) apply automatic weights proportional to sqrt(count) on edge fit ([`717b9c8`](https://github.com/kmnhan/erlabpy/commit/717b9c814ce6fd38567695adb515973e6097ec50))

* add class to handle momentum conversion offsets ([`15416e5`](https://github.com/kmnhan/erlabpy/commit/15416e5aeca748225e35659c65dcc07b82b40007))

* add translation layer between lmfit models and iminuit ([`0f2f894`](https://github.com/kmnhan/erlabpy/commit/0f2f894994a00714e50e934dd4d5b518540539df))

* include all cmaps by default ([`3afe72e`](https://github.com/kmnhan/erlabpy/commit/3afe72ece2474c9adcfb89cb5037c50af2e2f0e5))

* change default colormap to CET-L20 ([`274122a`](https://github.com/kmnhan/erlabpy/commit/274122a4b2e582a24385c4dd748206ae0165b4b8))

* (**goldtool**) can access fit result after window close ([`4c9f232`](https://github.com/kmnhan/erlabpy/commit/4c9f232adfde874d975f033dc9bdcc8bf4969787))

* (**io**) style summary ([`6919929`](https://github.com/kmnhan/erlabpy/commit/6919929815401aa5d6ee4a37398c0593fab3657d))

* (**io**) summarize will now default to set directory ([`ca5b65a`](https://github.com/kmnhan/erlabpy/commit/ca5b65a0e850058837945b7f53864cfc6b32e933))

* (**constants**) add neutron mass ([`379ef37`](https://github.com/kmnhan/erlabpy/commit/379ef3705eb2f952121c92c9b653f2ca715618ee))

* (**io**) make loaders accessible with __getattr__ ([`b4884e4`](https://github.com/kmnhan/erlabpy/commit/b4884e4989c3565fdeccb8e18ec6a840cc67a7d2))

* (**io**) add loader plugin for SSRL Beamline 5-2 ([`67ced64`](https://github.com/kmnhan/erlabpy/commit/67ced64352e0f071fad5ebe441348108b6834784))

* (**io**) show full table on summary in ipython ([`985046f`](https://github.com/kmnhan/erlabpy/commit/985046f8a13911dc2c0be7cc68e530e93de37683))

* (**io.dataloader**) allow loaders to specify mapping dictionary on cut postprocessing ([`73abb0d`](https://github.com/kmnhan/erlabpy/commit/73abb0d1a078779130e6f82fadc9f44e62619799))

* (**io**) new arg to get_files ([`a9be216`](https://github.com/kmnhan/erlabpy/commit/a9be2165b61082956d5118c328e6484bdd0694c5))

* (**io.igor**) remove dependency on find_first_file ([`1d95292`](https://github.com/kmnhan/erlabpy/commit/1d95292ff4ba1c6ad5fcc95cae2d36c452d52ecf))

* (**io**) dataloader now preserves more attributes ([`f1157ba`](https://github.com/kmnhan/erlabpy/commit/f1157ba7afcb0b259932062edb0dc2138c749c55))

* (**io**) new data loader! ([`6c85cba`](https://github.com/kmnhan/erlabpy/commit/6c85cba810936403734cfeefa65b005e2d2e329e))

  Implemented  new class-based data loader. Currently only implemented for ALS BL4.0.3.

* (**io**) add new utility function ([`a27a9f8`](https://github.com/kmnhan/erlabpy/commit/a27a9f89ce56b35a7a35fd1014c01269378a8148))

* (**igor**) remove attribute renaming, keep original attribute as much as possible ([`912376c`](https://github.com/kmnhan/erlabpy/commit/912376c0216c901e85dd68b03817cbc8b85522a7))

* show elapsed time for momentum conversion ([`8889737`](https://github.com/kmnhan/erlabpy/commit/8889737aa6a8c5da2ef2ba17e5201816c99c574b))

* add sample data generation in angles ([`1d2600d`](https://github.com/kmnhan/erlabpy/commit/1d2600d37b7fe323adb016a1daf2b9811d6f4593))

* (**constants**) add electron rest energy mc^2 ([`8e46445`](https://github.com/kmnhan/erlabpy/commit/8e46445efd3fa5fe623e2ab65ee939039b4dc149))

* (**ktool**) full kz support ([`c3284f7`](https://github.com/kmnhan/erlabpy/commit/c3284f785484b5f506bd2025afa57661ac947eec))

  Added inner potential spinbox and projected BZ overlay

* (**bz**) add BZ extending ([`cbbb11a`](https://github.com/kmnhan/erlabpy/commit/cbbb11ac8997951cda17ec976fe321b8699d1eff))

* (**itool**) allow different label maximum widths ([`3467ae1`](https://github.com/kmnhan/erlabpy/commit/3467ae1720363ebfc026e3f5fc4f9517a342a448))

  The controls took up too much space if one of the dim names was long

* change transpose button order for 4D data ([`018002c`](https://github.com/kmnhan/erlabpy/commit/018002cd5606a091a528f866b2edcff051ce7570))

* add support for automatic cut and hv-dependent momentum conversion ([`f5be05f`](https://github.com/kmnhan/erlabpy/commit/f5be05fb80a4e5dc4f2b976e8894bddc07d43d91))

* add bz masking function ([`f3a7d21`](https://github.com/kmnhan/erlabpy/commit/f3a7d21c44784e4191011e8e95202889afc15ccc))

* (**interpolate**) add more checks and warnings ([`f1c223f`](https://github.com/kmnhan/erlabpy/commit/f1c223fbce27d65ed196fafd196c8b9c28d978dd))

* (**ktool**) frontend tweaks ([`d0050d0`](https://github.com/kmnhan/erlabpy/commit/d0050d02270c5ad8e51af281cb7a9f7a1cb89913))

  Added wait dialog for showing imagetool and make labels prettier

* add new momentum conversion tool ([`dfa96fe`](https://github.com/kmnhan/erlabpy/commit/dfa96fe9f20b57699d43d3d23755c663c3793d30))

* implement new momentum conversion functions. Currently only supports kxky and kxkyE conversion. ([`956a0bd`](https://github.com/kmnhan/erlabpy/commit/956a0bd8e034754d0b208a7c654d77e690491b98))

* switch data loaders and plotting functions to new angle coordinate convention. ([`b5734df`](https://github.com/kmnhan/erlabpy/commit/b5734df4a4c0009131b0510645f4cb28eff702cb))

  Adopts the angle convention given by Y. Ishida and S. Shin, Rev. Sci. Instrum. 89, 043903 (2018). This is a breakin change, and momentum conversion will be entirely re-written. All angles are now given in degrees, goodbye to radians.

* misc. changes to momentum conversion tool ([`1221fd0`](https://github.com/kmnhan/erlabpy/commit/1221fd0403977b840e650221111ae3a72a69cb65))

* load new resistance data ([`782cdf8`](https://github.com/kmnhan/erlabpy/commit/782cdf82c8b6b12d15c3f964dfe4a924131e5d4a))

* add spline module (work in progress) ([`69f8e4c`](https://github.com/kmnhan/erlabpy/commit/69f8e4c60cd147fc677f27aee927607153db3ac6))

* (**goldtool**) scale roi initial position to data ([`65f71d2`](https://github.com/kmnhan/erlabpy/commit/65f71d2c3ec1888f8e8fecb84b0e98189859f191))

* (**goldtool**) add remaining time to progress bar ([`348aaf7`](https://github.com/kmnhan/erlabpy/commit/348aaf7acde9421156bb6027bc0853a4142f166f))

* add atom plotting module ([`2fcf012`](https://github.com/kmnhan/erlabpy/commit/2fcf012c2866b598a82741961b9ad29602a63748))

* minor tweaks to mplstyle ([`10972b8`](https://github.com/kmnhan/erlabpy/commit/10972b8f70c041835d0c8274448e53be67172d18))

* add crop to  plot_array ([`36d8536`](https://github.com/kmnhan/erlabpy/commit/36d85366ffd1c84298df79d6721b26f17d8cf031))

* (**colors**) make axes not required for nice_colorbar ([`48af74b`](https://github.com/kmnhan/erlabpy/commit/48af74b45e926f5ff9f2c0c717d27d05987837b8))

* (**goldtool**) add diffev to list of methods ([`d21c915`](https://github.com/kmnhan/erlabpy/commit/d21c915f7993c66f1a79dc7cfa28b06b63cc89f8))

* (**itool**) plot class customization ([`fee3913`](https://github.com/kmnhan/erlabpy/commit/fee39133c4208e7733a18b31d77a7898ac4b5713))

* better cache management for slicer ([`9f49374`](https://github.com/kmnhan/erlabpy/commit/9f49374ee98fb2cc101974292d959ee0f6898a9a))

* (**betterspinbox**) make keyboardTracking off by default ([`753ddcd`](https://github.com/kmnhan/erlabpy/commit/753ddcd344ff6859ff44cecaff4fb641e4e981dd))

* (**annotation**) add axis unit scaling ([`a61b8f3`](https://github.com/kmnhan/erlabpy/commit/a61b8f38165828d3f118c682cbfe05325c413e22))

* (**itool**) add option to decouple colors ([`3e2f132`](https://github.com/kmnhan/erlabpy/commit/3e2f1325baa1011540b29f1f36fd8b12a7770114))

* (**style**) update poster style ([`da58226`](https://github.com/kmnhan/erlabpy/commit/da582267a69fea8efb86e93478c398d70c7e411c))

* (**accessor**) use pyarpes for 2d kconv ([`9cbe422`](https://github.com/kmnhan/erlabpy/commit/9cbe422d19bb95e17a13b4757b8239d08069d850))

* improve extendability ([`ded8841`](https://github.com/kmnhan/erlabpy/commit/ded884139e25cea9549fd3e3e7b0519c03b1d447))

* (**itool**) pass parent ([`c8feac4`](https://github.com/kmnhan/erlabpy/commit/c8feac43871fba3a0c12db56d33c819efdfd3e41))

* (**itool**) add base imagetool with only controls ([`fdb8a6b`](https://github.com/kmnhan/erlabpy/commit/fdb8a6b0e52f83c05441d03a06bd15d83b8a261a))

* add option to return individual regions ([`7c5b081`](https://github.com/kmnhan/erlabpy/commit/7c5b08181f0cfc6d727fafe6a8789ec952ac895f))

* (**colors**) add function to unify color limits ([`33d571f`](https://github.com/kmnhan/erlabpy/commit/33d571f4dfcdecec02d536f5f5a4cd9af349d2df))

* (**colors**) add function that combines colormaps ([`f19d87c`](https://github.com/kmnhan/erlabpy/commit/f19d87ca5c3de53ca8623d7954a280b7050f2d15))

* add new colormap! ([`474ad52`](https://github.com/kmnhan/erlabpy/commit/474ad521c3c5726341c17b9b4759ee501661ce73))

* k range freedom in sample data generation ([`9dd5b00`](https://github.com/kmnhan/erlabpy/commit/9dd5b0025aaf52ef0bf4685ef5e380ccd5a72ac4))

* make BZ plotter standalone ([`3bec642`](https://github.com/kmnhan/erlabpy/commit/3bec64218de70cabf8a43b78bf6728bef6402b49))

* (**igorinterface**) add to load wave menu ([`da704c5`](https://github.com/kmnhan/erlabpy/commit/da704c5c083a41db3999b031a5676bb69e33f46a))

* (**io**) add DA30 loader ([`9afd149`](https://github.com/kmnhan/erlabpy/commit/9afd149ddfb21a64564f4d9d2bb1ccb6dcd6d808))

* (**goldtool**) add fit abort ([`693b1e4`](https://github.com/kmnhan/erlabpy/commit/693b1e474ebbc4989dba6ed900f85cad62b5cfcc))

* pass kwargs to broadcast_model ([`82d11f2`](https://github.com/kmnhan/erlabpy/commit/82d11f24b6412f8c774264f5941863eaff7cef6b))

* autolevel colorbar ([`f116155`](https://github.com/kmnhan/erlabpy/commit/f1161551fbf4665b65287d27a2312b3b052c1948))

* set colorbar width ([`b13b5d2`](https://github.com/kmnhan/erlabpy/commit/b13b5d2abd89e98bee6e55d5aeba135a16b3fb50))

* (**gold**) fully integrate spline fitting ([`d5b345c`](https://github.com/kmnhan/erlabpy/commit/d5b345cb5efc6e1e889c272d437e646935ff12e1))

* add lattice module ([`bafe36b`](https://github.com/kmnhan/erlabpy/commit/bafe36b7ab89f004f7ac0ac1d4aa2d1b5137698a))

* estimate k resolution from data ([`457edc7`](https://github.com/kmnhan/erlabpy/commit/457edc72ffa755ffc2e8d3fee039df97bab53ae0))

* add kspace conversion accessor ([`56da9e0`](https://github.com/kmnhan/erlabpy/commit/56da9e09c0dedc9cb066fbd641b5a92eca5b8635))

* (**itool**) save individual plot as hdf5 ([`a5a88b9`](https://github.com/kmnhan/erlabpy/commit/a5a88b9402b5d538386547083b8d7dcc2335e032))

* add ZT image view for 4D ([`bf66dfb`](https://github.com/kmnhan/erlabpy/commit/bf66dfbc948bbf585dde392b23b0e5586b2a1d35))

* add igor procedure to load dataarrays ([`f8b93e0`](https://github.com/kmnhan/erlabpy/commit/f8b93e0e79adc6052e577304dd856a146de0c521))

* (**itool**) enable sync across multiple windows! ([`d3a5056`](https://github.com/kmnhan/erlabpy/commit/d3a50561bd38c9040b9aae3b6c1b82453df1423b))

* (**gold**) automatic crop for corrected gold ([`dad4bb8`](https://github.com/kmnhan/erlabpy/commit/dad4bb886799fd11d5ecf6c8c18c4a118858d15f))

* (**io**) improve BL4 data loading, add basic log generator ([`e283262`](https://github.com/kmnhan/erlabpy/commit/e283262f138bc509d9f4481efd835626060f8b62))

* (**interp**) improve interpolator syntax ([`c7b322b`](https://github.com/kmnhan/erlabpy/commit/c7b322b6bf4106320270b39b4aa423d94ffa94be))

* (**gold**) configurable covariance matrix scaling ([`d051864`](https://github.com/kmnhan/erlabpy/commit/d051864da1dd2e3f5101d33743fa04720fab5411))

* (**fit**) add step function edge ([`3a7bc03`](https://github.com/kmnhan/erlabpy/commit/3a7bc03c666a946813ce79e95002475efb446046))

* handle rad2deg automatically ([`aea4a62`](https://github.com/kmnhan/erlabpy/commit/aea4a62d047cca0be231e0256ad3f519ef54eb0e))

* (**bz**) add option for clip path ([`52daa2a`](https://github.com/kmnhan/erlabpy/commit/52daa2afa4fe98acb1c5ac39dfe6362d348d2e11))

* add fast trilinear interpolation ([`ae77bee`](https://github.com/kmnhan/erlabpy/commit/ae77bee8050b3a80e5ab24bef447911d2e6a4ccd))

* (**io**) add function for pxp debugging ([`1802e09`](https://github.com/kmnhan/erlabpy/commit/1802e099689062bf7e7f531b726ea8ce7ada91e0))

* add W to ph/s conversion ([`a64039d`](https://github.com/kmnhan/erlabpy/commit/a64039d3ca2971b5a434a451fc8e956e40fa2e9b))

* add 2D fermi edge fitting function ([`90c65de`](https://github.com/kmnhan/erlabpy/commit/90c65de986074dcce3ba658a0ec9ce1581da1dd2))

* (**itool**) keep manual limits under transpose ([`bdec7e5`](https://github.com/kmnhan/erlabpy/commit/bdec7e5abe935072f91570af9fc58bb2f19582cb))

* add more colortables from igor ([`03a615f`](https://github.com/kmnhan/erlabpy/commit/03a615f2ffbbf522bc8421757b7def3db037eab4))

* add interactive brillouin zone plot ([`85f844f`](https://github.com/kmnhan/erlabpy/commit/85f844f5f7461489f924f8cf5096416791dec525))

* update style files ([`5278c6c`](https://github.com/kmnhan/erlabpy/commit/5278c6c04213ce4230797bc34028d8befa2923af))

* modernize plot_array ([`c5e7321`](https://github.com/kmnhan/erlabpy/commit/c5e7321dbbc124ae99d55b3959c020f59b1a2b7d))

* (**io**) igor-compatible output ([`2a654e2`](https://github.com/kmnhan/erlabpy/commit/2a654e26fb0e53f79b705bb62db437f5d16e5735))

* (**io**) rewrite igor related backend ([`8739bef`](https://github.com/kmnhan/erlabpy/commit/8739befc3f30661e4d2ffd680888c3b06f3230cb))

* polynomial model now takes positional args ([`a3000d5`](https://github.com/kmnhan/erlabpy/commit/a3000d55eca86fe2c1c1d8f7d468ab4051ceef08))

* (**goldtool**) add smoothing spline fit ([`b25cd7c`](https://github.com/kmnhan/erlabpy/commit/b25cd7c19ced68c028f63cd3fbac533ae7a026b3))

* (**itool**) add performance benchmarks ([`9634af0`](https://github.com/kmnhan/erlabpy/commit/9634af0232afbf7f845b9694a03739b9111d516f))

* default cmap for analysis is now terrain ([`c1cc21c`](https://github.com/kmnhan/erlabpy/commit/c1cc21c2e7660245d1d0bcd1c3e4a5f45b757e15))

* example dataset generator ([`6b6cadf`](https://github.com/kmnhan/erlabpy/commit/6b6cadf19d536dea990daacd687739953a33a2d1))

* add curve fitting tool (alpha) ([`68356ab`](https://github.com/kmnhan/erlabpy/commit/68356ab6d7fc4ecb6653d36213acc8c22ef4f23b))

* add new module for curve fitting ([`20c8be9`](https://github.com/kmnhan/erlabpy/commit/20c8be92116a0ce95f1dd55a2de131a1a9a16efe))

* added boltzmann const. at constants ([`038930e`](https://github.com/kmnhan/erlabpy/commit/038930ebd9a48c000990c006ca794f8fc34b3d94))

* add module for 3D plotting ([`4ec1bd2`](https://github.com/kmnhan/erlabpy/commit/4ec1bd2c81b17c775321d7b8197fe7ca84f1d03b))

* (**itool**) add copy index action ([`6f25677`](https://github.com/kmnhan/erlabpy/commit/6f25677d2ba5d8cf1172a937d3370f67a7220781))

* parallelize multidimensional mean ([`7cfae81`](https://github.com/kmnhan/erlabpy/commit/7cfae816761e3169d1eb52b24d7adbbe8aeef848))

* add 2D colormap ([`f5799d8`](https://github.com/kmnhan/erlabpy/commit/f5799d8d389ddf9c8cb03d7bfd05e9daa5331de0))

* (**itool**) intuitive transpose for 4D data ([`38315f8`](https://github.com/kmnhan/erlabpy/commit/38315f83f98fa92313c8f7857c5b100f1b6f4fed))

* (**polygon**) add convenience function ([`a9e64c7`](https://github.com/kmnhan/erlabpy/commit/a9e64c7213952754c2ad70c2f7b92c73a4a1a45d))

* gradient fill under line plot ([`6a951c1`](https://github.com/kmnhan/erlabpy/commit/6a951c10060d454c8f5335d9081d1764881d1fc2))

* plot 1D slices ([`36dee17`](https://github.com/kmnhan/erlabpy/commit/36dee178b9ffaa37216f74a4bf7810ff3021354c))

* (**io**) add convenience function for BL4 data loading ([`0f3a4ea`](https://github.com/kmnhan/erlabpy/commit/0f3a4eac22e4738fe860117c6474bdab7923b959))

* add igor colormaps ([`4bfcceb`](https://github.com/kmnhan/erlabpy/commit/4bfcceb8cca35df3fcb775743f73b127a7c48239))

* new module for some common constants ([`c6aaede`](https://github.com/kmnhan/erlabpy/commit/c6aaede496725cb59548e26fff45f933f71883da))

* update erplot ([`4bdbc10`](https://github.com/kmnhan/erlabpy/commit/4bdbc10e9556105df9f0250aeb1554ea3b862581))

* add some annotations ([`4a37666`](https://github.com/kmnhan/erlabpy/commit/4a37666dcbd732f8c14c95c2ac1a2c24dee46e3b))

* (**io**) add load functions ([`c654c44`](https://github.com/kmnhan/erlabpy/commit/c654c4444fd78ae64222b25559a881bdca1b6321))

* (**itool**) copy cursor position ([`cc39829`](https://github.com/kmnhan/erlabpy/commit/cc398299fb1d25ac1164a216cf68f68ec2e80555))

* attempt better colorbar... this is probably stupid, fix later ([`2b26a3c`](https://github.com/kmnhan/erlabpy/commit/2b26a3c12e52c11d0468fa1268611b58ecf82b19))

* fast peak fitting with broadened step edge ([`dc30094`](https://github.com/kmnhan/erlabpy/commit/dc300944a2ec0f25e0e4f16540b1edb42af0b154))

* AxisItem scientific labeling ([`42b8860`](https://github.com/kmnhan/erlabpy/commit/42b8860a722dc4e20206b835e7fccb926975c460))

* add diverging colormap normalizations ([`b085946`](https://github.com/kmnhan/erlabpy/commit/b085946ab554c6a4aaef19b4aa13d1071114dc3b))

* (**itool**) add working colorbar ([`c326727`](https://github.com/kmnhan/erlabpy/commit/c326727c126c5081f1d7e60c2ce806699b8fb36f))

* (**itool**) full support for non-uniform coords ([`d3eface`](https://github.com/kmnhan/erlabpy/commit/d3eface36de9c15d788ab479d0f0a6860b36c9c8))

* (**itool**) auto-convert non-uniform dimensions to indices ([`1ad940d`](https://github.com/kmnhan/erlabpy/commit/1ad940d8bbc2b430cc54b7d9228ddc87e5235432))

* (**bz**) add brillouin zone edge calculation ([`2b70d91`](https://github.com/kmnhan/erlabpy/commit/2b70d91ec8c77c27117637c1fec906a7a06c9571))

* (**colors**) add higher contrast PowerNorm ([`16b965c`](https://github.com/kmnhan/erlabpy/commit/16b965c224691103aabcfbb41c73524eb6b2244f))

* (**itool**) add file open dialog ([`ba50354`](https://github.com/kmnhan/erlabpy/commit/ba5035491fa87ab44103d7041a81e15dee7a029e))

* (**itool**) move all cursors on drag with alt modifier ([`6ad638d`](https://github.com/kmnhan/erlabpy/commit/6ad638d9ef07c7b5dac05acc4977a39954abe96a))

* (**itool**) add color limit lock and discrete cursor line ([`0929d61`](https://github.com/kmnhan/erlabpy/commit/0929d61973ef8ae62065820378d362a70cbbe110))

* (**itool**) parse ArrayLike input automatically ([`ce3d802`](https://github.com/kmnhan/erlabpy/commit/ce3d8027bf2ba2956ef80c8c28cbed28ab6d21be))

* (**itool**) add more menus ([`6b423eb`](https://github.com/kmnhan/erlabpy/commit/6b423eb0b56c0082ec0eee7de89761cbc56dfba5))

* add EDC fitting tool (WIP) ([`4fceef5`](https://github.com/kmnhan/erlabpy/commit/4fceef509c63b8f51661b37cb8470f5bcf514b4b))

* add and modify styles ([`079f007`](https://github.com/kmnhan/erlabpy/commit/079f0070da78c1c44dcb67c46b38e84490d67f16))

* (**io**) warn when modifying attrs ([`7692379`](https://github.com/kmnhan/erlabpy/commit/7692379416dda67c47a302d49b1c9a747514ef9c))

* (**itool**) add menubar ([`1614567`](https://github.com/kmnhan/erlabpy/commit/1614567a1287e3a099418190855ff54f8cadad95))

* (**itool**) better cursor colors ([`78dae09`](https://github.com/kmnhan/erlabpy/commit/78dae09619bfb79424eb0af930cc444467fce2c2))

* (**itool**) enable handling multiple cursors ([`e971af7`](https://github.com/kmnhan/erlabpy/commit/e971af767f78d72acee064a0e87ae5764fd13d98))

* pretty plot gold edge fit results ([`aa621c5`](https://github.com/kmnhan/erlabpy/commit/aa621c588b733eae683494422f1330da74e09b23))

* add vertical Fermi energy indicators ([`80e2694`](https://github.com/kmnhan/erlabpy/commit/80e2694ed19dc18375b2dc26531fbda37525ca9e))

* easier colorbar font size specification ([`5638d9c`](https://github.com/kmnhan/erlabpy/commit/5638d9c709c888fcfc24503196955e9d8df863c2))

* add interactive progressbar for joblib ([`3fd24c7`](https://github.com/kmnhan/erlabpy/commit/3fd24c7dd57e2139c6f16af88fe0773323b928b4))

* (**itool**) add axis labels ([`96f9e76`](https://github.com/kmnhan/erlabpy/commit/96f9e76315cbff362d11e16b2b33378c5d7deab0))

* (**itool**) minor adjustments to layout ([`c5dcbe2`](https://github.com/kmnhan/erlabpy/commit/c5dcbe2ec3c143579fe2c7969eda65dcb741e876))

* (**goldtool**) higher order polynomials ([`5f34c35`](https://github.com/kmnhan/erlabpy/commit/5f34c3579975ed48124a5a2cf89980735f8d53a8))

* (**gold**) get full results from Fermi edge fitting ([`ce0d853`](https://github.com/kmnhan/erlabpy/commit/ce0d853adf51df174d01098b703db21e26859882))

* colorbar tick label customization ([`03e82bf`](https://github.com/kmnhan/erlabpy/commit/03e82bf100b3ca62fe781e1c82d54fce0ecab9b7))

* (**io**) improve SSRL loader ([`8fe9a46`](https://github.com/kmnhan/erlabpy/commit/8fe9a467d348225d213909a95acf9d80d93d018b))

* (**itool**) multicursor binning ([`bb4430c`](https://github.com/kmnhan/erlabpy/commit/bb4430ced79bfec8850ef7724c6ce752b4cd707b))

* add autoscale convenience function ([`36f838d`](https://github.com/kmnhan/erlabpy/commit/36f838d47bd5fdf8554e59b327b9bed9df961a0f))

* delegate font handling to different styles ([`aabf87b`](https://github.com/kmnhan/erlabpy/commit/aabf87b05be49c8bfe19afaf570bb769656c5ccb))

* simplify getting foreground color based on image ([`8034bfd`](https://github.com/kmnhan/erlabpy/commit/8034bfd15452f6032b0b2ff64cb14dc92d607c8b))

* add fira font mplstyle ([`9c05702`](https://github.com/kmnhan/erlabpy/commit/9c05702d1ea822b5e138fca1e5f75c8d600397d7))

* (**itool**) bind autorange to keyboard ([`98a236d`](https://github.com/kmnhan/erlabpy/commit/98a236deb88a9d74bc6763f49a75d465a097a264))

* (**itool**) implement rad2deg ([`c074d74`](https://github.com/kmnhan/erlabpy/commit/c074d74a4ba62fda1e81e1dedf039ae939959975))

* include styles ([`5281177`](https://github.com/kmnhan/erlabpy/commit/528117714f3f7b08d253036b7492aba48dfe833b))

* add completely reimplemented imagetool with faster slicing and multicursor support ([`af0655e`](https://github.com/kmnhan/erlabpy/commit/af0655eebb1f27199eae5270c971b206cb3edb6b))

* added interactive gold edge fitting tool ([`c4017b6`](https://github.com/kmnhan/erlabpy/commit/c4017b699b27af76b9aa912f3231d1a19864e5e9))

* allow axes input to slice plotter ([`ab4c639`](https://github.com/kmnhan/erlabpy/commit/ab4c639c0b4d9e1cd34d03950846db2a4afdb138))

* multiple axes input to bz overlay plotter ([`9820e34`](https://github.com/kmnhan/erlabpy/commit/9820e340584299f5bbf95aea4193e6c47c026304))

* (**fermiline**) support multiple axes input ([`9dc59b1`](https://github.com/kmnhan/erlabpy/commit/9dc59b1fbb6605e82ed0dcf8e7f444978c2c05df))

* add interactive widget base classes ([`e6787cf`](https://github.com/kmnhan/erlabpy/commit/e6787cf567bfd1a1cc1a131a69700143eb6e17bb))

* (**colors**) add colorbar creation macro ([`811ed52`](https://github.com/kmnhan/erlabpy/commit/811ed526dc99d165f3f0cf77dd577744973cb539))

* remove pyarpes dependency on labeling ([`71081fa`](https://github.com/kmnhan/erlabpy/commit/71081faf8404d256fb5b7f57fdc2304de6674445))

* (**io**) silent loading of .pxp files ([`27d6b30`](https://github.com/kmnhan/erlabpy/commit/27d6b30fbff64237562ceaa9b3436d48b06a6e3b))

* parallel processing helpers wip ([`ac88e26`](https://github.com/kmnhan/erlabpy/commit/ac88e264463f4647a1195bd4eb93d617cd38c972))

* reliable gold edge and resolution fitting ([`653db58`](https://github.com/kmnhan/erlabpy/commit/653db5891f4573e069caaeb6f65edcffaa24b316))

* (**annotations**) better high symmetry marking ([`8e6dfcb`](https://github.com/kmnhan/erlabpy/commit/8e6dfcbc3f26f1d17a5bc1400a60512c51aa8a4d))

* (**mask**) completely reimplement masking ([`ccfb9b6`](https://github.com/kmnhan/erlabpy/commit/ccfb9b61b75aa5e1f17e88402fc6824fe14287e8))

* (**io**) data loading functions for igor ([`dd333e1`](https://github.com/kmnhan/erlabpy/commit/dd333e199c5e2718b0bb712b1842e37b6c099ea6))

* (**correlation**) rewrite based on scipy ([`14b1b29`](https://github.com/kmnhan/erlabpy/commit/14b1b29c1752ccb3d93d67de6de160e714606133))

* add callable models for edge correction ([`3a6d8c3`](https://github.com/kmnhan/erlabpy/commit/3a6d8c3b5cc44078cd4b8f5056ccacb21d335428))

* functions for gold edge related analysis ([`8c4941f`](https://github.com/kmnhan/erlabpy/commit/8c4941f6347c09a783b9e34ff17a132e05d16631))

* rudimentary loader for SSRL ([`7e18e13`](https://github.com/kmnhan/erlabpy/commit/7e18e1354f95341440da2f020d992c4a38403261))

* (**itool**) better dock, improved memory usage ([`04f0047`](https://github.com/kmnhan/erlabpy/commit/04f00472ce5b507c1d467620dcfe03bdde1700c0))

* (**itool**) support 4D input ([`33a252f`](https://github.com/kmnhan/erlabpy/commit/33a252f42f88b9a0c182a3d771644b8e937d57be))

* add sizebar ([`21f420f`](https://github.com/kmnhan/erlabpy/commit/21f420f362dbf2bb18d1b16e34051dc11034a08a))

* (**itool**) add gamma slider ([`f5efcb7`](https://github.com/kmnhan/erlabpy/commit/f5efcb71c21696ccb9ed1e9b90b3c9e976bac01e))

* (**itool**) subclass buttons for dark mode ([`e215470`](https://github.com/kmnhan/erlabpy/commit/e2154703986a8d35aa9b0b44b35a40c9dde62ec7))

* initial commit of general interactive tool ([`bf00956`](https://github.com/kmnhan/erlabpy/commit/bf009565f26a95ead1cda41bb000409bc435062e))

* (**itool**) add axes visibility controls ([`258ff3f`](https://github.com/kmnhan/erlabpy/commit/258ff3fcecf5743adb04c82606feeff54ed828eb))

* (**itool**) hide and show individual axes ([`6a8c71e`](https://github.com/kmnhan/erlabpy/commit/6a8c71e514c8fd4755c47ff521ed5f242c72056d))

* (**itool**) enable latex labels ([`be0b25d`](https://github.com/kmnhan/erlabpy/commit/be0b25dcad9abf49859c766668e969cc800406b9))

* (**itool**) add joystick for cursor ([`3faf410`](https://github.com/kmnhan/erlabpy/commit/3faf410676af1e85ebe27282a80d65c2d55c17a0))

* cursor binning ([`9b3de32`](https://github.com/kmnhan/erlabpy/commit/9b3de32f3b1ee098b646b0ad663cd8a95de02e11))

* new convenience function ([`e482cf6`](https://github.com/kmnhan/erlabpy/commit/e482cf6234e5b865b36ed2574c9566b199806c26))

* add Fermi edge correction from fit result ([`607e6ee`](https://github.com/kmnhan/erlabpy/commit/607e6eedef84b05f83cef7097c7ce0aaedbb2f97))

* add tool for analyzing dispersive features ([`7926238`](https://github.com/kmnhan/erlabpy/commit/792623836b77502f7d4ab065712465df007fb1ed))

* add pyqtgraph-based itool, WIP ([`de462c7`](https://github.com/kmnhan/erlabpy/commit/de462c7ab22d5ba47b686e05eaa0cc6558d2d4be))

* (**itool**) add invert colormap ([`bd616ad`](https://github.com/kmnhan/erlabpy/commit/bd616adf30cdee0f96433d9cb8ca8febedf5207c))

* (**itool**) add color picker ([`cb40537`](https://github.com/kmnhan/erlabpy/commit/cb405377302855b357387a539a37f7f466b83447))

* (**itool**) add binning ([`4d8155d`](https://github.com/kmnhan/erlabpy/commit/4d8155dd85da507ee87b09cb604ed2cba2ea7e42))

* add dark mode ([`a0dfefd`](https://github.com/kmnhan/erlabpy/commit/a0dfefdc65676b631ceb380216cfbdbdeab6f2f8))

* (**itool**) 2d image support ([`98d0af7`](https://github.com/kmnhan/erlabpy/commit/98d0af799729884f5ccec203ffacebbffbe72f91))

* add toggle for cursor snap ([`d9ccfbd`](https://github.com/kmnhan/erlabpy/commit/d9ccfbdd8b36fce57655ffe04109fffda5c86189))

* add energy and resolution slider to ktool ([`94e886a`](https://github.com/kmnhan/erlabpy/commit/94e886ab02d638059311b6c6cb43957c3bbde1a9))

* add qt and mpl-based interactive tools ([`823c509`](https://github.com/kmnhan/erlabpy/commit/823c50972fb5afe92f10c9409ed61a7c626971fb))

* added LabeledCursor ([`c970413`](https://github.com/kmnhan/erlabpy/commit/c970413dafd391c3da2e81efb1009e0452a6ec51))

* added annotation macros ([`dd36878`](https://github.com/kmnhan/erlabpy/commit/dd36878e25acd2b7cff5f34442f93b40307ff452))

* Add characterization module ([`e5c56e8`](https://github.com/kmnhan/erlabpy/commit/e5c56e8903d5bcfa95d0d665186b4c1bb33c81a7))

### Fix

* (**era.fit.models**) undefined name in all ([`308f525`](https://github.com/kmnhan/erlabpy/commit/308f525dd291247b631ac8a878d572ea3bfd2230))

* invalid escape ([`a0a33b6`](https://github.com/kmnhan/erlabpy/commit/a0a33b6695471fb85060c35f17c0cb39d9a6338b))

* (**interpolate**) make output shape consistent with scipy ([`0709493`](https://github.com/kmnhan/erlabpy/commit/07094935339bd29ab1c2fce5bf0a4478121a69c7))

* do not reset offsets on accessor initialization if exists ([`5328483`](https://github.com/kmnhan/erlabpy/commit/5328483587da21de3e11966d4baab4bc0fefce12))

* (**itool**) properly parse colorcet cmaps ([`cb81811`](https://github.com/kmnhan/erlabpy/commit/cb81811f4a1b9ea24657d2aa27a7455825fb4eff))

* add some more guess constraints, fix type ([`82825a7`](https://github.com/kmnhan/erlabpy/commit/82825a79af122602435802d3a3aab349ef2e37b7))

* (**goldtool**) round roi position to 3 decimal places ([`d83fafa`](https://github.com/kmnhan/erlabpy/commit/d83fafa0712726fee935755a0d3f816100f90666))

* (**goldtool**) disable memmapping for parallel fitting ([`0c8b053`](https://github.com/kmnhan/erlabpy/commit/0c8b053e2a741f28aa622a2d9ba7e847a82bd8d0))

* duplicated data_dir ([`4b08754`](https://github.com/kmnhan/erlabpy/commit/4b087543059e730c0085b0a656424329cf99bf08))

* missing import ([`a1241da`](https://github.com/kmnhan/erlabpy/commit/a1241da6cbdffb68cb9405236d0bbb40bcf16815))

* (**io**) fix duplicated loader aliases ([`d832a48`](https://github.com/kmnhan/erlabpy/commit/d832a487e6e63befa24049992eed8a6fbd2a2be7))

* syntax error ([`63214f3`](https://github.com/kmnhan/erlabpy/commit/63214f35fd9009e04d7ed4299bf9327ba18f08e0))

* (**io.plugins.merlin**) files with non-standard names  are properly summarized ([`388dd02`](https://github.com/kmnhan/erlabpy/commit/388dd0266e82557a250f9d20dff3a918b139cf00))

* move positional to keyword only ([`ae300b3`](https://github.com/kmnhan/erlabpy/commit/ae300b3a9158940e07ccd585fc76750e3652593c))

* (**io**) return full path for get_files ([`bbaab33`](https://github.com/kmnhan/erlabpy/commit/bbaab3328c09a385762c0d65d00e24ef50ed9273))

* make fit result accessible ([`a40ae66`](https://github.com/kmnhan/erlabpy/commit/a40ae6616721a320eac51e69b7206ec7bf70f243))

* typo in multipeakfunction ([`cafe6cc`](https://github.com/kmnhan/erlabpy/commit/cafe6cc405650921b0a28cc9831b53b448e620ff))

* (**ktool**) round angle offsets ([`31c4730`](https://github.com/kmnhan/erlabpy/commit/31c4730663b24a344ded9a658580981732445100))

* (**io**) when given path to file, skip regex parsing ([`92019bb`](https://github.com/kmnhan/erlabpy/commit/92019bb56b5b8f818bc311a294150644149899ec))

* more realistic angle data generation ([`498e7f5`](https://github.com/kmnhan/erlabpy/commit/498e7f52ce567efe8029f2d4baec7ccb12d607db))

* return type ([`c089f44`](https://github.com/kmnhan/erlabpy/commit/c089f4404c4e63172e1da1d06d167ea75fef5841))

* move doc comments to above ([`429c868`](https://github.com/kmnhan/erlabpy/commit/429c8681dd90bc3c7c890377eb37f4ad1414c3f7))

* (**docs**) avoid direct import in conf.py ([`db930f0`](https://github.com/kmnhan/erlabpy/commit/db930f03f7d308f677dfa5c1385e3dd6c9bf3d0e))

* (**ktool**) default values and data orientation ([`1dcf1f4`](https://github.com/kmnhan/erlabpy/commit/1dcf1f4ca773c5ea5e30146a8fec927851ec54a5))

* Update ktool.py to keep up with refactoring changes ([`f23afb2`](https://github.com/kmnhan/erlabpy/commit/f23afb25c5a2141a53e7427cb0e4ea1291e008f3))

* fix typo in documentation ([`5c527a4`](https://github.com/kmnhan/erlabpy/commit/5c527a4da449867231cf4ca548cd147cd7657edb))

* properly execute in ipython ([`262b965`](https://github.com/kmnhan/erlabpy/commit/262b96518e38cecacf36141614da44d7bf7e3deb))

* momentum conversion offset ([`7836ce1`](https://github.com/kmnhan/erlabpy/commit/7836ce137d12f8b68725ca333ce2536a9e0ab24c))

* Update pyproject.toml with package-dir ([`c56fe0f`](https://github.com/kmnhan/erlabpy/commit/c56fe0f28b537d7e0601d33e9f20efc3f3c87310))

* Update requirements.txt ([`6b3150a`](https://github.com/kmnhan/erlabpy/commit/6b3150a1f49feb349a8fe5e3aa5d07b53d058a00))

* Update docs/requirements.txt ([`5835db9`](https://github.com/kmnhan/erlabpy/commit/5835db9d53c13bc6da16d2b0f5f2e4695a5cd6fd))

* enable dynamic versioning ([`cc0fc96`](https://github.com/kmnhan/erlabpy/commit/cc0fc96a191abf807a80433025b243fc9d11431c))

* Add packages to setuptools configuration ([`a2bbdcf`](https://github.com/kmnhan/erlabpy/commit/a2bbdcf6950de42c56c0f558d0d13c6521f7887f))

* Add version number to pyproject.toml ([`0caf0c4`](https://github.com/kmnhan/erlabpy/commit/0caf0c40f3dc943f786fe06e68d54b779367edad))

* update version ([`d394f20`](https://github.com/kmnhan/erlabpy/commit/d394f204387a1f3fdb18006298002d03b0a8b0d3))

* Update Sphinx configuration path in .readthedocs.yaml ([`21c1b1f`](https://github.com/kmnhan/erlabpy/commit/21c1b1f02b7e8b356d86074fe5c0f89895afbc90))

* (**itool**) ignore zerodivision ([`97e2bf5`](https://github.com/kmnhan/erlabpy/commit/97e2bf56c6e351dcfe6bd382c5b888ef12b44f4a))

* try to make autoscale_off context more reliable, needs testing ([`3a4726a`](https://github.com/kmnhan/erlabpy/commit/3a4726a90343bee7af916490a76679a027fc6aa1))

* gradient_fill now doesn&#39;t mess with autoscale ([`040db5a`](https://github.com/kmnhan/erlabpy/commit/040db5a6aeaebbdb628a5ff05bddce53f5f825bd))

* proper file handler termination ([`2ca7593`](https://github.com/kmnhan/erlabpy/commit/2ca7593b3036f8c13dda8ec7ae34a18ad8ef572c))

* acf2 stack dimension mismatch resolved ([`8dde2da`](https://github.com/kmnhan/erlabpy/commit/8dde2da853bc756cabc79e0587b93d5f99c92efb))

* validation changed, fixes #11 ([`8479814`](https://github.com/kmnhan/erlabpy/commit/84798146b8596ee4cd75e89282180f4e858176b3))

* (**interactive**) override copy, temporarily fixes #10 ([`4e99863`](https://github.com/kmnhan/erlabpy/commit/4e99863b2ce5b6fe82f0fbe3658cf9b024f69fb0))

* subclass scalarformatter for better compat ([`b69bfd6`](https://github.com/kmnhan/erlabpy/commit/b69bfd6607aa283c1e717e8183982ef1c40d9749))

* nice horizontal colorbar ([`2766e36`](https://github.com/kmnhan/erlabpy/commit/2766e3689ef3ebe5152ec5ce8fe22373127a507b))

* (**colors**) handle callable segmented cmaps ([`8135d73`](https://github.com/kmnhan/erlabpy/commit/8135d73e3d11f7e8126576b7b2281a15ec6e2920))

* (**itool**) keyboard modifier syntax ([`d593258`](https://github.com/kmnhan/erlabpy/commit/d5932583b6b36f288a72a5d4965c81c805c55c66))

* (**io**) fix da30 loading ([`ee0615d`](https://github.com/kmnhan/erlabpy/commit/ee0615dd31fff487139350df692045b646402c03))

* fix typo ([`6a84557`](https://github.com/kmnhan/erlabpy/commit/6a845573f8c0bc0d03c6003a218e6d1322c6a05e))

* load da30 map angle in radians ([`8d08c65`](https://github.com/kmnhan/erlabpy/commit/8d08c6533cbb57b225d5ee9ae99d1795ab5ec300))

* remove duplicate star ([`a018691`](https://github.com/kmnhan/erlabpy/commit/a0186915be3fc7065fbb19af5ff68d491167a468))

* (**itool**) update io ([`8c082b0`](https://github.com/kmnhan/erlabpy/commit/8c082b0f2c814e101e25dd99002e5fb45fccd744))

* (**itool**) handle ambiguous datasets ([`9226c12`](https://github.com/kmnhan/erlabpy/commit/9226c12f93ddceb35ea5b8852989dcf15620f9c6))

* (**itool**) catch overflow ([`ae1afe1`](https://github.com/kmnhan/erlabpy/commit/ae1afe16e67b76789c5adc59f1e0b149010cfc2f))

* patch breaking changes in lmfit 1.2.2 ([`8179fc3`](https://github.com/kmnhan/erlabpy/commit/8179fc308976d3ac8ca8575136bea7382ee85eac))

* make colorbar more robust ([`3a4968d`](https://github.com/kmnhan/erlabpy/commit/3a4968dd5983fc4bcde43bdf3056c3cc467be070))

* resistance data loading ([`7db3c4a`](https://github.com/kmnhan/erlabpy/commit/7db3c4abae28ae2ef99d1e8a54aa2744a3bde025))

* typo ([`a74732d`](https://github.com/kmnhan/erlabpy/commit/a74732d8429beb94a3009cbfde004387e4fae762))

* fix critical typo ([`3e10834`](https://github.com/kmnhan/erlabpy/commit/3e10834a8f9f5f78a9a41f3567f2fb4a9d092245))

* handle undetermined spectrum type ([`b59c3e9`](https://github.com/kmnhan/erlabpy/commit/b59c3e9068de1c26c1efced8ee0ffdc62c256f1c))

* (**ssrl52**) improve compatibility with old data ([`135b022`](https://github.com/kmnhan/erlabpy/commit/135b022996259da7ee379a49a0bb13327182ef51))

* remove now-redundant patches ([`687ece9`](https://github.com/kmnhan/erlabpy/commit/687ece99866628da3fbb302981d005cf57552cfb))

* improve ZT image view for 4D ([`c8d07b5`](https://github.com/kmnhan/erlabpy/commit/c8d07b522216f191a94a7ef40d7beda478e46074))

* keep slicer object for expected signal behavior ([`e0ebe85`](https://github.com/kmnhan/erlabpy/commit/e0ebe8519d972467332965be4ac097df0c26d530))

* keep slicer object for expected signal behavior ([`2c401e6`](https://github.com/kmnhan/erlabpy/commit/2c401e643f08eadbf3f3c84854332ee95d89451d))

* indexerror on loading higher dimensional data ([`e068bec`](https://github.com/kmnhan/erlabpy/commit/e068bec807b8f744c5b5925ef1208130c9136c2b))

* binning control order not updating on transpose ([`bdfda97`](https://github.com/kmnhan/erlabpy/commit/bdfda979475ecc8efeb1e8c80cf45a006f2ff85f))

* labels should not displace subplots ([`43b9258`](https://github.com/kmnhan/erlabpy/commit/43b925846306b311fcd309bc14a89390a51fb99e))

* revert clean_labels ([`1316fce`](https://github.com/kmnhan/erlabpy/commit/1316fce90142b41e423b0f7464768ea8b178599e))

* dimension mismatch ([`68fc306`](https://github.com/kmnhan/erlabpy/commit/68fc3063294bee4c89581b252cb49e71fc40b377))

* circular import ([`186cb03`](https://github.com/kmnhan/erlabpy/commit/186cb03f4a93cba01773b2a5dac848ad16a761cd))

* make qt progressbar more accurate ([`7c83b31`](https://github.com/kmnhan/erlabpy/commit/7c83b3100d7880577c533c3b17b77d6b643759c8))

* revert default pad ([`8e5715c`](https://github.com/kmnhan/erlabpy/commit/8e5715ce5f2c89c236ae7791820dcf5a7411fcc1))

* (**goldtool**) keyerror on code generation ([`e421719`](https://github.com/kmnhan/erlabpy/commit/e4217193441cabca6f8bce3e4001a530dea02678))

* (**io**) make save and load work with datasets ([`a4f1a12`](https://github.com/kmnhan/erlabpy/commit/a4f1a1279fb0070f38119e80b04bdbf19739b50c))

* (**style**) nonzero pad on savefig ([`b21a178`](https://github.com/kmnhan/erlabpy/commit/b21a17829d95d69aee2affe4c590c082b1cb22ca))

* gold fit autoscale ([`b81e4e8`](https://github.com/kmnhan/erlabpy/commit/b81e4e86f72eedbc63e256de14b70298c2c591f4))

* (**io**) can now load BL4 pxt files ([`d318c91`](https://github.com/kmnhan/erlabpy/commit/d318c91bdaa4adc2fb54be647e5932463e84474c))

* gold fit autoscaling ([`9ed8f86`](https://github.com/kmnhan/erlabpy/commit/9ed8f86f68006a648e0ed2868ff65dd451b9f2ff))

* disable covariance matrix scaling ([`a3264c7`](https://github.com/kmnhan/erlabpy/commit/a3264c70bf9d2bb502b83f7265482ff462dba11a))

* try to fix random segfault with numba ([`3505b42`](https://github.com/kmnhan/erlabpy/commit/3505b4291b34697732f2dccc33f81f5c32e5fcad))

* stupid regression in plot_array ([`2ef2ad6`](https://github.com/kmnhan/erlabpy/commit/2ef2ad62227798c0ab3b72ccd9132086fd8a3e45))

* stupid regression on rename ([`cc37d8e`](https://github.com/kmnhan/erlabpy/commit/cc37d8e86e7a0ea36a89b265b6409aa2898c302c))

* wrong attributes ([`01812ba`](https://github.com/kmnhan/erlabpy/commit/01812bad65847dbe33d1afe013c564898c3c99a6))

* invert before gamma ([`c70bff5`](https://github.com/kmnhan/erlabpy/commit/c70bff5b59a17d59260a1289e8c3df007194b762))

* (**io**) fix livexy and livepolar loader dims ([`edcf9a8`](https://github.com/kmnhan/erlabpy/commit/edcf9a8640169bebb427f768baeb82735494ab40))

* (**itool**) restore compatibility for float64 data ([`4e8baab`](https://github.com/kmnhan/erlabpy/commit/4e8baabb0f592294862bcbda41bdf1422748ab62))

* resolve type related problems ([`4ec5bdd`](https://github.com/kmnhan/erlabpy/commit/4ec5bddc2c82076eab3e973845d21f188cf09161))

* fix typo in comment ([`8cd71df`](https://github.com/kmnhan/erlabpy/commit/8cd71df415b5715a64c81dcf338f44abee21fa7d))

* remove type hints, were causing thread errors ([`1ed309c`](https://github.com/kmnhan/erlabpy/commit/1ed309c58ea3c8da696d352e1df0ba03903c223b))

* edge correction with callable ([`8a1052f`](https://github.com/kmnhan/erlabpy/commit/8a1052fe2678baca09042d8eb644d41a9ab48f56))

* default pad changed ([`a2039c2`](https://github.com/kmnhan/erlabpy/commit/a2039c2b9db91950d7d627b82614fad0a536f906))

* fix curve fitting on notebook ([`4126aa0`](https://github.com/kmnhan/erlabpy/commit/4126aa0fa6061c880b03d188199b3d40d64c4b84))

* compatibility with PyQt6 ([`01f550e`](https://github.com/kmnhan/erlabpy/commit/01f550e463ea14720d0e84cd43bf98501cf0b554))

* stupid commit ([`d3b1f53`](https://github.com/kmnhan/erlabpy/commit/d3b1f53448805691c3de358e2cdfca3cb631866a))

* (**io**) add compatibiity check, fixes #9 ([`35c6bd7`](https://github.com/kmnhan/erlabpy/commit/35c6bd73753af06f49130dddc642baca008044e4))

* correct color limits for new cursors ([`25e54f4`](https://github.com/kmnhan/erlabpy/commit/25e54f46bbca24ac54aa2cccf1abaacf9403ec68))

* add PyQt6 compatibility ([`713cea2`](https://github.com/kmnhan/erlabpy/commit/713cea28683643f27c5b163c1aba0e332c168b30))

* pyqt-compatible multiple inheritance, fixes #7 ([`de37753`](https://github.com/kmnhan/erlabpy/commit/de377534db051b852d528ab8dd6abf212bf3e1a0))

* pyqt-compatible multiple inheritance, fixes #7 ([`7fb28d8`](https://github.com/kmnhan/erlabpy/commit/7fb28d834b149c1a7dd8bd09162d09fcf4890004))

* minor fixes ([`2b4e61c`](https://github.com/kmnhan/erlabpy/commit/2b4e61cb086ae5f813aa1d784c722fbbc8613330))

* 2d colormap incorrect normalization ([`a425cb1`](https://github.com/kmnhan/erlabpy/commit/a425cb1bd560aa2e7047321ec2b21ecd0bc0779d))

* set tol to 10x eps float32, fixes #5 ([`d4d2dc7`](https://github.com/kmnhan/erlabpy/commit/d4d2dc72fda8ece6875d50f03e337d1b9dd222de))

* convert everything to float32, fixes #2 ([`8f2b58f`](https://github.com/kmnhan/erlabpy/commit/8f2b58f5218a4bdcb2d6fb8c8d6d8ef798c84a03))

* fixes #3 ([`1ae6ff8`](https://github.com/kmnhan/erlabpy/commit/1ae6ff8cd58db64fc439cd034f2f93f6961bd6c9))

* fixes #1 along with some memory optimization ([`a667c6a`](https://github.com/kmnhan/erlabpy/commit/a667c6a44fda268b7a450b4522d2ada193de0639))

* choose nearest for zero width when plotting slices ([`6ba03da`](https://github.com/kmnhan/erlabpy/commit/6ba03da2c6054044701accc20763bac424ac3bcf))

* (**itool**) multicursor colorbar ([`65bd992`](https://github.com/kmnhan/erlabpy/commit/65bd9923680a11e96350b8f9dab079934199bd3a))

* shift by DataArray ([`fd38eaf`](https://github.com/kmnhan/erlabpy/commit/fd38eaf479b1109f57f3dc30e7b4cc5964459bf7))

* force qt api ([`3613853`](https://github.com/kmnhan/erlabpy/commit/36138538c31ba1baaf6c5606372039aeb5dcfb95))

* temperature not required when fitting with broadened step edge ([`4b83d87`](https://github.com/kmnhan/erlabpy/commit/4b83d8713044e8dfdaf605653cc60f8fe6057ea5))

* fix colorbar conflict with multiple cursors ([`6cb35b6`](https://github.com/kmnhan/erlabpy/commit/6cb35b68b9643dc19d9e197e67554c203e5c4b99))

* wrong sign in powernorm ([`b4df421`](https://github.com/kmnhan/erlabpy/commit/b4df42102822da0e41991b9c176a24b900220088))

* rewrite pyqtgraph colormap normalization ([`e2a807e`](https://github.com/kmnhan/erlabpy/commit/e2a807e7a0c7e556a6d2750cd4052df911b5ec3c))

* (**itool**) fix misc. bugs ([`15c737c`](https://github.com/kmnhan/erlabpy/commit/15c737cd0096a27d5582ab1f084a7ad070b28bcc))

* retain clipboard after window close ([`e800bc7`](https://github.com/kmnhan/erlabpy/commit/e800bc72fae148e0f9fc4df646566013bf7082b8))

* (**bz**) input reciprocal lattice vectors ([`32df4d7`](https://github.com/kmnhan/erlabpy/commit/32df4d7e673fbcfb886c959450d30c9bf1fa1d27))

* automatic figure detection ([`f0f2ef9`](https://github.com/kmnhan/erlabpy/commit/f0f2ef9810ec637a46d316b37578bc45307ea881))

* (**itool**) regression: aspect ratio for 2D arrays ([`087ba24`](https://github.com/kmnhan/erlabpy/commit/087ba24cb99b12901ae928cd2b3288c25570c9eb))

* (**itool**) better handle drag ([`23cdbf0`](https://github.com/kmnhan/erlabpy/commit/23cdbf039118980c50760a7ebf94b24b48a6b6c8))

* replace bitwise inversion on boolean ([`5569060`](https://github.com/kmnhan/erlabpy/commit/5569060a356aad3d1d3141b5accce61f66e78418))

* works properly with integer coordinates ([`adc0074`](https://github.com/kmnhan/erlabpy/commit/adc00746668456576709b1db9525ea5bb6e5bb11))

* update some deprecated syntax ([`839f1a6`](https://github.com/kmnhan/erlabpy/commit/839f1a60132ff03b578224109553491e49a80aae))

* patch for PySide6 6.4 ([`6e9d1b9`](https://github.com/kmnhan/erlabpy/commit/6e9d1b9ed5899cb4f639806281da55d987e01f1d))

* better aspect ratio for 2D arrays ([`b002350`](https://github.com/kmnhan/erlabpy/commit/b002350164c4f9c7cd233f50488937f55dff8e46))

* regression as per pyqtgraph/pyqtgraph@cead5cd ([`74c9f81`](https://github.com/kmnhan/erlabpy/commit/74c9f8181033c9f7cfbe8eaa15b64172dc287601))

* stupid rad2deg handling ([`4a6d629`](https://github.com/kmnhan/erlabpy/commit/4a6d629261e6bc0da286b3c687d8fe8af46ee589))

* (**goldtool**) catch varname exceptions ([`aa22dd7`](https://github.com/kmnhan/erlabpy/commit/aa22dd7833e1fd7e3f1a36fa48ccee7c6420dffe))

* (**itool**) wrong signals ([`a0f0036`](https://github.com/kmnhan/erlabpy/commit/a0f00366be305f302a52e84eac5e4fc97d7d9d54))

* colorbar aspect specification ([`456f370`](https://github.com/kmnhan/erlabpy/commit/456f370907ca163c30567992da8edbfebcf18caf))

* docstring and labeling ([`4bbedda`](https://github.com/kmnhan/erlabpy/commit/4bbedda90b83f7be32494cfd63bc9ac7dfb94327))

* attempts to overwrite read-only object ([`0cf8606`](https://github.com/kmnhan/erlabpy/commit/0cf8606cd69ccdeaae4d84945e181b9e2b07a846))

* (**plot_array**) colorbar extents ([`2fe8a93`](https://github.com/kmnhan/erlabpy/commit/2fe8a930f2cf8fd37d88958231388e4a446b1443))

* broken binning ([`f217938`](https://github.com/kmnhan/erlabpy/commit/f217938379ea7d69613fdef462b26ed41d4850cd))

* (**itool**) isocurve wrong orientation ([`dfd6aac`](https://github.com/kmnhan/erlabpy/commit/dfd6aac29b846e5b187791aa7fe2708dc95abe76))

* (**itool**) tab position ([`a70af2b`](https://github.com/kmnhan/erlabpy/commit/a70af2be59c80231be605e49304d226ed5c24f0c))

* fine-tune automatic colormap assignment ([`1d035bb`](https://github.com/kmnhan/erlabpy/commit/1d035bb6ce178f7c13b4c2cd35b20fb69342a627))

* (**itool**) smarter mouse detection ([`3032c0c`](https://github.com/kmnhan/erlabpy/commit/3032c0cc15c23a3f345300de0eb57080d31b7604))

* noisetool import ([`c99bfd2`](https://github.com/kmnhan/erlabpy/commit/c99bfd2980127f07fc6b361d9156e5a68be2711c))

* invalid import ([`5fd9010`](https://github.com/kmnhan/erlabpy/commit/5fd90104cd8e2a2b8158df6acf88dfa7408bc102))

* flickering cursor when moving ([`439157b`](https://github.com/kmnhan/erlabpy/commit/439157bf3fb3b377ebb17a9f6c7a37b87fafc905))

* (**itool**) fix transpose ([`0b23f61`](https://github.com/kmnhan/erlabpy/commit/0b23f61dfb7c3dec8a534d5b5c7dae560418376f))

* (**itool**) change wrong 2D layout ([`e04f162`](https://github.com/kmnhan/erlabpy/commit/e04f162ac0f88f1c576942e9eb820044a17ea791))

* (**itool**) revert ([`ca4335a`](https://github.com/kmnhan/erlabpy/commit/ca4335a77ba467d46fa0d68e80927aa6ee8e8a24))

* (**itool**) adjust blitting ([`0873882`](https://github.com/kmnhan/erlabpy/commit/0873882a7692da94ec05ee659030d0e26be51585))

* (**itool**) properly functioning pan &amp; zoom ([`26ce4e8`](https://github.com/kmnhan/erlabpy/commit/26ce4e8efce62eb2ab38f14b097f0121ef6205d4))

* fix offset not syncing across energy ([`7ec4072`](https://github.com/kmnhan/erlabpy/commit/7ec4072c93e1b41f24e4247afc414244a43badac))

* fix cursor issue and home resetting limits ([`f3374ab`](https://github.com/kmnhan/erlabpy/commit/f3374abaa8643aef78f33366dfab9a0994b475ce))

* simplify cursor customization ([`a5ff97f`](https://github.com/kmnhan/erlabpy/commit/a5ff97f6cffcd2cc3da0a31f44d62ac590c663e3))

* make plotting imports backwards compatible ([`82f132b`](https://github.com/kmnhan/erlabpy/commit/82f132b89ffd3c911208ae93fa8d47b1825e7928))

* (**plotting**) fix UnboundLocalError ([`e4f5bca`](https://github.com/kmnhan/erlabpy/commit/e4f5bca381c881d159c9f6a09ebdc86616a4f467))

* fix typo ([`39d160a`](https://github.com/kmnhan/erlabpy/commit/39d160a15a6d8990061de48a08b431a2ff35f699))

### Performance

* (**interpolate**) make some jitted functions always inlined ([`4624b16`](https://github.com/kmnhan/erlabpy/commit/4624b16ec926546876a98864abe3c1d47b6fc221))

* (**itool**) fps optimization, add proper support for nonuniform dimensions ([`6df84db`](https://github.com/kmnhan/erlabpy/commit/6df84db7156de88b2ee8d100a2ce0c45f8b2135a))

* cleanup, reduce import time ([`dbfcce3`](https://github.com/kmnhan/erlabpy/commit/dbfcce38f9b874b8532666ff7479dbc789fef657))

* (**itool**) add cached properties ([`0124093`](https://github.com/kmnhan/erlabpy/commit/0124093a25ba47e5bc304125fcfd62f6768beb13))

* limit fps with SignalProxy ([`f7ce099`](https://github.com/kmnhan/erlabpy/commit/f7ce099f87e97adeb27a8e4f2d760b6ddd4f4d22))

* get coords efficiently ([`5582afc`](https://github.com/kmnhan/erlabpy/commit/5582afc9e86e8306e6499da52b2f2c06604b029b))

* better min/max performance, fixes #4 ([`3c4aa13`](https://github.com/kmnhan/erlabpy/commit/3c4aa1369fbbf10fa5a2597dc958bb559e4bb26a))

* (**slicer**) contiguity optimizations ([`6f2b543`](https://github.com/kmnhan/erlabpy/commit/6f2b543a11d89d744961e1f1716ad542a9bab163))

* (**itool**) update only relevant axes ([`c561650`](https://github.com/kmnhan/erlabpy/commit/c5616502767e8f14425ba07bed4660355eba1203))

### Refactor

* apply linter suggestions ([`edfc91a`](https://github.com/kmnhan/erlabpy/commit/edfc91a6712620588106471ae03975a76976f634))

* apply linter suggestions ([`231f794`](https://github.com/kmnhan/erlabpy/commit/231f794a3aaf6575528df63b70a4478cb9769fe8))

* apply some linter suggestions ([`4e1f66c`](https://github.com/kmnhan/erlabpy/commit/4e1f66c6b19eb2e3674511e19309c9127d610369))

* (**goldtool**) ui changes ([`464d05e`](https://github.com/kmnhan/erlabpy/commit/464d05ee270cf601c322536b3dafd3c7bf0e9f7f))

* rename variable ([`32a901e`](https://github.com/kmnhan/erlabpy/commit/32a901e0de1b7d0c5fb6858a275b8bf3cb69e801))

* cleanup namespace ([`4779e46`](https://github.com/kmnhan/erlabpy/commit/4779e46920cf88a49f211d1d7f856f61e6c84c2a))

* (**io**) minor changes to summary format ([`d26a8f7`](https://github.com/kmnhan/erlabpy/commit/d26a8f78f4d86f7b0654a0949e644d2f96652b50))

* move functions  to submodule ([`824a2fb`](https://github.com/kmnhan/erlabpy/commit/824a2fb4847d5f29dc51f50517a51aae3709d3df))

* fit functions submodule ([`2bc555c`](https://github.com/kmnhan/erlabpy/commit/2bc555cfb050d28523f805b75053dcf836759a7f))

* (**io.dataloader**) fix _repr_html_ to return valid html table ([`73adb0f`](https://github.com/kmnhan/erlabpy/commit/73adb0ffffc618d28903a4c5814923dac5502186))

* (**io.dataloader**) make reverse_mapping a staticmethod ([`983c02b`](https://github.com/kmnhan/erlabpy/commit/983c02bda02c97a7eb179c35f98d0a89c0814cd9))

* (**io**) change dict format ([`a13b064`](https://github.com/kmnhan/erlabpy/commit/a13b06465f53c78b9edea538e9825d83f66b86f0))

* add type annotation ([`7e08658`](https://github.com/kmnhan/erlabpy/commit/7e08658093fe24a383366f722f044029f097cb69))

* use match-case for enum matching ([`cc9e112`](https://github.com/kmnhan/erlabpy/commit/cc9e1126f7571df3e68e568375773a3a4dce63b5))

* change package directory structure; BREAKING CHANGE ([`5385ec7`](https://github.com/kmnhan/erlabpy/commit/5385ec70b23775ddd19b02459cbb0d0630143454))

* format with black ([`1655eec`](https://github.com/kmnhan/erlabpy/commit/1655eec321bd12acc37681e1d55e21155ec34252))

* remove code trying to infer spectrum from dataset ([`3fa6b1b`](https://github.com/kmnhan/erlabpy/commit/3fa6b1b25dc70add644e9719488ae55df314238c))

  From now on all data should be strictly a xr.DataArray

* deprecate old ktool, replace ([`dbd972f`](https://github.com/kmnhan/erlabpy/commit/dbd972f0e56197c1796695921b6cc021b3f4d190))

* (**gold**) add type annotation ([`06c39a7`](https://github.com/kmnhan/erlabpy/commit/06c39a790109e23a81e88c885a1fa0ee1f615e41))

* Update requirements.txt so that igor2 is not editable ([`0e20bc4`](https://github.com/kmnhan/erlabpy/commit/0e20bc48d458a363b4c2e2b829a59ac6e1aba6cf))

* temporarily disable annotate_cuts_erlab ([`08e4b52`](https://github.com/kmnhan/erlabpy/commit/08e4b527316b94b0a1d2912b67d1b67ea4571755))

* (**itool**) modify test code ([`8934fba`](https://github.com/kmnhan/erlabpy/commit/8934fba1b20cf182455d2a70e24ff835d6ca96be))

* (**exampledata**) tweak defaults ([`73eb495`](https://github.com/kmnhan/erlabpy/commit/73eb4955ff3ed7f568136efce5efaf9ab929dfba))

* cleanup ([`88b418f`](https://github.com/kmnhan/erlabpy/commit/88b418f61491917e275ec7b5cc544c157617429d))

* try garbage collection, failed ([`ac75267`](https://github.com/kmnhan/erlabpy/commit/ac75267921884299970c5d78b1633835700dce6e))

* typo ([`f1df9ea`](https://github.com/kmnhan/erlabpy/commit/f1df9ea0a4e128013d21e2f2b5e9a64f9acf57a3))

* reorder functions ([`ee6886f`](https://github.com/kmnhan/erlabpy/commit/ee6886fa7810c3bac26e05b1407ceebc1563ad54))

* organize imports ([`70b2b9b`](https://github.com/kmnhan/erlabpy/commit/70b2b9bed7e395a824ee3e1b240937b570e670ec))

* cleanup ([`8b01e73`](https://github.com/kmnhan/erlabpy/commit/8b01e7393b7d7a6ae23b85e6f784a7f760cdd74a))

* relocate color related classes ([`b322fb8`](https://github.com/kmnhan/erlabpy/commit/b322fb8ced0b41a1ec8e7dbaf9ac50c8f7aa853c))

* (**io**) cleanup imports ([`125f672`](https://github.com/kmnhan/erlabpy/commit/125f672edd3b6f94943c2a3d221e626f9a58c820))

* add submodules to analysis initialization ([`30ab7d0`](https://github.com/kmnhan/erlabpy/commit/30ab7d0f71a6c09633e54345c7317215934c1ca6))

* rename igor procedure file ([`92d495c`](https://github.com/kmnhan/erlabpy/commit/92d495cb114a921a8ca9d1fde81a4e6400d53089))

* cleanup ([`025b39b`](https://github.com/kmnhan/erlabpy/commit/025b39bb557ab364e6c2f5ff14ddb2956ba52467))

* move colormap controls up ([`13a44fe`](https://github.com/kmnhan/erlabpy/commit/13a44febb156fe657ed39636323c124650010501))

* follow pep8 dunder names ([`f954b7b`](https://github.com/kmnhan/erlabpy/commit/f954b7ba76b877a7376b45a9a91871e9b8ea4d25))

* replace deprecated syntax ([`6d48a08`](https://github.com/kmnhan/erlabpy/commit/6d48a088f632c0a6d467067463e706c410ad8978))

* cleanup ([`5b24106`](https://github.com/kmnhan/erlabpy/commit/5b241068cbd60de73da420149fbd4ba2f09a1ac9))

* cleanup and add some type annotation ([`daabfb8`](https://github.com/kmnhan/erlabpy/commit/daabfb87b07fa71a618b127c2cd6a51bbc0e3e18))

* move DictMenuBar to utilities ([`2b3fcb1`](https://github.com/kmnhan/erlabpy/commit/2b3fcb1776115cfee987db4e74c4d34a5d679bcf))

* change gold fitting syntax ([`e6043d6`](https://github.com/kmnhan/erlabpy/commit/e6043d6ae3aa0bc3b90db2d0378c9bd3c7a7c948))

* update clean_labels with matplotlib API ([`13013dd`](https://github.com/kmnhan/erlabpy/commit/13013dd7ebb16e03b1609e4e44315b22139e8f28))

* move some functions ([`866eca5`](https://github.com/kmnhan/erlabpy/commit/866eca5c5a5b0ccf38383aead2b5871903c5c50a))

* follow pep8 dunder names ([`b043701`](https://github.com/kmnhan/erlabpy/commit/b043701e428c7e1c461f64f57667d8b25642a5ce))

* imagetool is now a package ([`1a39b82`](https://github.com/kmnhan/erlabpy/commit/1a39b8283582cccacd448addb3566f6f1e882744))

* follow pep8 dunder names ([`dbdc0f6`](https://github.com/kmnhan/erlabpy/commit/dbdc0f62cb048ce5c2f248a21e112476d5ebb3df))

* cleanup ([`aace486`](https://github.com/kmnhan/erlabpy/commit/aace486d8652e07ab9007f89a49a29cca3d4e1af))

* cleanup annotations ([`12e590c`](https://github.com/kmnhan/erlabpy/commit/12e590c2a771d089e29aff02492851448640b605))

* transition to new polynomial api ([`5461c34`](https://github.com/kmnhan/erlabpy/commit/5461c342f7a84d1ba98f924c74be6ecde16cf6e4))

* cleanup syntax ([`310c2ba`](https://github.com/kmnhan/erlabpy/commit/310c2babf15d7a22900806cac308643bb22f8bce))

* (**itool**) changes to layout ([`7a684e4`](https://github.com/kmnhan/erlabpy/commit/7a684e400772036dde7af4f8a4747c8db208eb0f))

* remove dependency on darkdetect ([`81fa963`](https://github.com/kmnhan/erlabpy/commit/81fa963a91fddd1eb2b7f316fcec04eb27716d13))

* format code ([`adf6cb3`](https://github.com/kmnhan/erlabpy/commit/adf6cb3cd48e494d708c79906347e3c54fcd3e20))

* remove io module, add as package ([`5d1280a`](https://github.com/kmnhan/erlabpy/commit/5d1280a3c7893bb86a42fdf2a71b6825c5db6986))

* try to merge commit error ([`2c61d45`](https://github.com/kmnhan/erlabpy/commit/2c61d4575bfcafe22b63527a5dc3f318147cb8c4))

* menubar cleanup ([`f147f11`](https://github.com/kmnhan/erlabpy/commit/f147f114364bd9fab0ba456a24113d58dc549652))

* organize code ([`93ebd5b`](https://github.com/kmnhan/erlabpy/commit/93ebd5b59307b8524452a3c155a5851988070923))

* rename constants ([`ebaeaa1`](https://github.com/kmnhan/erlabpy/commit/ebaeaa11a0c718266512261f640441a06e2703e1))

* cleanup ([`f29bdf3`](https://github.com/kmnhan/erlabpy/commit/f29bdf31e035fc8891f95b8479bf2c05f39ffc03))

* syntax cleanup ([`2678967`](https://github.com/kmnhan/erlabpy/commit/26789677f4fb489af39edcab5b0f73714c1e52b4))

* interactive is no longer a submodule of plotting ([`3da646c`](https://github.com/kmnhan/erlabpy/commit/3da646cf4f13c717cf99ec9cc2e3e449e4d4bfdb))

* remove relative imports ([`96a2095`](https://github.com/kmnhan/erlabpy/commit/96a209572843c35dff50dc383c00bf83c48eb263))

* lint with flake8 ([`ca34ea1`](https://github.com/kmnhan/erlabpy/commit/ca34ea1b82fccf4ed73b5fdf4f8dfedca303cfa2))

* remove sandbox notebook ([`504cbc9`](https://github.com/kmnhan/erlabpy/commit/504cbc9f58eb43fb142541c1a5c6b97b4ced3c64))

* minor changes ([`692fbc0`](https://github.com/kmnhan/erlabpy/commit/692fbc0d95657c2fcd0646e5a8a65aa0161d2aca))

* expose property label ([`9e2dafd`](https://github.com/kmnhan/erlabpy/commit/9e2dafd4948437b3e3b308a7a53a2a331b551941))

* syntax cleanup ([`37166af`](https://github.com/kmnhan/erlabpy/commit/37166af6b1cbb7bdebccdd8a90e4953685859ab0))

* syntax cleanup ([`19884df`](https://github.com/kmnhan/erlabpy/commit/19884df9909684f2352e6583cad0abb523e958de))

* syntax cleanup ([`59e2921`](https://github.com/kmnhan/erlabpy/commit/59e2921c5c7d897f3cfdb17417d46127bb95af60))

* change clipboard handler ([`d687499`](https://github.com/kmnhan/erlabpy/commit/d687499b43a027ea97a05243d57c88ae6f756398))

* some changes regarding colorbar, may be reverted ([`bc913c0`](https://github.com/kmnhan/erlabpy/commit/bc913c07dcd9bbe36f6a28112f32d5d8ff077152))

* deprecate old imagetool ([`26e20ec`](https://github.com/kmnhan/erlabpy/commit/26e20ec474776274b07d4221976c5d5cee3d2b8b))

* cleanup enums ([`ec42673`](https://github.com/kmnhan/erlabpy/commit/ec4267373c9f3caf36c2ef4a5f7870e4fa205bdf))

* cleanup ([`9eef9f9`](https://github.com/kmnhan/erlabpy/commit/9eef9f94d54badb74578fac96dd8854fcfc1f393))

* cleanup axis labels and signals ([`3aec658`](https://github.com/kmnhan/erlabpy/commit/3aec658a2c93a3244f8053b28f27b8446fbef61e))

* add typing ([`4951388`](https://github.com/kmnhan/erlabpy/commit/4951388b93ec37fffaf04e44ad02ee67b6f92d6d))

* change class name ([`a7d51e5`](https://github.com/kmnhan/erlabpy/commit/a7d51e5c7b5eead59f3a7a0ecd667adae6d90c3a))

* minor improvement and cleanup ([`b2886e4`](https://github.com/kmnhan/erlabpy/commit/b2886e447737c954d01eac2a016ef68ecbfc97b6))

* move spinbox to utilities ([`906a0f7`](https://github.com/kmnhan/erlabpy/commit/906a0f7ed919f7f800007633f27d5be0f1f33a2a))

* remove dependency on pyqtgraph dock ([`35b5c3f`](https://github.com/kmnhan/erlabpy/commit/35b5c3f15e7ad2524c239e270dbbecfde5287f78))

* preparation and cleanup for leaving pyqtgraph dock ([`5721243`](https://github.com/kmnhan/erlabpy/commit/572124321b63c805a94aefb371653918cc174b9a))

* cleanup ([`1ac0de1`](https://github.com/kmnhan/erlabpy/commit/1ac0de10ddd215d208eb4ca7038e35d35abd826a))

* remove font refresh macro ([`3780f23`](https://github.com/kmnhan/erlabpy/commit/3780f23b4b29b271bc4a640e5c187e176f3813b8))

* restructure plotting module ([`7259ab3`](https://github.com/kmnhan/erlabpy/commit/7259ab3d34ad289225f2972dacdfd6aa8a827007))

* split interactive colors module ([`a06439c`](https://github.com/kmnhan/erlabpy/commit/a06439cfdb044d379df25fc6643250662dcc81c4))

* change to updated colorbar ([`cdd7a8e`](https://github.com/kmnhan/erlabpy/commit/cdd7a8ed04b1c16a3dfd293668604d00e9d3926f))

* prevent namespace collision ([`f3599b2`](https://github.com/kmnhan/erlabpy/commit/f3599b2d465f4c56db38046f8e9127fe519c78dd))

* clean up comments and format with black ([`90d67b9`](https://github.com/kmnhan/erlabpy/commit/90d67b96448c1786a1fc7dabd5215ab7b6d0cfcb))

* format with black ([`7f79731`](https://github.com/kmnhan/erlabpy/commit/7f7973120c674d1d34f574aa2fedc8ac240bc402))

* remove deprecated ([`a655778`](https://github.com/kmnhan/erlabpy/commit/a655778161208a9c2f605a27343afe82792c4218))

* format with black ([`88548cb`](https://github.com/kmnhan/erlabpy/commit/88548cbeda20da21ffd02c284a4086e0e50d3bbe))

* format with black ([`306e530`](https://github.com/kmnhan/erlabpy/commit/306e5304c34b3e7c648fe2ea030259fa7a892a42))

* reduce pyarpes dependency ([`14e85fb`](https://github.com/kmnhan/erlabpy/commit/14e85fbcfa7d95df674381f8882db80f39e7bda3))

* format with black ([`7696b66`](https://github.com/kmnhan/erlabpy/commit/7696b66c6ef93a04c353e8e199c096c2458794ff))

* format with black ([`f6fc29c`](https://github.com/kmnhan/erlabpy/commit/f6fc29ce22a541d9143397ffb902c34d752cdad3))

* (**itool**) cleanup hide buttons ([`2649fe9`](https://github.com/kmnhan/erlabpy/commit/2649fe9ae12893e719a456f0cf22b7eee097df0f))

* format code ([`c69b9cc`](https://github.com/kmnhan/erlabpy/commit/c69b9ccca3ee911b5b07bd6393a43444ba771f33))

* trivial changes ([`6d5df90`](https://github.com/kmnhan/erlabpy/commit/6d5df9019932c4c6ad5930d952714e0bd181cacf))

* easy import ([`b308d3d`](https://github.com/kmnhan/erlabpy/commit/b308d3d546ee92b04299d679042564149759206b))

* cleanup and reorganize ([`136715b`](https://github.com/kmnhan/erlabpy/commit/136715b46da525bf7b8e28d2e26c306754d93e61))

* update with new imagetool ([`6ec6f65`](https://github.com/kmnhan/erlabpy/commit/6ec6f65ca653ef1d90807c2f8448d12ae57d69c2))

* deprecate mpl-based imagetool ([`5a28947`](https://github.com/kmnhan/erlabpy/commit/5a289471c08022bd4ac4502b176af73d4157c31c))

* sort imports ([`60f52f5`](https://github.com/kmnhan/erlabpy/commit/60f52f50feb68b48c1e3eaf965becf565129b8c0))

* cleanup code ([`58ca4ed`](https://github.com/kmnhan/erlabpy/commit/58ca4edeab44c0e665e0b272c1a42a0376d2223a))

* reorganize plotting routines ([`f5ef54f`](https://github.com/kmnhan/erlabpy/commit/f5ef54f22c7a2b51c794fe00816fa8778da53060))

### Style

* format with black ([`82b2938`](https://github.com/kmnhan/erlabpy/commit/82b29382c3c01f7aea4c0ec511c3828de1177e10))

* remove relative imports ([`096a78d`](https://github.com/kmnhan/erlabpy/commit/096a78df4e09643961f358cf13d200d2428afc6f))

* remove relative imports ([`9891273`](https://github.com/kmnhan/erlabpy/commit/9891273e568c6673541866d3475990aed965fd5e))

* add typing ([`1f16ebb`](https://github.com/kmnhan/erlabpy/commit/1f16ebb398dc29a4d0bddf77a422af926b7952da))

* remove unused imports ([`68ef006`](https://github.com/kmnhan/erlabpy/commit/68ef00632f75978e380c15bd6ad9040ef6565520))

### Test

* tests initial commit ([`fde2283`](https://github.com/kmnhan/erlabpy/commit/fde2283c646ce3b5335c6a8a2960da19380e7827))

  Add basic tests for fast binning, momentum conversion, and interpolation.

### Unknown

* Merge branch &#39;main&#39; of https://github.com/kmnhan/erlabpy ([`ffb7de8`](https://github.com/kmnhan/erlabpy/commit/ffb7de8345100a5f935f9a97739f549f31bb014d))

* Create LICENSE ([`2ee8814`](https://github.com/kmnhan/erlabpy/commit/2ee881436ded6073c889693af3e50f6124d651ce))

* Refactor atoms and add docstrings ([`115e710`](https://github.com/kmnhan/erlabpy/commit/115e710db379d2d7e46da5cfd9e13625932a0226))

* refactro: code cleanup ([`df22454`](https://github.com/kmnhan/erlabpy/commit/df224544650888d39c42d4f6752f1b845e53f750))

* Merge branch &#39;main&#39; of https://github.com/kmnhan/erlabpy ([`a122847`](https://github.com/kmnhan/erlabpy/commit/a122847c4e18bb0ef977c366b8290dbcf02819f1))

* merge conflicts ([`056268f`](https://github.com/kmnhan/erlabpy/commit/056268f40e92a2fa2d23f2fcbdd2e008f8dfd43c))

* merge conflicts ([`6f39ac2`](https://github.com/kmnhan/erlabpy/commit/6f39ac2690dfbcc9fc458702f0d0f0065dbeeff3))

* refactor : constants ([`1c6fe03`](https://github.com/kmnhan/erlabpy/commit/1c6fe03201d3df0cf7016d34767d973f548963ad))

* minor changes ([`28c4e0d`](https://github.com/kmnhan/erlabpy/commit/28c4e0db9c575b7c095c58a34f1744917935dd67))

* Merge branch &#39;main&#39; of https://github.com/kmnhan/erlabpy ([`17ea8e3`](https://github.com/kmnhan/erlabpy/commit/17ea8e3924b66a4fa05df35e4382e5637e183d64))

* doc: add documentation ([`0952a41`](https://github.com/kmnhan/erlabpy/commit/0952a414613121fdbc579ff114c8d5aa1116a593))

* doc: documentation for sizebar ([`bd4da7c`](https://github.com/kmnhan/erlabpy/commit/bd4da7ca074f5080a7792b76d79949adfdd5663c))

* doc(itool): add docstrings ([`0d3237a`](https://github.com/kmnhan/erlabpy/commit/0d3237a09af5af8164696197e49c6c2bfbacbc5b))

* add transform ([`0587967`](https://github.com/kmnhan/erlabpy/commit/0587967324f5d43d64255a1800101ed9a5e10fc3))

* fixes and refactors ([`a46cea0`](https://github.com/kmnhan/erlabpy/commit/a46cea07b8b4342ee722f238b9e99f7052423987))

* doc: update docstring (WIP) ([`5bf5438`](https://github.com/kmnhan/erlabpy/commit/5bf5438fc930f5c6bb38fd98d955c23f9474a7d3))

* update .gitignore ([`c3ba12e`](https://github.com/kmnhan/erlabpy/commit/c3ba12e80af6af3826f6b1b5d670880025c257fa))

* fix and refactor ([`7c53791`](https://github.com/kmnhan/erlabpy/commit/7c537910017973a8269f95c346648ccc174ead15))

* doc: update docstring ([`3446806`](https://github.com/kmnhan/erlabpy/commit/34468065afa8906e240a1e252b627edebd3ef53d))

* doc: update README and installation ([`cbdf8c4`](https://github.com/kmnhan/erlabpy/commit/cbdf8c4547019977f1dc3a892d46ee762c54c4d3))

* Delete erlab.egg-info directory ([`737347b`](https://github.com/kmnhan/erlabpy/commit/737347b34f9128eefb7b61480c38becea40a6854))

* update .gitignore ([`b3b84a6`](https://github.com/kmnhan/erlabpy/commit/b3b84a66bc8dd1a226dfaa06e45f74ebc163a520))

* Initial commit ([`85e2ff8`](https://github.com/kmnhan/erlabpy/commit/85e2ff8e0f3b7f383e999b56cc821e1ed443698b))

* first commit ([`1b2e6a8`](https://github.com/kmnhan/erlabpy/commit/1b2e6a8b51398b0a2b5a2a3dd56db7bda92047bd))
