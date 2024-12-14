## v3.2.0 (2024-12-14)

### ✨ Features

- **io.dataloader:** enhance data combination logic ([80f2772](https://github.com/kmnhan/erlabpy/commit/80f2772f2e991032c795b317d9502f7846f5766f))

  When given multi-file data with multiple coordinates, the previous behavior was to include every coordinate as a dimension. This is logical for scans such as 4D position-dependent scans, but unnecessary for data like hv and angle dependent scans. Now, the loader will concatenate only along one axis if all motor coordinates are strictly monotonic.
- **constants:** add Bohr radius ([f8e4ca2](https://github.com/kmnhan/erlabpy/commit/f8e4ca2245e3e2a922dc26387fa97bdc62c14340))
- **imagetool.manager:** enable concatenating selected data ([44d61ba](https://github.com/kmnhan/erlabpy/commit/44d61ba559d26a11b72d6151e08d490a91f5ca9f))
- **imagetool:** add normalization option for 1D plot data ([5417a32](https://github.com/kmnhan/erlabpy/commit/5417a32f4ea4fb661036279e00289d2a9012ddc1))

  Adds an option to normalize 1D data with its mean to the right-click menu of 1D plots.
- add lazy-loader support ([e5ec658](https://github.com/kmnhan/erlabpy/commit/e5ec65895f347ce69c70bf999ec876261abbdd2b))

  Properly implements [SPEC 1](https://scientific-python.org/specs/spec-0001/) lazy-loading to top-level modules and the analysis module. Users can now directly access submodules after importing the top-level module only:

  ```

  import erlab

  erlab.analysis.transform.rotate(...)

  ```
- **imagetool.manager:** add console ([470808f](https://github.com/kmnhan/erlabpy/commit/470808fca209d93d18db041eb5a3a0cbb71f4672))

  Adds a python console to the manager that can be triggered from the `View` menu.
- **imagetool:** add keyboard shortcut to close ImageTool window ([97a7533](https://github.com/kmnhan/erlabpy/commit/97a7533d578b2e28dc5e05c9ab4484762bb77817))
- **interactive.imagetool:** add info box to manager ([0918a5b](https://github.com/kmnhan/erlabpy/commit/0918a5b357298bf4bf943e785c2f4cfa111dce29))

  Adds a textbox to ImageTool manager that shows coordinates and attributes of the selected window.

### 🐞 Bug Fixes

- **kspace:** fix broken hv-dependent data momentum conversion ([4695583](https://github.com/kmnhan/erlabpy/commit/46955838e18d9ab940e2f786711a9a78626da3d3))

  Fixes completely wrong implementation of kz-dependent momentum conversion. I can't believe this went unnoticed!
- **imagetool:** remove and reapply filter upon transformation ([af54a1d](https://github.com/kmnhan/erlabpy/commit/af54a1d52b8f8b53250aae9c9a950590a97853ad))
- **imagetool:** fix nonuniform data io and cropping ([8b538e6](https://github.com/kmnhan/erlabpy/commit/8b538e68312c3c92f223be0d8d1653c36c664abf))
- **imagetool:** fix wrong cursor position when loading ImageTool state from file ([e8191a8](https://github.com/kmnhan/erlabpy/commit/e8191a87aa9663c4e0ac5098ce0fcfe623fe5396))
- **imagetool:** resolve menu widgets losing keyboard focus ([90f8868](https://github.com/kmnhan/erlabpy/commit/90f8868de7a22b997cefd5b74a1f5a7f2e592c67))

### ⚡️ Performance

- **imagetool.manager:** accelerate opening new windows within the manager ([d4380b7](https://github.com/kmnhan/erlabpy/commit/d4380b7fca42e857f8cace0b6bd0d42a20bdbbda))
- delay imports for performance optimization in interactive tools ([abac874](https://github.com/kmnhan/erlabpy/commit/abac87401237d7f695261c51cb1a98d56841b49a))

### 🛠 Code Refactor

- **plotting:** update import statements to use `erlab.plotting` directly and deprecate `erlab.plotting.erplot` ([6a19f6a](https://github.com/kmnhan/erlabpy/commit/6a19f6ab57c9f65fdc855d1f1a155adc52421316))

  The import convention `import erlab.plotting.erplot as eplt` is now deprecated. Users should replace them with `import erlab.plotting as eplt`.
- **imagetool:** streamline namespace handling and improve layout structure ([5506f18](https://github.com/kmnhan/erlabpy/commit/5506f18888093da7842b98ed87b768f317bff38e))
- **imagetool:** move center zero button to context menu ([48deb6d](https://github.com/kmnhan/erlabpy/commit/48deb6d41a0a31c0815c9cdb7553783bc185f552))
- improve cli interface ([34a4db1](https://github.com/kmnhan/erlabpy/commit/34a4db1f120ae5839fa8c0375d7d951d9407e36b))
- cleanup some function signatures ([b04df05](https://github.com/kmnhan/erlabpy/commit/b04df05559f75b882075ab223adc8013f25241bc))
- **analysis.fit:** enable lazy loading for fit functions ([e877e12](https://github.com/kmnhan/erlabpy/commit/e877e12360157e8303ec0177e8a8999b4c307ab4))
- **imagetool.manager:** add ipython-based console ([f0b0adf](https://github.com/kmnhan/erlabpy/commit/f0b0adfefc5bd4e71ed0be427dd530bca89ff024))
- **analysis.gold:** adjust resolution plot cosmetics ([5d4a486](https://github.com/kmnhan/erlabpy/commit/5d4a4863391e08392f3b65e11e9b086cf98d831f))
- **interactive.colors:** minimize number of default colormaps ([a4c750c](https://github.com/kmnhan/erlabpy/commit/a4c750c21e8fbb11a93b3bc2f33cfb0477c4cf5a))

  Reduces the number of colormaps initially available in ImageTool. All colormaps can be loaded from the right-click menu of the colormap selection widget.
- **interactive.imagetool:** simplify method names for clarity ([48d0453](https://github.com/kmnhan/erlabpy/commit/48d04531c9b11ed120fbecac7885a7503c732973))

## v3.1.2 (2024-12-05)

### 🐞 Bug Fixes

- **plotting:** correct axis labels in `plot_array_2d` ([2fa358a](https://github.com/kmnhan/erlabpy/commit/2fa358af3c1866c6f886f03f23ec03eebdde08ec))
- **interactive.imagetool:** fix selection and io for non-uniform data ([5670a15](https://github.com/kmnhan/erlabpy/commit/5670a153fd19fcca93c9a926e57c32d29b25aac3))
- **accessors.kspace:** fix binding energy detection ([ed26162](https://github.com/kmnhan/erlabpy/commit/ed26162724ffb1fb263347f1a1c21d26602b6784))
- **interactive.imagetool:** show unarchiving message on double click ([d3dd3ee](https://github.com/kmnhan/erlabpy/commit/d3dd3eeffc25ccb10d507962815019deb961fa3c))
- **interactive.imagetool:** fix saving and loading non-uniform data ([73a1d4b](https://github.com/kmnhan/erlabpy/commit/73a1d4b23d2a043f3400b609e8f6343beda7c5d2))

## v3.1.1 (2024-11-28)

### 🐞 Bug Fixes

- **interactive.imagetool:** fix compatibility issues with Windows ([8691014](https://github.com/kmnhan/erlabpy/commit/86910140b25c62a093f68bd34efd1bce68587cba))

## v3.1.0 (2024-11-28)

### ✨ Features

- **interactive.imagetool:** overhaul manager UI ([a2ce551](https://github.com/kmnhan/erlabpy/commit/a2ce551067e05dba018bfef8dcae14d7a836afe9))

  Replaced widget-based implemenation with a model/view architecture. As a consequence, the displayed interface looks like a list. The user can now click a selected item to rename it, and drag items to reorder them. Each item now has a dedicated right-click menu.
- **interactive.imagetool.manager:** improve file opening ([da3425f](https://github.com/kmnhan/erlabpy/commit/da3425f6ec66364bfdb909c4742d6447d441c1b3))

  Implements opening multiple files at once through the open menu of the manager. Also, add support for opening data files by dragging them into the manager window.
- **interactive.imagetool:** add save and load functionality for workspace ([a5d38af](https://github.com/kmnhan/erlabpy/commit/a5d38affa85db0a213eecc86e48741b7506892fa))

  Enables users to save multiple ImageTool windows to a single file using the manager.
- **interactive.imagetool:** show dialog when data is being loaded ([5577249](https://github.com/kmnhan/erlabpy/commit/557724964f60db7f8c587d0c0b2d76b0a21e0dd6))
- **interactive.imagetool:** change manager icon ([3e20e63](https://github.com/kmnhan/erlabpy/commit/3e20e6342c5cf164dc2662e1de7fe5bceefa3007))
- **interactive.imagetool.manager:** add menubar to manager ([59326a1](https://github.com/kmnhan/erlabpy/commit/59326a1e0b37291a1c920c80c948dd23e5a678f7))
- **io.plugins:** add summary generation to maestro loader ([aa6f5d2](https://github.com/kmnhan/erlabpy/commit/aa6f5d28e1dd0a441e6440eb0fab61db5544896d))
- **plotting:** add fine-grained control over color limit normalization ([46c962f](https://github.com/kmnhan/erlabpy/commit/46c962f9131f9a1a691167eb192a13c2a5e7d2fa))
- **interactive.imagetool:** open ktool from imagetool ([d2cb8a7](https://github.com/kmnhan/erlabpy/commit/d2cb8a734f51af68e2ab7cbc50cf43a50d61a136))
- **interactive.imagetool:** add equal aspect ratio checkbox to right-click menu ([d4db0cf](https://github.com/kmnhan/erlabpy/commit/d4db0cf71124870778ee3af1e90d5d8d319fa486))
- **interactive.imagetool:** add crop menu ([639749f](https://github.com/kmnhan/erlabpy/commit/639749f30966181b619796d0b6bb2485db931e19))
- **interactive.utils:** add qobject that handles mutually exclusive selection comboboxes ([33b5f6b](https://github.com/kmnhan/erlabpy/commit/33b5f6beead46b57bb3bdc54cffe29ec9862c0c5))
- **interactive.imagetool:** add rename button to manager ([56ac884](https://github.com/kmnhan/erlabpy/commit/56ac88463b25257b3fc29fbbf78aa4670a5091b9))
- **interactive.imagetool:** add open in new window option to right-click menu of each plot ([8742659](https://github.com/kmnhan/erlabpy/commit/874265909f8036da9e68b36046982f6d25ec3311))
- **accessors.general:** added `qshow.params` accessors for fit results ([2592e5a](https://github.com/kmnhan/erlabpy/commit/2592e5a480970252b8ecd744c208a9cf3366c6ae))

  Calling `ds.qshow.params()` on a fit result dataset will now plot the coefficient value and errorbars as a function of fit coordinates.
- **analysis.fit:** add Fermi-Dirac distribution to MultiPeakFunction components ([65a1e8c](https://github.com/kmnhan/erlabpy/commit/65a1e8cdce5fc67a3a21c0019775cb380f8ff2aa))
- **io.dataloader:** add itool button to interactive summary ([ba3aa15](https://github.com/kmnhan/erlabpy/commit/ba3aa15d84a675612cd174968c319e753c2c4f81))

  A button that can open the data directly in ImageTool has been added to the interactive summary. The button is displayed when the interactive summary is called while the ImageTool manager is running.

### 🐞 Bug Fixes

- **io.plugins.lorea:** fix file dialog method ([4c74105](https://github.com/kmnhan/erlabpy/commit/4c741055d156bb4121fb7420ff1106ffe16282f3))
- **interactive.imagetool:** resolve segfault on save current data ([5699fa3](https://github.com/kmnhan/erlabpy/commit/5699fa3960b1a876c70771d37033aff8c610829b))
- **interactive.imagetool:** retain axis order when opening dtool and goldtool ([e14c9fc](https://github.com/kmnhan/erlabpy/commit/e14c9fc778759a961154f4360d69c80c8ec098d6))
- **io.plugins.merlin:** fix match signature ([5586cce](https://github.com/kmnhan/erlabpy/commit/5586cce9fa32d2ce6b3d6ef0538f1894d80bfe92))
- **interactive.imagetool:** cursor sync for non-uniform coords ([4aa1425](https://github.com/kmnhan/erlabpy/commit/4aa1425e1009ae50d40485dfa99fecc6146f9c0a))
- **io.plugins.maestro:** fix wrong temperature attribute ([6ed2a70](https://github.com/kmnhan/erlabpy/commit/6ed2a7075c71af972eff36250b073f6f7cba5566))
- **io.dataloader:** allow dimensions without coordinates in output data ([752facf](https://github.com/kmnhan/erlabpy/commit/752facfeda2f2a5ce2312a6fe355654baf4f2423))
- **interactive:** improve ktool compatibility with manager ([4c775cf](https://github.com/kmnhan/erlabpy/commit/4c775cf1b4018777275c95625915d23641e42bf0))
- **interactive.imagetool:** fix opening slice in new tool when manager is running ([db8e0af](https://github.com/kmnhan/erlabpy/commit/db8e0afaeb705a09a5395e67b83646e2aba4d6d0))
- **interactive.utils:** fix opening with manager in tools ([3726049](https://github.com/kmnhan/erlabpy/commit/3726049c4383da5fa2224efa4a4596ba635ecb16))
- **accessors.general:** allow `qshow` for fit results from multivariable Datasets ([f5b88e7](https://github.com/kmnhan/erlabpy/commit/f5b88e742f34819bbad6a46caebbcc54bed771bb))
- **interactive.imagetool:** fix wrong decimals for rotation center ([3fc3a50](https://github.com/kmnhan/erlabpy/commit/3fc3a500da532d73792a77ac769d15349cd36622))
- **accessors.general:** fix component plotting for concatenated fit datasets with multiple models ([7d2976d](https://github.com/kmnhan/erlabpy/commit/7d2976db0b7cc5721b192285170941b8434e6225))

### ⚡️ Performance

- **interactive.imagetool:** improve associated tool garbage collection ([839dab0](https://github.com/kmnhan/erlabpy/commit/839dab01682d4903ede92144a3337a69684c3a38))
- **io:** implement lazy loading for h5netcdf and nexusformat imports ([3f219ae](https://github.com/kmnhan/erlabpy/commit/3f219aef8764859f1eaccb2e354c7780f41133bb))
- speed up initial import ([d7f3b3c](https://github.com/kmnhan/erlabpy/commit/d7f3b3c2b5a1265c52f6416c7050ccda724532a2))

  Accelerates initial import time by refactoring heavy imports to reside inside functions.

  Importing the plotting module no longer automatically imports the colormap packages `cmocean`, `cmasher`, and `colorcet`. The user must add manual import statements.
- **analysis.interpolate:** cache jitted interpolation functions ([34521ef](https://github.com/kmnhan/erlabpy/commit/34521ef485a1ec0d3123885c6290f10410ce347e))
- **interactive.imagetool:** optimize memory usage by reducing circular references ([a675e1a](https://github.com/kmnhan/erlabpy/commit/a675e1a9d1d96f4ad4fa8d4853c2cca78864342e))

### 🛠 Code Refactor

- **interactive.imagetool:** make it easier to show and hide windows in manager ([aefc560](https://github.com/kmnhan/erlabpy/commit/aefc5600d82d0cfea24d2f4f1ef5b0e29b212f69))
- **interactive.imagetool:** use `QSharedMemory` instead of `multiprocessing` ([234c19f](https://github.com/kmnhan/erlabpy/commit/234c19f25ef2669cc32f45c4c56d9f40566a082b))
- **plotting.general:** use matplotlib api instead of xarray plot for 1D ([9e38b2c](https://github.com/kmnhan/erlabpy/commit/9e38b2cab68386443dfe04ce829032fe5836fcdf))
- **io.dataloader:** improve warning messages and error handling ([1ebfa72](https://github.com/kmnhan/erlabpy/commit/1ebfa7257b2238f8df34fd687f26f8a8a2c6c2ea))
- **interactive:** move IconButton to interactive utils and add IconActionButton ([e78190f](https://github.com/kmnhan/erlabpy/commit/e78190f286f5cf79978d499276be41c7900b76ae))
- **interactive.imagetool:** use HDF5 files instead of pickle to cache tools ([be66297](https://github.com/kmnhan/erlabpy/commit/be66297c76ed2062cc281ff9c21e761b94d2616b))
- move `AxesConfiguration` from `erlab.analysis.kspace` to `erlab.constants` ([3593d41](https://github.com/kmnhan/erlabpy/commit/3593d41e864caa68400fb806b5ce0c86de86e29d))
- **plotting:** streamline igor CT loading ([e1e8baa](https://github.com/kmnhan/erlabpy/commit/e1e8baac003be0a763cc22bc9cf2d7e3e9ce03cb))
- **io:** update type hints for file handling functions to use Iterable ([e3caf83](https://github.com/kmnhan/erlabpy/commit/e3caf83735561de57d00b5088a34a884edfb4c00))
- **io:** streamline file identification logic and add user warnings for multiple file scenarios ([05b7e6c](https://github.com/kmnhan/erlabpy/commit/05b7e6ccd2305aec0a743e36f1d6f53a1eeebdd6))
- **io:** replace several `os.path` calls with pathlib ([bdfdd22](https://github.com/kmnhan/erlabpy/commit/bdfdd22536271bbbeafca8ef7f23fbed928130fa))
- **dataloader:** make some methods private ([31cf008](https://github.com/kmnhan/erlabpy/commit/31cf00855d563536b4a410bf6257093b50c1601b))

  This makes all dataloader methods and attributes that are not meant to be overriden private. Affected methods and properties are `combine_multiple`, `generate_summary`, and `name_map_reversed`.
- **accessors.kspace:** cleanup namespace ([7af0d66](https://github.com/kmnhan/erlabpy/commit/7af0d66cf356f923014d1144c150ed2e85b954a9))

  Withdraws some internal properties and methods from public API.
- directly import `_THIS_ARRAY` from xarray core ([3bd72ec](https://github.com/kmnhan/erlabpy/commit/3bd72ecc8fd57abc6abe6624c360a3a06f72ae23))
- **interactive.imagetool:** improve dialog code structure ([5a16686](https://github.com/kmnhan/erlabpy/commit/5a16686df302ecf69098f542179bedc20aa99226))
- **interactive.imagetool:** move dialogs into new dedicated module ([a90a735](https://github.com/kmnhan/erlabpy/commit/a90a735874e16769b135df1d2e70ef04922f27c8))
- **interactive.imagetool:** improve error messages for invalid data ([a715ba1](https://github.com/kmnhan/erlabpy/commit/a715ba131b032ba7b2258b7c934764bde6424f62))

## v3.0.0 (2024-11-06)

### 🚨 Breaking Changes

- Deprecated module `erlab.io.utilities` is removed. Use `erlab.io.utils` instead. ([e189722](https://github.com/kmnhan/erlabpy/commit/e189722f129d55cab0d2ec279e5303929cb09979))
- Deprecated module `erlab.interactive.utilities` is removed. Use `erlab.interactive.utils` instead. ([af2c81c](https://github.com/kmnhan/erlabpy/commit/af2c81c676455ddfa19ae9bbbbdbdd68d257f26c))
- Deprecated module `erlab.characterization` is removed. Use `erlab.io.characterization` instead. ([8d770bf](https://github.com/kmnhan/erlabpy/commit/8d770bfe298253c020aeda6d61a9eab625facf6c))
- Deprecated module `erlab.analysis.utils` is removed. Use `erlab.analysis.transform.shift` and `erlab.analysis.gold.correct_with_edge`. ([0b2ca44](https://github.com/kmnhan/erlabpy/commit/0b2ca44844cc5802d32d9ed949e831b534525183))
- Deprecated alias `slice_along_path` in `erlab.analysis` is removed. Call from `erlab.analysis.interpolate` instead. ([305832b](https://github.com/kmnhan/erlabpy/commit/305832b7bb18aa3d1fda21f4cd0c0992b174d639))
- Deprecated aliases `correct_with_edge` and `quick_resolution` in `erlab.analysis` are removed. Call from `erlab.analysis.gold` instead. ([075eaf8](https://github.com/kmnhan/erlabpy/commit/075eaf8cd222044aa5cc0c3459698ab33568958c))
- Removed deprecated aliases `load_igor_ibw` and `load_igor_pxp`. Use `xarray.open_dataarray` and `xarray.open_dataset` instead. ([7f07ad2](https://github.com/kmnhan/erlabpy/commit/7f07ad2c46f80d48c255d408f3f200ae01930060))
- The default attribute name for the sample temperature is changed to `sample_temp` from `temp_sample`. This will unfortunately break a lot of code that relies on the key `temp_sample`, but will be easy to refactor with find and replace. ([32e1cd5](https://github.com/kmnhan/erlabpy/commit/32e1cd5fb45bce12cfa83c520e8c61af96a8cb39))
- All dataloaders must now add a new keyword argument to `load_single`, but implementing it is not mandatory.

  Also, dataloaders that implements summary generation by overriding `generate_summary` must now switch to the new method.

  See the summary generation section in the updated user guide.

  Furthermore, the `isummarize` method is no longer public; code that uses this method should use `summarize` instead.

  The `usecache` argument to the `summarize` method is no longer available, and the cache will be updated whenever it is outdated. ([0f5dab4](https://github.com/kmnhan/erlabpy/commit/0f5dab46e3d3a75fc77908b4072f08aa89059acd))

### ✨ Features

- **io.igor:** enable loading experiment files to DataTree ([1835be0](https://github.com/kmnhan/erlabpy/commit/1835be0d08ed899b2edbb06fb442cd9addb40929))

  Added methods to the backend to allow using `xarray.open_datatree` and `xarray.open_groups` with Igor packed experiment files. Closes [#29](https://github.com/kmnhan/erlabpy/issues/29)
- add `qinfo` accessor ([eb3a742](https://github.com/kmnhan/erlabpy/commit/eb3a74297211aae8f13e6974563e6da819bfbedb))

  Adds a `qinfo` accessor that prints a table summarizing the data in a human readable format. Closes [#27](https://github.com/kmnhan/erlabpy/issues/27)
- **interactive.kspace:** pass lattice parameters and colormap info to `ktool` ([6830af3](https://github.com/kmnhan/erlabpy/commit/6830af343326e0367a6dfb016728a6cf1325cf64))

  Added the ability to pass lattice vectors and colormaps to `ktool`.
- **interactive.kspace:** add circle ROI to ktool ([304e1a5](https://github.com/kmnhan/erlabpy/commit/304e1a53f189ebed9a890680c3499a756c586498))

  Added a button to the visualization tab which creates a circle ROI. The position and radius can be edited by right-clicking on the roi.
- **interactive.colors:** add zero center button to right-click colorbar ([c037de1](https://github.com/kmnhan/erlabpy/commit/c037de1f4387c0daf7cc7aa252124f01269bc633))
- **interactive.imagetool:** add `.ibw` and `.pxt` files to load menu ([73c3afe](https://github.com/kmnhan/erlabpy/commit/73c3afef306109be858d23dbf8511617c5d203dd))
- **io.dataloader:** allow passing rcParams to interactive summary plot ([a348366](https://github.com/kmnhan/erlabpy/commit/a34836673315fdc9acc0ed52d8e56edc90c18456))
- **io.dataloader:** implement automatic summary generation ([0f5dab4](https://github.com/kmnhan/erlabpy/commit/0f5dab46e3d3a75fc77908b4072f08aa89059acd))

  It is now much easier to implement a summary generation mechanism. This commit also adds a new keyword argument to `load_single` that can greatly speed up summary generation.
- **io.dataloader:** support callable objects in `additional_attrs` ([e209499](https://github.com/kmnhan/erlabpy/commit/e209499c8044f0085fda74b7dc491517a695099c))

### 🐞 Bug Fixes

- **interactive.imagetool:** fix copy cursor value for numpy 2 ([dc19c82](https://github.com/kmnhan/erlabpy/commit/dc19c827c4082989e47b0f8e2d7adda45ad62aaa))
- **io.dataloader:** retain selected dimension in interactive summary ([9d54f8b](https://github.com/kmnhan/erlabpy/commit/9d54f8b3402767cf15e6cf5ab00ee5a1b766d172))
- **accessors.general:** keep associated coords in `qsel` when averaging ([03a7b4a](https://github.com/kmnhan/erlabpy/commit/03a7b4a30b4c6a635f904fcab377298b06b86f66))
- **io.dataloader:** ignore old summary files ([bda95fc](https://github.com/kmnhan/erlabpy/commit/bda95fc1f0aaec73c179fd47258f6fde8056aaf9))
- **io.plugins.kriss:** fix KRISS ibw file match pattern ([7ced571](https://github.com/kmnhan/erlabpy/commit/7ced57152edb802bd14f831c77494a6f805f5097))
- **analysis.gold:** retain attributes in `quick_resolution` ([504acdc](https://github.com/kmnhan/erlabpy/commit/504acdc1d7d9b8dcd4613ca97551d78c366f0337))
- do not require qt libs on initial import ([118ead6](https://github.com/kmnhan/erlabpy/commit/118ead603b89867e56b29932f59bd02b476ab43b))

### ⚡️ Performance

- **io.plugins.da30:** faster summary generation for DA30 zip files ([22b77bf](https://github.com/kmnhan/erlabpy/commit/22b77bf0ee787fe1236fb85262702b79265e3b8d))
- **io.igor:** suppress `igor2` logging ([5cd3a8c](https://github.com/kmnhan/erlabpy/commit/5cd3a8c273b143d1a83f3286678638fede1ddd01))
- **analysis.interpolate:** extend acceleration ([84daa88](https://github.com/kmnhan/erlabpy/commit/84daa8866ec4223555568f441b6010bb4936a413))

  The fast linear interpolator now allows more general interpolation points like interpolating 3D data on a 2D grid. This means that passing `method='linearfast'` to `DataArray.interp` is faster in many cases.

### 🛠 Code Refactor

- **io.igor:** change wave dimension name handling ([3e0586a](https://github.com/kmnhan/erlabpy/commit/3e0586ae34893698317136bfccc5fd839b91332e))

  Waves with both dim and unit labels  provided were given a dim label formatted like `dim(unit)`. This update changes this so that the dim label is just `dim`, and the unit is inserted to coordinate attrs.
- **io:** remove deprecated module ([e189722](https://github.com/kmnhan/erlabpy/commit/e189722f129d55cab0d2ec279e5303929cb09979))
- **interactive:** remove deprecated module ([af2c81c](https://github.com/kmnhan/erlabpy/commit/af2c81c676455ddfa19ae9bbbbdbdd68d257f26c))
- remove deprecated module `erlab.characterization` ([8d770bf](https://github.com/kmnhan/erlabpy/commit/8d770bfe298253c020aeda6d61a9eab625facf6c))
- **analysis:** remove deprecated module ([0b2ca44](https://github.com/kmnhan/erlabpy/commit/0b2ca44844cc5802d32d9ed949e831b534525183))
- **analysis:** remove deprecated alias ([305832b](https://github.com/kmnhan/erlabpy/commit/305832b7bb18aa3d1fda21f4cd0c0992b174d639))
- **analysis:** remove deprecated aliases ([075eaf8](https://github.com/kmnhan/erlabpy/commit/075eaf8cd222044aa5cc0c3459698ab33568958c))
- **interactive.imagetool.manager:** add prefix to temporary directories for better identification ([e56163b](https://github.com/kmnhan/erlabpy/commit/e56163ba7fe7d92f3a01ec78098c2d0194ea0302))
- **io.plugins:** implement DA30 file identification patterns in superclass ([f6dfc44](https://github.com/kmnhan/erlabpy/commit/f6dfc4412b56fc1d83efceb4a65070eb9ef1c2b1))
- **io:** remove deprecated aliases ([7f07ad2](https://github.com/kmnhan/erlabpy/commit/7f07ad2c46f80d48c255d408f3f200ae01930060))
- change temperature attribute name ([32e1cd5](https://github.com/kmnhan/erlabpy/commit/32e1cd5fb45bce12cfa83c520e8c61af96a8cb39))

  Changes `temp_sample` to `sample_temp` for all data loaders and analysis code.
- **utils.formatting:** change formatting for numpy arrays ([95d9f0b](https://github.com/kmnhan/erlabpy/commit/95d9f0b602551141232eb5a2fa10c421d11d2233))

  For arrays with 2 or more dimensions upon squeezing, only the minimum and maximum values are shown. Also, arrays with only two entries are displayed as a list.
- **io.dataloader:** disable parallel loading by default ([fed2428](https://github.com/kmnhan/erlabpy/commit/fed2428229e3ef70fc95a35670fc75ace44024bd))

  Parallel loading is now disabled by default since the overhead is larger than the performance gain in most cases.
- change some warnings to emit from user level ([e81f2b1](https://github.com/kmnhan/erlabpy/commit/e81f2b121d2931b327d30b146db1e77e7a3b3ec2))
- **io.dataloader:** cache summary only if directory is writable ([85bcb80](https://github.com/kmnhan/erlabpy/commit/85bcb80bdf27ea12edb9314247a978f71c8be6dc))
- **io.plugins:** improve warning message when a plugin fails to load ([9ee0b90](https://github.com/kmnhan/erlabpy/commit/9ee0b901b1b904dabb38d29f4c166dca07c9a7e9))
- **io:** update datatree to use public api ([6c27e07](https://github.com/kmnhan/erlabpy/commit/6c27e074c5aceb16eb9808cca38b8ba73748f07e))

  Also bumps the minimum supported xarray version to 2024.10.0.
- **io.dataloader:** make `RegistryBase` private ([df7079e](https://github.com/kmnhan/erlabpy/commit/df7079e4fc96b195d34436bcc93684e10ddecdad))
- **io.dataloader:** rename loader registry attribute `default_data_dir` to `current_data_dir` ([d87eba7](https://github.com/kmnhan/erlabpy/commit/d87eba7db6cea051f76b61ea7b0834e439460810))

  The attribute `default_data_dir` has been renamed to `current_data_dir` so that it is consistent with `current_loader`. Accessing the old name is now deprecated.

  Also, the `current_loader` and `current_data_dir` can now be assigned directly with a syntax like `erlab.io.loaders.current_loader = "merlin"`.

## v2.12.0 (2024-10-22)

### ✨ Features

- **interactive.imagetool:** add normalization option to View menu ([53e2cf2](https://github.com/kmnhan/erlabpy/commit/53e2cf2b6e49de70a6857b782acc8ec0815a93b5))
- **io.dataloader:** allow passing additional arguments to `load_single` ([1652c20](https://github.com/kmnhan/erlabpy/commit/1652c20de102f7bb5fa4a26360652341f3249c2e))
- **io.plugins:** add support for two new beamlines, closes [#61](https://github.com/kmnhan/erlabpy/issues/61) ([368263e](https://github.com/kmnhan/erlabpy/commit/368263ef282ef3ec39cc6c8bbe23f26ddcf18b8f))

  Added plugins with preliminary support for Diamond I05 and ALBA BL20 LOREA.
- **io:** add `nexusutils` module for working with NeXus files ([2532941](https://github.com/kmnhan/erlabpy/commit/25329416d837cd318bab51fe7ff747e5f83cbc19))

  This commit adds a new submodule `io.nexusutils` that contains utilities for converting NeXus data to xarray data structures.

### 🐞 Bug Fixes

- **erlab.io.plugins.merlin:** resolve typo in file dialog methods ([39caa99](https://github.com/kmnhan/erlabpy/commit/39caa998cc6857cd197dc1a0a7efc961ad3afe46))
- **erlab.accessors.general:** make `qsel` accessor work along dimensions with no coordinates ([7f0d259](https://github.com/kmnhan/erlabpy/commit/7f0d259f75626316d28497a3e25e1009d7c05851))
- **interactive.imagetool:** avoid errors on termination ([9fd044b](https://github.com/kmnhan/erlabpy/commit/9fd044bae170e025f5e27eb392fc600c19ec30b8))

### 🛠 Code Refactor

- **io:** return path-like objects instead of strings in `get_files` ([2eb9166](https://github.com/kmnhan/erlabpy/commit/2eb9166e4183222e9cb282fc0bf4984d76bf3124))

## v2.11.2 (2024-10-14)

### 🐞 Bug Fixes

- **io.dataloader:** fix `coordinate_attrs` not being propagated ([278675b](https://github.com/kmnhan/erlabpy/commit/278675b54d2e12471ce8629fbd6d249aa7184c0e))

## v2.11.1 (2024-10-14)

### 🛠 Code Refactor

- add app icon for imagetool manager ([e1cbcd2](https://github.com/kmnhan/erlabpy/commit/e1cbcd29b045cb2d586baa3c6272fd60cfd05979))

## v2.11.0 (2024-10-13)

### ✨ Features

- **io.dataloader:** add new argument that can control combining ([bdec5ff](https://github.com/kmnhan/erlabpy/commit/bdec5ff24e02e82597cef10d225997599efaa257))

  Adds a new parameter `combine` to `io.load`. If `False`, returns a list of post-processed files without attempting to concatenate or merge the data into a single object. If `True`, retains the current default behavior.

### 🐞 Bug Fixes

- **imagetool:** allow coords of any dtype coercible to float64 ([4342ebc](https://github.com/kmnhan/erlabpy/commit/4342ebc1bc4be01fcc9c7883ecfbaef0f5857e5d))
- **io.dataloader:** properly handle combining multi-axis scans ([2cd22c7](https://github.com/kmnhan/erlabpy/commit/2cd22c7998cd22d399e59a86131e4c5712127b23))

### 🛠 Code Refactor

- **io.plugins:** update type hints ([54d0c5d](https://github.com/kmnhan/erlabpy/commit/54d0c5d7e55cc7e6af6b4d83feeb9d6c863e52f6))
- remove unused imports ([f1e35de](https://github.com/kmnhan/erlabpy/commit/f1e35ded993f1fb2be04c549aa241809c7d68a4d))
- **interactive:** add informative error message for missing pyqt ([1347a02](https://github.com/kmnhan/erlabpy/commit/1347a0231698f8104be71116823f797caeccc9a6))
- **io.plugins:** add warning when plugin load fails ([ed5b184](https://github.com/kmnhan/erlabpy/commit/ed5b184538cde60c1fa2ba2421ff30c27acb1eed))

## v2.10.0 (2024-10-08)

### ✨ Features

- **io.plugins:** add loader for beamline ID21 ESM at NSLS-II ([c07e490](https://github.com/kmnhan/erlabpy/commit/c07e490c37b52706aa9407d6adb5aa7787e2c1b0))

  This commit adds a new data loader for beamline ID21 ESM at NSLS-II, Brookhaven National Laboratory.
- **io.dataloader:** add formatters ([2ee9a4a](https://github.com/kmnhan/erlabpy/commit/2ee9a4a2361b91727c4e964adc00a26398863f2f))

  A new attribute named `formatters` and a new method `get_formatted_attr_or_coord` has been added to loaders. This allows custom per-attribute pretty-printing behavior.
- **io:** add parallel argument to `load` ([88cd924](https://github.com/kmnhan/erlabpy/commit/88cd924efd9375ebe89417b504c80e55f7071404))
- **io:** add xarray backend for igor files ([1fe5ca5](https://github.com/kmnhan/erlabpy/commit/1fe5ca514777f4356f1b67251dc1a6f21b320d48))

  `.pxt`, `.pxp`, and `.ibw` files can now be opened with xarray methods such as `xr.open_dataset` and `xr.open_dataarray`. See the updated user guide for more information.

### 🐞 Bug Fixes

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

### ⚡️ Performance

- **interactive.imagetool:** improve manager speed ([891c4ee](https://github.com/kmnhan/erlabpy/commit/891c4eed0921fdb5b4ebd2cb17e6c932afa79ccc))

### 🛠 Code Refactor

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

## v2.9.1 (2024-09-03)

### 🐞 Bug Fixes

- ui file compatibility with Qt5 ([66c776d](https://github.com/kmnhan/erlabpy/commit/66c776d912e5cf3ba1be819bd46b972f22cbb560))

## v2.9.0 (2024-08-30)

### ✨ Features

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

### 🐞 Bug Fixes

- **interactive.imagetool:** properly disconnect signals ([dce236f](https://github.com/kmnhan/erlabpy/commit/dce236f1da1aae44bc3210aac1ff2eb710d71f41))
- **interactive.imagetool:** fix autoscale when loading data ([2c12f59](https://github.com/kmnhan/erlabpy/commit/2c12f592e3d1d57de649aabae742eee230545387))
- **interactive.imagetool:** scale spinbox decimals relative to coordinate step size ([9a801a5](https://github.com/kmnhan/erlabpy/commit/9a801a5391cd35e1fdeaf9377fd8b220420d4829))
- **interactive.utils:** update `BetterSpinBox` width on changing decimals ([0a70884](https://github.com/kmnhan/erlabpy/commit/0a70884eb7742cea53962beb80ba42cad723fe4e))
- **interactive:** fix compatibility issue with PySide6 ([da5f4af](https://github.com/kmnhan/erlabpy/commit/da5f4af139988e38f1c9b0534a5957644e01b9aa))
- **interactive.imagetool:** do not copy code when unnecessary ([9131029](https://github.com/kmnhan/erlabpy/commit/91310295fb9ffab994b0b681c84a4646680583b0))
- **accessors.general:** qshow now triggers hvplot properly for 1D data ([8a84813](https://github.com/kmnhan/erlabpy/commit/8a84813e561a52eaf5ef5fc7118b992e7537b1f6))
- **interactive.imagetool:** make manager socket use default backlog ([0ac7f0b](https://github.com/kmnhan/erlabpy/commit/0ac7f0b35b3288a62708c1c5e2c54a483e563bd7))
- **interactive.imagetool:** ensure proper socket termination in manager ([2cceb27](https://github.com/kmnhan/erlabpy/commit/2cceb27a7d0e9db80fc39aa65340df6612587206))

### 🛠 Code Refactor

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

### 🐞 Bug Fixes

- **plotting.annotations:** properly pass keyword arguments in `mark_points_outside` ([2136939](https://github.com/kmnhan/erlabpy/commit/2136939e09656f921aed7204ca11cc6615605b7f))
- **plotting.annotations:** expose property label generation to public api ([545781d](https://github.com/kmnhan/erlabpy/commit/545781d1aa5d04b5dd3bf6d0498821d104f837ac))

  A new `property_labels` function can be used to generate strings that are used by `label_subplot_properties` so that the labels can be used as titles easily through `eplt.set_titles`. Also, label generation now recognizes time, given as 't' with default unit seconds.

## v2.8.4 (2024-07-26)

### 🐞 Bug Fixes

- **erlab.plotting.general:** improve `plot_array` keyword versatility ([1dc41cd](https://github.com/kmnhan/erlabpy/commit/1dc41cd52f8d7f879cfe54f2adf3a512b78ac007))

  Enables additional kwargs with valid data dimensions as the key to be passed onto `qsel`.
- **erlab.analysis.gold:** fix `quick_fit` attribute detection ([3797f93](https://github.com/kmnhan/erlabpy/commit/3797f93e1a578e356ce21e7b7c933341099ab156))
- **interactive.imagetool:** retain window title upon archiving ([b5d8aa4](https://github.com/kmnhan/erlabpy/commit/b5d8aa4884562ba4b53351baf58d598e27a1e757))

### 🛠 Code Refactor

- **plotting.general:** remove `LabeledCursor` ([912b4fb](https://github.com/kmnhan/erlabpy/commit/912b4fb73f88e3a529d1e3880a2253e0cb26e7ae))

  We skip the deprecation step since nobody is likely to be using it anyway.
- **accessors:** split submodule ([6ed5c03](https://github.com/kmnhan/erlabpy/commit/6ed5c039889624d3589d9ce71a75ed6047f4406f))

  Accessors in `utils.py` has been moved to `general.py`, so that `utils.py` only contains utilities for creating accessors.
- improve type annotations ([b242f44](https://github.com/kmnhan/erlabpy/commit/b242f44d49239e51b4bd9e4b1ae7fd952c59e2c2))

## v2.8.3 (2024-07-08)

### 🐞 Bug Fixes

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

### 🛠 Code Refactor

- satisfy type checker ([042a7b1](https://github.com/kmnhan/erlabpy/commit/042a7b1f72a9a29b93736fe1eea61f18cc8ea49d))
- **interactive.imagetool:** add batch close button to manager ([efc6089](https://github.com/kmnhan/erlabpy/commit/efc6089669d73ec5ba39acbbeb08720f0543fe3e))

## v2.8.2 (2024-07-01)

### 🐞 Bug Fixes

- **interactive.imagetool:** fix crash while linking more than 3 tools ([d5f8a30](https://github.com/kmnhan/erlabpy/commit/d5f8a30224f72d7159216fa5638056569521f75f))
- update resistance loader ([6fcf2ab](https://github.com/kmnhan/erlabpy/commit/6fcf2abe797313ee3c21fd3cd2f4daebf412225f))

### 🛠 Code Refactor

- **interactive.imagetool:** show error message in GUI when opening file ([287a7e8](https://github.com/kmnhan/erlabpy/commit/287a7e84e5110ac08e17d9a852b0d2b0da830e42))

## v2.8.1 (2024-06-21)

### 🐞 Bug Fixes

- **interactive.imagetool:** properly implement caching and linking from GUI ([ffacdce](https://github.com/kmnhan/erlabpy/commit/ffacdce93d1ff89e1be823317a6d59a400a6dee2))
- **plotting.general:** pass DataArray to `func` argument to `plot_array` ([ed76e64](https://github.com/kmnhan/erlabpy/commit/ed76e64e45eb3ea93fba61380bc0d63864446fd3))

### ⚡️ Performance

- **interactive.imagetool:** speedup file loading and saving ([a6c869b](https://github.com/kmnhan/erlabpy/commit/a6c869b7d6ce0419d84a46086004d451845c23e3))

  Use pickle to save and load files instead of `erlab.io.load_hdf5` and `erlab.io.save_as_hdf5`.

## v2.8.0 (2024-06-17)

### ✨ Features

- **erlab.io.plugins.ssrl52:** changes to loader ([512a89b](https://github.com/kmnhan/erlabpy/commit/512a89b051911c88bafd59bdc9bd993ec727321a))

  The loader now promotes all attributes that varies during the scan to coordinates. Also, if the energy axis is given in kinetic energy and the work function is inferrable from the data attributes, the energy values are automatically converted to binding energy. This may require changes to existing code. This commit also includes a fix for hv-dependent swept cuts.
- **erlab.io.dataloader:** reorder output coordinates ([178edd2](https://github.com/kmnhan/erlabpy/commit/178edd27f3e58387b12b7a7928a26e87766fa9be))

  Coordinates on the loaded data will now respect the order given in `name_map` and `additional_coords`, improving readability.
- **interactive.imagetool:** add ImageTool window manager ([b52d249](https://github.com/kmnhan/erlabpy/commit/b52d2490ec61053b7b933e274a68a163761827ce))

  Start the manager with the cli command `itool-manager`. While running, all calls to `erlab.interactive.imagetool.itool` will make the ImageTool open in a separate process. The behavior can be controlled with a new keyword argument, `use_manager`.
- **interactive.imagetool:** add undo and redo ([e7e8213](https://github.com/kmnhan/erlabpy/commit/e7e8213964c9739468b65e6a56dcc1a0d9d648e4))

  Adjustments made in ImageTool can now be undone with Ctrl+Z. Virtually all actions except window size change and splitter position change should be undoable. Up to 1000 recent actions are stored in memory.
- **interactive.imagetool:** remember last used loader for each tool ([eb0cd2f](https://github.com/kmnhan/erlabpy/commit/eb0cd2f41992845988f5e500416ed98f5d078c14))

### 🐞 Bug Fixes

- **interactive.imagetool:** fix code generation behaviour for non-uniform coordinates ([3652a21](https://github.com/kmnhan/erlabpy/commit/3652a21cf126ebcde015d5b7373bf5d5a675b177))

### 🛠 Code Refactor

- **interactive.imagetool:** preparation for saving and loading state ([eca8262](https://github.com/kmnhan/erlabpy/commit/eca8262defe8d135168ca7da115d947bda3c1040))

## v2.7.2 (2024-06-14)

### 🐞 Bug Fixes

- **erlab.io:** regression in handling getattr of dataloader ([dd0a568](https://github.com/kmnhan/erlabpy/commit/dd0a5680c6aed6e3b7ab391a10fbeb5c3cdc9c14))

## v2.7.1 (2024-06-14)

### 🐞 Bug Fixes

- **interactive.imagetool:** Integrate data loaders to imagetool ([7e7ea25](https://github.com/kmnhan/erlabpy/commit/7e7ea25a8fbe3a43222fbc7baedaa04c6522e24d))

  A new property called `file_dialog_methods` can be set in each loader which determines the method and name that is used in the file chooser window in imagetool.
- **accessors.kspace:** `hv_to_kz` now accepts iterables ([36770d7](https://github.com/kmnhan/erlabpy/commit/36770d723b1e3592bf83750f7559603026059bb1))

## v2.7.0 (2024-06-09)

### ✨ Features

- **analysis.gold:** add function for quick resolution fitting ([2fae1c3](https://github.com/kmnhan/erlabpy/commit/2fae1c351f29b2fb1ceef39a69706b3f198e4659))
- **analysis.fit:** Add background option to `MultiPeakModel` and `MultiPeakFunction` ([2ccd8ad](https://github.com/kmnhan/erlabpy/commit/2ccd8ad835cbc8de9764d2f8bbadda425babddb1))

### 🐞 Bug Fixes

- **erlab.io.plugins:** fix for hv-dependent data ([d52152f](https://github.com/kmnhan/erlabpy/commit/d52152f24807b9334ad5ffcc22c45a4af7a8d9ec))

## v2.6.3 (2024-06-07)

### 🐞 Bug Fixes

- **erlab.io.plugins:** support SSRL hv dependent data ([1529b6a](https://github.com/kmnhan/erlabpy/commit/1529b6a0af43f09c51691ad8bebf9208d421940a))

### 🛠 Code Refactor

- cleanup namespace ([847fbbe](https://github.com/kmnhan/erlabpy/commit/847fbbe4975b507905dc85ca5ae75fe16f5f887e))

## v2.6.2 (2024-06-03)

### 🐞 Bug Fixes

- **interactive.imagetool:** fix regression with nonuniform data ([67df972](https://github.com/kmnhan/erlabpy/commit/67df9720193611816e2a562ce71d080fccbbec5e))

## v2.6.1 (2024-05-30)

### 🐞 Bug Fixes

- re-trigger due to CI failure ([b6d69b5](https://github.com/kmnhan/erlabpy/commit/b6d69b556e3d0dbe6d8d71596e32d9d7cfdc5267))

## v2.6.0 (2024-05-30)

### ✨ Features

- **interactive.imagetool:** add bin amount label to binning controls ([7a7a692](https://github.com/kmnhan/erlabpy/commit/7a7a692b881e4cc1bd49342f31f3fe50407d72b5))
- add accessor for selecting around a point ([aa24457](https://github.com/kmnhan/erlabpy/commit/aa244576fcfa17f71be0a765be8f270a6ae28080))
- **accessors.fit:** add support for background models ([550be2d](https://github.com/kmnhan/erlabpy/commit/550be2deebf54fab77bef591ccbe059b5b219937))

  If one coordinate is given but there are two independent variables are present in the model,  the second one will be treated as the data. This makes the accessor compatible with y-dependent background models, such as the Shirley background provided in `lmfitxps`.
- **io:** make the dataloader behavior more customizable ([4824127](https://github.com/kmnhan/erlabpy/commit/4824127181b4383788f6dbe5cbeae4b2060f1f4f))

  Now, a new `average_attrs` class attribute exists for attributes that would be averaged over multiple file scans. The current default just takes the attributes from the first file. This also works when you wish to demote a coordinate to an attribute while averaging over its values.

  For more fine-grained control of the resulting data attributes, a new method `combine_attrs` can be overridden to take care of attributes for scans over multiple files. The default behavior is to just use the attributes from the first file.

### 🐞 Bug Fixes

- **plotting:** make `gradient_fill` keep axis scaling ([51507dd](https://github.com/kmnhan/erlabpy/commit/51507dd966a0ce2db4aabff2aac8222bee184cf8))

### 🛠 Code Refactor

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

### 🐞 Bug Fixes

- **io.plugins.maestro:** load correct beta for non deflector scans ([5324c36](https://github.com/kmnhan/erlabpy/commit/5324c362d7bdd1dcf0c21303f370fd2e77f12448))

## v2.5.3 (2024-05-22)

### 🐞 Bug Fixes

- **io.utilities:** `get_files` now only list files, not directories ([60f9230](https://github.com/kmnhan/erlabpy/commit/60f92307f94484361e0ba11b10a52be4c4cc05a1))
- **accessors.fit:** add `make_params` call before determining param names, closes [#38](https://github.com/kmnhan/erlabpy/issues/38) ([f1d161d](https://github.com/kmnhan/erlabpy/commit/f1d161de089b93e16b2947b126ac075764d98f75))
- **analysis.fit:** make some models more robust to DataArray input ([afe5ddd](https://github.com/kmnhan/erlabpy/commit/afe5ddd9d1e6796ba0261a147c2733d607916d81))

### 🛠 Code Refactor

- add loader for ALS BL7 MAESTRO `.h5` files ([4f33402](https://github.com/kmnhan/erlabpy/commit/4f3340228ae2e1cbd8baf57d5d426043f5e28688))
- **interactive:** add informative error message for missing Qt bindings ([560615b](https://github.com/kmnhan/erlabpy/commit/560615bb89d2646965d1a2a967133f0df08e3f6e))
- **io:** rename some internal variables and reorder ([76fe284](https://github.com/kmnhan/erlabpy/commit/76fe284b4bc9f1e0c3cb94857a65599b07ee04df))

  Also added a check for astropy in FITS file related utility.

## v2.5.2 (2024-05-16)

### 🐞 Bug Fixes

- make mathtext copy default to svg ([2f6e0e5](https://github.com/kmnhan/erlabpy/commit/2f6e0e558f251c846bc3dec39cd150391802460d))
- resolve MemoryError in prominent color estimation ([3bdcd03](https://github.com/kmnhan/erlabpy/commit/3bdcd0341c41b424ebbcb565b7cda0db839e4cb8))

  Due to [numpy/numpy/#11879](https://github.com/numpy/numpy/issues/11879) changed the auto method to sqrt. This should also improve memory usage and speed, with little to no impact on the end result.

## v2.5.1 (2024-05-15)

### 🐞 Bug Fixes

- **plotting:** fixes [#35](https://github.com/kmnhan/erlabpy/issues/35) ([a67be68](https://github.com/kmnhan/erlabpy/commit/a67be6869c2d25780f8a56794aad0386379202dd))

  Gradient fill disappears upon adding labels
- **fit.models:** wrong StepEdgeModel guess with DataArray input ([6778c8d](https://github.com/kmnhan/erlabpy/commit/6778c8dd2c048b0cab67c6d3668b25b3f79a71da))

### 🛠 Code Refactor

- **plotting:** code cleanup ([aef10e4](https://github.com/kmnhan/erlabpy/commit/aef10e472a3ebc935711253e91124cfd87beb9cc))

## v2.5.0 (2024-05-13)

### ✨ Features

- extended interactive accessor ([f6f19ab](https://github.com/kmnhan/erlabpy/commit/f6f19abd8edfb33585b5e19040a2ebaff39b2b70))

  The `qshow` accessor has been updated so that it calls `hvplot` (if installed) for data not supported by ImageTool.

  Also, the `qshow` accessor has been introduced to Datasets. For valid fit result datasets produced by the `modelfit` accessor, calling `qshow` will now show an `hvplot`-based interactive visualization of the fit result.
- **itool:** make itool accept Datasets ([f77b699](https://github.com/kmnhan/erlabpy/commit/f77b699abdf312a23832611052d67e8c4c8d4930))

  When a Dataset is passed to `itool`, each data variable will be shown in a separate ImageTool window.
- **analysis.image:** add multidimensional Savitzky-Golay filter ([131b32d](https://github.com/kmnhan/erlabpy/commit/131b32d9e562693e98a2f9e45cf6db4635405b44))

### 🐞 Bug Fixes

- **itool:** add input data dimension check ([984f2db](https://github.com/kmnhan/erlabpy/commit/984f2db0f69db2b5b99211728840447d9617f8bf))
- **analysis.image:** correct argument order parsing in some filters ([6043413](https://github.com/kmnhan/erlabpy/commit/60434136224c0875ed8fba41d24e32fc6868127c))
- **interactive:** improve formatting for code copied to clipboard ([d8b6d91](https://github.com/kmnhan/erlabpy/commit/d8b6d91a4d2688486886f2464426935fdf8cabc2))

### 🛠 Code Refactor

- **plotting:** update `clean_labels` to use `Axes.label_outer` ([0c64756](https://github.com/kmnhan/erlabpy/commit/0c647564c6027f5b60f9ff288f13019e0e5933b6))

## v2.4.2 (2024-05-07)

### 🐞 Bug Fixes

- **ktool:** resolve ktool initialization problem, closes [#32](https://github.com/kmnhan/erlabpy/issues/32) ([e88a58e](https://github.com/kmnhan/erlabpy/commit/e88a58e6aaed326af1a68aa33322d6ea9f0e800d))
- **itool:** disable flag checking for non-numpy arrays ([da6eb1d](https://github.com/kmnhan/erlabpy/commit/da6eb1db9e81d51b52d4b361de938bcf7ba45e68))

## v2.4.1 (2024-05-03)

### 🐞 Bug Fixes

- **plotting:** fix wrong regex in `scale_units` ([d7826d0](https://github.com/kmnhan/erlabpy/commit/d7826d0269214dfd822a4d0293e42a9840015ce8))
- fix bug in `modelfit` parameter concatenation ([edaa556](https://github.com/kmnhan/erlabpy/commit/edaa5566c6e3817e1d9220f7a96e8e731cf7eede))
- **itool:** ensure DataArray is readable on load ([5a0ff00](https://github.com/kmnhan/erlabpy/commit/5a0ff002802cdf5bd3ceb34f9cddc53c9674e7bd))

## v2.4.0 (2024-05-02)

### ✨ Features

- **imagetool:** add method to update only the values ([ca40fe4](https://github.com/kmnhan/erlabpy/commit/ca40fe41a0320fd7843c86f95b68f8b6e19adca8))
- add interpolation along a path ([7366ec4](https://github.com/kmnhan/erlabpy/commit/7366ec4db600617e585c724d05aafea387456cf2))

  The `slice_along_path` function has been added to `analysis.interpolate`, which enables easy interpolation along a evenly spaced path that is specified by its vertices and step size. The path can have an arbitrary number of dimensions and points.

### 🐞 Bug Fixes

- **io:** remove direct display call in interactive summary ([d44b3a5](https://github.com/kmnhan/erlabpy/commit/d44b3a56aecfb054a38d944c5c8b7f45d362cf3b))

  This was causing duplicated plots.
- **plotting:** add some validation checks to `plot_array` ([2e0753c](https://github.com/kmnhan/erlabpy/commit/2e0753c90ffbe6fdd05af210ac6a4dbfa9aba899))

  The functions `plot_array` and `plot_array_2d` now checks if the input array coordinates are uniformly spaced. If they are not, a warning is issued and the user is informed that the plot may not be accurate.
- **plotting:** increase default colorbar size ([3208399](https://github.com/kmnhan/erlabpy/commit/32083990e9e77df6e94b2b0836bc1f9764cfaaf7))

  The default `width` argument to `nice_colorbar` is changed to 8 points. This ensures visibility in subplots, especially when constrained layout is used.
- delay interactive imports until called ([ad15910](https://github.com/kmnhan/erlabpy/commit/ad15910f921cb5ffffc388e7a5e02832935f8547))

### 🛠 Code Refactor

- various cleanup ([2b38397](https://github.com/kmnhan/erlabpy/commit/2b383970b602507b6efedbf396f14d470db60d8f))

  Improve docstring formatting and tweak linter settings

## v2.3.2 (2024-04-25)

### 🐞 Bug Fixes

- **io:** make summary caching togglable ([99b8e22](https://github.com/kmnhan/erlabpy/commit/99b8e221e75db73382bf599170c58d8a68ca049e))

  Also fixes a bug where interactive summary plots were duplicated
- **io:** data loader related fixes ([da08e90](https://github.com/kmnhan/erlabpy/commit/da08e9076e59895b35c393c8e2556c3592adf4a5))

  DA30 dataloader now preserves case for attribute names from zip files.

  Post processing for datasets now works properly

## v2.3.1 (2024-04-25)

### 🐞 Bug Fixes

- **interactive:** keep pointer for imagetool, fix typing issues ([c98c38e](https://github.com/kmnhan/erlabpy/commit/c98c38ea11bce50ed9bfd8d374064bb2b1659d0c))

### 🛠 Code Refactor

- move `characterization` to `io` ([9c30f1b](https://github.com/kmnhan/erlabpy/commit/9c30f1b7df51460f502dcbf999e3fac34be1cf99))

## v2.3.0 (2024-04-22)

### ✨ Features

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

### 🐞 Bug Fixes

- **kspace:** allow explicit coordinate kwargs ([fe47efc](https://github.com/kmnhan/erlabpy/commit/fe47efcde941767c02b582ce8b29d4b3678fd843))
- **exampledata:** change noise generation parameters ([b213f11](https://github.com/kmnhan/erlabpy/commit/b213f1151ed2555fc80374e9ebe3fc0856a13948))
- **fit:** make FermiEdge2dModel compatible with flattened meshgrid-like input arrays ([c0dba26](https://github.com/kmnhan/erlabpy/commit/c0dba261670774862f2dfae62c770bbab81aac2f))
- fix progress bar for parallel objects that return generators ([23d41b3](https://github.com/kmnhan/erlabpy/commit/23d41b31a3f3ee6c7343d471f7cec34dc374bafa))

  Tqdm imports are also simplified. We no longer handle `is_notebook` ourselves, but just import from `tqdm.auto`
- **plotting:** fix 2d colormaps ([8299576](https://github.com/kmnhan/erlabpy/commit/8299576ce3cbcbaec106bef952c6df148bb7ca18))

  Allow images including nan to be plotted with gen_2d_colormap, also handle plot_array_2d colorbar aspect

### 🛠 Code Refactor

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

### 🐞 Bug Fixes

- **io:** unify call signature for summarize ([e2782c8](https://github.com/kmnhan/erlabpy/commit/e2782c898d5aaaa1443b2bc82bb61fb40a28d232))
- resolve failing tests due to changes in sample data generation ([80f0045](https://github.com/kmnhan/erlabpy/commit/80f004574950834e42dbfa7677031d0f9f113bda))
- **interactive.exampledata:** properly generate 2D data ([825260c](https://github.com/kmnhan/erlabpy/commit/825260c8ceb0a79b8c071750003529b91cda3573))

### ⚡️ Performance

- **io:** speedup merlin summary generation by excluding duplicates ([d6b4253](https://github.com/kmnhan/erlabpy/commit/d6b42537ce48232b5112daef8f31e5cf86ea921a))

### 🛠 Code Refactor

- **io:** allow for more complex setups ([f67b2e4](https://github.com/kmnhan/erlabpy/commit/f67b2e4c7b092b7ca2db00ce02a23647879c514b))

  LoaderBase.infer_index now returns a second argument, which is a dictionary containing optional keyword arguments to load.
- **io:** provide rich interactive summary ([b075a9e](https://github.com/kmnhan/erlabpy/commit/b075a9ee59b61892462fc475e78b036a54408099))
- **io:** include "Path" column in ssrl loader summary ([ae1d8ae](https://github.com/kmnhan/erlabpy/commit/ae1d8aee051aa71563f6a6009ce9672e56edfae7))
- **io:** improve array formatting in summary ([1718529](https://github.com/kmnhan/erlabpy/commit/171852957db7fe53ff6a5c5c5f843530078d4b46))

## v2.2.1 (2024-04-14)

### 🐞 Bug Fixes

- **fit:** add sigma and amplitude expressions to MultiPeakModel parameters ([3f6ba5e](https://github.com/kmnhan/erlabpy/commit/3f6ba5e84922129296183e02255506df73da0276))
- **fit.minuit:** properly handle parameters constrained with expressions ([d03f012](https://github.com/kmnhan/erlabpy/commit/d03f012b4fde92f445a24657dca1fb5b3600fa45))

### 🛠 Code Refactor

- set informative model name for MultiPeakModel ([d14ee9d](https://github.com/kmnhan/erlabpy/commit/d14ee9d6ac7962207700de50039a5b7a858fea6a))
- add gaussian and lorentzian for consistency ([07c0dfb](https://github.com/kmnhan/erlabpy/commit/07c0dfb9ecfb882e4f5f0ccfe942c1a835b613b2))

## v2.2.0 (2024-04-12)

### ✨ Features

- enable component evaluation for MultiPeakModel ([8875b74](https://github.com/kmnhan/erlabpy/commit/8875b7443d26313156fcdcc43586d40af4ff4f00))
- **analysis.fit:** add BCS gap equation and Dynes formula ([f862aa4](https://github.com/kmnhan/erlabpy/commit/f862aa4af4d2ba470f1ea074fc90442d9b18b336))

### 🐞 Bug Fixes

- curvefittingtool errors ([9abb99c](https://github.com/kmnhan/erlabpy/commit/9abb99c35633bc722469276d4837a2372c132042))

### 🛠 Code Refactor

- cleanup fit namespace ([906aa99](https://github.com/kmnhan/erlabpy/commit/906aa99193f78577e705218b2d6c22378611f84b))
- rename ExtendedAffineBroadenedFD to FermiEdgeModel ([a98aa82](https://github.com/kmnhan/erlabpy/commit/a98aa82bcbdf22ff8a156d800e336653f9afba07))
- **interactive:** exclude bad colormaps ([877c915](https://github.com/kmnhan/erlabpy/commit/877c915def6eb3dddb3862d6ac64c8c70f456ad3))

## v2.1.3 (2024-04-11)

### 🐞 Bug Fixes

- **interactive:** update data load functions used in imagetool ([c3abe35](https://github.com/kmnhan/erlabpy/commit/c3abe3517046ed603a9221de38b22257322d3a51))

## v2.1.2 (2024-04-11)

### 🐞 Bug Fixes

- **io:** prevent specifying invalid data_dir ([701b011](https://github.com/kmnhan/erlabpy/commit/701b011339ecba657a0f4a14e2fef19adeb4bf2b))
- **io:** fixes merlin summary data type resolving ([a91ad3d](https://github.com/kmnhan/erlabpy/commit/a91ad3d4387a23d25ac1b208cba8217e67efbec0))
- **io:** fix summary loading ([a5dd84a](https://github.com/kmnhan/erlabpy/commit/a5dd84af9eec0f835b3116bc7c470e57ef3f3e02))

## v2.1.1 (2024-04-10)

### 🐞 Bug Fixes

- **io:** enable specifying data_dir in loader context manager ([37913b8](https://github.com/kmnhan/erlabpy/commit/37913b80a1d7c6313a5b6cc4a3ab614565274c81))
- **io:** allow loader_class aliases to be None ([7eae2eb](https://github.com/kmnhan/erlabpy/commit/7eae2ebf13f972d368ddb9922a71fd3bbed014e5))

### 🛠 Code Refactor

- remove igor2 import checking ([b64d8f7](https://github.com/kmnhan/erlabpy/commit/b64d8f7fe22ebc1c4818e26f93f864fd402bbd05))
- **io:** default to always_single=True ([007bb3b](https://github.com/kmnhan/erlabpy/commit/007bb3b2703a647856c0a85e89075cf6572d263a))

## v2.1.0 (2024-04-09)

### ✨ Features

- **interactive:** overhaul dtool ([8e5ec38](https://github.com/kmnhan/erlabpy/commit/8e5ec3827dd2bd52475d454d5c5ef8aef7d665aa))

  Now supports interpolation, copying code, opening in imagetool, and 2D laplacian method.
- **interactive:** improve code generation ([7cbe857](https://github.com/kmnhan/erlabpy/commit/7cbe8572272f6c84a486599a990098ce8e3ff754))

  Automatically shortens code and allows literals in kwargs
- **interactive:** extend xImageItem, add right-click menu to open imagetool ([2b5bb2d](https://github.com/kmnhan/erlabpy/commit/2b5bb2dfc3d4173d950135306b3b30a018c6d389))

### 🐞 Bug Fixes

- sign error in minimum gradient ([c45be0c](https://github.com/kmnhan/erlabpy/commit/c45be0cf1a025c67e8af959ff83a9339cddbaaaa))
- **analysis.image:** normalize data for mingrad output for numerical stability ([0fc3711](https://github.com/kmnhan/erlabpy/commit/0fc3711a521ffb0cbb4f5206c06d923eced1200c))

### 🛠 Code Refactor

- **io:** validation now defaults to warning instead of raising an error ([8867a07](https://github.com/kmnhan/erlabpy/commit/8867a07304129beda749fa82d3909bf920fdb975))

## v2.0.0 (2024-04-08)

### 🚨 Breaking Changes

- `PolyFunc` is now `PolynomialFunction`, and `FermiEdge2dFunc` is now `FermiEdge2dFunction`. The corresponding model names are unchanged. ([20d784c](https://github.com/kmnhan/erlabpy/commit/20d784c1d8fdcd786ab73b3ae03d3e331dc04df5))
- This change disables the use of guess_fit. All fitting must be performed in the syntax recommended by lmfit. Addition of a accessor or a convenience function for coordinate-aware fitting is planned in the next release. ([59163d5](https://github.com/kmnhan/erlabpy/commit/59163d5f0e000d65aa53690a51b6db82df1ce5f1))

### ✨ Features

- **itool:** add copy code to PlotItem vb menu ([7b4f30a](https://github.com/kmnhan/erlabpy/commit/7b4f30ada21c5accc1d3824ad3d0f8097f9a99c1))

  For each plot in imagetool, a new 'copy selection code' button has been added to the right-click menu that copies the code that can slice the data to recreate the data shown in the plot.
- add 2D curvature, finally closes [#14](https://github.com/kmnhan/erlabpy/issues/14) ([7fe95ff](https://github.com/kmnhan/erlabpy/commit/7fe95ffcdf0531e456cfc97ae605467e4ae433c0))
- **plotting:** add N argument to plot_array_2d ([2cd79f7](https://github.com/kmnhan/erlabpy/commit/2cd79f7ee007058da09aff244cd75748698444ee))
- add scaled laplace ([079e1d2](https://github.com/kmnhan/erlabpy/commit/079e1d21201c7523877b06a0f04f7640027b0614))
- add gaussian filter and laplacian ([8628d33](https://github.com/kmnhan/erlabpy/commit/8628d336ff5b4219e4fd382293736e4cbf026d56))
- add derivative module with minimum gradient implementation ([e0eabde](https://github.com/kmnhan/erlabpy/commit/e0eabde60e6860c3827959b45be6d4f491918363))
- **fit:** directly base models on lmfit.Model ([59163d5](https://github.com/kmnhan/erlabpy/commit/59163d5f0e000d65aa53690a51b6db82df1ce5f1))

### 🐞 Bug Fixes

- **dynamic:** properly broadcast xarray input ([2f6672f](https://github.com/kmnhan/erlabpy/commit/2f6672f3b003792ecd98b4fbc99fb11fcc0efb8b))
- **fit.functions:** polynomial function now works for xarray input ([3eb80de](https://github.com/kmnhan/erlabpy/commit/3eb80dea31b6414fa9a694049b92b7334a4e10f5))
- **analysis.image:** remove critical typo ([fb7de0f](https://github.com/kmnhan/erlabpy/commit/fb7de0fc3ba9049c488a90bef8ee3c4feb935341))
- **analysis.image:** dtype safety of cfunc ([b4f9b17](https://github.com/kmnhan/erlabpy/commit/b4f9b17656c64be4cff876843ed0f3491d8310d4))
- set autodownsample off for colorbar ([256bf2d](https://github.com/kmnhan/erlabpy/commit/256bf2dc8c368d093a3578d7f9279b1ee4653534))
- disable itool downsample ([e626bba](https://github.com/kmnhan/erlabpy/commit/e626bba9fcd4fd31387ca3a07a9a33b7690f3645))

### ⚡️ Performance

- **itool:** add explicit signatures to fastbinning ([62e1d51](https://github.com/kmnhan/erlabpy/commit/62e1d516f0260f661fe9cd8f1fae9cb81afbcabe))

  Speedup initial binning by providing explicit signatures.

### 🛠 Code Refactor

- **fit:** unify dynamic function names ([20d784c](https://github.com/kmnhan/erlabpy/commit/20d784c1d8fdcd786ab73b3ae03d3e331dc04df5))
- update dtool to use new functions ([a6e46bb](https://github.com/kmnhan/erlabpy/commit/a6e46bb8b19512e438291afbbd5e0e9a4eb4fe87))
- **analysis.image:** add documentation and reorder functions ([340665d](https://github.com/kmnhan/erlabpy/commit/340665dc507a99acc7d56c46a2a2326fbb56b1e3))
- rename module to image and add citation ([b74a654](https://github.com/kmnhan/erlabpy/commit/b74a654e07d9f4522cee2db0b897f1ffcdb86e94))
- **dtool:** cleanup unused code ([f4abd34](https://github.com/kmnhan/erlabpy/commit/f4abd34bbf3130c0ec0fd2f9c830c8da43849f13))

## v1.6.5 (2024-04-03)

### 🐞 Bug Fixes

- make imports work without optional pip dependencies ([b8ac11d](https://github.com/kmnhan/erlabpy/commit/b8ac11d8fb4379f70a39c817332382c352391a64))

## v1.6.4 (2024-04-03)

### 🐞 Bug Fixes

- load colormaps only when igor2 is  available ([7927c7d](https://github.com/kmnhan/erlabpy/commit/7927c7db264bedb1a27b980d820d352f779b64c9))

## v1.6.3 (2024-04-03)

### 🐞 Bug Fixes

- leave out type annotation for passing tests ([eb25008](https://github.com/kmnhan/erlabpy/commit/eb2500838820172529ee751b5d8a624c950f66d2))

## v1.6.2 (2024-04-03)

### 🐞 Bug Fixes

- igor2 does not have to be installed on import time ([186727a](https://github.com/kmnhan/erlabpy/commit/186727ac8d50b662efeba8bee567cf1013ca936a))

## v1.6.1 (2024-04-03)

### 🐞 Bug Fixes

- remove all pypi dependencies from pyproject.toml ([1b2fd55](https://github.com/kmnhan/erlabpy/commit/1b2fd5594f00bba8367419cd00919eba45cde5a7))

### 🛠 Code Refactor

- remove ktool_old ([18ea072](https://github.com/kmnhan/erlabpy/commit/18ea0723fdf538bdbf2789ca73b2b962839ca3e5))

## v1.6.0 (2024-04-02)

### ✨ Features

- add mdctool ([a4976f9](https://github.com/kmnhan/erlabpy/commit/a4976f93cde51a41d667321a93dc2a90f23bddc3))

### 🛠 Code Refactor

- remove deprecated function and dependencies ([4b9c7b1](https://github.com/kmnhan/erlabpy/commit/4b9c7b1629d99fbf0108ca33791d1bfd59632199))

## v1.5.2 (2024-04-01)

### 🐞 Bug Fixes

- set values after setting bounds ([ab6d682](https://github.com/kmnhan/erlabpy/commit/ab6d682d0afafefcaec4c1ab6d673a39a75f40a6))
- proper patch all interpolator selection functions ([b91834e](https://github.com/kmnhan/erlabpy/commit/b91834e1b0be200bafb86ed3581f08cf1a5d42ef))
- make bz voronoi robust ([8259760](https://github.com/kmnhan/erlabpy/commit/8259760249be45892cd32f143b1b83aefe166c49))

### 🛠 Code Refactor

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

### 🐞 Bug Fixes

- restore argname detection that was broken with namespace changes ([863b702](https://github.com/kmnhan/erlabpy/commit/863b702b6373f9a219a1e770aa49c71145371681))
- namespace collision ([10edcdc](https://github.com/kmnhan/erlabpy/commit/10edcdc8b06425c380ca6caa2d3f5f2be5c13733))
- followup namespace change ([4c5222c](https://github.com/kmnhan/erlabpy/commit/4c5222cc93196f0b6a75a0101107a37e73748eeb))

### 🛠 Code Refactor

- allow offsetview upate chaining ([8d5ca4f](https://github.com/kmnhan/erlabpy/commit/8d5ca4f5b12c7d7060ea444773a9851f23db9850))

  This also means that _repr_html_ is automatically displayed when update or reset is called.
- improve consistency in accessors ([9596fd7](https://github.com/kmnhan/erlabpy/commit/9596fd723206f3e992fe00990f73364a61604cd6))

  Added setter method for configuration too.
- make prints consistent ([0021302](https://github.com/kmnhan/erlabpy/commit/002130224e3efc01615948a6443516e29d333cf5))
- change module names to prevent conflict with function names ([493a5aa](https://github.com/kmnhan/erlabpy/commit/493a5aab19c0d66851ca068e286a6aec92131e33))

  Cleanup erplot namespace and move tools to interactive.
- follow class naming conventions ([efb9610](https://github.com/kmnhan/erlabpy/commit/efb9610a864ef637f424c2f1b2871add7324b090))

## v1.5.0 (2024-03-27)

### ✨ Features

- add interactive tool to kspace accessor ([fb91cdb](https://github.com/kmnhan/erlabpy/commit/fb91cdb50229154c070df8dfaa80cddc8520ae6d))

### 🛠 Code Refactor

- accessors are now registered upon package import ([d79fee2](https://github.com/kmnhan/erlabpy/commit/d79fee2a28dd5ee59bfc6bd1ce224a44c5f40a24))

## v1.4.1 (2024-03-26)

### 🐞 Bug Fixes

- update package metadata ([ecfb88f](https://github.com/kmnhan/erlabpy/commit/ecfb88f2c23a7681e12d6f2dedcc316a28aa22c7))

  This should be classified as chore, but commiting as a fix to trigger CI

## v1.4.0 (2024-03-26)

### ✨ Features

- calculate kz in MomentumAccessor ([46979f9](https://github.com/kmnhan/erlabpy/commit/46979f907b120e5a4a88fdacd7d74a4b9dd41d6d))

  Add method that calculates kz array from given photon energy float
- make momentum conversion functions xarray compatible ([a7aa34b](https://github.com/kmnhan/erlabpy/commit/a7aa34ba983d3159c555ed66579d46eaf9e993aa))

## v1.3.1 (2024-03-25)

### 🐞 Bug Fixes

- fixes [#12](https://github.com/kmnhan/erlabpy/issues/12) ([02b49a1](https://github.com/kmnhan/erlabpy/commit/02b49a1da7550ae2b07819e6ccde3dcf750fc527))

## v1.3.0 (2024-03-25)

### ✨ Features

- **io:** add new data loader plugin for DA30 + SES ([7a27a2f](https://github.com/kmnhan/erlabpy/commit/7a27a2f27d9658f1091aaa48bcc78dea562898d8))

### 🐞 Bug Fixes

- **io:** properly handle registry getattr ([499526f](https://github.com/kmnhan/erlabpy/commit/499526fc1705bfbfbf8d3b80d50d65450dec7eae))

  This fixes an issue where _repr_html_ will fallback to __repr__.

  Additionally, `get` will now raise a KeyError instead of a ValueError.
