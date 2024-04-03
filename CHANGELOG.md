# CHANGELOG



## v1.6.4 (2024-04-03)

### Fix

* fix: load colormaps only when igor2 is  available ([`7927c7d`](https://github.com/kmnhan/erlabpy/commit/7927c7db264bedb1a27b980d820d352f779b64c9))


## v1.6.3 (2024-04-03)

### Fix

* fix: leave out type annotation for passing tests ([`eb25008`](https://github.com/kmnhan/erlabpy/commit/eb2500838820172529ee751b5d8a624c950f66d2))


## v1.6.2 (2024-04-03)

### Fix

* fix: igor2 does not have to be installed on import time ([`186727a`](https://github.com/kmnhan/erlabpy/commit/186727ac8d50b662efeba8bee567cf1013ca936a))


## v1.6.1 (2024-04-03)

### Chore

* chore(deps): add pre-commit to dev dependency ([`3a2fccd`](https://github.com/kmnhan/erlabpy/commit/3a2fccd978d23d806d2088ebd9ef60c7a2b20902))

* chore: make csaps optional ([`db31b06`](https://github.com/kmnhan/erlabpy/commit/db31b064c1f46edef7743fdd1c3ab7984e170b3c))

* chore: update issue templates ([`dfc2ab0`](https://github.com/kmnhan/erlabpy/commit/dfc2ab0fdfcf1fd5ab83dac2c9d6473b4d2cb7e1))

### Ci

* ci(github): remove linting, let pre-commit handle it ([`b209ecb`](https://github.com/kmnhan/erlabpy/commit/b209ecbb3c0a35d2bbeba8155bea3da9ffa58fe1))

* ci(pre-commit): add hooks ([`9b401c3`](https://github.com/kmnhan/erlabpy/commit/9b401c328bb3ff18dddcce40b935afa2b6e2624a))

### Documentation

* docs: rephrase kconv guide ([`dd2c022`](https://github.com/kmnhan/erlabpy/commit/dd2c022e42e692c2af640a1fc8d21c3e429781b2))

* docs: add ipykernel dependency to resolve failing builds ([`e5774a5`](https://github.com/kmnhan/erlabpy/commit/e5774a51c14ef6df190eb9f6198c274d2061cdd5))

* docs: add hvplot example ([`6997020`](https://github.com/kmnhan/erlabpy/commit/69970208ba6658f15e900ee6b9367177fcd86d29))

### Fix

* fix: remove all pypi dependencies from pyproject.toml ([`1b2fd55`](https://github.com/kmnhan/erlabpy/commit/1b2fd5594f00bba8367419cd00919eba45cde5a7))

### Refactor

* refactor: remove ktool_old ([`18ea072`](https://github.com/kmnhan/erlabpy/commit/18ea0723fdf538bdbf2789ca73b2b962839ca3e5))

### Style

* style: apply ruff to deprecated imagetools ([`b2c7596`](https://github.com/kmnhan/erlabpy/commit/b2c7596ed12d89edaa2be3fe2923388014c68007))

* style: apply pre-commit fixes ([`12b6441`](https://github.com/kmnhan/erlabpy/commit/12b6441419ed6c4ff4da921790c57a599032dba7))


## v1.6.0 (2024-04-02)

### Ci

* ci: speedup tests ([`618851e`](https://github.com/kmnhan/erlabpy/commit/618851e74d94301ec4f85a46facd46d3b6272571))

* ci: parallelize tests ([`232301a`](https://github.com/kmnhan/erlabpy/commit/232301a0ab26c9c32a355af11b5458395a1cd832))

* ci: migrate from pylint to ruff ([`2acd5e3`](https://github.com/kmnhan/erlabpy/commit/2acd5e3177f97f196d94644d75e3566a2714bf40))

* ci: add pre-commit configuration ([`063067d`](https://github.com/kmnhan/erlabpy/commit/063067dfdedefefc47e55096d310a4df54a5b999))

### Documentation

* docs: add pre-commit ci status badge ([`ae39d3d`](https://github.com/kmnhan/erlabpy/commit/ae39d3dbb0a058b59493b97507f88576f6b1737a))

* docs: add pre-commit badges ([`1b6702b`](https://github.com/kmnhan/erlabpy/commit/1b6702b9615c9881afb86883466f3e8846a2db12))

* docs: replace black with ruff ([`cb1a4b5`](https://github.com/kmnhan/erlabpy/commit/cb1a4b56a1b11b6d4630e5a36307befc48270294))

### Feature

* feat: add mdctool ([`a4976f9`](https://github.com/kmnhan/erlabpy/commit/a4976f93cde51a41d667321a93dc2a90f23bddc3))

### Refactor

* refactor: remove deprecated function and dependencies ([`4b9c7b1`](https://github.com/kmnhan/erlabpy/commit/4b9c7b1629d99fbf0108ca33791d1bfd59632199))

### Style

* style: remove unnecessary dict call ([`ea0e0e8`](https://github.com/kmnhan/erlabpy/commit/ea0e0e822f8487ec5238b651f3d72aafac5c6bcb))

* style: apply formatting ([`12e3a16`](https://github.com/kmnhan/erlabpy/commit/12e3a1649ce03792f79df8220f70572ff0ecc97a))

* style: remove implicit optionals and apply more linter suggestions ([`798508c`](https://github.com/kmnhan/erlabpy/commit/798508c6a65ac439be70f9b7cc32c801ae8632cb))

* style: reduce indentation ([`274a330`](https://github.com/kmnhan/erlabpy/commit/274a33037b0155b82d8f9eb5ec542568c54da1db))

* style: move imports to type-checking block ([`e1f4005`](https://github.com/kmnhan/erlabpy/commit/e1f400516dcbc220979346f25a7dcfe4018df906))

* style: cleanup kwargs and unnecessary pass statements ([`7867623`](https://github.com/kmnhan/erlabpy/commit/7867623e779636531cdf1e0675846d22d0045249))

* style: make collections literal ([`74a8878`](https://github.com/kmnhan/erlabpy/commit/74a887853c2e84f315d45e52844a9c0fa7b46e28))

* style: rewrite unnecessary dict calls as literal ([`10637f6`](https://github.com/kmnhan/erlabpy/commit/10637f622b29703a02b4666c5712e8cf03a96066))

* style: format with ruff ([`64f3fed`](https://github.com/kmnhan/erlabpy/commit/64f3fed42e4766c1fe70d6a9488b75179a905314))

* style: fix flake8-bugbear violations ([`4aade97`](https://github.com/kmnhan/erlabpy/commit/4aade97013cea63e20895fb39b43c04953a67984))

* style: apply ruff unsafe fixes ([`a1a7d9a`](https://github.com/kmnhan/erlabpy/commit/a1a7d9ae79d3afa88cffe7423bb942aca29bfd09))

* style: lint with pyupgrade and ruff ([`244e053`](https://github.com/kmnhan/erlabpy/commit/244e05305ce2e0b72c54e3eb7c96befb97762f87))

* style: apply linter suggestions ([`7295cbc`](https://github.com/kmnhan/erlabpy/commit/7295cbc5b08065d75447f80ab1d84eb1c15255f3))

### Unknown

* [pre-commit.ci] auto fixes from pre-commit.com hooks

for more information, see https://pre-commit.ci ([`b86c995`](https://github.com/kmnhan/erlabpy/commit/b86c9952be94b4b7f5e5918ed28cbf39b750ef09))


## v1.5.2 (2024-04-01)

### Documentation

* docs: update user guide notebooks ([`80ab771`](https://github.com/kmnhan/erlabpy/commit/80ab7717539e95c2cfe4a15f0713f259dfe04da3))

* docs: update docstring ([`b262765`](https://github.com/kmnhan/erlabpy/commit/b2627651648066dc8b98f023c5028c11f2929426))

* docs: update documentation ([`9051ed8`](https://github.com/kmnhan/erlabpy/commit/9051ed8d406c06ae4a037b65ed648a16843a0655))

### Fix

* fix: set values after setting bounds ([`ab6d682`](https://github.com/kmnhan/erlabpy/commit/ab6d682d0afafefcaec4c1ab6d673a39a75f40a6))

* fix: proper patch all interpolator selection functions ([`b91834e`](https://github.com/kmnhan/erlabpy/commit/b91834e1b0be200bafb86ed3581f08cf1a5d42ef))

* fix: make bz voronoi robust ([`8259760`](https://github.com/kmnhan/erlabpy/commit/8259760249be45892cd32f143b1b83aefe166c49))

### Refactor

* refactor: remove debug print statement in FastInterpolator class ([`712bd2c`](https://github.com/kmnhan/erlabpy/commit/712bd2ce90ad3534212d8a63c3fe10d780e243f5))

* refactor: add edge correction ([`87adcef`](https://github.com/kmnhan/erlabpy/commit/87adceffda2364f404de0860bfe8bf36b4cc1394))

* refactor: change variable name ([`b68949e`](https://github.com/kmnhan/erlabpy/commit/b68949ec59fd6bd7d7dad4ff9cc232b0e1ce4fba))

* refactor: make rotation transformations try fast interpolator first ([`e0a7908`](https://github.com/kmnhan/erlabpy/commit/e0a790833025f0c7e952ad17d120f46de3100555))

* refactor: update warning message ([`af67c1a`](https://github.com/kmnhan/erlabpy/commit/af67c1a507be35348b58862b6b51b92fac52781b))

* refactor: add several new accessors ([`664e92a`](https://github.com/kmnhan/erlabpy/commit/664e92a3e171512be26ea957df945e84134c880a))

* refactor: use new accessors and attrs ([`8e1dee2`](https://github.com/kmnhan/erlabpy/commit/8e1dee22d9d716f7e9bce29a1be3e68311494aa1))

* refactor: add qplot accessor ([`cb9aa01`](https://github.com/kmnhan/erlabpy/commit/cb9aa017bebd2ee6661f4eb87b988509d28a37a5))

* refactor: remove annotate_cuts ([`004ee80`](https://github.com/kmnhan/erlabpy/commit/004ee808dab13073cb3d2021d331767f6c28388a))

* refactor: dataloader cleanup ([`fd97780`](https://github.com/kmnhan/erlabpy/commit/fd977800a504256afd6018e9991b2d1e996277df))


## v1.5.1 (2024-03-28)

### Documentation

* docs: update README screenshots ([`04d6b44`](https://github.com/kmnhan/erlabpy/commit/04d6b443dc077cbf056dae9b2bf9630284e707ee))

* docs: use svg plots ([`aaa4842`](https://github.com/kmnhan/erlabpy/commit/aaa48420f69c71eb08180934ef2051819df92c03))

* docs: improve momentum conversion documentation ([`c315a1a`](https://github.com/kmnhan/erlabpy/commit/c315a1a6e4d6365a6cc02e861dae84daf9e0cc14))

* docs: update dev docs ([`7406308`](https://github.com/kmnhan/erlabpy/commit/740630899108d562bcc542bd6ae9d147b893c27d))

### Fix

* fix: restore argname detection that was broken with namespace changes ([`863b702`](https://github.com/kmnhan/erlabpy/commit/863b702b6373f9a219a1e770aa49c71145371681))

* fix: namespace collision ([`10edcdc`](https://github.com/kmnhan/erlabpy/commit/10edcdc8b06425c380ca6caa2d3f5f2be5c13733))

* fix: followup namespace change ([`4c5222c`](https://github.com/kmnhan/erlabpy/commit/4c5222cc93196f0b6a75a0101107a37e73748eeb))

### Refactor

* refactor: allow offsetview upate chaining

This also means that _repr_html_ is automatically displayed when update or reset is called. ([`8d5ca4f`](https://github.com/kmnhan/erlabpy/commit/8d5ca4f5b12c7d7060ea444773a9851f23db9850))

* refactor: improve consistency in accessors

Added setter method for configuration too. ([`9596fd7`](https://github.com/kmnhan/erlabpy/commit/9596fd723206f3e992fe00990f73364a61604cd6))

* refactor: make prints consistent ([`0021302`](https://github.com/kmnhan/erlabpy/commit/002130224e3efc01615948a6443516e29d333cf5))

* refactor: change module names to prevent conflict with function names

Cleanup erplot namespace and move tools to interactive. ([`493a5aa`](https://github.com/kmnhan/erlabpy/commit/493a5aab19c0d66851ca068e286a6aec92131e33))

* refactor: follow class naming conventions ([`efb9610`](https://github.com/kmnhan/erlabpy/commit/efb9610a864ef637f424c2f1b2871add7324b090))


## v1.5.0 (2024-03-27)

### Chore

* chore: remove unnecessary dependency on colorcet, cmasher, cmocean and seaborn ([`5fd2d61`](https://github.com/kmnhan/erlabpy/commit/5fd2d614f97e8bba4f34a9277c70835214a95be7))

* chore: add isort profile to project configuration ([`df269a9`](https://github.com/kmnhan/erlabpy/commit/df269a990e642135c76a60bfd19e0a6767974a40))

* chore: update dependencies and environment files

Fix python version and remove editable installs ([`6ec32dd`](https://github.com/kmnhan/erlabpy/commit/6ec32ddedb342d0556aacec0625c889b01f18b62))

* chore: change pyclip dependency to pyperclip

Although pyclip supports copying bytes, it&#39;s not on conda-forge. Using pyperclip instead. ([`db78f8e`](https://github.com/kmnhan/erlabpy/commit/db78f8e5a8be47ca4f23aa560e8aef88efb58c5b))

### Documentation

* docs: add momentum conversion documentation draft ([`5410763`](https://github.com/kmnhan/erlabpy/commit/54107632edd5a7a911a1c8d06c663fc48d5217a0))

* docs: add installation and contribution information ([`93a4e7c`](https://github.com/kmnhan/erlabpy/commit/93a4e7c4f43a8133f3f2149eb638261a9d56cfe6))

* docs: fix typo in README ([`2b5e2cf`](https://github.com/kmnhan/erlabpy/commit/2b5e2cf3d5dd9e93d34da578e5689f14d490405b))

### Feature

* feat: add interactive tool to kspace accessor ([`fb91cdb`](https://github.com/kmnhan/erlabpy/commit/fb91cdb50229154c070df8dfaa80cddc8520ae6d))

### Refactor

* refactor: accessors are now registered upon package import ([`d79fee2`](https://github.com/kmnhan/erlabpy/commit/d79fee2a28dd5ee59bfc6bd1ce224a44c5f40a24))

### Style

* style: apply linter suggestions ([`fe35da9`](https://github.com/kmnhan/erlabpy/commit/fe35da9a3494af28420ead2d8d40c5339788ac80))


## v1.4.1 (2024-03-26)

### Fix

* fix: update package metadata

This should be classified as chore, but commiting as a fix to trigger CI ([`ecfb88f`](https://github.com/kmnhan/erlabpy/commit/ecfb88f2c23a7681e12d6f2dedcc316a28aa22c7))


## v1.4.0 (2024-03-26)

### Chore

* chore: update workflow triggers ([`fb158f3`](https://github.com/kmnhan/erlabpy/commit/fb158f3a6b6ded4ed2d573f4d33f85fbd36809b5))

* chore: update build command ([`a22b8e5`](https://github.com/kmnhan/erlabpy/commit/a22b8e58bb744d02c2e0214af6185da8c66cbe29))

* chore: update CI/CD badge urls ([`db61b29`](https://github.com/kmnhan/erlabpy/commit/db61b29fa0d92f54f7134ce5bd1c021aacfae647))

* chore: make pyproject.toml compatible

README file link fixed, and remove direct dependencies.
Add build command for automatic building ([`959f687`](https://github.com/kmnhan/erlabpy/commit/959f6874f421ddd7bdf816f96c78d1533081b24d))

* chore: update workflows to upload to pypi ([`2902b68`](https://github.com/kmnhan/erlabpy/commit/2902b683051ce651be6d5e38c6bdf6e55a9681f1))

### Documentation

* docs: update docstring and apply linter suggestions ([`de3ee01`](https://github.com/kmnhan/erlabpy/commit/de3ee01dd35973186d69125f24d9527cfa8abd94))

* docs: update README ([`8bd239f`](https://github.com/kmnhan/erlabpy/commit/8bd239f562d2d2345178c339a455ec23a5aa8082))

### Feature

* feat: calculate kz in MomentumAccessor

Add method that calculates kz array from given photon energy float ([`46979f9`](https://github.com/kmnhan/erlabpy/commit/46979f907b120e5a4a88fdacd7d74a4b9dd41d6d))

* feat: make momentum conversion functions xarray compatible ([`a7aa34b`](https://github.com/kmnhan/erlabpy/commit/a7aa34ba983d3159c555ed66579d46eaf9e993aa))


## v1.3.1 (2024-03-25)

### Documentation

* docs: update documentation

- Move rst README contents to docs, replace with newly written markdown.
- Add screenshot images to  documentation and README ([`69a02fa`](https://github.com/kmnhan/erlabpy/commit/69a02fa3591720cf79b01289fd9dfb9cf55c26db))

* docs: update README ([`15f61bf`](https://github.com/kmnhan/erlabpy/commit/15f61bfe7a1734cece17479064c6d7946e2701f9))

### Fix

* fix: fixes #12 ([`02b49a1`](https://github.com/kmnhan/erlabpy/commit/02b49a1da7550ae2b07819e6ccde3dcf750fc527))


## v1.3.0 (2024-03-25)

### Chore

* chore: fix wrong branch name in release workflow ([`76a51b8`](https://github.com/kmnhan/erlabpy/commit/76a51b87180065631b6f5ca0678a87dfaa7e267e))

* chore: configure semantic release ([`3ebdecb`](https://github.com/kmnhan/erlabpy/commit/3ebdecb45b510ed5e45e25fbc10d58ebc0b4ce20))

* chore: bump version to 1.2.1 ([`30ec306`](https://github.com/kmnhan/erlabpy/commit/30ec3065234b6f727ed8f74daa1a866b82b0abc7))

### Documentation

* docs: update README ([`79ba5b4`](https://github.com/kmnhan/erlabpy/commit/79ba5b42f5089d9fd81ccfc69dadda21914b42a7))

### Feature

* feat(io): add new data loader plugin for DA30 + SES ([`7a27a2f`](https://github.com/kmnhan/erlabpy/commit/7a27a2f27d9658f1091aaa48bcc78dea562898d8))

### Fix

* fix(io): properly handle registry getattr

This fixes an issue where _repr_html_ will fallback to __repr__.
Additionally, `get` will now raise a KeyError instead of a ValueError. ([`499526f`](https://github.com/kmnhan/erlabpy/commit/499526fc1705bfbfbf8d3b80d50d65450dec7eae))

### Style

* style: adjust loader registry repr ([`1fc31af`](https://github.com/kmnhan/erlabpy/commit/1fc31af083654a6c093bf343a881fcab37f9fbe2))

* style: remove incorrect type annotation ([`69dbf8a`](https://github.com/kmnhan/erlabpy/commit/69dbf8a1041ab22ea5d928623adae497b5ecd919))


## v1.2.1 (2024-03-25)

### Build

* build: drop python 3.10 support ([`183769f`](https://github.com/kmnhan/erlabpy/commit/183769f9af371f5e3a910976356ac2ac384c9ebb))

* build: update pyproject.toml to properly include dependencies ([`d39e69e`](https://github.com/kmnhan/erlabpy/commit/d39e69e7f5a6cf1b9b6088689f1f7756e25edc4f))

* build: update requirements.txt ([`91cac05`](https://github.com/kmnhan/erlabpy/commit/91cac05fdade8c5ce21a9e28d311159f57351ac9))

* build: update requirements.txt ([`bf2e534`](https://github.com/kmnhan/erlabpy/commit/bf2e5346d4c3030a4ef8061e5f463491725e9f9c))

* build: bump version ([`a68a6ea`](https://github.com/kmnhan/erlabpy/commit/a68a6ea1e061ffbb6c4ae966a6b23c6710175744))

* build: bump version to 1.1.0 ([`791af02`](https://github.com/kmnhan/erlabpy/commit/791af027a4e3ebf3cbad4bcb9af490986a2be2c0))

* build: bump setuptools minver ([`b55da17`](https://github.com/kmnhan/erlabpy/commit/b55da17c30ec94e897cf2ee8abf151796f3f78b7))

* build: try automatic discovery ([`05040e3`](https://github.com/kmnhan/erlabpy/commit/05040e3f98b6094258d4a7be7a33feeefa1fd44b))

* build: modify to updated directory structure ([`625ffcf`](https://github.com/kmnhan/erlabpy/commit/625ffcf2eb5a2914024efd815de40a910b1ae040))

* build: add setuptools-scm as build dependency ([`4ff4791`](https://github.com/kmnhan/erlabpy/commit/4ff47910022571d0d37ddc923755705dfa8a549e))

* build: fix typo in requirements.txt ([`29571de`](https://github.com/kmnhan/erlabpy/commit/29571de3bec9a9bbbcdff850b34e6caec167a109))

* build: update README, remove setup.cfg and update pyproject.toml ([`3736f46`](https://github.com/kmnhan/erlabpy/commit/3736f4629621d036ec071f27b0660bf7e99f8e86))

* build: add reqquirements.txt ([`dacd5be`](https://github.com/kmnhan/erlabpy/commit/dacd5be9985e28d1c67ac46d2badd24f148d4062))

* build: add yml file for intel ([`4844091`](https://github.com/kmnhan/erlabpy/commit/48440915782e1e8b918c88e4d9dce38b9703e43c))

* build: update readme type ([`40cf807`](https://github.com/kmnhan/erlabpy/commit/40cf807e522a59e9aad8ff6196788fe58846584b))

* build: update dependencies ([`e1bb13a`](https://github.com/kmnhan/erlabpy/commit/e1bb13ae6a350adf51e1b08ebe83f2732d1797f8))

* build: update instructions, cleanup dependencies ([`66d0e25`](https://github.com/kmnhan/erlabpy/commit/66d0e25d07393dc4b16a4d5e5994ea4fa822dd33))

* build: add yml for intel ([`8db3b26`](https://github.com/kmnhan/erlabpy/commit/8db3b269544c2f086f98fa302beb594f10573f65))

* build: update dependencies ([`21e8fab`](https://github.com/kmnhan/erlabpy/commit/21e8fabc2896d2b1fac5d2567eecf2df63047fe4))

* build: update dependencies ([`88a0170`](https://github.com/kmnhan/erlabpy/commit/88a01706b4f49d3b56f017badfd69af63df897f9))

* build: update dependencies
dropped python 3.9 support (temporary) ([`832e1ed`](https://github.com/kmnhan/erlabpy/commit/832e1ed25c299f7838dee1bc3d862a43ffe0bbb7))

* build: update requirements ([`fe26fc2`](https://github.com/kmnhan/erlabpy/commit/fe26fc2d17e4c107ef9b9a2773ce6c6318e18b2e))

* build: refactor requirements ([`81d3038`](https://github.com/kmnhan/erlabpy/commit/81d3038b03a717cf18fff655dba4224a7e67570a))

* build: add environment.yml for conda env ([`ea7bbf4`](https://github.com/kmnhan/erlabpy/commit/ea7bbf4309efeecd0ed1dcbd1791e482ab4d3ced))

* build: fix dependencies ([`e344f99`](https://github.com/kmnhan/erlabpy/commit/e344f990e55e7ddcc7e6ef5c87999d2c777b44a5))

* build: fix dependencies ([`5a6e38c`](https://github.com/kmnhan/erlabpy/commit/5a6e38c8de1f554819aa74d9e9e5a4f4524a2f6f))

### Chore

* chore: update .gitignore ([`a4b2cbc`](https://github.com/kmnhan/erlabpy/commit/a4b2cbc322a4e031d75db87e87e7d532c585709f))

* chore: update .gitignore ([`221b623`](https://github.com/kmnhan/erlabpy/commit/221b6232a6dc39b42ac1a3fd1a5abd0e2f1441d4))

* chore: add some type hints remove deprecated ([`79d7349`](https://github.com/kmnhan/erlabpy/commit/79d7349953a1747e177796d14da44dbed5b02874))

### Ci

* ci: update flake8 args ([`d720ee2`](https://github.com/kmnhan/erlabpy/commit/d720ee2b27df60fb42eca954ae7691641669828d))

* ci: install qt runtime dependency ([`9e18d14`](https://github.com/kmnhan/erlabpy/commit/9e18d1499cddf987f8d267d8f358646b2a23ae32))

* ci: create test.yml ([`c788eef`](https://github.com/kmnhan/erlabpy/commit/c788eef6025382a744d944e9e64c38b935324158))

### Documentation

* docs: update docstring

Changed configuration so that type annotations appear in both the signature and the description. ([`f7bf9cb`](https://github.com/kmnhan/erlabpy/commit/f7bf9cb2f51fc8a57aa9a84b2c690b06cd029ec8))

* docs: update plotting examples with display_expand_data=False ([`79117c4`](https://github.com/kmnhan/erlabpy/commit/79117c4814d66dd5dc3d7cfc85af8f9b662651bf))

* docs: update dosctring ([`2eccbf3`](https://github.com/kmnhan/erlabpy/commit/2eccbf3b18b7c52250b52c958cf5899fd05ea9a4))

* docs: update docstring ([`4e8be19`](https://github.com/kmnhan/erlabpy/commit/4e8be19c1153d816e06d41d446bef7d057a4eb9b))

* docs: add link to api reference in guide ([`e750519`](https://github.com/kmnhan/erlabpy/commit/e750519f3936ad4b7d4d532daebbe72c1b35f3d3))

* docs: add update instructions ([`71bb0c6`](https://github.com/kmnhan/erlabpy/commit/71bb0c66815eef6acf1222c1f6e1081b753db493))

* docs: update installation instructions ([`64f9a3b`](https://github.com/kmnhan/erlabpy/commit/64f9a3bf39a4c0ef49fd23ead37e34bf7a94f620))

* docs: update README ([`428ea67`](https://github.com/kmnhan/erlabpy/commit/428ea671748f7062872533603b954ed27933a72a))

* docs: cleanup top-level headers ([`39e9fcf`](https://github.com/kmnhan/erlabpy/commit/39e9fcf8962f0282902a90595d52a7d576fe35dd))

* docs: update documentation

Cleanup header styles and add cards to index page ([`32ca369`](https://github.com/kmnhan/erlabpy/commit/32ca369ee6238696792229b63a29c27aa4060ddc))

* docs: update docstrings for some functions ([`d6a8e7d`](https://github.com/kmnhan/erlabpy/commit/d6a8e7d408d795586bd6b6bc6fc6d4f48443a853))

* docs: cleanup conf.py ([`eab5e60`](https://github.com/kmnhan/erlabpy/commit/eab5e6078850de71c447cc1aa42b266bc10caf3b))

* docs: update documentation ([`e03d118`](https://github.com/kmnhan/erlabpy/commit/e03d1182f294ef909b03ff04ea3d805349304bc1))

* docs: update to use bibtex ([`c571b11`](https://github.com/kmnhan/erlabpy/commit/c571b11774c2dc2c152ccace4953976ad29600ef))

* docs: update accessor docstring ([`d32f352`](https://github.com/kmnhan/erlabpy/commit/d32f35283208b6fbc72264a8f966beef4c39e0c9))

* docs: update documentation ([`ec7a47e`](https://github.com/kmnhan/erlabpy/commit/ec7a47e28eee265ec01a66419fd32412ed565ba2))

* docs: update documentation ([`0390eb3`](https://github.com/kmnhan/erlabpy/commit/0390eb37d758459a83cbb20b9d567d6f4937619d))

* docs: update documentation ([`a7bfea2`](https://github.com/kmnhan/erlabpy/commit/a7bfea28f2f9cef1a32d591d3ebbf32c0d968450))

* docs: update docstring to use PEP annotation ([`fc496df`](https://github.com/kmnhan/erlabpy/commit/fc496df1a9efe0ca345a183decdddaf54cdb0cc6))

* docs: add copybutton ([`5e14247`](https://github.com/kmnhan/erlabpy/commit/5e142478568d17e4160f7ea52b99fa4b15a75b0d))

* docs: update docstring ([`9b9eaff`](https://github.com/kmnhan/erlabpy/commit/9b9eaff437f39e4b684a2e1ef1f0d91a2762507b))

* docs: use default font for docs plot generation ([`d36ce1d`](https://github.com/kmnhan/erlabpy/commit/d36ce1d5687168958da9cd33c9e225a52443c6ef))

* docs: revert to pip due to slow build time, keep at py311 until 3.12.2 is available ([`954f357`](https://github.com/kmnhan/erlabpy/commit/954f3573e578da9a73a3256011b9016df01f6f30))

* docs: build with conda ([`d2b8c40`](https://github.com/kmnhan/erlabpy/commit/d2b8c408d75c5d1dcedef6f7cfbdd12d93a57d34))

* docs: retry build with py312 ([`bb12503`](https://github.com/kmnhan/erlabpy/commit/bb12503c5c6d67166a78de986df70778902c65cf))

* docs: update requirements.txt ([`0b2eaad`](https://github.com/kmnhan/erlabpy/commit/0b2eaad96ff270d688f7b0549f57f8a753730b38))

* docs: Add varname to requirements.txt ([`90df11e`](https://github.com/kmnhan/erlabpy/commit/90df11e1dd07ac8c3ea009b0b6c98d42014910d7))

* docs: Add h5netcdf to requirements.txt ([`6be918c`](https://github.com/kmnhan/erlabpy/commit/6be918c905f463dc171c9df63065f0e87bd1811f))

* docs: try build with py311 ([`233553d`](https://github.com/kmnhan/erlabpy/commit/233553d56d7d83692c4b9a9e7a35848554e21846))

* docs: comment out code for latex generation and add new dependencies ([`ad9b974`](https://github.com/kmnhan/erlabpy/commit/ad9b97470ea45c7f1abc3e3ccaeccc5f761e5fc1))

* docs: update requirements ([`fde56f2`](https://github.com/kmnhan/erlabpy/commit/fde56f2c7ac7c146dac0a28e76decfc250d400dc))

* docs: Add Sphinx documentation dependencies ([`11d200e`](https://github.com/kmnhan/erlabpy/commit/11d200e113a54c047acaccd430f90da6e9c19f1f))

* docs: Add .readthedocs.yaml and docs/requirements.txt files ([`9296c71`](https://github.com/kmnhan/erlabpy/commit/9296c710fb082442bc4226b82f383e4e6fddf45c))

* docs: update documentation ([`fd5fc1a`](https://github.com/kmnhan/erlabpy/commit/fd5fc1a616e4b76b6cd37e3065f58e49e94b53b5))

* docs: Add figmpl_directive to extensions in conf.py ([`3ca728f`](https://github.com/kmnhan/erlabpy/commit/3ca728f0aa2d6904c8bf31791275992b27ce1ebe))

* docs: update documentation ([`fa312c0`](https://github.com/kmnhan/erlabpy/commit/fa312c05cc9dabb858d9268808b3efd2bead1a3e))

* docs: update documentation ([`2e4e573`](https://github.com/kmnhan/erlabpy/commit/2e4e5736cec670f51e55e81db603c82b98a9e78b))

* docs: add some comments ([`e66bc31`](https://github.com/kmnhan/erlabpy/commit/e66bc31cd9f9905a16a94f3b491170556b29adbc))

* docs(itool): improve tooltip ([`97ad19b`](https://github.com/kmnhan/erlabpy/commit/97ad19b1581613b6eb7012d4e04d79209f12075d))

* docs: update docstring ([`700a9a3`](https://github.com/kmnhan/erlabpy/commit/700a9a3982ddcd16756efacb53bd5be861f7ed18))

* docs: update documentation ([`e12b9af`](https://github.com/kmnhan/erlabpy/commit/e12b9afc1750823db4edb8034f4b225e51460cca))

* docs: update docstring and formatting ([`dc8e544`](https://github.com/kmnhan/erlabpy/commit/dc8e54415fd26090782075847fd78fa942f7efca))

* docs: update docstring ([`9d7139c`](https://github.com/kmnhan/erlabpy/commit/9d7139ceb90676189e300479d7775c26a4c107ed))

* docs: update README ([`8ca11fe`](https://github.com/kmnhan/erlabpy/commit/8ca11fec492fd1b29737cec5df3dc8535e23b9bb))

* docs: update documentation ([`dc857f7`](https://github.com/kmnhan/erlabpy/commit/dc857f73931413a40eb1b14cced16dda9fe5f4c8))

* docs: update documentation ([`dec0b39`](https://github.com/kmnhan/erlabpy/commit/dec0b39a4d6182af688f7030f88e747cab7b255d))

* docs: update documentation ([`01d0cb4`](https://github.com/kmnhan/erlabpy/commit/01d0cb4a868076e4e60851a58606df6d5b00ba4e))

* docs: update documentation ([`cd914c6`](https://github.com/kmnhan/erlabpy/commit/cd914c650b651ad97ce6f4083c999c5251be02fc))

* docs: update documentation ([`1944109`](https://github.com/kmnhan/erlabpy/commit/194410948df009ea56c91b4fbacafd5dd07f6537))

* docs: update documentation ([`83615f6`](https://github.com/kmnhan/erlabpy/commit/83615f67bc27ee8d1bf9dcf2a2fd3a9e62b1ef51))

* docs: update documentation ([`c28e6c4`](https://github.com/kmnhan/erlabpy/commit/c28e6c4e52817ed2cdec7f7902529bb7f99b829a))

* docs: update documentation ([`0577134`](https://github.com/kmnhan/erlabpy/commit/05771342c12225536328c5bb959ff4bc1898ee34))

* docs: update README ([`ccd3c05`](https://github.com/kmnhan/erlabpy/commit/ccd3c05c9a46110b3b740ea0ee061cf8ba595661))

* docs: update documentation ([`e4df36f`](https://github.com/kmnhan/erlabpy/commit/e4df36f98c0031cd5a25738fa4698fd77836e79b))

* docs: update documentation ([`5388383`](https://github.com/kmnhan/erlabpy/commit/5388383fd27be899d9e749e3cf3423c3e9a07e0c))

* docs: update docstring ([`793f1de`](https://github.com/kmnhan/erlabpy/commit/793f1de1b1be734343f7c67a2d59f93a955adeac))

* docs: update documentation ([`1ff5855`](https://github.com/kmnhan/erlabpy/commit/1ff5855f70b33e47034c03347e6d531f5553b431))

* docs: update documentation ([`d01edcb`](https://github.com/kmnhan/erlabpy/commit/d01edcbffeb38d0e9473e865053f52d646bfda55))

* docs: update docstring ([`d9df4f6`](https://github.com/kmnhan/erlabpy/commit/d9df4f67d596cde8efed7bb5a07451be2eb8924b))

* docs: update documentation ([`00a9472`](https://github.com/kmnhan/erlabpy/commit/00a947298a42c857574a0f1fa0c8a10c1e02dd27))

* docs: update README ([`7f62e74`](https://github.com/kmnhan/erlabpy/commit/7f62e7416ad2514372379f0d8a0a33c2946848af))

* docs: update README ([`f774da7`](https://github.com/kmnhan/erlabpy/commit/f774da73f7bf11cc2b2c6740552fb6f53acb2bb9))

* docs: update README ([`e1aa11d`](https://github.com/kmnhan/erlabpy/commit/e1aa11dedba93636431b761160d883e36189050e))

* docs: update docstring ([`ee31e1f`](https://github.com/kmnhan/erlabpy/commit/ee31e1fe997c245981057c59a602fe1b7a253501))

* docs: update docstring ([`b32e92f`](https://github.com/kmnhan/erlabpy/commit/b32e92f0cee5c5aa2b7793a17d95c21ff49ce94d))

### Feature

* feat(gold): apply automatic weights proportional to sqrt(count) on edge fit ([`717b9c8`](https://github.com/kmnhan/erlabpy/commit/717b9c814ce6fd38567695adb515973e6097ec50))

* feat: add class to handle momentum conversion offsets ([`15416e5`](https://github.com/kmnhan/erlabpy/commit/15416e5aeca748225e35659c65dcc07b82b40007))

* feat: add translation layer between lmfit models and iminuit ([`0f2f894`](https://github.com/kmnhan/erlabpy/commit/0f2f894994a00714e50e934dd4d5b518540539df))

* feat: include all cmaps by default ([`3afe72e`](https://github.com/kmnhan/erlabpy/commit/3afe72ece2474c9adcfb89cb5037c50af2e2f0e5))

* feat: change default colormap to CET-L20 ([`274122a`](https://github.com/kmnhan/erlabpy/commit/274122a4b2e582a24385c4dd748206ae0165b4b8))

* feat(goldtool): can access fit result after window close ([`4c9f232`](https://github.com/kmnhan/erlabpy/commit/4c9f232adfde874d975f033dc9bdcc8bf4969787))

* feat(io): style summary ([`6919929`](https://github.com/kmnhan/erlabpy/commit/6919929815401aa5d6ee4a37398c0593fab3657d))

* feat(io): summarize will now default to set directory ([`ca5b65a`](https://github.com/kmnhan/erlabpy/commit/ca5b65a0e850058837945b7f53864cfc6b32e933))

* feat(constants): add neutron mass ([`379ef37`](https://github.com/kmnhan/erlabpy/commit/379ef3705eb2f952121c92c9b653f2ca715618ee))

* feat(io): make loaders accessible with __getattr__ ([`b4884e4`](https://github.com/kmnhan/erlabpy/commit/b4884e4989c3565fdeccb8e18ec6a840cc67a7d2))

* feat(io): add loader plugin for SSRL Beamline 5-2 ([`67ced64`](https://github.com/kmnhan/erlabpy/commit/67ced64352e0f071fad5ebe441348108b6834784))

* feat(io): show full table on summary in ipython ([`985046f`](https://github.com/kmnhan/erlabpy/commit/985046f8a13911dc2c0be7cc68e530e93de37683))

* feat(io.dataloader): allow loaders to specify mapping dictionary on cut postprocessing ([`73abb0d`](https://github.com/kmnhan/erlabpy/commit/73abb0d1a078779130e6f82fadc9f44e62619799))

* feat(io): new arg to get_files ([`a9be216`](https://github.com/kmnhan/erlabpy/commit/a9be2165b61082956d5118c328e6484bdd0694c5))

* feat(io.igor): remove dependency on find_first_file ([`1d95292`](https://github.com/kmnhan/erlabpy/commit/1d95292ff4ba1c6ad5fcc95cae2d36c452d52ecf))

* feat(io): dataloader now preserves more attributes ([`f1157ba`](https://github.com/kmnhan/erlabpy/commit/f1157ba7afcb0b259932062edb0dc2138c749c55))

* feat(io): new data loader!

Implemented  new class-based data loader. Currently only implemented for ALS BL4.0.3. ([`6c85cba`](https://github.com/kmnhan/erlabpy/commit/6c85cba810936403734cfeefa65b005e2d2e329e))

* feat(io): add new utility function ([`a27a9f8`](https://github.com/kmnhan/erlabpy/commit/a27a9f89ce56b35a7a35fd1014c01269378a8148))

* feat(igor): remove attribute renaming, keep original attribute as much as possible ([`912376c`](https://github.com/kmnhan/erlabpy/commit/912376c0216c901e85dd68b03817cbc8b85522a7))

* feat: show elapsed time for momentum conversion ([`8889737`](https://github.com/kmnhan/erlabpy/commit/8889737aa6a8c5da2ef2ba17e5201816c99c574b))

* feat: add sample data generation in angles ([`1d2600d`](https://github.com/kmnhan/erlabpy/commit/1d2600d37b7fe323adb016a1daf2b9811d6f4593))

* feat(constants): add electron rest energy mc^2 ([`8e46445`](https://github.com/kmnhan/erlabpy/commit/8e46445efd3fa5fe623e2ab65ee939039b4dc149))

* feat(ktool): full kz support

Added inner potential spinbox and projected BZ overlay ([`c3284f7`](https://github.com/kmnhan/erlabpy/commit/c3284f785484b5f506bd2025afa57661ac947eec))

* feat(bz): add BZ extending ([`cbbb11a`](https://github.com/kmnhan/erlabpy/commit/cbbb11ac8997951cda17ec976fe321b8699d1eff))

* feat(itool): allow different label maximum widths

The controls took up too much space if one of the dim names was long ([`3467ae1`](https://github.com/kmnhan/erlabpy/commit/3467ae1720363ebfc026e3f5fc4f9517a342a448))

* feat: change transpose button order for 4D data ([`018002c`](https://github.com/kmnhan/erlabpy/commit/018002cd5606a091a528f866b2edcff051ce7570))

* feat: add support for automatic cut and hv-dependent momentum conversion ([`f5be05f`](https://github.com/kmnhan/erlabpy/commit/f5be05fb80a4e5dc4f2b976e8894bddc07d43d91))

* feat: add bz masking function ([`f3a7d21`](https://github.com/kmnhan/erlabpy/commit/f3a7d21c44784e4191011e8e95202889afc15ccc))

* feat(interpolate): add more checks and warnings ([`f1c223f`](https://github.com/kmnhan/erlabpy/commit/f1c223fbce27d65ed196fafd196c8b9c28d978dd))

* feat(ktool): frontend tweaks

Added wait dialog for showing imagetool and make labels prettier ([`d0050d0`](https://github.com/kmnhan/erlabpy/commit/d0050d02270c5ad8e51af281cb7a9f7a1cb89913))

* feat: add new momentum conversion tool ([`dfa96fe`](https://github.com/kmnhan/erlabpy/commit/dfa96fe9f20b57699d43d3d23755c663c3793d30))

* feat: implement new momentum conversion functions. Currently only supports kxky and kxkyE conversion. ([`956a0bd`](https://github.com/kmnhan/erlabpy/commit/956a0bd8e034754d0b208a7c654d77e690491b98))

* feat: switch data loaders and plotting functions to new angle coordinate convention.

Adopts the angle convention given by Y. Ishida and S. Shin, Rev. Sci. Instrum. 89, 043903 (2018). This is a breakin change, and momentum conversion will be entirely re-written.
All angles are now given in degrees, goodbye to radians. ([`b5734df`](https://github.com/kmnhan/erlabpy/commit/b5734df4a4c0009131b0510645f4cb28eff702cb))

* feat: misc. changes to momentum conversion tool ([`1221fd0`](https://github.com/kmnhan/erlabpy/commit/1221fd0403977b840e650221111ae3a72a69cb65))

* feat: load new resistance data ([`782cdf8`](https://github.com/kmnhan/erlabpy/commit/782cdf82c8b6b12d15c3f964dfe4a924131e5d4a))

* feat: add spline module (work in progress) ([`69f8e4c`](https://github.com/kmnhan/erlabpy/commit/69f8e4c60cd147fc677f27aee927607153db3ac6))

* feat(goldtool): scale roi initial position to data ([`65f71d2`](https://github.com/kmnhan/erlabpy/commit/65f71d2c3ec1888f8e8fecb84b0e98189859f191))

* feat(goldtool): add remaining time to progress bar ([`348aaf7`](https://github.com/kmnhan/erlabpy/commit/348aaf7acde9421156bb6027bc0853a4142f166f))

* feat: add atom plotting module ([`2fcf012`](https://github.com/kmnhan/erlabpy/commit/2fcf012c2866b598a82741961b9ad29602a63748))

* feat: minor tweaks to mplstyle ([`10972b8`](https://github.com/kmnhan/erlabpy/commit/10972b8f70c041835d0c8274448e53be67172d18))

* feat: add crop to  plot_array ([`36d8536`](https://github.com/kmnhan/erlabpy/commit/36d85366ffd1c84298df79d6721b26f17d8cf031))

* feat(colors): make axes not required for nice_colorbar ([`48af74b`](https://github.com/kmnhan/erlabpy/commit/48af74b45e926f5ff9f2c0c717d27d05987837b8))

* feat(goldtool): add diffev to list of methods ([`d21c915`](https://github.com/kmnhan/erlabpy/commit/d21c915f7993c66f1a79dc7cfa28b06b63cc89f8))

* feat(itool): plot class customization ([`fee3913`](https://github.com/kmnhan/erlabpy/commit/fee39133c4208e7733a18b31d77a7898ac4b5713))

* feat: better cache management for slicer ([`9f49374`](https://github.com/kmnhan/erlabpy/commit/9f49374ee98fb2cc101974292d959ee0f6898a9a))

* feat(betterspinbox): make keyboardTracking off by default ([`753ddcd`](https://github.com/kmnhan/erlabpy/commit/753ddcd344ff6859ff44cecaff4fb641e4e981dd))

* feat(annotation): add axis unit scaling ([`a61b8f3`](https://github.com/kmnhan/erlabpy/commit/a61b8f38165828d3f118c682cbfe05325c413e22))

* feat(itool): add option to decouple colors ([`3e2f132`](https://github.com/kmnhan/erlabpy/commit/3e2f1325baa1011540b29f1f36fd8b12a7770114))

* feat(style): update poster style ([`da58226`](https://github.com/kmnhan/erlabpy/commit/da582267a69fea8efb86e93478c398d70c7e411c))

* feat(accessor): use pyarpes for 2d kconv ([`9cbe422`](https://github.com/kmnhan/erlabpy/commit/9cbe422d19bb95e17a13b4757b8239d08069d850))

* feat: improve extendability ([`ded8841`](https://github.com/kmnhan/erlabpy/commit/ded884139e25cea9549fd3e3e7b0519c03b1d447))

* feat(itool): pass parent ([`c8feac4`](https://github.com/kmnhan/erlabpy/commit/c8feac43871fba3a0c12db56d33c819efdfd3e41))

* feat(itool): add base imagetool with only controls ([`fdb8a6b`](https://github.com/kmnhan/erlabpy/commit/fdb8a6b0e52f83c05441d03a06bd15d83b8a261a))

* feat: add option to return individual regions ([`7c5b081`](https://github.com/kmnhan/erlabpy/commit/7c5b08181f0cfc6d727fafe6a8789ec952ac895f))

* feat(colors): add function to unify color limits ([`33d571f`](https://github.com/kmnhan/erlabpy/commit/33d571f4dfcdecec02d536f5f5a4cd9af349d2df))

* feat(colors): add function that combines colormaps ([`f19d87c`](https://github.com/kmnhan/erlabpy/commit/f19d87ca5c3de53ca8623d7954a280b7050f2d15))

* feat: add new colormap! ([`474ad52`](https://github.com/kmnhan/erlabpy/commit/474ad521c3c5726341c17b9b4759ee501661ce73))

* feat: k range freedom in sample data generation ([`9dd5b00`](https://github.com/kmnhan/erlabpy/commit/9dd5b0025aaf52ef0bf4685ef5e380ccd5a72ac4))

* feat: make BZ plotter standalone ([`3bec642`](https://github.com/kmnhan/erlabpy/commit/3bec64218de70cabf8a43b78bf6728bef6402b49))

* feat(igorinterface): add to load wave menu ([`da704c5`](https://github.com/kmnhan/erlabpy/commit/da704c5c083a41db3999b031a5676bb69e33f46a))

* feat(io): add DA30 loader ([`9afd149`](https://github.com/kmnhan/erlabpy/commit/9afd149ddfb21a64564f4d9d2bb1ccb6dcd6d808))

* feat(goldtool): add fit abort ([`693b1e4`](https://github.com/kmnhan/erlabpy/commit/693b1e474ebbc4989dba6ed900f85cad62b5cfcc))

* feat: pass kwargs to broadcast_model ([`82d11f2`](https://github.com/kmnhan/erlabpy/commit/82d11f24b6412f8c774264f5941863eaff7cef6b))

* feat: autolevel colorbar ([`f116155`](https://github.com/kmnhan/erlabpy/commit/f1161551fbf4665b65287d27a2312b3b052c1948))

* feat: set colorbar width ([`b13b5d2`](https://github.com/kmnhan/erlabpy/commit/b13b5d2abd89e98bee6e55d5aeba135a16b3fb50))

* feat(gold): fully integrate spline fitting ([`d5b345c`](https://github.com/kmnhan/erlabpy/commit/d5b345cb5efc6e1e889c272d437e646935ff12e1))

* feat: add lattice module ([`bafe36b`](https://github.com/kmnhan/erlabpy/commit/bafe36b7ab89f004f7ac0ac1d4aa2d1b5137698a))

* feat: estimate k resolution from data ([`457edc7`](https://github.com/kmnhan/erlabpy/commit/457edc72ffa755ffc2e8d3fee039df97bab53ae0))

* feat: add kspace conversion accessor ([`56da9e0`](https://github.com/kmnhan/erlabpy/commit/56da9e09c0dedc9cb066fbd641b5a92eca5b8635))

* feat(itool): save individual plot as hdf5 ([`a5a88b9`](https://github.com/kmnhan/erlabpy/commit/a5a88b9402b5d538386547083b8d7dcc2335e032))

* feat: add ZT image view for 4D ([`bf66dfb`](https://github.com/kmnhan/erlabpy/commit/bf66dfbc948bbf585dde392b23b0e5586b2a1d35))

* feat: add igor procedure to load dataarrays ([`f8b93e0`](https://github.com/kmnhan/erlabpy/commit/f8b93e0e79adc6052e577304dd856a146de0c521))

* feat(itool): enable sync across multiple windows! ([`d3a5056`](https://github.com/kmnhan/erlabpy/commit/d3a50561bd38c9040b9aae3b6c1b82453df1423b))

* feat(gold): automatic crop for corrected gold ([`dad4bb8`](https://github.com/kmnhan/erlabpy/commit/dad4bb886799fd11d5ecf6c8c18c4a118858d15f))

* feat(io): improve BL4 data loading, add basic log generator ([`e283262`](https://github.com/kmnhan/erlabpy/commit/e283262f138bc509d9f4481efd835626060f8b62))

* feat(interp): improve interpolator syntax ([`c7b322b`](https://github.com/kmnhan/erlabpy/commit/c7b322b6bf4106320270b39b4aa423d94ffa94be))

* feat(gold): configurable covariance matrix scaling ([`d051864`](https://github.com/kmnhan/erlabpy/commit/d051864da1dd2e3f5101d33743fa04720fab5411))

* feat(fit): add step function edge ([`3a7bc03`](https://github.com/kmnhan/erlabpy/commit/3a7bc03c666a946813ce79e95002475efb446046))

* feat: handle rad2deg automatically ([`aea4a62`](https://github.com/kmnhan/erlabpy/commit/aea4a62d047cca0be231e0256ad3f519ef54eb0e))

* feat(bz): add option for clip path ([`52daa2a`](https://github.com/kmnhan/erlabpy/commit/52daa2afa4fe98acb1c5ac39dfe6362d348d2e11))

* feat: add fast trilinear interpolation ([`ae77bee`](https://github.com/kmnhan/erlabpy/commit/ae77bee8050b3a80e5ab24bef447911d2e6a4ccd))

* feat(io): add function for pxp debugging ([`1802e09`](https://github.com/kmnhan/erlabpy/commit/1802e099689062bf7e7f531b726ea8ce7ada91e0))

* feat: add W to ph/s conversion ([`a64039d`](https://github.com/kmnhan/erlabpy/commit/a64039d3ca2971b5a434a451fc8e956e40fa2e9b))

* feat: add 2D fermi edge fitting function ([`90c65de`](https://github.com/kmnhan/erlabpy/commit/90c65de986074dcce3ba658a0ec9ce1581da1dd2))

* feat(itool): keep manual limits under transpose ([`bdec7e5`](https://github.com/kmnhan/erlabpy/commit/bdec7e5abe935072f91570af9fc58bb2f19582cb))

* feat: add more colortables from igor ([`03a615f`](https://github.com/kmnhan/erlabpy/commit/03a615f2ffbbf522bc8421757b7def3db037eab4))

* feat: add interactive brillouin zone plot ([`85f844f`](https://github.com/kmnhan/erlabpy/commit/85f844f5f7461489f924f8cf5096416791dec525))

* feat: update style files ([`5278c6c`](https://github.com/kmnhan/erlabpy/commit/5278c6c04213ce4230797bc34028d8befa2923af))

* feat: modernize plot_array ([`c5e7321`](https://github.com/kmnhan/erlabpy/commit/c5e7321dbbc124ae99d55b3959c020f59b1a2b7d))

* feat(io): igor-compatible output ([`2a654e2`](https://github.com/kmnhan/erlabpy/commit/2a654e26fb0e53f79b705bb62db437f5d16e5735))

* feat(io): rewrite igor related backend ([`8739bef`](https://github.com/kmnhan/erlabpy/commit/8739befc3f30661e4d2ffd680888c3b06f3230cb))

* feat: polynomial model now takes positional args ([`a3000d5`](https://github.com/kmnhan/erlabpy/commit/a3000d55eca86fe2c1c1d8f7d468ab4051ceef08))

* feat(goldtool): add smoothing spline fit ([`b25cd7c`](https://github.com/kmnhan/erlabpy/commit/b25cd7c19ced68c028f63cd3fbac533ae7a026b3))

* feat(itool): add performance benchmarks ([`9634af0`](https://github.com/kmnhan/erlabpy/commit/9634af0232afbf7f845b9694a03739b9111d516f))

* feat: default cmap for analysis is now terrain ([`c1cc21c`](https://github.com/kmnhan/erlabpy/commit/c1cc21c2e7660245d1d0bcd1c3e4a5f45b757e15))

* feat: example dataset generator ([`6b6cadf`](https://github.com/kmnhan/erlabpy/commit/6b6cadf19d536dea990daacd687739953a33a2d1))

* feat: add curve fitting tool (alpha) ([`68356ab`](https://github.com/kmnhan/erlabpy/commit/68356ab6d7fc4ecb6653d36213acc8c22ef4f23b))

* feat: add new module for curve fitting ([`20c8be9`](https://github.com/kmnhan/erlabpy/commit/20c8be92116a0ce95f1dd55a2de131a1a9a16efe))

* feat: added boltzmann const. at constants ([`038930e`](https://github.com/kmnhan/erlabpy/commit/038930ebd9a48c000990c006ca794f8fc34b3d94))

* feat: add module for 3D plotting ([`4ec1bd2`](https://github.com/kmnhan/erlabpy/commit/4ec1bd2c81b17c775321d7b8197fe7ca84f1d03b))

* feat(itool): add copy index action ([`6f25677`](https://github.com/kmnhan/erlabpy/commit/6f25677d2ba5d8cf1172a937d3370f67a7220781))

* feat: parallelize multidimensional mean ([`7cfae81`](https://github.com/kmnhan/erlabpy/commit/7cfae816761e3169d1eb52b24d7adbbe8aeef848))

* feat: add 2D colormap ([`f5799d8`](https://github.com/kmnhan/erlabpy/commit/f5799d8d389ddf9c8cb03d7bfd05e9daa5331de0))

* feat(itool): intuitive transpose for 4D data ([`38315f8`](https://github.com/kmnhan/erlabpy/commit/38315f83f98fa92313c8f7857c5b100f1b6f4fed))

* feat(polygon): add convenience function ([`a9e64c7`](https://github.com/kmnhan/erlabpy/commit/a9e64c7213952754c2ad70c2f7b92c73a4a1a45d))

* feat: gradient fill under line plot ([`6a951c1`](https://github.com/kmnhan/erlabpy/commit/6a951c10060d454c8f5335d9081d1764881d1fc2))

* feat: plot 1D slices ([`36dee17`](https://github.com/kmnhan/erlabpy/commit/36dee178b9ffaa37216f74a4bf7810ff3021354c))

* feat(io): add convenience function for BL4 data loading ([`0f3a4ea`](https://github.com/kmnhan/erlabpy/commit/0f3a4eac22e4738fe860117c6474bdab7923b959))

* feat: add igor colormaps ([`4bfcceb`](https://github.com/kmnhan/erlabpy/commit/4bfcceb8cca35df3fcb775743f73b127a7c48239))

* feat: new module for some common constants ([`c6aaede`](https://github.com/kmnhan/erlabpy/commit/c6aaede496725cb59548e26fff45f933f71883da))

* feat: update erplot ([`4bdbc10`](https://github.com/kmnhan/erlabpy/commit/4bdbc10e9556105df9f0250aeb1554ea3b862581))

* feat: add some annotations ([`4a37666`](https://github.com/kmnhan/erlabpy/commit/4a37666dcbd732f8c14c95c2ac1a2c24dee46e3b))

* feat(io): add load functions ([`c654c44`](https://github.com/kmnhan/erlabpy/commit/c654c4444fd78ae64222b25559a881bdca1b6321))

* feat(itool): copy cursor position ([`cc39829`](https://github.com/kmnhan/erlabpy/commit/cc398299fb1d25ac1164a216cf68f68ec2e80555))

* feat: attempt better colorbar... this is probably stupid, fix later ([`2b26a3c`](https://github.com/kmnhan/erlabpy/commit/2b26a3c12e52c11d0468fa1268611b58ecf82b19))

* feat: fast peak fitting with broadened step edge ([`dc30094`](https://github.com/kmnhan/erlabpy/commit/dc300944a2ec0f25e0e4f16540b1edb42af0b154))

* feat: AxisItem scientific labeling
feat: increased degree of freedom for ParameterGroup
fix: wrong colorbar levels ([`42b8860`](https://github.com/kmnhan/erlabpy/commit/42b8860a722dc4e20206b835e7fccb926975c460))

* feat: add diverging colormap normalizations ([`b085946`](https://github.com/kmnhan/erlabpy/commit/b085946ab554c6a4aaef19b4aa13d1071114dc3b))

* feat(itool): add working colorbar ([`c326727`](https://github.com/kmnhan/erlabpy/commit/c326727c126c5081f1d7e60c2ce806699b8fb36f))

* feat(itool): full support for non-uniform coords ([`d3eface`](https://github.com/kmnhan/erlabpy/commit/d3eface36de9c15d788ab479d0f0a6860b36c9c8))

* feat(itool): auto-convert non-uniform dimensions to indices ([`1ad940d`](https://github.com/kmnhan/erlabpy/commit/1ad940d8bbc2b430cc54b7d9228ddc87e5235432))

* feat(bz): add brillouin zone edge calculation ([`2b70d91`](https://github.com/kmnhan/erlabpy/commit/2b70d91ec8c77c27117637c1fec906a7a06c9571))

* feat(colors): add higher contrast PowerNorm ([`16b965c`](https://github.com/kmnhan/erlabpy/commit/16b965c224691103aabcfbb41c73524eb6b2244f))

* feat(itool): add file open dialog ([`ba50354`](https://github.com/kmnhan/erlabpy/commit/ba5035491fa87ab44103d7041a81e15dee7a029e))

* feat(itool): move all cursors on drag with alt modifier ([`6ad638d`](https://github.com/kmnhan/erlabpy/commit/6ad638d9ef07c7b5dac05acc4977a39954abe96a))

* feat(itool): add color limit lock and discrete cursor line ([`0929d61`](https://github.com/kmnhan/erlabpy/commit/0929d61973ef8ae62065820378d362a70cbbe110))

* feat(itool): parse ArrayLike input automatically ([`ce3d802`](https://github.com/kmnhan/erlabpy/commit/ce3d8027bf2ba2956ef80c8c28cbed28ab6d21be))

* feat(itool): add more menus ([`6b423eb`](https://github.com/kmnhan/erlabpy/commit/6b423eb0b56c0082ec0eee7de89761cbc56dfba5))

* feat: add EDC fitting tool (WIP) ([`4fceef5`](https://github.com/kmnhan/erlabpy/commit/4fceef509c63b8f51661b37cb8470f5bcf514b4b))

* feat: add and modify styles ([`079f007`](https://github.com/kmnhan/erlabpy/commit/079f0070da78c1c44dcb67c46b38e84490d67f16))

* feat(io): warn when modifying attrs ([`7692379`](https://github.com/kmnhan/erlabpy/commit/7692379416dda67c47a302d49b1c9a747514ef9c))

* feat(itool): add menubar ([`1614567`](https://github.com/kmnhan/erlabpy/commit/1614567a1287e3a099418190855ff54f8cadad95))

* feat(itool): better cursor colors ([`78dae09`](https://github.com/kmnhan/erlabpy/commit/78dae09619bfb79424eb0af930cc444467fce2c2))

* feat(itool): enable handling multiple cursors ([`e971af7`](https://github.com/kmnhan/erlabpy/commit/e971af767f78d72acee064a0e87ae5764fd13d98))

* feat: pretty plot gold edge fit results ([`aa621c5`](https://github.com/kmnhan/erlabpy/commit/aa621c588b733eae683494422f1330da74e09b23))

* feat: add vertical Fermi energy indicators ([`80e2694`](https://github.com/kmnhan/erlabpy/commit/80e2694ed19dc18375b2dc26531fbda37525ca9e))

* feat: easier colorbar font size specification ([`5638d9c`](https://github.com/kmnhan/erlabpy/commit/5638d9c709c888fcfc24503196955e9d8df863c2))

* feat: add interactive progressbar for joblib ([`3fd24c7`](https://github.com/kmnhan/erlabpy/commit/3fd24c7dd57e2139c6f16af88fe0773323b928b4))

* feat(itool): add axis labels ([`96f9e76`](https://github.com/kmnhan/erlabpy/commit/96f9e76315cbff362d11e16b2b33378c5d7deab0))

* feat(itool): minor adjustments to layout ([`c5dcbe2`](https://github.com/kmnhan/erlabpy/commit/c5dcbe2ec3c143579fe2c7969eda65dcb741e876))

* feat(goldtool): higher order polynomials ([`5f34c35`](https://github.com/kmnhan/erlabpy/commit/5f34c3579975ed48124a5a2cf89980735f8d53a8))

* feat(gold): get full results from Fermi edge fitting ([`ce0d853`](https://github.com/kmnhan/erlabpy/commit/ce0d853adf51df174d01098b703db21e26859882))

* feat: colorbar tick label customization ([`03e82bf`](https://github.com/kmnhan/erlabpy/commit/03e82bf100b3ca62fe781e1c82d54fce0ecab9b7))

* feat(io): improve SSRL loader ([`8fe9a46`](https://github.com/kmnhan/erlabpy/commit/8fe9a467d348225d213909a95acf9d80d93d018b))

* feat(itool): multicursor binning ([`bb4430c`](https://github.com/kmnhan/erlabpy/commit/bb4430ced79bfec8850ef7724c6ce752b4cd707b))

* feat: add autoscale convenience function ([`36f838d`](https://github.com/kmnhan/erlabpy/commit/36f838d47bd5fdf8554e59b327b9bed9df961a0f))

* feat: delegate font handling to different styles ([`aabf87b`](https://github.com/kmnhan/erlabpy/commit/aabf87b05be49c8bfe19afaf570bb769656c5ccb))

* feat: simplify getting foreground color based on image ([`8034bfd`](https://github.com/kmnhan/erlabpy/commit/8034bfd15452f6032b0b2ff64cb14dc92d607c8b))

* feat: add fira font mplstyle ([`9c05702`](https://github.com/kmnhan/erlabpy/commit/9c05702d1ea822b5e138fca1e5f75c8d600397d7))

* feat(itool): bind autorange to keyboard ([`98a236d`](https://github.com/kmnhan/erlabpy/commit/98a236deb88a9d74bc6763f49a75d465a097a264))

* feat(itool): implement rad2deg ([`c074d74`](https://github.com/kmnhan/erlabpy/commit/c074d74a4ba62fda1e81e1dedf039ae939959975))

* feat: include styles ([`5281177`](https://github.com/kmnhan/erlabpy/commit/528117714f3f7b08d253036b7492aba48dfe833b))

* feat: add completely reimplemented imagetool with faster slicing and multicursor support ([`af0655e`](https://github.com/kmnhan/erlabpy/commit/af0655eebb1f27199eae5270c971b206cb3edb6b))

* feat: added interactive gold edge fitting tool ([`c4017b6`](https://github.com/kmnhan/erlabpy/commit/c4017b699b27af76b9aa912f3231d1a19864e5e9))

* feat: allow axes input to slice plotter ([`ab4c639`](https://github.com/kmnhan/erlabpy/commit/ab4c639c0b4d9e1cd34d03950846db2a4afdb138))

* feat: multiple axes input to bz overlay plotter ([`9820e34`](https://github.com/kmnhan/erlabpy/commit/9820e340584299f5bbf95aea4193e6c47c026304))

* feat(fermiline): support multiple axes input ([`9dc59b1`](https://github.com/kmnhan/erlabpy/commit/9dc59b1fbb6605e82ed0dcf8e7f444978c2c05df))

* feat: add interactive widget base classes
refactor: subclass interactive widgets ([`e6787cf`](https://github.com/kmnhan/erlabpy/commit/e6787cf567bfd1a1cc1a131a69700143eb6e17bb))

* feat(colors): add colorbar creation macro ([`811ed52`](https://github.com/kmnhan/erlabpy/commit/811ed526dc99d165f3f0cf77dd577744973cb539))

* feat: remove pyarpes dependency on labeling ([`71081fa`](https://github.com/kmnhan/erlabpy/commit/71081faf8404d256fb5b7f57fdc2304de6674445))

* feat(io): silent loading of .pxp files ([`27d6b30`](https://github.com/kmnhan/erlabpy/commit/27d6b30fbff64237562ceaa9b3436d48b06a6e3b))

* feat: parallel processing helpers wip ([`ac88e26`](https://github.com/kmnhan/erlabpy/commit/ac88e264463f4647a1195bd4eb93d617cd38c972))

* feat: reliable gold edge and resolution fitting ([`653db58`](https://github.com/kmnhan/erlabpy/commit/653db5891f4573e069caaeb6f65edcffaa24b316))

* feat(annotations): better high symmetry marking ([`8e6dfcb`](https://github.com/kmnhan/erlabpy/commit/8e6dfcbc3f26f1d17a5bc1400a60512c51aa8a4d))

* feat(mask): completely reimplement masking
based on CGAL C++ library ([`ccfb9b6`](https://github.com/kmnhan/erlabpy/commit/ccfb9b61b75aa5e1f17e88402fc6824fe14287e8))

* feat(io): data loading functions for igor ([`dd333e1`](https://github.com/kmnhan/erlabpy/commit/dd333e199c5e2718b0bb712b1842e37b6c099ea6))

* feat(correlation): rewrite based on scipy ([`14b1b29`](https://github.com/kmnhan/erlabpy/commit/14b1b29c1752ccb3d93d67de6de160e714606133))

* feat: add callable models for edge correction ([`3a6d8c3`](https://github.com/kmnhan/erlabpy/commit/3a6d8c3b5cc44078cd4b8f5056ccacb21d335428))

* feat: functions for gold edge related analysis ([`8c4941f`](https://github.com/kmnhan/erlabpy/commit/8c4941f6347c09a783b9e34ff17a132e05d16631))

* feat: rudimentary loader for SSRL ([`7e18e13`](https://github.com/kmnhan/erlabpy/commit/7e18e1354f95341440da2f020d992c4a38403261))

* feat(itool): better dock, improved memory usage ([`04f0047`](https://github.com/kmnhan/erlabpy/commit/04f00472ce5b507c1d467620dcfe03bdde1700c0))

* feat(itool): support 4D input
known issue: hiding slices are buggy with 4D ([`33a252f`](https://github.com/kmnhan/erlabpy/commit/33a252f42f88b9a0c182a3d771644b8e937d57be))

* feat: add sizebar ([`21f420f`](https://github.com/kmnhan/erlabpy/commit/21f420f362dbf2bb18d1b16e34051dc11034a08a))

* feat(itool): add gamma slider ([`f5efcb7`](https://github.com/kmnhan/erlabpy/commit/f5efcb71c21696ccb9ed1e9b90b3c9e976bac01e))

* feat(itool): subclass buttons for dark mode
feat(itool): flow layout with customizable spacing
refactor(itool): subclassed some button layouts
refactor(itool): remove pyarpes dependency
feat(itool): lazy load colormaps
feat(itool): replace tabs with docks ([`e215470`](https://github.com/kmnhan/erlabpy/commit/e2154703986a8d35aa9b0b44b35a40c9dde62ec7))

* feat: initial commit of general interactive tool ([`bf00956`](https://github.com/kmnhan/erlabpy/commit/bf009565f26a95ead1cda41bb000409bc435062e))

* feat(itool): add axes visibility controls ([`258ff3f`](https://github.com/kmnhan/erlabpy/commit/258ff3fcecf5743adb04c82606feeff54ed828eb))

* feat(itool): hide and show individual axes ([`6a8c71e`](https://github.com/kmnhan/erlabpy/commit/6a8c71e514c8fd4755c47ff521ed5f242c72056d))

* feat(itool): enable latex labels
fix(itool): better grid size adjustment ([`be0b25d`](https://github.com/kmnhan/erlabpy/commit/be0b25dcad9abf49859c766668e969cc800406b9))

* feat(itool):
add joystick for cursor
add axis stretch sliders
add keyboard shortcuts
changed layout margins ([`3faf410`](https://github.com/kmnhan/erlabpy/commit/3faf410676af1e85ebe27282a80d65c2d55c17a0))

* feat: cursor binning ([`9b3de32`](https://github.com/kmnhan/erlabpy/commit/9b3de32f3b1ee098b646b0ad663cd8a95de02e11))

* feat: new convenience function ([`e482cf6`](https://github.com/kmnhan/erlabpy/commit/e482cf6234e5b865b36ed2574c9566b199806c26))

* feat: add Fermi edge correction from fit result ([`607e6ee`](https://github.com/kmnhan/erlabpy/commit/607e6eedef84b05f83cef7097c7ce0aaedbb2f97))

* feat: add tool for analyzing dispersive features ([`7926238`](https://github.com/kmnhan/erlabpy/commit/792623836b77502f7d4ab065712465df007fb1ed))

* feat: add pyqtgraph-based itool, WIP ([`de462c7`](https://github.com/kmnhan/erlabpy/commit/de462c7ab22d5ba47b686e05eaa0cc6558d2d4be))

* feat(itool): add invert colormap ([`bd616ad`](https://github.com/kmnhan/erlabpy/commit/bd616adf30cdee0f96433d9cb8ca8febedf5207c))

* feat(itool): add color picker ([`cb40537`](https://github.com/kmnhan/erlabpy/commit/cb405377302855b357387a539a37f7f466b83447))

* feat(itool): add binning ([`4d8155d`](https://github.com/kmnhan/erlabpy/commit/4d8155dd85da507ee87b09cb604ed2cba2ea7e42))

* feat: add dark mode ([`a0dfefd`](https://github.com/kmnhan/erlabpy/commit/a0dfefdc65676b631ceb380216cfbdbdeab6f2f8))

* feat(itool): 2d image support ([`98d0af7`](https://github.com/kmnhan/erlabpy/commit/98d0af799729884f5ccec203ffacebbffbe72f91))

* feat: add toggle for cursor snap ([`d9ccfbd`](https://github.com/kmnhan/erlabpy/commit/d9ccfbdd8b36fce57655ffe04109fffda5c86189))

* feat: add energy and resolution slider to ktool ([`94e886a`](https://github.com/kmnhan/erlabpy/commit/94e886ab02d638059311b6c6cb43957c3bbde1a9))

* feat: add qt and mpl-based interactive tools ([`823c509`](https://github.com/kmnhan/erlabpy/commit/823c50972fb5afe92f10c9409ed61a7c626971fb))

* feat: added LabeledCursor ([`c970413`](https://github.com/kmnhan/erlabpy/commit/c970413dafd391c3da2e81efb1009e0452a6ec51))

* feat: added annotation macros ([`dd36878`](https://github.com/kmnhan/erlabpy/commit/dd36878e25acd2b7cff5f34442f93b40307ff452))

* feat: Add characterization module ([`e5c56e8`](https://github.com/kmnhan/erlabpy/commit/e5c56e8903d5bcfa95d0d665186b4c1bb33c81a7))

### Fix

* fix(era.fit.models): undefined name in all ([`308f525`](https://github.com/kmnhan/erlabpy/commit/308f525dd291247b631ac8a878d572ea3bfd2230))

* fix: invalid escape ([`a0a33b6`](https://github.com/kmnhan/erlabpy/commit/a0a33b6695471fb85060c35f17c0cb39d9a6338b))

* fix(interpolate): make output shape consistent with scipy ([`0709493`](https://github.com/kmnhan/erlabpy/commit/07094935339bd29ab1c2fce5bf0a4478121a69c7))

* fix: do not reset offsets on accessor initialization if exists ([`5328483`](https://github.com/kmnhan/erlabpy/commit/5328483587da21de3e11966d4baab4bc0fefce12))

* fix(itool): properly parse colorcet cmaps ([`cb81811`](https://github.com/kmnhan/erlabpy/commit/cb81811f4a1b9ea24657d2aa27a7455825fb4eff))

* fix: add some more guess constraints, fix type ([`82825a7`](https://github.com/kmnhan/erlabpy/commit/82825a79af122602435802d3a3aab349ef2e37b7))

* fix(goldtool): round roi position to 3 decimal places ([`d83fafa`](https://github.com/kmnhan/erlabpy/commit/d83fafa0712726fee935755a0d3f816100f90666))

* fix(goldtool): disable memmapping for parallel fitting ([`0c8b053`](https://github.com/kmnhan/erlabpy/commit/0c8b053e2a741f28aa622a2d9ba7e847a82bd8d0))

* fix: duplicated data_dir ([`4b08754`](https://github.com/kmnhan/erlabpy/commit/4b087543059e730c0085b0a656424329cf99bf08))

* fix: missing import ([`a1241da`](https://github.com/kmnhan/erlabpy/commit/a1241da6cbdffb68cb9405236d0bbb40bcf16815))

* fix(io): fix duplicated loader aliases ([`d832a48`](https://github.com/kmnhan/erlabpy/commit/d832a487e6e63befa24049992eed8a6fbd2a2be7))

* fix: syntax error ([`63214f3`](https://github.com/kmnhan/erlabpy/commit/63214f35fd9009e04d7ed4299bf9327ba18f08e0))

* fix(io.plugins.merlin): files with non-standard names  are properly summarized ([`388dd02`](https://github.com/kmnhan/erlabpy/commit/388dd0266e82557a250f9d20dff3a918b139cf00))

* fix: move positional to keyword only ([`ae300b3`](https://github.com/kmnhan/erlabpy/commit/ae300b3a9158940e07ccd585fc76750e3652593c))

* fix(io): return full path for get_files ([`bbaab33`](https://github.com/kmnhan/erlabpy/commit/bbaab3328c09a385762c0d65d00e24ef50ed9273))

* fix: make fit result accessible ([`a40ae66`](https://github.com/kmnhan/erlabpy/commit/a40ae6616721a320eac51e69b7206ec7bf70f243))

* fix: typo in multipeakfunction ([`cafe6cc`](https://github.com/kmnhan/erlabpy/commit/cafe6cc405650921b0a28cc9831b53b448e620ff))

* fix(ktool): round angle offsets ([`31c4730`](https://github.com/kmnhan/erlabpy/commit/31c4730663b24a344ded9a658580981732445100))

* fix(io): when given path to file, skip regex parsing ([`92019bb`](https://github.com/kmnhan/erlabpy/commit/92019bb56b5b8f818bc311a294150644149899ec))

* fix: more realistic angle data generation ([`498e7f5`](https://github.com/kmnhan/erlabpy/commit/498e7f52ce567efe8029f2d4baec7ccb12d607db))

* fix: return type ([`c089f44`](https://github.com/kmnhan/erlabpy/commit/c089f4404c4e63172e1da1d06d167ea75fef5841))

* fix: move doc comments to above ([`429c868`](https://github.com/kmnhan/erlabpy/commit/429c8681dd90bc3c7c890377eb37f4ad1414c3f7))

* fix(docs): avoid direct import in conf.py ([`db930f0`](https://github.com/kmnhan/erlabpy/commit/db930f03f7d308f677dfa5c1385e3dd6c9bf3d0e))

* fix(ktool): default values and data orientation ([`1dcf1f4`](https://github.com/kmnhan/erlabpy/commit/1dcf1f4ca773c5ea5e30146a8fec927851ec54a5))

* fix: Update ktool.py to keep up with refactoring changes ([`f23afb2`](https://github.com/kmnhan/erlabpy/commit/f23afb25c5a2141a53e7427cb0e4ea1291e008f3))

* fix: fix typo in documentation ([`5c527a4`](https://github.com/kmnhan/erlabpy/commit/5c527a4da449867231cf4ca548cd147cd7657edb))

* fix: properly execute in ipython ([`262b965`](https://github.com/kmnhan/erlabpy/commit/262b96518e38cecacf36141614da44d7bf7e3deb))

* fix: momentum conversion offset ([`7836ce1`](https://github.com/kmnhan/erlabpy/commit/7836ce137d12f8b68725ca333ce2536a9e0ab24c))

* fix: Update pyproject.toml with package-dir ([`c56fe0f`](https://github.com/kmnhan/erlabpy/commit/c56fe0f28b537d7e0601d33e9f20efc3f3c87310))

* fix: Update requirements.txt ([`6b3150a`](https://github.com/kmnhan/erlabpy/commit/6b3150a1f49feb349a8fe5e3aa5d07b53d058a00))

* fix: Update docs/requirements.txt ([`5835db9`](https://github.com/kmnhan/erlabpy/commit/5835db9d53c13bc6da16d2b0f5f2e4695a5cd6fd))

* fix: enable dynamic versioning ([`cc0fc96`](https://github.com/kmnhan/erlabpy/commit/cc0fc96a191abf807a80433025b243fc9d11431c))

* fix: Add packages to setuptools configuration ([`a2bbdcf`](https://github.com/kmnhan/erlabpy/commit/a2bbdcf6950de42c56c0f558d0d13c6521f7887f))

* fix: Add version number to pyproject.toml ([`0caf0c4`](https://github.com/kmnhan/erlabpy/commit/0caf0c40f3dc943f786fe06e68d54b779367edad))

* fix: update version ([`d394f20`](https://github.com/kmnhan/erlabpy/commit/d394f204387a1f3fdb18006298002d03b0a8b0d3))

* fix: Update Sphinx configuration path in .readthedocs.yaml ([`21c1b1f`](https://github.com/kmnhan/erlabpy/commit/21c1b1f02b7e8b356d86074fe5c0f89895afbc90))

* fix(itool): ignore zerodivision ([`97e2bf5`](https://github.com/kmnhan/erlabpy/commit/97e2bf56c6e351dcfe6bd382c5b888ef12b44f4a))

* fix: try to make autoscale_off context more reliable, needs testing ([`3a4726a`](https://github.com/kmnhan/erlabpy/commit/3a4726a90343bee7af916490a76679a027fc6aa1))

* fix: gradient_fill now doesn&#39;t mess with autoscale ([`040db5a`](https://github.com/kmnhan/erlabpy/commit/040db5a6aeaebbdb628a5ff05bddce53f5f825bd))

* fix: proper file handler termination ([`2ca7593`](https://github.com/kmnhan/erlabpy/commit/2ca7593b3036f8c13dda8ec7ae34a18ad8ef572c))

* fix: acf2 stack dimension mismatch resolved ([`8dde2da`](https://github.com/kmnhan/erlabpy/commit/8dde2da853bc756cabc79e0587b93d5f99c92efb))

* fix: validation changed, fixes #11 ([`8479814`](https://github.com/kmnhan/erlabpy/commit/84798146b8596ee4cd75e89282180f4e858176b3))

* fix(interactive): override copy, temporarily fixes #10 ([`4e99863`](https://github.com/kmnhan/erlabpy/commit/4e99863b2ce5b6fe82f0fbe3658cf9b024f69fb0))

* fix: subclass scalarformatter for better compat ([`b69bfd6`](https://github.com/kmnhan/erlabpy/commit/b69bfd6607aa283c1e717e8183982ef1c40d9749))

* fix: nice horizontal colorbar ([`2766e36`](https://github.com/kmnhan/erlabpy/commit/2766e3689ef3ebe5152ec5ce8fe22373127a507b))

* fix(colors): handle callable segmented cmaps ([`8135d73`](https://github.com/kmnhan/erlabpy/commit/8135d73e3d11f7e8126576b7b2281a15ec6e2920))

* fix(itool): keyboard modifier syntax ([`d593258`](https://github.com/kmnhan/erlabpy/commit/d5932583b6b36f288a72a5d4965c81c805c55c66))

* fix(io): fix da30 loading ([`ee0615d`](https://github.com/kmnhan/erlabpy/commit/ee0615dd31fff487139350df692045b646402c03))

* fix: fix typo ([`6a84557`](https://github.com/kmnhan/erlabpy/commit/6a845573f8c0bc0d03c6003a218e6d1322c6a05e))

* fix: load da30 map angle in radians ([`8d08c65`](https://github.com/kmnhan/erlabpy/commit/8d08c6533cbb57b225d5ee9ae99d1795ab5ec300))

* fix: remove duplicate star ([`a018691`](https://github.com/kmnhan/erlabpy/commit/a0186915be3fc7065fbb19af5ff68d491167a468))

* fix(itool): update io ([`8c082b0`](https://github.com/kmnhan/erlabpy/commit/8c082b0f2c814e101e25dd99002e5fb45fccd744))

* fix(itool): handle ambiguous datasets ([`9226c12`](https://github.com/kmnhan/erlabpy/commit/9226c12f93ddceb35ea5b8852989dcf15620f9c6))

* fix(itool): catch overflow ([`ae1afe1`](https://github.com/kmnhan/erlabpy/commit/ae1afe16e67b76789c5adc59f1e0b149010cfc2f))

* fix: patch breaking changes in lmfit 1.2.2 ([`8179fc3`](https://github.com/kmnhan/erlabpy/commit/8179fc308976d3ac8ca8575136bea7382ee85eac))

* fix: make colorbar more robust ([`3a4968d`](https://github.com/kmnhan/erlabpy/commit/3a4968dd5983fc4bcde43bdf3056c3cc467be070))

* fix: resistance data loading ([`7db3c4a`](https://github.com/kmnhan/erlabpy/commit/7db3c4abae28ae2ef99d1e8a54aa2744a3bde025))

* fix: typo ([`a74732d`](https://github.com/kmnhan/erlabpy/commit/a74732d8429beb94a3009cbfde004387e4fae762))

* fix: fix critical typo ([`3e10834`](https://github.com/kmnhan/erlabpy/commit/3e10834a8f9f5f78a9a41f3567f2fb4a9d092245))

* fix: handle undetermined spectrum type ([`b59c3e9`](https://github.com/kmnhan/erlabpy/commit/b59c3e9068de1c26c1efced8ee0ffdc62c256f1c))

* fix(ssrl52): improve compatibility with old data ([`135b022`](https://github.com/kmnhan/erlabpy/commit/135b022996259da7ee379a49a0bb13327182ef51))

* fix: remove now-redundant patches ([`687ece9`](https://github.com/kmnhan/erlabpy/commit/687ece99866628da3fbb302981d005cf57552cfb))

* fix: improve ZT image view for 4D ([`c8d07b5`](https://github.com/kmnhan/erlabpy/commit/c8d07b522216f191a94a7ef40d7beda478e46074))

* fix: keep slicer object for expected signal behavior ([`e0ebe85`](https://github.com/kmnhan/erlabpy/commit/e0ebe8519d972467332965be4ac097df0c26d530))

* fix: keep slicer object for expected signal behavior ([`2c401e6`](https://github.com/kmnhan/erlabpy/commit/2c401e643f08eadbf3f3c84854332ee95d89451d))

* fix: indexerror on loading higher dimensional data ([`e068bec`](https://github.com/kmnhan/erlabpy/commit/e068bec807b8f744c5b5925ef1208130c9136c2b))

* fix: binning control order not updating on transpose ([`bdfda97`](https://github.com/kmnhan/erlabpy/commit/bdfda979475ecc8efeb1e8c80cf45a006f2ff85f))

* fix: labels should not displace subplots ([`43b9258`](https://github.com/kmnhan/erlabpy/commit/43b925846306b311fcd309bc14a89390a51fb99e))

* fix: revert clean_labels ([`1316fce`](https://github.com/kmnhan/erlabpy/commit/1316fce90142b41e423b0f7464768ea8b178599e))

* fix: dimension mismatch ([`68fc306`](https://github.com/kmnhan/erlabpy/commit/68fc3063294bee4c89581b252cb49e71fc40b377))

* fix: circular import ([`186cb03`](https://github.com/kmnhan/erlabpy/commit/186cb03f4a93cba01773b2a5dac848ad16a761cd))

* fix: make qt progressbar more accurate ([`7c83b31`](https://github.com/kmnhan/erlabpy/commit/7c83b3100d7880577c533c3b17b77d6b643759c8))

* fix: revert default pad ([`8e5715c`](https://github.com/kmnhan/erlabpy/commit/8e5715ce5f2c89c236ae7791820dcf5a7411fcc1))

* fix(goldtool): keyerror on code generation ([`e421719`](https://github.com/kmnhan/erlabpy/commit/e4217193441cabca6f8bce3e4001a530dea02678))

* fix(io): make save and load work with datasets ([`a4f1a12`](https://github.com/kmnhan/erlabpy/commit/a4f1a1279fb0070f38119e80b04bdbf19739b50c))

* fix(style): nonzero pad on savefig ([`b21a178`](https://github.com/kmnhan/erlabpy/commit/b21a17829d95d69aee2affe4c590c082b1cb22ca))

* fix: gold fit autoscale ([`b81e4e8`](https://github.com/kmnhan/erlabpy/commit/b81e4e86f72eedbc63e256de14b70298c2c591f4))

* fix(io): can now load BL4 pxt files ([`d318c91`](https://github.com/kmnhan/erlabpy/commit/d318c91bdaa4adc2fb54be647e5932463e84474c))

* fix: gold fit autoscaling ([`9ed8f86`](https://github.com/kmnhan/erlabpy/commit/9ed8f86f68006a648e0ed2868ff65dd451b9f2ff))

* fix: disable covariance matrix scaling ([`a3264c7`](https://github.com/kmnhan/erlabpy/commit/a3264c70bf9d2bb502b83f7265482ff462dba11a))

* fix: try to fix random segfault with numba ([`3505b42`](https://github.com/kmnhan/erlabpy/commit/3505b4291b34697732f2dccc33f81f5c32e5fcad))

* fix: stupid regression in plot_array ([`2ef2ad6`](https://github.com/kmnhan/erlabpy/commit/2ef2ad62227798c0ab3b72ccd9132086fd8a3e45))

* fix: stupid regression on rename ([`cc37d8e`](https://github.com/kmnhan/erlabpy/commit/cc37d8e86e7a0ea36a89b265b6409aa2898c302c))

* fix: wrong attributes ([`01812ba`](https://github.com/kmnhan/erlabpy/commit/01812bad65847dbe33d1afe013c564898c3c99a6))

* fix: invert before gamma ([`c70bff5`](https://github.com/kmnhan/erlabpy/commit/c70bff5b59a17d59260a1289e8c3df007194b762))

* fix(io): fix livexy and livepolar loader dims ([`edcf9a8`](https://github.com/kmnhan/erlabpy/commit/edcf9a8640169bebb427f768baeb82735494ab40))

* fix(itool): restore compatibility for float64 data ([`4e8baab`](https://github.com/kmnhan/erlabpy/commit/4e8baabb0f592294862bcbda41bdf1422748ab62))

* fix: resolve type related problems ([`4ec5bdd`](https://github.com/kmnhan/erlabpy/commit/4ec5bddc2c82076eab3e973845d21f188cf09161))

* fix: fix typo in comment ([`8cd71df`](https://github.com/kmnhan/erlabpy/commit/8cd71df415b5715a64c81dcf338f44abee21fa7d))

* fix: remove type hints, were causing thread errors ([`1ed309c`](https://github.com/kmnhan/erlabpy/commit/1ed309c58ea3c8da696d352e1df0ba03903c223b))

* fix: edge correction with callable ([`8a1052f`](https://github.com/kmnhan/erlabpy/commit/8a1052fe2678baca09042d8eb644d41a9ab48f56))

* fix: default pad changed ([`a2039c2`](https://github.com/kmnhan/erlabpy/commit/a2039c2b9db91950d7d627b82614fad0a536f906))

* fix: fix curve fitting on notebook ([`4126aa0`](https://github.com/kmnhan/erlabpy/commit/4126aa0fa6061c880b03d188199b3d40d64c4b84))

* fix: compatibility with PyQt6 ([`01f550e`](https://github.com/kmnhan/erlabpy/commit/01f550e463ea14720d0e84cd43bf98501cf0b554))

* fix: stupid commit ([`d3b1f53`](https://github.com/kmnhan/erlabpy/commit/d3b1f53448805691c3de358e2cdfca3cb631866a))

* fix(io): add compatibiity check, fixes #9 ([`35c6bd7`](https://github.com/kmnhan/erlabpy/commit/35c6bd73753af06f49130dddc642baca008044e4))

* fix: correct color limits for new cursors ([`25e54f4`](https://github.com/kmnhan/erlabpy/commit/25e54f46bbca24ac54aa2cccf1abaacf9403ec68))

* fix: add PyQt6 compatibility ([`713cea2`](https://github.com/kmnhan/erlabpy/commit/713cea28683643f27c5b163c1aba0e332c168b30))

* fix: pyqt-compatible multiple inheritance, fixes #7 ([`de37753`](https://github.com/kmnhan/erlabpy/commit/de377534db051b852d528ab8dd6abf212bf3e1a0))

* fix: pyqt-compatible multiple inheritance, fixes #7 ([`7fb28d8`](https://github.com/kmnhan/erlabpy/commit/7fb28d834b149c1a7dd8bd09162d09fcf4890004))

* fix: minor fixes ([`2b4e61c`](https://github.com/kmnhan/erlabpy/commit/2b4e61cb086ae5f813aa1d784c722fbbc8613330))

* fix: 2d colormap incorrect normalization ([`a425cb1`](https://github.com/kmnhan/erlabpy/commit/a425cb1bd560aa2e7047321ec2b21ecd0bc0779d))

* fix: set tol to 10x eps float32, fixes #5 ([`d4d2dc7`](https://github.com/kmnhan/erlabpy/commit/d4d2dc72fda8ece6875d50f03e337d1b9dd222de))

* fix: convert everything to float32, fixes #2 ([`8f2b58f`](https://github.com/kmnhan/erlabpy/commit/8f2b58f5218a4bdcb2d6fb8c8d6d8ef798c84a03))

* fix: fixes #3 ([`1ae6ff8`](https://github.com/kmnhan/erlabpy/commit/1ae6ff8cd58db64fc439cd034f2f93f6961bd6c9))

* fix: fixes #1 along with some memory optimization ([`a667c6a`](https://github.com/kmnhan/erlabpy/commit/a667c6a44fda268b7a450b4522d2ada193de0639))

* fix: choose nearest for zero width when plotting slices ([`6ba03da`](https://github.com/kmnhan/erlabpy/commit/6ba03da2c6054044701accc20763bac424ac3bcf))

* fix(itool): multicursor colorbar ([`65bd992`](https://github.com/kmnhan/erlabpy/commit/65bd9923680a11e96350b8f9dab079934199bd3a))

* fix: shift by DataArray ([`fd38eaf`](https://github.com/kmnhan/erlabpy/commit/fd38eaf479b1109f57f3dc30e7b4cc5964459bf7))

* fix: force qt api ([`3613853`](https://github.com/kmnhan/erlabpy/commit/36138538c31ba1baaf6c5606372039aeb5dcfb95))

* fix: temperature not required when fitting with broadened step edge ([`4b83d87`](https://github.com/kmnhan/erlabpy/commit/4b83d8713044e8dfdaf605653cc60f8fe6057ea5))

* fix: fix colorbar conflict with multiple cursors ([`6cb35b6`](https://github.com/kmnhan/erlabpy/commit/6cb35b68b9643dc19d9e197e67554c203e5c4b99))

* fix: wrong sign in powernorm ([`b4df421`](https://github.com/kmnhan/erlabpy/commit/b4df42102822da0e41991b9c176a24b900220088))

* fix: rewrite pyqtgraph colormap normalization ([`e2a807e`](https://github.com/kmnhan/erlabpy/commit/e2a807e7a0c7e556a6d2750cd4052df911b5ec3c))

* fix(itool): fix misc. bugs ([`15c737c`](https://github.com/kmnhan/erlabpy/commit/15c737cd0096a27d5582ab1f084a7ad070b28bcc))

* fix: retain clipboard after window close ([`e800bc7`](https://github.com/kmnhan/erlabpy/commit/e800bc72fae148e0f9fc4df646566013bf7082b8))

* fix(bz): input reciprocal lattice vectors ([`32df4d7`](https://github.com/kmnhan/erlabpy/commit/32df4d7e673fbcfb886c959450d30c9bf1fa1d27))

* fix: automatic figure detection ([`f0f2ef9`](https://github.com/kmnhan/erlabpy/commit/f0f2ef9810ec637a46d316b37578bc45307ea881))

* fix(itool): regression: aspect ratio for 2D arrays ([`087ba24`](https://github.com/kmnhan/erlabpy/commit/087ba24cb99b12901ae928cd2b3288c25570c9eb))

* fix(itool): better handle drag ([`23cdbf0`](https://github.com/kmnhan/erlabpy/commit/23cdbf039118980c50760a7ebf94b24b48a6b6c8))

* fix: replace bitwise inversion on boolean ([`5569060`](https://github.com/kmnhan/erlabpy/commit/5569060a356aad3d1d3141b5accce61f66e78418))

* fix: works properly with integer coordinates ([`adc0074`](https://github.com/kmnhan/erlabpy/commit/adc00746668456576709b1db9525ea5bb6e5bb11))

* fix: update some deprecated syntax ([`839f1a6`](https://github.com/kmnhan/erlabpy/commit/839f1a60132ff03b578224109553491e49a80aae))

* fix: patch for PySide6 6.4 ([`6e9d1b9`](https://github.com/kmnhan/erlabpy/commit/6e9d1b9ed5899cb4f639806281da55d987e01f1d))

* fix: better aspect ratio for 2D arrays ([`b002350`](https://github.com/kmnhan/erlabpy/commit/b002350164c4f9c7cd233f50488937f55dff8e46))

* fix: regression as per pyqtgraph/pyqtgraph@cead5cd ([`74c9f81`](https://github.com/kmnhan/erlabpy/commit/74c9f8181033c9f7cfbe8eaa15b64172dc287601))

* fix: stupid rad2deg handling ([`4a6d629`](https://github.com/kmnhan/erlabpy/commit/4a6d629261e6bc0da286b3c687d8fe8af46ee589))

* fix(goldtool): catch varname exceptions ([`aa22dd7`](https://github.com/kmnhan/erlabpy/commit/aa22dd7833e1fd7e3f1a36fa48ccee7c6420dffe))

* fix(itool): wrong signals ([`a0f0036`](https://github.com/kmnhan/erlabpy/commit/a0f00366be305f302a52e84eac5e4fc97d7d9d54))

* fix: colorbar aspect specification ([`456f370`](https://github.com/kmnhan/erlabpy/commit/456f370907ca163c30567992da8edbfebcf18caf))

* fix: docstring and labeling ([`4bbedda`](https://github.com/kmnhan/erlabpy/commit/4bbedda90b83f7be32494cfd63bc9ac7dfb94327))

* fix: attempts to overwrite read-only object ([`0cf8606`](https://github.com/kmnhan/erlabpy/commit/0cf8606cd69ccdeaae4d84945e181b9e2b07a846))

* fix(plot_array): colorbar extents ([`2fe8a93`](https://github.com/kmnhan/erlabpy/commit/2fe8a930f2cf8fd37d88958231388e4a446b1443))

* fix: broken binning ([`f217938`](https://github.com/kmnhan/erlabpy/commit/f217938379ea7d69613fdef462b26ed41d4850cd))

* fix(itool): isocurve wrong orientation ([`dfd6aac`](https://github.com/kmnhan/erlabpy/commit/dfd6aac29b846e5b187791aa7fe2708dc95abe76))

* fix(itool): tab position ([`a70af2b`](https://github.com/kmnhan/erlabpy/commit/a70af2be59c80231be605e49304d226ed5c24f0c))

* fix: fine-tune automatic colormap assignment ([`1d035bb`](https://github.com/kmnhan/erlabpy/commit/1d035bb6ce178f7c13b4c2cd35b20fb69342a627))

* fix(itool): smarter mouse detection ([`3032c0c`](https://github.com/kmnhan/erlabpy/commit/3032c0cc15c23a3f345300de0eb57080d31b7604))

* fix: noisetool import ([`c99bfd2`](https://github.com/kmnhan/erlabpy/commit/c99bfd2980127f07fc6b361d9156e5a68be2711c))

* fix: invalid import ([`5fd9010`](https://github.com/kmnhan/erlabpy/commit/5fd90104cd8e2a2b8158df6acf88dfa7408bc102))

* fix: flickering cursor when moving ([`439157b`](https://github.com/kmnhan/erlabpy/commit/439157bf3fb3b377ebb17a9f6c7a37b87fafc905))

* fix(itool): fix transpose ([`0b23f61`](https://github.com/kmnhan/erlabpy/commit/0b23f61dfb7c3dec8a534d5b5c7dae560418376f))

* fix(itool): change wrong 2D layout ([`e04f162`](https://github.com/kmnhan/erlabpy/commit/e04f162ac0f88f1c576942e9eb820044a17ea791))

* fix(itool): revert ([`ca4335a`](https://github.com/kmnhan/erlabpy/commit/ca4335a77ba467d46fa0d68e80927aa6ee8e8a24))

* fix(itool): adjust blitting ([`0873882`](https://github.com/kmnhan/erlabpy/commit/0873882a7692da94ec05ee659030d0e26be51585))

* fix(itool): properly functioning pan &amp; zoom ([`26ce4e8`](https://github.com/kmnhan/erlabpy/commit/26ce4e8efce62eb2ab38f14b097f0121ef6205d4))

* fix: fix offset not syncing across energy ([`7ec4072`](https://github.com/kmnhan/erlabpy/commit/7ec4072c93e1b41f24e4247afc414244a43badac))

* fix: fix cursor issue and home resetting limits ([`f3374ab`](https://github.com/kmnhan/erlabpy/commit/f3374abaa8643aef78f33366dfab9a0994b475ce))

* fix: simplify cursor customization ([`a5ff97f`](https://github.com/kmnhan/erlabpy/commit/a5ff97f6cffcd2cc3da0a31f44d62ac590c663e3))

* fix: make plotting imports backwards compatible ([`82f132b`](https://github.com/kmnhan/erlabpy/commit/82f132b89ffd3c911208ae93fa8d47b1825e7928))

* fix(plotting): fix UnboundLocalError ([`e4f5bca`](https://github.com/kmnhan/erlabpy/commit/e4f5bca381c881d159c9f6a09ebdc86616a4f467))

* fix: fix typo ([`39d160a`](https://github.com/kmnhan/erlabpy/commit/39d160a15a6d8990061de48a08b431a2ff35f699))

### Performance

* perf(interpolate): make some jitted functions always inlined ([`4624b16`](https://github.com/kmnhan/erlabpy/commit/4624b16ec926546876a98864abe3c1d47b6fc221))

* perf(itool): fps optimization, add proper support for nonuniform dimensions ([`6df84db`](https://github.com/kmnhan/erlabpy/commit/6df84db7156de88b2ee8d100a2ce0c45f8b2135a))

* perf: cleanup, reduce import time ([`dbfcce3`](https://github.com/kmnhan/erlabpy/commit/dbfcce38f9b874b8532666ff7479dbc789fef657))

* perf(itool): add cached properties ([`0124093`](https://github.com/kmnhan/erlabpy/commit/0124093a25ba47e5bc304125fcfd62f6768beb13))

* perf: limit fps with SignalProxy ([`f7ce099`](https://github.com/kmnhan/erlabpy/commit/f7ce099f87e97adeb27a8e4f2d760b6ddd4f4d22))

* perf: get coords efficiently ([`5582afc`](https://github.com/kmnhan/erlabpy/commit/5582afc9e86e8306e6499da52b2f2c06604b029b))

* perf: better min/max performance, fixes #4 ([`3c4aa13`](https://github.com/kmnhan/erlabpy/commit/3c4aa1369fbbf10fa5a2597dc958bb559e4bb26a))

* perf(slicer): contiguity optimizations ([`6f2b543`](https://github.com/kmnhan/erlabpy/commit/6f2b543a11d89d744961e1f1716ad542a9bab163))

* perf(itool): update only relevant axes ([`c561650`](https://github.com/kmnhan/erlabpy/commit/c5616502767e8f14425ba07bed4660355eba1203))

### Refactor

* refactor: apply linter suggestions ([`edfc91a`](https://github.com/kmnhan/erlabpy/commit/edfc91a6712620588106471ae03975a76976f634))

* refactor: apply linter suggestions ([`231f794`](https://github.com/kmnhan/erlabpy/commit/231f794a3aaf6575528df63b70a4478cb9769fe8))

* refactor: apply some linter suggestions ([`4e1f66c`](https://github.com/kmnhan/erlabpy/commit/4e1f66c6b19eb2e3674511e19309c9127d610369))

* refactor(goldtool): ui changes ([`464d05e`](https://github.com/kmnhan/erlabpy/commit/464d05ee270cf601c322536b3dafd3c7bf0e9f7f))

* refactor: rename variable ([`32a901e`](https://github.com/kmnhan/erlabpy/commit/32a901e0de1b7d0c5fb6858a275b8bf3cb69e801))

* refactor: cleanup namespace ([`4779e46`](https://github.com/kmnhan/erlabpy/commit/4779e46920cf88a49f211d1d7f856f61e6c84c2a))

* refactor(io): minor changes to summary format ([`d26a8f7`](https://github.com/kmnhan/erlabpy/commit/d26a8f78f4d86f7b0654a0949e644d2f96652b50))

* refactor: move functions  to submodule ([`824a2fb`](https://github.com/kmnhan/erlabpy/commit/824a2fb4847d5f29dc51f50517a51aae3709d3df))

* refactor: fit functions submodule ([`2bc555c`](https://github.com/kmnhan/erlabpy/commit/2bc555cfb050d28523f805b75053dcf836759a7f))

* refactor(io.dataloader): fix _repr_html_ to return valid html table ([`73adb0f`](https://github.com/kmnhan/erlabpy/commit/73adb0ffffc618d28903a4c5814923dac5502186))

* refactor(io.dataloader): make reverse_mapping a staticmethod ([`983c02b`](https://github.com/kmnhan/erlabpy/commit/983c02bda02c97a7eb179c35f98d0a89c0814cd9))

* refactor(io): change dict format ([`a13b064`](https://github.com/kmnhan/erlabpy/commit/a13b06465f53c78b9edea538e9825d83f66b86f0))

* refactor: add type annotation ([`7e08658`](https://github.com/kmnhan/erlabpy/commit/7e08658093fe24a383366f722f044029f097cb69))

* refactor: use match-case for enum matching ([`cc9e112`](https://github.com/kmnhan/erlabpy/commit/cc9e1126f7571df3e68e568375773a3a4dce63b5))

* refactor: change package directory structure; BREAKING CHANGE ([`5385ec7`](https://github.com/kmnhan/erlabpy/commit/5385ec70b23775ddd19b02459cbb0d0630143454))

* refactor: format with black ([`1655eec`](https://github.com/kmnhan/erlabpy/commit/1655eec321bd12acc37681e1d55e21155ec34252))

* refactor: remove code trying to infer spectrum from dataset

From now on all data should be strictly a xr.DataArray ([`3fa6b1b`](https://github.com/kmnhan/erlabpy/commit/3fa6b1b25dc70add644e9719488ae55df314238c))

* refactor: deprecate old ktool, replace ([`dbd972f`](https://github.com/kmnhan/erlabpy/commit/dbd972f0e56197c1796695921b6cc021b3f4d190))

* refactor(gold): add type annotation ([`06c39a7`](https://github.com/kmnhan/erlabpy/commit/06c39a790109e23a81e88c885a1fa0ee1f615e41))

* refactor: Update requirements.txt so that igor2 is not editable ([`0e20bc4`](https://github.com/kmnhan/erlabpy/commit/0e20bc48d458a363b4c2e2b829a59ac6e1aba6cf))

* refactor: temporarily disable annotate_cuts_erlab ([`08e4b52`](https://github.com/kmnhan/erlabpy/commit/08e4b527316b94b0a1d2912b67d1b67ea4571755))

* refactor(itool): modify test code ([`8934fba`](https://github.com/kmnhan/erlabpy/commit/8934fba1b20cf182455d2a70e24ff835d6ca96be))

* refactor(exampledata): tweak defaults ([`73eb495`](https://github.com/kmnhan/erlabpy/commit/73eb4955ff3ed7f568136efce5efaf9ab929dfba))

* refactor: cleanup ([`88b418f`](https://github.com/kmnhan/erlabpy/commit/88b418f61491917e275ec7b5cc544c157617429d))

* refactor: try garbage collection, failed ([`ac75267`](https://github.com/kmnhan/erlabpy/commit/ac75267921884299970c5d78b1633835700dce6e))

* refactor: typo ([`f1df9ea`](https://github.com/kmnhan/erlabpy/commit/f1df9ea0a4e128013d21e2f2b5e9a64f9acf57a3))

* refactor: reorder functions ([`ee6886f`](https://github.com/kmnhan/erlabpy/commit/ee6886fa7810c3bac26e05b1407ceebc1563ad54))

* refactor: organize imports ([`70b2b9b`](https://github.com/kmnhan/erlabpy/commit/70b2b9bed7e395a824ee3e1b240937b570e670ec))

* refactor: cleanup ([`8b01e73`](https://github.com/kmnhan/erlabpy/commit/8b01e7393b7d7a6ae23b85e6f784a7f760cdd74a))

* refactor: relocate color related classes ([`b322fb8`](https://github.com/kmnhan/erlabpy/commit/b322fb8ced0b41a1ec8e7dbaf9ac50c8f7aa853c))

* refactor(io): cleanup imports ([`125f672`](https://github.com/kmnhan/erlabpy/commit/125f672edd3b6f94943c2a3d221e626f9a58c820))

* refactor: add submodules to analysis initialization ([`30ab7d0`](https://github.com/kmnhan/erlabpy/commit/30ab7d0f71a6c09633e54345c7317215934c1ca6))

* refactor: rename igor procedure file ([`92d495c`](https://github.com/kmnhan/erlabpy/commit/92d495cb114a921a8ca9d1fde81a4e6400d53089))

* refactor: cleanup ([`025b39b`](https://github.com/kmnhan/erlabpy/commit/025b39bb557ab364e6c2f5ff14ddb2956ba52467))

* refactor: move colormap controls up ([`13a44fe`](https://github.com/kmnhan/erlabpy/commit/13a44febb156fe657ed39636323c124650010501))

* refactor: follow pep8 dunder names ([`f954b7b`](https://github.com/kmnhan/erlabpy/commit/f954b7ba76b877a7376b45a9a91871e9b8ea4d25))

* refactor: replace deprecated syntax ([`6d48a08`](https://github.com/kmnhan/erlabpy/commit/6d48a088f632c0a6d467067463e706c410ad8978))

* refactor: cleanup ([`5b24106`](https://github.com/kmnhan/erlabpy/commit/5b241068cbd60de73da420149fbd4ba2f09a1ac9))

* refactor: cleanup and add some type annotation ([`daabfb8`](https://github.com/kmnhan/erlabpy/commit/daabfb87b07fa71a618b127c2cd6a51bbc0e3e18))

* refactor: move DictMenuBar to utilities ([`2b3fcb1`](https://github.com/kmnhan/erlabpy/commit/2b3fcb1776115cfee987db4e74c4d34a5d679bcf))

* refactor: change gold fitting syntax ([`e6043d6`](https://github.com/kmnhan/erlabpy/commit/e6043d6ae3aa0bc3b90db2d0378c9bd3c7a7c948))

* refactor: update clean_labels with matplotlib API ([`13013dd`](https://github.com/kmnhan/erlabpy/commit/13013dd7ebb16e03b1609e4e44315b22139e8f28))

* refactor: move some functions ([`866eca5`](https://github.com/kmnhan/erlabpy/commit/866eca5c5a5b0ccf38383aead2b5871903c5c50a))

* refactor: follow pep8 dunder names ([`b043701`](https://github.com/kmnhan/erlabpy/commit/b043701e428c7e1c461f64f57667d8b25642a5ce))

* refactor: imagetool is now a package ([`1a39b82`](https://github.com/kmnhan/erlabpy/commit/1a39b8283582cccacd448addb3566f6f1e882744))

* refactor: follow pep8 dunder names ([`dbdc0f6`](https://github.com/kmnhan/erlabpy/commit/dbdc0f62cb048ce5c2f248a21e112476d5ebb3df))

* refactor: cleanup ([`aace486`](https://github.com/kmnhan/erlabpy/commit/aace486d8652e07ab9007f89a49a29cca3d4e1af))

* refactor: cleanup annotations ([`12e590c`](https://github.com/kmnhan/erlabpy/commit/12e590c2a771d089e29aff02492851448640b605))

* refactor: transition to new polynomial api ([`5461c34`](https://github.com/kmnhan/erlabpy/commit/5461c342f7a84d1ba98f924c74be6ecde16cf6e4))

* refactor: cleanup syntax ([`310c2ba`](https://github.com/kmnhan/erlabpy/commit/310c2babf15d7a22900806cac308643bb22f8bce))

* refactor(itool): changes to layout ([`7a684e4`](https://github.com/kmnhan/erlabpy/commit/7a684e400772036dde7af4f8a4747c8db208eb0f))

* refactor: remove dependency on darkdetect ([`81fa963`](https://github.com/kmnhan/erlabpy/commit/81fa963a91fddd1eb2b7f316fcec04eb27716d13))

* refactor: format code ([`adf6cb3`](https://github.com/kmnhan/erlabpy/commit/adf6cb3cd48e494d708c79906347e3c54fcd3e20))

* refactor: remove io module, add as package ([`5d1280a`](https://github.com/kmnhan/erlabpy/commit/5d1280a3c7893bb86a42fdf2a71b6825c5db6986))

* refactor: try to merge commit error ([`2c61d45`](https://github.com/kmnhan/erlabpy/commit/2c61d4575bfcafe22b63527a5dc3f318147cb8c4))

* refactor: menubar cleanup ([`f147f11`](https://github.com/kmnhan/erlabpy/commit/f147f114364bd9fab0ba456a24113d58dc549652))

* refactor: organize code ([`93ebd5b`](https://github.com/kmnhan/erlabpy/commit/93ebd5b59307b8524452a3c155a5851988070923))

* refactor: rename constants ([`ebaeaa1`](https://github.com/kmnhan/erlabpy/commit/ebaeaa11a0c718266512261f640441a06e2703e1))

* refactor: cleanup ([`f29bdf3`](https://github.com/kmnhan/erlabpy/commit/f29bdf31e035fc8891f95b8479bf2c05f39ffc03))

* refactor: syntax cleanup ([`2678967`](https://github.com/kmnhan/erlabpy/commit/26789677f4fb489af39edcab5b0f73714c1e52b4))

* refactor: interactive is no longer a submodule of plotting ([`3da646c`](https://github.com/kmnhan/erlabpy/commit/3da646cf4f13c717cf99ec9cc2e3e449e4d4bfdb))

* refactor: remove relative imports ([`96a2095`](https://github.com/kmnhan/erlabpy/commit/96a209572843c35dff50dc383c00bf83c48eb263))

* refactor: lint with flake8 ([`ca34ea1`](https://github.com/kmnhan/erlabpy/commit/ca34ea1b82fccf4ed73b5fdf4f8dfedca303cfa2))

* refactor: remove sandbox notebook ([`504cbc9`](https://github.com/kmnhan/erlabpy/commit/504cbc9f58eb43fb142541c1a5c6b97b4ced3c64))

* refactor: minor changes ([`692fbc0`](https://github.com/kmnhan/erlabpy/commit/692fbc0d95657c2fcd0646e5a8a65aa0161d2aca))

* refactor: expose property label ([`9e2dafd`](https://github.com/kmnhan/erlabpy/commit/9e2dafd4948437b3e3b308a7a53a2a331b551941))

* refactor: syntax cleanup ([`37166af`](https://github.com/kmnhan/erlabpy/commit/37166af6b1cbb7bdebccdd8a90e4953685859ab0))

* refactor: syntax cleanup ([`19884df`](https://github.com/kmnhan/erlabpy/commit/19884df9909684f2352e6583cad0abb523e958de))

* refactor: syntax cleanup ([`59e2921`](https://github.com/kmnhan/erlabpy/commit/59e2921c5c7d897f3cfdb17417d46127bb95af60))

* refactor: change clipboard handler ([`d687499`](https://github.com/kmnhan/erlabpy/commit/d687499b43a027ea97a05243d57c88ae6f756398))

* refactor: some changes regarding colorbar, may be reverted ([`bc913c0`](https://github.com/kmnhan/erlabpy/commit/bc913c07dcd9bbe36f6a28112f32d5d8ff077152))

* refactor: deprecate old imagetool ([`26e20ec`](https://github.com/kmnhan/erlabpy/commit/26e20ec474776274b07d4221976c5d5cee3d2b8b))

* refactor: cleanup enums ([`ec42673`](https://github.com/kmnhan/erlabpy/commit/ec4267373c9f3caf36c2ef4a5f7870e4fa205bdf))

* refactor: cleanup ([`9eef9f9`](https://github.com/kmnhan/erlabpy/commit/9eef9f94d54badb74578fac96dd8854fcfc1f393))

* refactor: cleanup axis labels and signals ([`3aec658`](https://github.com/kmnhan/erlabpy/commit/3aec658a2c93a3244f8053b28f27b8446fbef61e))

* refactor: add typing ([`4951388`](https://github.com/kmnhan/erlabpy/commit/4951388b93ec37fffaf04e44ad02ee67b6f92d6d))

* refactor: change class name ([`a7d51e5`](https://github.com/kmnhan/erlabpy/commit/a7d51e5c7b5eead59f3a7a0ecd667adae6d90c3a))

* refactor: minor improvement and cleanup ([`b2886e4`](https://github.com/kmnhan/erlabpy/commit/b2886e447737c954d01eac2a016ef68ecbfc97b6))

* refactor: move spinbox to utilities ([`906a0f7`](https://github.com/kmnhan/erlabpy/commit/906a0f7ed919f7f800007633f27d5be0f1f33a2a))

* refactor: remove dependency on pyqtgraph dock ([`35b5c3f`](https://github.com/kmnhan/erlabpy/commit/35b5c3f15e7ad2524c239e270dbbecfde5287f78))

* refactor: preparation and cleanup for leaving pyqtgraph dock ([`5721243`](https://github.com/kmnhan/erlabpy/commit/572124321b63c805a94aefb371653918cc174b9a))

* refactor: cleanup ([`1ac0de1`](https://github.com/kmnhan/erlabpy/commit/1ac0de10ddd215d208eb4ca7038e35d35abd826a))

* refactor: remove font refresh macro
Just delete cache... ([`3780f23`](https://github.com/kmnhan/erlabpy/commit/3780f23b4b29b271bc4a640e5c187e176f3813b8))

* refactor: restructure plotting module ([`7259ab3`](https://github.com/kmnhan/erlabpy/commit/7259ab3d34ad289225f2972dacdfd6aa8a827007))

* refactor: split interactive colors module
the whole package will need a complete restructuring. ([`a06439c`](https://github.com/kmnhan/erlabpy/commit/a06439cfdb044d379df25fc6643250662dcc81c4))

* refactor: change to updated colorbar ([`cdd7a8e`](https://github.com/kmnhan/erlabpy/commit/cdd7a8ed04b1c16a3dfd293668604d00e9d3926f))

* refactor: prevent namespace collision ([`f3599b2`](https://github.com/kmnhan/erlabpy/commit/f3599b2d465f4c56db38046f8e9127fe519c78dd))

* refactor: clean up comments and format with black ([`90d67b9`](https://github.com/kmnhan/erlabpy/commit/90d67b96448c1786a1fc7dabd5215ab7b6d0cfcb))

* refactor: format with black ([`7f79731`](https://github.com/kmnhan/erlabpy/commit/7f7973120c674d1d34f574aa2fedc8ac240bc402))

* refactor: remove deprecated ([`a655778`](https://github.com/kmnhan/erlabpy/commit/a655778161208a9c2f605a27343afe82792c4218))

* refactor: format with black ([`88548cb`](https://github.com/kmnhan/erlabpy/commit/88548cbeda20da21ffd02c284a4086e0e50d3bbe))

* refactor: format with black ([`306e530`](https://github.com/kmnhan/erlabpy/commit/306e5304c34b3e7c648fe2ea030259fa7a892a42))

* refactor: reduce pyarpes dependency ([`14e85fb`](https://github.com/kmnhan/erlabpy/commit/14e85fbcfa7d95df674381f8882db80f39e7bda3))

* refactor: format with black ([`7696b66`](https://github.com/kmnhan/erlabpy/commit/7696b66c6ef93a04c353e8e199c096c2458794ff))

* refactor: format with black ([`f6fc29c`](https://github.com/kmnhan/erlabpy/commit/f6fc29ce22a541d9143397ffb902c34d752cdad3))

* refactor(itool): cleanup hide buttons ([`2649fe9`](https://github.com/kmnhan/erlabpy/commit/2649fe9ae12893e719a456f0cf22b7eee097df0f))

* refactor: format code ([`c69b9cc`](https://github.com/kmnhan/erlabpy/commit/c69b9ccca3ee911b5b07bd6393a43444ba771f33))

* refactor: trivial changes ([`6d5df90`](https://github.com/kmnhan/erlabpy/commit/6d5df9019932c4c6ad5930d952714e0bd181cacf))

* refactor: easy import ([`b308d3d`](https://github.com/kmnhan/erlabpy/commit/b308d3d546ee92b04299d679042564149759206b))

* refactor: cleanup and reorganize ([`136715b`](https://github.com/kmnhan/erlabpy/commit/136715b46da525bf7b8e28d2e26c306754d93e61))

* refactor: update with new imagetool ([`6ec6f65`](https://github.com/kmnhan/erlabpy/commit/6ec6f65ca653ef1d90807c2f8448d12ae57d69c2))

* refactor: deprecate mpl-based imagetool ([`5a28947`](https://github.com/kmnhan/erlabpy/commit/5a289471c08022bd4ac4502b176af73d4157c31c))

* refactor: sort imports ([`60f52f5`](https://github.com/kmnhan/erlabpy/commit/60f52f50feb68b48c1e3eaf965becf565129b8c0))

* refactor: cleanup code ([`58ca4ed`](https://github.com/kmnhan/erlabpy/commit/58ca4edeab44c0e665e0b272c1a42a0376d2223a))

* refactor: reorganize plotting routines ([`f5ef54f`](https://github.com/kmnhan/erlabpy/commit/f5ef54f22c7a2b51c794fe00816fa8778da53060))

### Style

* style: format with black ([`82b2938`](https://github.com/kmnhan/erlabpy/commit/82b29382c3c01f7aea4c0ec511c3828de1177e10))

* style: remove relative imports ([`096a78d`](https://github.com/kmnhan/erlabpy/commit/096a78df4e09643961f358cf13d200d2428afc6f))

* style: remove relative imports ([`9891273`](https://github.com/kmnhan/erlabpy/commit/9891273e568c6673541866d3475990aed965fd5e))

* style: add typing ([`1f16ebb`](https://github.com/kmnhan/erlabpy/commit/1f16ebb398dc29a4d0bddf77a422af926b7952da))

* style: remove unused imports ([`68ef006`](https://github.com/kmnhan/erlabpy/commit/68ef00632f75978e380c15bd6ad9040ef6565520))

### Test

* test: tests initial commit

Add basic tests for fast binning, momentum conversion, and interpolation. ([`fde2283`](https://github.com/kmnhan/erlabpy/commit/fde2283c646ce3b5335c6a8a2960da19380e7827))

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
