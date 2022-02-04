# ERLab Python Macros
Python macros for ERLab.

## Requirements
Installation process requires `git` and `conda`. 

### Installing Git
- macOS: get Xcode Command Line Tools by running in your terminal window:
   ```bash
   xcode-select --install
   ```
- Windows 10 1709 (build 16299) or later: type this command in command prompt or Powershell.
   ```cmd
   winget install --id Git.Git -e --source winget
   ```
- Otherwise: [Install git](https://git-scm.com/downloads)

### Installing Conda
 - Apple silicon macs: Download [Miniforge](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh) and run in your terminal window:
   ```bash
   chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh
   sh ~/Downloads/Miniforge3-MacOSX-arm64.sh
   source ~/miniforge3/bin/activate
   ```
 - Otherwise: [Install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

## Installation
1. Create and activate a conda environment. Replace `envname` with the environment name.
   ```bash
   conda create -n envname python=3.9
   conda activate envname
   ```

2. Dependencies for **Apple silicon macs** only.
   
   1. Install [homebrew](https://brew.sh).

   2. Install pyqt5 with brew.
      ```bash
      brew install pyqt@5
      ```
   
   3. Optional: Install numpy with BLAS interface specified as vecLib. Not necessary but optimizes linear algebra computations on Apple ARM.
      ```bash
      conda install cython pybind11
      pip install --no-binary :all: --no-use-pep517 numpy<1.22
      ```
      
3. Install dependencies.
   ```bash
   conda install numba bottleneck scipy astropy joblib xarray h5py netCDF4 pint pandas scikit-learn matplotlib bokeh ipywidgets packaging colorama imageio titlecase tqdm rx dill ase pyqtgraph
   ```

4. Change the current working directory to the location where you want the cloned directories to be. 
   
5. Install erlabpy.
   ```bash
   git clone https://github.com/kmnhan/erlabpy.git
   cd erlabpy
   pip install -e .
   cd ../
   ```

6. Install igor dependency of PyARPES.
   ```bash
   pip install https://github.com/chstan/igorpy/tarball/712a4c4#egg=igor
   ```

7. Install modified PyARPES and return to previous directory. On Apple silicon macs, this may take long(~20 min) due to PyQt5 requiring to be rebuilt.
   ```bash
   git clone https://github.com/kmnhan/arpes.git
   cd arpes
   pip install -e .
   cd ../
   ```

8. Install modified PyImagetool.
   ```bash
   git clone https://github.com/kmnhan/PyImageTool.git
   cd PyImageTool
   pip install -e .
   ```