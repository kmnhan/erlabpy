# ERLab Python Macros

Python macros for ERLab.
## Installation
1. Install conda.
   - Apple silicon macs
     1. Download [Miniforge](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh) 
     2. In your terminal window, run
        ```bash
        chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh
        sh ~/Downloads/Miniforge3-MacOSX-arm64.sh
        source ~/miniforge3/bin/activate
        ```
   - PC | Linux | Intel macs: [Install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

2. Create and activate a conda environment. Replace `envname` with the environment name.
   ```bash
   conda env create -n envname python=3.9
   conda activate envname
   ```

3. **(Optional, for Apple silicon macs)** Install numpy with BLAS interface specified as vecLib. Not necessary but greatly speeds up linear algebra computations.
   ```bash
   conda install cython pybind11
   pip install --no-binary :all: --no-use-pep517 numpy<1.22
   ```

4. Install PyARPES.

5. Install PyImagetool.
   ```bash
   git clone https://github.com/kgord831/PyImageTool.git
   cd path/to/PyImageTool
   pip install .
   ```




