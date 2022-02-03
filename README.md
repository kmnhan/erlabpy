# ERLab Python Macros

Python macros for ERLab.
## Installation
1. Install conda.
   - Apple silicon macs
      
      Download [Miniforge](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh) and run in your terminal window:
      ```bash
      chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh
      sh ~/Downloads/Miniforge3-MacOSX-arm64.sh
      source ~/miniforge3/bin/activate
      ```
   - PC | Linux | Intel macs
  
     [Get conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

2. Create and activate a conda environment. Replace `envname` with the environment name.
   ```bash
   conda env create -n envname python=3.9
   conda activate envname
   ```

3. **(Optional, for Apple silicon macs only)** Install numpy with BLAS interface specified as vecLib. Not necessary but optimizes linear algebra computations on Apple ARM.
   ```bash
   conda install cython pybind11
   pip install --no-binary :all: --no-use-pep517 numpy<1.22
   ```

4. Change the current working directory to the location where you want the cloned directories to be. 

5. Install PyARPES.
   ```bash
   git clone https://github.com/kmnhan/arpes.git
   cd arpes
   pip install -e .
   ```

6. Install igor dependency of PyARPES.
   ```bash
   pip install https://github.com/chstan/igorpy/tarball/712a4c4#egg=igor
   ```

7. Install PyImagetool.
   ```bash
   git clone https://github.com/kgord831/PyImageTool.git
   cd PyImageTool
   pip install -e .
   ```