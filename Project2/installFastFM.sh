# Install cblas and python-dev header (Linux only).
# - cblas can be installed with libatlas-base-dev or libopenblas-dev (Ubuntu)
sudo apt-get install python-dev libopenblas-dev

# Clone the repo including submodules (or clone + `git submodule update --init --recursive`)
git clone --recursive https://github.com/ibayer/fastFM.git

# Enter the root directory
cd fastFM

# Install Python dependencies (Cython>=0.22, numpy, pandas, scipy, scikit-learn)
pip install -r ./requirements.txt

# Compile the C extension.
make                      # build with default python version (python)
PYTHON=python3 make       # build with custom python version (python3)

# Install fastFM
pip install .