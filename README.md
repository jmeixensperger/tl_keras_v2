### TL_Keras_V2
This project aims to classify images with/without various 

### Setup
Install [miniconda](http://conda.pydata.org/miniconda.html).

Create a conda environment:

    conda create -n Keras python=3.5 numpy scipy yaml h5py scikit-learn pillow
    source activate Keras 
    pip install tensorflow
    pip install keras

*Note some of the required conda packages might be missing from this list!*

### Usage
Activate the conda environment:

    source activate Keras

See ssu_talisman.py for a usage example. Also most functions have up to date docstrings on the master branch.

### Preparing Images
Put the images in folders named with the image class label. 
Each class needs at least 25 images.
Then put these folders in a parent directory so the directory structure looks something like:

    images
    ├── healthy
    ├── emphysema
    ├── fibrosis
    ├── ground_glass
    └── micronodules

### Research Directory
