### TL_Keras_V2
This project aims to use "Transfer Learning" with Keras Models to accurately classify different types of Interstitial Lung Disease (ILD). By providing CT scan patches in RGB format, the Convolutional Neural Networks can learn to identify each class of ILD, as well as those without any disease. Transfer Learning allows the program to pull intermediate outputs from the network and compare classes using simpler features.

### Setup
Install [miniconda](http://conda.pydata.org/miniconda.html).

Create a conda environment:

    conda create -n Keras python=3.5 numpy scipy yaml h5py scikit-learn pillow
    source activate Keras
    pip install Tensorflow-gpu
    pip install Keras
    pip install cython
    


*Note some of the required conda packages might be missing from this list!*

Unzip images:
    unzip binary_images.zip
    unzip multiclass_images.zip
  


### Usage
Activate the conda environment:

    source activate Keras

See talisman-test-suite.py for a usage example. Also most functions have up to date docstrings on the master branch.

### SVC
Dictionary of images are saved in svc file located in tl_keras_v2/research. 


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
The research directory comes pre-loaded with 2 .csv groups files:

    binary_patient_groups.csv
    multiclass_patient_groups.csv
    
Correctly assign the groups_file variable in talisman-test-suite.py based on desired classification.
