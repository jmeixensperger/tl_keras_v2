"""
This file handles the setup and pre-processing done before testing/cross-validation.
Chunks of commented code below represent the options for different Keras Models (https://keras.io/applications/). Choose a Keras Model and comment/uncomment all necessary lines in this file.
If you have already generated bottlenecks for a given Model/Output-Layer combination, make sure to specify the bottleneck file below. It will reduce the run-time significantly.
Make sure to double check the groups file as well to ensure that it isn't created every run.
"""

from keras import backend as K
from keras import optimizers
from retrain import create_bottlenecks, cross_validate, \
        load_base_model, train_and_evaluate, group_dict
import report

from keras.models import Model, Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Input

import numpy as np
import os.path
import shutil
import sys

# Assign our program a single GPU to avoid taking all workstation resources
# Only allows program to use GPU_0 on the workstation
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def create_groups(data_dir, groups_file):
    """Save csv (file_name,patient) of patient grouping."""

    print("\nCreating patient groups...")
    if not os.path.exists(groups_file):
        # load images
        from keras.preprocessing.image import ImageDataGenerator
        datagen = ImageDataGenerator()
        images = datagen.flow_from_directory(
            data_dir,
            batch_size=1,
            class_mode='categorical',
            shuffle=False)

        file_names = images.filenames

        group = []    
        for name in file_names:
            group.append(name.split('patient')[1][0:-4])

        groups = np.hstack((np.array(file_names).reshape((-1,1)), np.array(group).reshape((-1,1))))
        np.savetxt(groups_file, groups, delimiter=',', fmt='%s')
        print("Done.")
    else:
        print("Patient groups already exist.")


# load base model with the model's name
input_shape = None
#base_model = load_base_model('InceptionV3', input_shape)
#base_model = load_base_model('ResNet50', input_shape)
#base_model = load_base_model('VGG16')
#base_model = load_base_model('VGG19', input_shape)
#base_model = load_base_model('Xception', input_shape)
base_model = load_base_model('InceptionResNetV2', input_shape)

# create a Model and specify which layer of your network will be the output
#base_model = Model(inputs=base_model.input, outputs=base_model.get_layer('mixed10').output, name='inception_v3')
#base_model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output, name='resnet50')
#base_model = Model(input=base_model.input, outputs=base_model.get_layer('block5_pool').output, name='vgg16')
#base_model = Model(input=base_model.input, outputs=base_model.get_layer('block5_pool').output, name='vgg19')
#base_model = Model(input=base_model.input, outputs=base_model.get_layer('block14_sepconv2_act').output, name='xception')
base_model = Model(input=base_model.input, outputs=base_model.get_layer('conv_7b_ac').output, name='inceptionresnetv2')
print(base_model.output.name, "layer will be used for creating bottlenecks.")  

# Perform average pooling on new Model
x = base_model.output
x = GlobalAveragePooling2D()(x)
#base_model = Model(inputs=base_model.input, outputs=x, name='inception_v3')
#base_model = Model(inputs=base_model.input, outputs=x, name='resnet50')
#base_model = Model(inputs=base_model.input, outputs=x, name='vgg16')
#base_model = Model(inputs=base_model.input, outputs=x, name='vgg19')
#base_model = Model(inputs=base_model.input, outputs=x, name='xception')
base_model = Model(inputs=base_model.input, outputs=x, name='inception_resnet_v2')
#base_model.summary()


# setup missing file paths
data_dir = '../binary2'
tmp_dir = './research/tmp/'
log_dir = tmp_dir + 'logs/'
groups_file = './research/patient-groups.csv' # csv -> file_name,group
if os.path.exists(tmp_dir + 'results'):
	shutil.rmtree(tmp_dir + 'results')
os.makedirs(tmp_dir + 'results')
print(tmp_dir + 'results/')

# create group file csv
create_groups(data_dir, groups_file)
print()
report.data_summary(data_dir, groups_file, csv=tmp_dir+'data_summary.csv')

# create bottleneck file if not present
#bottleneck_file = './research/tmp/inception_v3-mixed10.h5'
#bottleneck_file = './research/tmp/resnet50-converted.h5'
#bottleneck_file = './research/tmp/vgg16-converted.h5'
#bottleneck_file = './research/tmp/vgg19.h5'
#bottleneck_file = './research/tmp/xception.h5'
bottleneck_file = './research/tmp/inception_resnet_v2.h5'
groups_files = [groups_file]
bottlenecks = create_bottlenecks(bottleneck_file, data_dir, base_model, groups_files)

# perform tests
cv = True
groups = "patient-groups"

if not cv:
    train_and_evaluate(
            base_model, bottlenecks, tmp_dir, log_dir, 
            test_size=0.3, groups=groups, use_weights=True,
            optimizer=None, dropout_rate=0.5, epochs=20, batch_size=512,
            save_model=False)
else:
    cross_validate(
            base_model, bottlenecks, tmp_dir, data_dir, groups=groups, 
            num_folds=5, logo=True, use_weights=False, resample=1.0,
            optimizer=None, dropout_rate=0.5, epochs=20, batch_size=512,
            summarize_model=True, summarize_misclassified_images=True)

K.clear_session() # prevent TensorFlow error message

