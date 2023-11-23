!apt-get install -qq sox

# Cloning the provided github repository
!git clone https://github.com/AdroitAnandAI/Indian-Accent-Speech-Recognition.git

# Mounting the drive
import os
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
PATH = os.getcwd()

import tarfile

model_tar_path = '/content/gdrive/MyDrive/1StopASR/DeepSpeech-0.9.3.tar.gz'
model_dir = '/content/gdrive/MyDrive/1StopASR'
model= '/content/gdrive/MyDrive/1StopASR/deepspeech-0.9.3-model.pbmm'
# Create the model directory if it doesn't exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Extract the model files from the tar.gz file
with tarfile.open(model_tar_path, 'r:gz') as tar:
    tar.extractall(path=model_dir)

cd /content/gdrive/MyDrive/1StopASR/DeepSpeech-0.9.3

# Installing the requirements
pip install -r requirements_transcribe.txt

# Installing DeepSpeech 
pip install deepspeech==0.9.3

!apt-get install -qq sox

import deepspeech

!deepspeech --model 1StopASR/output_graph.pbmm --lm speech/lm.binary --trie speech/trie --audio /content/06_M_artic_01_004.wav

#to use deepspeech decoder function to do STT
./DeepSpeech.py --train_files ../data/CV/en/clips/train.csv --dev_files ../data/CV/en/clips/dev.csv --test_files ../data/CV/en/clips/test.csv

from data_generator import vis_train_features

# extract label and audio features for a single training example
vis_text, vis_raw_audio, vis_mfcc_feature, vis_spectrogram_feature, vis_audio_path = vis_train_features()


from IPython.display import Markdown, display
from data_generator import vis_train_features, plot_raw_audio
from IPython.display import Audio
%matplotlib inline

# plot audio signal
plot_raw_audio(vis_raw_audio)
# print length of audio signal
display(Markdown('**Shape of Audio Signal** : ' + str(vis_raw_audio.shape)))
# print transcript corresponding to audio clip
display(Markdown('**Transcript** : ' + str(vis_text)))
# play the audio file
Audio(vis_audio_path)

# Plotting Spectogram
from data_generator import plot_spectrogram_feature

# plot normalized spectrogram
plot_spectrogram_feature(vis_spectrogram_feature)
# print shape of spectrogram
display(Markdown('**Shape of Spectrogram** : ' + str(vis_spectrogram_feature.shape)))

 
#Plottinng MFCC
from data_generator import plot_mfcc_feature

# plot normalized MFCC
plot_mfcc_feature(vis_mfcc_feature)
# print shape of MFCC
display(Markdown('**Shape of MFCC** : ' + str(vis_mfcc_feature.shape)))


%reload_ext autoreload
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf 
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

# watch for any changes in the sample_models module, and reload it automatically
%load_ext autoreload
%autoreload 2
# import NN architectures for speech recognition
from sample_models import *
# import function for training acoustic model
from train_utils import train_model

# specify the model
model_end = final_model(input_dim=161,
                        filters=200,
                        kernel_size=11, 
                        conv_stride=2,
                        conv_border_mode='valid', units=200, recur_layers=2)


# Training the model
from keras.optimizers import SGD

train_model(input_to_softmax=model_end, 
            pickle_path='model_end.pickle', 
            save_model_path='model_end.h5', 
            spectrogram=True,# change to False if you would like to use MFCC features
            optimizer=SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1)) 
# Plotting 

from glob import glob
import numpy as np
import _pickle as pickle
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set_style(style='white')

# obtain the paths for the saved model history
all_pickles = sorted(glob("results/*.pickle"))
# extract the name of  model
model_names = [item[8:-7] for item in all_pickles]
# extract the loss history for  model
valid_loss = [pickle.load( open( i, "rb" ) )['val_loss'] for i in all_pickles]
train_loss = [pickle.load( open( i, "rb" ) )['loss'] for i in all_pickles]
# save the number of epochs used to train model
num_epochs = [len(valid_loss[i]) for i in range(len(valid_loss))]

# plot the training loss vs. epoch for model
ax1 = fig.add_subplot(121)
for i in range(len(all_pickles)):
    ax1.plot(np.linspace(1, num_epochs[i], num_epochs[i]), 
            train_loss[i], label=model_names[i])

# plot the validation loss vs. epoch for model
ax2 = fig.add_subplot(122)
for i in range(len(all_pickles)):
    ax2.plot(np.linspace(1, num_epochs[i], num_epochs[i]), 
            valid_loss[i], label=model_names[i])
# clean up the plot
ax2.legend()  
ax2.set_xlim([1, max(num_epochs)])
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.show()


# Obtain predictions
import numpy as np
from data_generator import AudioGenerator
from keras import backend as K
from utils import int_sequence_to_text
from IPython.display import Audio

def get_predictions(index, partition, input_to_softmax, model_path):
    """ Print a model's decoded predictions
    Params:
        index (int): The example you would like to visualize
        partition (str): One of 'train' or 'validation'
        input_to_softmax (Model): The acoustic model
        model_path (str): Path to saved acoustic model's weights
    """
    # load the train and test data
    data_gen = AudioGenerator()
    data_gen.load_train_data()
    data_gen.load_validation_data()
    
    # obtain the true transcription and the audio features 
    if partition == 'validation':
        transcr = data_gen.valid_texts[index]
        audio_path = data_gen.valid_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    elif partition == 'train':
        transcr = data_gen.train_texts[index]
        audio_path = data_gen.train_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    else:
        raise Exception('Invalid partition!  Must be "train" or "validation"')
        
    # obtain and decode the acoustic model's predictions
    input_to_softmax.load_weights(model_path)
    prediction = input_to_softmax.predict(np.expand_dims(data_point, axis=0))
    output_length = [input_to_softmax.output_length(data_point.shape[0])] 
    pred_ints = (K.eval(K.ctc_decode(
                prediction, output_length)[0][0])+1).flatten().tolist()
    
    # play the audio file, and display the true and predicted transcriptions
    print('-'*80)
    Audio(audio_path)
    print('True transcription:\n' + '\n' + transcr)
    print('-'*80)
    print('Predicted transcription:\n' + '\n' + ''.join(int_sequence_to_text(pred_ints)))
    print('-'*80)


get_predictions(index=0, 
                partition='train',
                input_to_softmax=deep_rnn_model(input_dim=161,
                         units=200,
                         recur_layers=2), 
                model_path='results/model_3.h5')

get_predictions(index=0, 
                partition='validation',
                input_to_softmax=deep_rnn_model(input_dim=161,
                         units=200,
                         recur_layers=2), 
                model_path='results/model_3.h5')
