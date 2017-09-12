import hashlib
import os
import pickle
from urllib.request import urlretrieve

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
from tqdm import tqdm
from zipfile import ZipFile

ARR = os.listdir('data/')

def download_one(url, file):
    """
    Download file from <url>
    :param url: URL to file
    :param file: Local file path
    """
    if not os.path.isfile(file):
        print('Downloading ' + file + '...')
        urlretrieve(url, file)
        print('Download Finished')

def download_test_data():
    if 'notMNIST_train.zip' not in ARR:
        download_one('https://s3.amazonaws.com/udacity-sdc/notMNIST_train.zip', 'notMNIST_train.zip')
        assert hashlib.md5(open('notMNIST_train.zip', 'rb').read()).hexdigest() == 'c8673b3f28f489e9cdf3a3d74e2ac8fa',\
                'notMNIST_train.zip file is corrupted.  Remove the file and try again.'
        print('Train file downloaded.')

    if 'notMNIST_test.zip' not in ARR:
        download_one('https://s3.amazonaws.com/udacity-sdc/notMNIST_test.zip', 'notMNIST_test.zip')
        assert hashlib.md5(open('notMNIST_test.zip', 'rb').read()).hexdigest() == '5d3c7e653e63471c88df796156a9dfa9',\
            'notMNIST_test.zip file is corrupted.  Remove the file and try again.'
        print('Test file downloaded.')

def uncompress_features_labels(file):
    """
    Uncompress features and labels from a zip file
    :param file: The zip file to extract the data from
    """
    features = []
    labels = []

    with ZipFile(file) as zipf:
        # Progress Bar
        filenames_pbar = tqdm(zipf.namelist(), unit='files')

        # Get features and labels from all files
        for filename in filenames_pbar:
            # Check if the file is a directory
            if not filename.endswith('/'):
                with zipf.open(filename) as image_file:
                    image = Image.open(image_file)
                    image.load()
                    # Load image data as 1 dimensional array
                    # We're using float32 to save on memory space
                    feature = np.array(image, dtype=np.float32).flatten()

                # Get the the letter from the filename.  This is the letter of the image.
                label = os.path.split(filename)[1][0]

                features.append(feature)
                labels.append(label)
    return np.array(features), np.array(labels)

def extract_data_from_zip(docker_size_limit=150000):
    npy_set = set(['train_labels.npy', 'train_features.npy','test_labels.npy', 'test_features.npy'])
    if npy_set.intersection(set(ARR)) == npy_set:
        train_labels = np.load('train_labels.npy')
        train_features = np.load('train_features.npy')
        test_labels = np.load('test_labels.npy')
        test_features = np.load('test_features.npy')

    else:
        # Get the features and labels from the zip files
        train_features, train_labels = uncompress_features_labels('notMNIST_train.zip')
        test_features, test_labels = uncompress_features_labels('notMNIST_test.zip')

        # Limit the amount of data to work with a docker container
        train_features, train_labels = resample(train_features, train_labels, n_samples=docker_size_limit)

        np.save('train_labels.npy',train_labels)
        np.save('train_features.npy',train_features)
        np.save('test_labels.npy',test_labels)
        np.save('test_features.npy',test_features)

        # Wait until you see that all features and labels have been uncompressed.
        print('All features and labels uncompressed.')
    return train_features, train_labels, test_features, test_labels

def normalize_grayscale(image_data, a = 0.1, b = 0.9):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    x_min = 0
    x_max = 255
    return a + (image_data - x_min)*(b-a) / (x_max - x_min)

def prep_data_for_training():
    download_test_data()

    train_features, train_labels, test_features, test_labels = extract_data_from_zip()

    train_features = normalize_grayscale(train_features)
    test_features = normalize_grayscale(test_features)

    # Turn labels into numbers and apply One-Hot Encoding
    encoder = LabelBinarizer()
    encoder.fit(train_labels)
    train_labels = encoder.transform(train_labels)
    test_labels = encoder.transform(test_labels)

    # Change to float32, so it can be multiplied against the features in TensorFlow, which are float32
    train_labels = train_labels.astype(np.float32)
    test_labels = test_labels.astype(np.float32)

    train_features, valid_features, train_labels, valid_labels = train_test_split(
    train_features,
    train_labels,
    test_size=0.05,
    random_state=832289)

    if not os.path.isfile(pickle_file):
        print('Saving data to pickle file...')
        try:
            with open('notMNIST.pickle', 'wb') as pfile:
                pickle.dump(
                    {
                        'train_dataset': train_features,
                        'train_labels': train_labels,
                        'valid_dataset': valid_features,
                        'valid_labels': valid_labels,
                        'test_dataset': test_features,
                        'test_labels': test_labels,
                    },
                    pfile, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise

    print('Data cached in pickle file.')

def load_data():
    pickle_file = 'notMNIST.pickle'
    if pickle_file not in ARR:
        prep_data_for_training()
    with open('data/' + pickle_file, 'rb') as f:
      pickle_data = pickle.load(f)
      train_features = pickle_data['train_dataset']
      train_labels = pickle_data['train_labels']
      valid_features = pickle_data['valid_dataset']
      valid_labels = pickle_data['valid_labels']
      test_features = pickle_data['test_dataset']
      test_labels = pickle_data['test_labels']
      del pickle_data  # Free up memory
      return train_features, train_labels, valid_features, valid_labels, test_features, test_labels
