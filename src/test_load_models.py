#%%
import glob
import pathlib
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import soundfile as sf
import os
import random
import librosa
import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Flatten, Dense,DepthwiseConv2D, SeparableConv2D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
import pandas as pd
import seaborn as sns
import argparse



#%%

class AudioDataset:
    def __init__(self, data_dir, n_samples_per_class):
        self.data_dir = data_dir
        self.n_samples_per_class = n_samples_per_class
        self.label_to_class_name = {}  # Mapping of numeric labels to class names
        self.dataset = self._create_dataset()

    def _get_file_paths(self):
        file_paths = []
        labels = []
        class_folders = os.listdir(self.data_dir)

        for label, class_folder in tqdm.tqdm(enumerate(class_folders)):
            full_path = os.path.join(self.data_dir, class_folder)
            if os.path.isdir(full_path):
                self.label_to_class_name[label] = class_folder  # Store class name mapping
                all_files = [os.path.join(full_path, f) for f in tqdm.tqdm(os.listdir(full_path)) if f.endswith('.wav')]
                random.seed(42)
                sampled_files = random.sample(all_files, min(len(all_files), self.n_samples_per_class))
                file_paths.extend(sampled_files)
                labels.extend([label] * len(sampled_files))

        return file_paths, labels

    def resample_audio(self, file_path, new_sr):
        audio, sr = sf.read(file_path)
        resampled_audio = librosa.resample(audio, orig_sr=sr, target_sr=new_sr)
        return resampled_audio, new_sr

    def _extract_feature(self, audio=None, sr=None, file_path=None, hop_length=512):
        if audio is None and file_path is not None:
            audio, sr = librosa.load(file_path, sr=None)

        # Check if audio was successfully loaded
        if audio is None:
            raise ValueError("Audio data is not provided.")

        # Rest of the feature extraction logic
        n_fft = int(0.040 * sr)  # 40 ms window
        hop_length = int(0.020 * sr)  # 50% overlap, 20 ms hop
        n_mels = 40

        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        return log_mel_spec

    # def _process_audio(self, file_path):
    #     audio, sr = sf.read(file_path, always_2d=True)
    #     embedding = openl3.get_audio_embedding(audio, sr, input_repr="mel256", content_type="music", embedding_size=512)[0]
    #     return embedding
    
    def _create_dataset(self):
        file_paths, labels = self._get_file_paths()
        log_mel_specs = [self._extract_feature(file_path=fp) for fp in tqdm.tqdm(file_paths)]
        one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=len(set(labels)))
        return tf.data.Dataset.from_tensor_slices((log_mel_specs, one_hot_labels))
        
    def _create_dataset_l3(self):
        file_paths, labels = self._get_file_paths()
        embeddings = [self._process_audio(fp) for fp in tqdm.tqdm(file_paths)]
        one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=len(set(labels)))
        return tf.data.Dataset.from_tensor_slices((embeddings, one_hot_labels))

    def plot_raw_audio(self, file_path=None, audio=None, sr=None):
        
        if audio is not None and sr is not None:
            # Use provided audio and sample rate
            audio_feature = audio
        elif file_path is not None:
            # Load audio from file path
            audio, sr = librosa.load(file_path, sr=None)
        else:
            raise ValueError("No valid input provided for plotting.")

        file_name = os.path.basename(file_path) if file_path else None
        
        time_steps = librosa.core.frames_to_time(range(len(audio)), sr=sr, hop_length=1)
        
        # Plotting logic
        plt.figure(figsize=(12, 4))
        plt.plot(time_steps, audio)
        plt.xlim(0, max(time_steps))
        plt.title(f'Raw waveform -- {file_name or ""}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.savefig(f'raw_waveform_{file_name or ""}.png')
        print(f'Saved raw waveform to raw_waveform_{file_name or ""}.png')
        plt.close()

    def plot_log_mel_spectrogram(self, file_path=None, audio=None, sr=None, index=None):
        if index is not None:
            # Access audio feature by index from the dataset
            for i, (audio_feature, _) in enumerate(self.dataset):
                if i == index:
                    audio_feature = audio_feature.numpy().squeeze()  # Remove batch dimension and channel dimension
                    break
        elif audio is not None and sr is not None:
            # Use provided audio and sample rate
            audio_feature = self._extract_feature(audio=audio, sr=sr)
        elif file_path is not None:
            # Load audio from file path
            audio_feature = self._extract_feature(file_path=file_path)
        else:
            raise ValueError("No valid input provided for plotting.")

        file_name = os.path.basename(file_path) if file_path else None
        # Plotting logic
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(audio_feature, sr=sr or 16000, hop_length=512, x_axis='time', y_axis='mel')
        plt.colorbar()
        plt.title(f'Mel spectrogram: {file_name or ""}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.savefig(f'mel_spec_{file_name or ""}.png')
        print(f'Saved mel spectrogram to mel_spec_{file_name or ""}.png')
        plt.close()
    
    def plot_l3_features(self, index):
        for i, (audio_feature, label) in enumerate(self.dataset):
            if i == index:
                plt.imshow(audio_feature.numpy().T, aspect='auto', origin='lower')
                class_name = self.decode_label(label)
                plt.title(f'Class: {class_name}, One-hot: {label.numpy()}')
                plt.show()
                break
        else:
            print(f"No audio feature found at index {index}")


    def decode_label(self, one_hot_label):
        label_index = np.argmax(one_hot_label)
        return self.label_to_class_name.get(label_index)


def build_baseline_model(input_shape, num_classes, multi_gpu=False):
    # Define the model architecture within a strategy scope if multi_gpu is True
    strategy = tf.distribute.MirroredStrategy() if multi_gpu else None
    with strategy.scope() if strategy else tf.device('/cpu:0'):
        name = 'baseline'
        
        model = Sequential(name=name)

        # Input shape should be in the format (frequency bins, time steps, channels)
        # Assuming input_shape is (40, 501, 1) or similar for frequency x time x channels

        # CNN layer #1
        model.add(Conv2D(32, kernel_size=(7, 7), input_shape=input_shape, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(5, 5)))
        model.add(Dropout(0.3))

        # CNN layer #2
        model.add(Conv2D(64, kernel_size=(7, 7), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(4, 100)))
        model.add(Dropout(0.3))

        # Flatten
        model.add(Flatten())

        # Dense layer #1
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.3))

        # Output layer
        model.add(Dense(num_classes, activation='softmax'))


    return model



def get_keras_model_size(keras_model, verbose=True, excluded_layers=None):
    """Calculate Keras model size (non-zero parameters on disk)

    Parameters
    ----------
    keras_model : keras.models.Model
        Keras model for the size calculation

    verbose : bool
        Print layer by layer information
        Default value True

    excluded_layers : list
        List of layers to be excluded from the calculation
        Default value None

    Returns
    -------
    dict
        Model size details

    """
    model_name = keras_model.name
    parameters_count = 0
    parameters_count_nonzero = 0
    parameters_bytes = 0
    parameters_bytes_nonzero = 0

    if excluded_layers is None:
        # Define excluded layers
        excluded_layers = [
            tf.keras.layers.BatchNormalization
            # Add other layers to exclude if needed
        ]

    if verbose:
        print("{:<30} {:<12} {:<12} {:<30} {:<30}".format('Name', 'Param', 'NZ Param', 'Size', 'NZ Size'))
        print("-" * 114)

    for layer in keras_model.layers:
        # Skip excluded layers
        if type(layer) in excluded_layers:
            continue

        for weight in layer.get_weights():
            # Calculate parameters
            shape = weight.shape
            current_parameters_count = np.prod(shape)
            current_parameters_count_nonzero = np.count_nonzero(weight)

            # Calculate size in bytes based on data type
            bytes_per_element = weight.dtype.itemsize
            current_parameters_bytes = current_parameters_count * bytes_per_element
            current_parameters_bytes_nonzero = current_parameters_count_nonzero * bytes_per_element

            # Accumulate totals
            parameters_count += current_parameters_count
            parameters_count_nonzero += current_parameters_count_nonzero
            parameters_bytes += current_parameters_bytes
            parameters_bytes_nonzero += current_parameters_bytes_nonzero

            if verbose:
                print("{:<30} {:<12} {:<12} {:<30} {:<30}".format(
                    layer.name,
                    current_parameters_count,
                    current_parameters_count_nonzero,
                    f"{current_parameters_bytes / 1024:.2f} KB",
                    f"{current_parameters_bytes_nonzero / 1024:.2f} KB"
                ))

    # Summary
    if verbose:
        print("-" * 114)
        print("{:<30} {:<12} {:<12} {:<30} {:<30}".format(
            'Total',
            parameters_count,
            parameters_count_nonzero,
            f"{parameters_bytes / 1024:.2f} KB",
            f"{parameters_bytes_nonzero / 1024:.2f} KB"
        ))
    info = {
        'parameters': {
            'all': {
                'count': parameters_count,
                'bytes': parameters_bytes
            },
            'non_zero': {
                'count': parameters_count_nonzero,
                'bytes': parameters_bytes_nonzero
            }
        }
    }
    
    # save as csv
    df = pd.DataFrame(info)
    df.to_csv(f'{model_name}_model_size.csv')
    
    return info

def plot_history(history, model_name):
    # convert history to dataframe
    history_df = pd.DataFrame(history.history)
    # plot train and val loss and accuracy and save plot
    history_df[['loss', 'val_loss']].plot()
    plt.savefig(f'{model_name}_loss.png')
    plt.close()
    history_df[['accuracy', 'val_accuracy']].plot()
    plt.savefig(f'{model_name}_accuracy.png')
    print(f'Saved loss and accuracy plots to {model_name}_loss.png and {model_name}_accuracy.png')
    plt.close()


#%%
def split_dataset(batched_dataset, verbose=False):
    dataset_size = len(list(batched_dataset))
    train_size = int(0.75 * dataset_size)
    test_size = int(0.15 * dataset_size)
    val_size = int(0.1 * dataset_size)

    train_dataset = batched_dataset.take(train_size)
    test_dataset = batched_dataset.skip(train_size)
    val_dataset = test_dataset.skip(test_size)
    test_dataset = test_dataset.take(test_size)

    if verbose:
        print(f"Train size: {len(list(train_dataset))}")
        print(f"Test size: {len(list(test_dataset))}")
        print(f"Val size: {len(list(val_dataset))}")
        
    return train_dataset,test_dataset,val_dataset


def decode_label(one_hot_label):
    label_index = np.argmax(one_hot_label)
    classes = ['indoor', 'outdoor', 'transportation']
    return classes[label_index]

#%%
def try_on_test(test_dataset, model):
    test_data = []
    target_classes = ['indoor', 'outdoor', 'transportation']
    for audio_features, labels in test_dataset:
        for i in range(len(audio_features)):
            label = labels[i]
            audio_feature = audio_features[i]
            class_name = decode_label(label)
            if class_name in target_classes:
                test_data.append((audio_feature, label))
                target_classes.remove(class_name)
            if not target_classes:
                break
    
    for audio_feature, label in test_data:
        audio_feature = np.expand_dims(audio_feature, axis=0) # Add batch dimension
        audio_feature = np.expand_dims(audio_feature, axis=-1) # Add channel dimension
        print(f'audio feature shape: {audio_feature.shape}')
        start = time.time()
        pred = model.predict(audio_feature)
        end = time.time()
        # decode prediction
        pred_class = decode_label(pred[0])
        gt = decode_label(label)
        print(f"{model.name} -- Predicted class (in {(end - start) :.2f}s): {pred_class} GT: {gt}")


def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    # warm up with random prediction
    for _ in range(3):
        model.predict(np.random.randn(1, 40, 501, 1))
    print(f'Loaded model from {model_path}')
    return model


def build_small_student_model(input_shape, num_classes, multi_gpu=False):
    strategy = tf.distribute.MirroredStrategy() if multi_gpu else None
    with strategy.scope() if strategy else tf.device('/cpu:0'):
        model = Sequential([
                SeparableConv2D(16, (3, 3), activation='relu', input_shape=input_shape),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                SeparableConv2D(32, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dropout(0.4),
                Dense(num_classes, activation='softmax')
            ], name='low_student_model')

    return model




def extract_features(audio, sr, hop_length=512):
    # Check if audio was successfully loaded
    if audio is None:
        raise ValueError("Audio data is not provided.")

    # Rest of the feature extraction logic
    n_fft = int(0.040 * sr)  # 40 ms window
    hop_length = int(0.020 * sr)  # 50% overlap, 20 ms hop
    n_mels = 40

    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return log_mel_spec

def infer_and_plot_from_file(model, file_path):
    audio, sr = librosa.load(file_path, sr=None)

    # Extract features
    audio_feature = extract_features(audio, sr)
    
    # Add batch dimension
    audio_feature = np.expand_dims(audio_feature, axis=0)
    # Add channel dimension
    audio_feature = np.expand_dims(audio_feature, axis=-1)
    
    # Predict
    pred = model.predict(audio_feature)
    # Decode prediction
    pred_class = decode_label(pred[0])
    
    return pred_class, (audio_feature, sr)

def infer_from_folder(model, folder, limit=10):
    file_paths = glob.glob(os.path.join(folder, '*.wav'))
    
    for i, file_path in tqdm.tqdm(enumerate(file_paths)):
        pred_class, (audio_feature, sr) = infer_and_plot_from_file(model, file_path)
        print(f'Predicted class: {pred_class}')
    
        # Plot log mel spectrogram with prediction and ground truth
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(audio_feature.squeeze(), sr=sr, hop_length=512, x_axis='time', y_axis='mel')
        plt.colorbar()
        pth = pathlib.Path(file_path).stem
        clas = pathlib.Path(file_path).parent.name
        plt.title(f'Prediction: {pred_class}, File: {clas}_{pth}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        # save in test_restults folder
        
        os.makedirs('test_results', exist_ok=True)
        plt.savefig(f'test_results/pred_{pred_class}_gt_{clas}_{pth}_pred.png')
        plt.close()
        if i >= limit:
            break

def dump_set(test_dataset):
    os.makedirs('test_folder', exist_ok=True)
    # dump features and labels into test_folder
    for i, batch in enumerate(test_dataset):
        audio_features, labels = batch
        for j in range(len(audio_features)):
            audio_feature = audio_features[j]
            label = labels[j]
            np.save(f'test_folder/audio_feature_{i*len(audio_features)+j}.npy', audio_feature)
            np.save(f'test_folder/label_{i*len(audio_features)+j}.npy', label)
        
def read_dump_set(data_dir, batch_size=16):
    # read dumped features and labels
    audio_features = []
    labels = []
    for file_path in glob.glob(os.path.join(data_dir, '*.npy')):
        if 'audio_feature' in file_path:
            audio_feature = np.load(file_path)
            audio_features.append(audio_feature)
        elif 'label' in file_path:
            label = np.load(file_path)
            labels.append(label)
    print(f'Number of audio features: {len(audio_features)}')
    print(f'Number of labels: {len(labels)}')
    # batch into batch size of 16
    batch_size = batch_size
    audio_features = np.array(audio_features)
    labels = np.array(labels)
    audio_features = tf.data.Dataset.from_tensor_slices(audio_features)
    labels = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((audio_features, labels))
    dataset = dataset.batch(batch_size)
    
    return dataset

#%%
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--hpc', action='store_true', help='Use cluster', default=False)
    args = parser.parse_args()
    is_local = not args.hpc
    # is_local = True
    # test_folder contains log mel spectrograms and labels
    test_dataset = read_dump_set('test_folder')
    
    print('*=*'*20)
    print(f'Baseline model')
    print('*=*'*20)
    
    # load baseline model
    baseline_model_path = 'baseline_model.keras'
    baseline_model_checkpoint_path = 'baseline_checkpoint.keras'
    
    baseline_model = load_model(baseline_model_path)
    baseline_model_checkpoint = load_model(baseline_model_checkpoint_path)
    
    # evaluate baseline model
    loss, accuracy = baseline_model.evaluate(test_dataset)
    print(f'Baseline model loss: {loss:.2f}')
    print(f'Baseline model accuracy: {accuracy * 100:.2f}%')
    
    # evaluate baseline model checkpoint
    loss, accuracy = baseline_model_checkpoint.evaluate(test_dataset)

    print(f'Baseline model checkpoint loss: {loss:.2f}')
    print(f'Baseline model checkpoint accuracy: {accuracy * 100:.2f}%')
    
    try_on_test(test_dataset, baseline_model)
    try_on_test(test_dataset, baseline_model_checkpoint)
    #%%
    print('*=*'*20)
    print(f'Low student model')
    print('*=*'*20)
    # load low student model
    low_student_model_path = 'low_student_model.keras'
    low_student_model_checkpoint_path = 'low_student_model_checkpoint.keras'
    
    low_student_model = load_model(low_student_model_path)
    low_student_model_checkpoint = load_model(low_student_model_checkpoint_path)

    
    # evaluate low student model
    loss, accuracy = low_student_model.evaluate(test_dataset)
    print(f'Low student model loss: {loss:.2f}')
    print(f'Low student model accuracy: {accuracy * 100:.2f}%')
    
    # evaluate low student model checkpoint
    loss, accuracy = low_student_model_checkpoint.evaluate(test_dataset)
    print(f'Low student model checkpoint loss: {loss:.2f}')
    print(f'Low student model checkpoint accuracy: {accuracy * 100:.2f}%')
    
    try_on_test(test_dataset, low_student_model)
    try_on_test(test_dataset, low_student_model_checkpoint)

    
    # confusion matrix using sklearn
    # imports 
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    
    # get predictions
    y_pred_baseline = []
    y_pred_low_student = []

    for audio_features, labels in test_dataset:
        for i in tqdm.tqdm(range(len(audio_features))):
            audio_feature = audio_features[i]
            label = labels[i]
            audio_feature = np.expand_dims(audio_feature, axis=0)
            audio_feature = np.expand_dims(audio_feature, axis=-1)
            pred_baseline = baseline_model.predict(audio_feature, verbose=0)
            pred_low_student = low_student_model.predict(audio_feature, verbose=0)
            y_pred_baseline.append(decode_label(pred_baseline[0]))
            y_pred_low_student.append(decode_label(pred_low_student[0]))
            
    y_true = []
    for audio_features, labels in test_dataset:
        for i in tqdm.tqdm(range(len(audio_features))):
            label = labels[i]
            y_true.append(decode_label(label))  
            
    # confusion matrix
    cm_baseline = confusion_matrix(y_true, y_pred_baseline)
    cm_low_student = confusion_matrix(y_true, y_pred_low_student)
    
    # plot confusion matrix
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    disp_baseline = ConfusionMatrixDisplay(confusion_matrix=cm_baseline, display_labels=['indoor', 'outdoor', 'transportation'])
    
    disp_baseline.plot(ax=ax[0], cmap=plt.cm.Blues)
    disp_baseline.ax_.set_title('Baseline model')
    disp_baseline.im_.colorbar.remove()
    disp_baseline.ax_.set_xlabel('Predicted label')
    disp_baseline.ax_.set_ylabel('True label')
    
    disp_low_student = ConfusionMatrixDisplay(confusion_matrix=cm_low_student, display_labels=['indoor', 'outdoor', 'transportation'])
    
    disp_low_student.plot(ax=ax[1], cmap=plt.cm.Blues)
    disp_low_student.ax_.set_title('Low student model')
    disp_low_student.im_.colorbar.remove()
    disp_low_student.ax_.set_xlabel('Predicted label')
    disp_low_student.ax_.set_ylabel('True label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
# %%
