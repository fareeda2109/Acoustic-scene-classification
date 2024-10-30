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
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Flatten, Dense, SeparableConv2D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
import pandas as pd
import seaborn as sns
import argparse



#%%
#set random seed for random and numpy and tensorflow
random.seed(33)
np.random.seed(33)
tf.random.set_seed(33)


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
                # set seed to get same samples
                random.seed(33)
                sampled_files = random.sample(all_files, min(len(all_files), self.n_samples_per_class))
                self.n_samples_per_class = min(len(all_files), self.n_samples_per_class)
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
        plt.savefig(f'plots/raw_waveform_{file_name or ""}.png')
        print(f'Saved raw waveform to plots/raw_waveform_{file_name or ""}.png')
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
        plt.savefig(f'plots/mel_spec_{file_name or ""}.png')
        print(f'Saved mel spectrogram to plots/mel_spec_{file_name or ""}.png')
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

    
    return info

def plot_history(history, model_name):
    # convert history to dataframe
    history_df = pd.DataFrame(history.history)
    # plot train and val loss and accuracy and save plot
    history_df[['loss', 'val_loss']].plot()
    plt.savefig(f'plots/{model_name}_loss.png')
    plt.close()
    history_df[['accuracy', 'val_accuracy']].plot()
    plt.savefig(f'plots/{model_name}_accuracy.png')
    print(f'Saved loss and accuracy plots to plots/{model_name}_loss.png and plots/{model_name}_accuracy.png')
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

def plot_class_dist(sets, sets_names):
    df_dict = {'feature': [], 'label': [], 'set': []}
    for i, dataset in tqdm.tqdm(enumerate(sets)):
        for audio_feature, label in dataset:
            df_dict['feature'].append(audio_feature.numpy().squeeze())
            df_dict['label'].append(audio_dataset.decode_label(label))
            df_dict['set'].append(sets_names[i])
            
    df = pd.DataFrame(df_dict)
    print(f'df head: {df.head()}')
    
    
    # plot class distribution
    sns.set(style='whitegrid', palette='muted', font_scale=1.2)
    sns.countplot(x='label', hue='set', data=df, palette='Spectral')
    plt.gca().set_ylabel('Count')
    plt.gca().set_title('Class Distribution')
    plt.tight_layout()
    sets_names_str = '_'.join(sets_names)
    plt.savefig(f'plots/{sets_names_str}_class_dist.png')
    plt.close()

#%%
def try_on_test(audio_dataset, test_dataset, model):
    test_data = []
    target_classes = list(audio_dataset.label_to_class_name.values())
    for audio_features, labels in test_dataset:
        for i in range(len(audio_features)):
            label = labels[i]
            audio_feature = audio_features[i]
            class_name = audio_dataset.decode_label(label)
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
        pred_class = audio_dataset.decode_label(pred[0])
        gt = audio_dataset.decode_label(label)
        print(f"{model.name} -- Predicted class (in {(end - start) :.2f}s): {pred_class} GT: {gt}")


def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    # warm up with random prediction
    for _ in range(3):
        model.predict(np.random.randn(1, 40, 501, 1))
    print(f'Loaded model from {model_path}')
    return model

def build_low_comp_model(input_shape, num_classes, multi_gpu=False):
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
            ], name='low_complexity_model')

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

def decode_label(one_hot_label):
    label_index = np.argmax(one_hot_label)
    classes = ['indoor', 'outdoor', 'transportation']
    return classes[label_index]

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

def infer_from_folder(model_path, folder, limit=20):
    model = load_model(model_path)
    file_paths = glob.glob(os.path.join(folder, '*.wav'))
    
    for i, file_path in tqdm.tqdm(enumerate(file_paths)):
        pred_class, (audio_feature, sr) = infer_and_plot_from_file(model, file_path)
        print(f'Predicted class: {pred_class}')
    
        # Plot log mel spectrogram with prediction and ground truth
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(audio_feature.squeeze(), sr=sr, hop_length=512, x_axis='time', y_axis='mel')
        plt.colorbar()
        pth = pathlib.Path(file_path).stem
        # clas = pathlib.Path(file_path).parent.name

        # increase title font size
        plt.title(f'Prediction: {pred_class}, File: {pth}', fontsize=20)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        # save in test_restults folder
        
        os.makedirs('plots/test_results', exist_ok=True)
        plt.savefig(f'plots/test_results/pred_{pred_class}_gt_{pth}.png')
        plt.close()
        print(f'Saved mel spectrogram to plots/test_results/pred_{pred_class}_gt_{pth}_pred.png')
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
    parser.add_argument('-m', '--model_path', type=str, help='Path to model')
    parser.add_argument('-f', '--test_audio_folder', type=str, help='Path to log mel /labels folder')
    parser.add_argument('-x', '--train_baseline', action='store_true', help='Train baseline model', default=False)
    parser.add_argument('-d', '--data_dir', type=str, help='Path to data dir')
    
    args = parser.parse_args()
    
    data_dir = args.data_dir if args.data_dir else '../small_dataset'
    
    if args.test_audio_folder:
        if not args.model_path:
            print('Please provide path to model')
            exit(1)
        infer_from_folder(args.model_path, args.test_audio_folder)
        exit(0)
    
    train_baseline = args.train_baseline

    
    is_local = not args.hpc
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # is_local = True
    # data_dir = '../small_dataset' if is_local else '../3class_dataset'
    n_samples_per_class = 100 if is_local else 5000
    audio_dataset = AudioDataset(data_dir, n_samples_per_class)
    # %% explore dataset
    print(f'Number of classes: {len(audio_dataset.label_to_class_name)}')
    print(f'Number of samples per class: {audio_dataset.n_samples_per_class}')
    print(f'Number of samples in dataset: {len(list(audio_dataset.dataset))}')

    # plot log mel spectrogram
    # glob.glob(os.path.join(data_dir, '*/.wav'))
    # take first sample from glob
    file_path = glob.glob(os.path.join(data_dir, '*/*.wav'))[0]
    audio_dataset.plot_log_mel_spectrogram(file_path=file_path)
    audio_dataset.plot_raw_audio(file_path=file_path)
    
    
    # shape of dataset
    audio_dataset.dataset.element_spec
    # shape of features
    audio_dataset.dataset.element_spec[0].shape
    # shape of labels
    audio_dataset.dataset.element_spec[1].shape
    

    # %% split dataset
    batch_size = 2 if is_local else 16
    # cache and prefetch dataset
    audio_dataset.dataset = audio_dataset.dataset.cache().prefetch(tf.data.experimental.AUTOTUNE)

    # shuffle dataset
    # set random seed to get same shuffle tensorflow
    tf.random.set_seed(33)
    audio_dataset.dataset = audio_dataset.dataset.shuffle(buffer_size=len(list(audio_dataset.dataset)))
    batched_dataset = audio_dataset.dataset.batch(batch_size)

    train_dataset, test_dataset, val_dataset = split_dataset(batched_dataset, verbose=True)
    # write out test set into test_folder
    # dump_set(test_dataset)
    
    # take one sample and print feature and label shapes
    for audio_feature, label in train_dataset.take(1):
        print(f"Audio feature shape: {audio_feature.shape}")
        print(f"Label shape: {label.shape}")
        print(f'Label: {label}')

    #%% plot class distribution using pandas and seaborn
    # convert dataset to pandas dataframe
    plot_class_dist([train_dataset, test_dataset, val_dataset], ['train', 'test', 'val'])
    
    
    input_shape = (40, 501, 1)  
    num_classes = len(audio_dataset.label_to_class_name)
    multi_gpu = not is_local  # Set to True to use multiple GPUs

    # %% build model
    

    if train_baseline:

        print('-*-' * 20)
        print(f'Baseline model section starts...')
        print('-*-' * 20)
        model = build_baseline_model(input_shape, num_classes, multi_gpu=multi_gpu)

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

        # Summary of the model
        model.summary()
        model_name = model.name
        checkpoint_path = f'models/{model_name}.keras'  # File path to save the model
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        
        # Training the model
        epochs = 20 if is_local else 200
        history = model.fit(
            train_dataset,  # Your training dataset
            epochs=epochs,  # Number of epochs
            batch_size=batch_size,  # Batch size
            validation_data=val_dataset,  # Your validation dataset
            verbose=1,  # Verbosity mode (1 = progress bar)
            callbacks=[checkpoint]
        )
        plot_history(history, model_name)

        # use checkpoint to load best model
        model = load_model(checkpoint_path)
        
        # Evaluate the model on the test dataset
        loss, accuracy = model.evaluate(test_dataset)
            
        print(f"Test loss: {loss}")
        print(f"Test accuracy: {accuracy}")

        # Test on a single audio file
        # select a random audio file from testset from each class
        try_on_test(audio_dataset, test_dataset, model)
        # play audio
        # import IPython.display as ipd
        # ipd.Audio(file_path)

        model_size_details = get_keras_model_size(model)

        print(f'Saved baseline model to {checkpoint_path}')
        

    #%%
    print('-*-' * 20)
    print(f'Training low complexity model...')
    print('-*-' * 20)
    
    low_complexity_model = build_low_comp_model(input_shape, num_classes, multi_gpu)
    low_complexity_model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=.001),
                                loss='categorical_crossentropy',
                                metrics=['accuracy'])
    print(f'low complexity model summary: {low_complexity_model.summary()}')
    low_complexity_model_name = 'low_complexity_model'
    lc_checkpoint_path = f'models/{low_complexity_model_name}.keras'
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=lc_checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    # train the model
    history = low_complexity_model.fit(train_dataset,
                                    epochs=10 if is_local else 40,
                                    validation_data=test_dataset, callbacks=[checkpoint_callback])
    
    plot_history(history, low_complexity_model_name)

    # load best model
    low_complexity_model = load_model(f'models/{low_complexity_model_name}.keras')
    # evaluate the model
    loss, accuracy = low_complexity_model.evaluate(test_dataset)
    
    print(f'Loss: {loss:.2f}')
    print(f'Accuracy: {accuracy * 100:.2f}%')
    
    try_on_test(audio_dataset, test_dataset, low_complexity_model)
    
    low_complexity_model_size_details = get_keras_model_size(low_complexity_model)

    print(f'Saved low complexity model to {lc_checkpoint_path}')

            


# %%
