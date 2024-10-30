First clone the baseline implementation and install the requirements,
then to download the dataset run the python file task1b.py,
which will download the dataset into TAU-urban-acoustic-scenes-2020-3class-development.

To get all the audio files from the zip files, use our script which are **located in the scripts folder.**

```bash
python3 extract_and_move.py -d TAU-urban-acoustic-scenes-2020-3class-development  -o raw_dataset
```

39GB of audio files will be extracted to raw_dataset folder.

Then from the raw_dataset folder, run the following command to get the 3 class dataset:

'indoor': ['airport', 'shopping_mall', 'metro_station'],
'outdoor': ['street_pedestrian', 'public_square', 'street_traffic', 'urban_park'],
'transportation': ['bus', 'tram', 'underground_metro']

```bash
python3 get_3class_dataset_from_raw.py -d raw_dataset/ -o 3class_dataset
```

To copy n samples from a dataset to another folder, run the following command:
This gives us 1000 samples for each class, 3000 samples in total. 8GB

```bash
bash copy_n_samples.sh raw_dataset/ 1000 3class_dataset/
bash copy_n_samples.sh 3class_dataset/ 100 small_dataset/
```

This will give us small dataset to work in local machine.

---

## Before you Train, check the training log from HPC with 3000 samples, by openinig nohup.out file.

To train the new model run the following command:

```bash
python3 main.py -c -d 3class_dataset/
```

```bash
python3 main.py -c -d 3class_dataset/ -x
```

_-x to train baseline model along with the low model._

**-c is for HPC or gpu machine if not don't use it.**

Use this to evaluate the model on a **folder of audio files (wav)**:
this will save the resutls in plots/test_results folder.

```bash
python3 main.py -c -m models/low_complexity_model.keras -f test_folder
```

test_folder should contain wav files.

---

Model: "baseline"

---

# Layer (type) Output Shape Param

conv2d (Conv2D) (None, 40, 501, 32) 1600

batch_normalization (Batch (None, 40, 501, 32) 128
Normalization)

activation (Activation) (None, 40, 501, 32) 0

max_pooling2d (MaxPooling2 (None, 8, 100, 32) 0
D)

dropout (Dropout) (None, 8, 100, 32) 0

conv2d_1 (Conv2D) (None, 8, 100, 64) 100416

batch_normalization_1 (Bat (None, 8, 100, 64) 256
chNormalization)

activation_1 (Activation) (None, 8, 100, 64) 0

max_pooling2d_1 (MaxPoolin (None, 2, 1, 64) 0
g2D)

dropout_1 (Dropout) (None, 2, 1, 64) 0

flatten (Flatten) (None, 128) 0

dense (Dense) (None, 100) 12900

dropout_2 (Dropout) (None, 100) 0

dense_1 (Dense) (None, 3) 303

=================================================================
Total params: 115603 (451.57 KB)
Trainable params: 115411 (450.82 KB)
Non-trainable params: 192 (768.00 Byte)

our model size analysis:
Name Param NZ Param Size NZ Size

---

separable_conv2d 9 9 0.04 KB 0.04 KB
separable_conv2d 16 16 0.06 KB 0.06 KB
separable_conv2d 16 16 0.06 KB 0.06 KB
separable_conv2d_1 144 144 0.56 KB 0.56 KB
separable_conv2d_1 512 512 2.00 KB 2.00 KB
separable_conv2d_1 32 32 0.12 KB 0.12 KB
dense_2 94464 94464 369.00 KB 369.00 KB
dense_2 3 3 0.01 KB 0.01 KB

---

Total 95196 95196 371.86 KB 371.86 KB
