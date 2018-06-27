# OsarePoemGenerator_keras

This is Osare-Poem-Generator keras implementation.

[Description]
https://qiita.com/nagayosi/items/9cd8fb0445882ce09c94

[Reference] LSTM sample code of https://github.com/keras-team/keras


# Requirements
Python-3.6.4

Keras-2.2.0

TensorFlow-1.8.0

Numpy-1.14.0

# Training
Please prepare your own dataset, and you change "Train_data_path" to your dataset path.

Please type below command.

```
python3 main.py --train
```

# Testing

```
python3 main.py --test
```

# Model
You can change network model in "model.py"

# N-gram
You can change N-gram number and other parameters in "config.py"