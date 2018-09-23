# OsarePoemGenerator_keras

This is Osare-Poem-Generator keras implementation.

[Description]
https://qiita.com/nagayosi/items/79916363a9a5a36137bc

[Reference] LSTM sample code of https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py 

# Result
```:case1 
われらは すかくかくしとなくおかずたくおかくできまく ちりおかかとたなくきはいもい きことだおしならなたる にくなうは にいなうもうへつこと
```

```:case2
おれたちはもし みをふぬな、ことにきがういのもの はいまどのようにしろくかにやいくあらおよくら おれたちはたきのすのれいにべてならいならかからいきだとめくたねが はいほるるめうないりなら それがえいえんにまじわることのない そらとだいちをつなぎとめるように だれかのこころをつぶし ひれりあうと ひそきをあめるすざきむきをひとなすせきがあべとをよぎちこころと おさえのすべるをよくえく ほじりくのいるがるかり わましがだめええくえるそまげわよくは ないたりのないにいか よいをうる
```

```:case3
ああ おれたちは げっこうにどくされている
```

# Requirements

```
Python-3.6.4
Keras-2.2.0
TensorFlow-1.8.0
Numpy-1.14.0
```

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
