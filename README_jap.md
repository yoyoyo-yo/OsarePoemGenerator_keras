# OsarePoemGenerator_keras

Osare-Poem-Generator のKeras実装.

[Description]
https://qiita.com/nagayosi/items/79916363a9a5a36137bc

[Reference] LSTM sample code of https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py 

# Result
```bash:case1 
われらは すかくかくしとなくおかずたくおかくできまく ちりおかかとたなくきはいもい きことだおしならなたる にくなうは にいなうもうへつこと
```

```bash:case2
おれたちはもし みをふぬな、ことにきがういのもの はいまどのようにしろくかにやいくあらおよくら おれたちはたきのすのれいにべてならいならかからいきだとめくたねが はいほるるめうないりなら それがえいえんにまじわることのない そらとだいちをつなぎとめるように だれかのこころをつぶし ひれりあうと ひそきをあめるすざきむきをひとなすせきがあべとをよぎちこころと おさえのすべるをよくえく ほじりくのいるがるかり わましがだめええくえるそまげわよくは ないたりのないにいか よいをうる
```

```bash:case3
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

自分で用意したデータセットで学習するには、config.pyの"Train_data_path"を変える。

そして下記コマンドを打つで学習が始まる。

```
python3 main.py --train
```

# Testing

```
python3 main.py --test
```

# Model

"model.py"でネットワークを変更できる。

# N-gram

"config.py"のN-gramパラメータを変更できる。

