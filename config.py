## Config setting

import os

Voca = 'あいうおえかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽぁぃぅぇぉゃゅょっ、。ー!? @'

Vocabrary = [x for x in Voca]
Vocabrary_num = len(Vocabrary)

Train_data_path = 'bleach_poem_hiragana.txt'
    

N_gram = 5
Prefix = ['@']


# Training
Minibatch = 10
Epoch = 50
Learning_rate = 0.01
Weight_decay = 0.0005

# Test
Max_text_length = 1000
diversity = 0.3

Save_directory = 'output'
Save_model = 'osarePoem.h5'
Save_path = os.path.join(Save_directory, Save_model)

