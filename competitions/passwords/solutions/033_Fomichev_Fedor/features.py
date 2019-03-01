import pandas as pd
from collections import Counter
import numpy as np
from source.keyboard import keyboard_dist
import re
import time

year = re.compile('(19[0-9][0-9]|20[0-2][0-9])')

PATH_TO_TRAIN = './train.csv'
PATH_TO_TEST = './Xtest.csv'

PATH_TO_NAMES = './lists/names.txt'
PATH_TO_WORDS = './lists/words.txt'
PATH_TO_SURNAMES = './lists/surnames.txt'
PATH_TO_WORST_ONE = './lists/worst.txt'
PATH_TO_WORST_TWO = './lists/worst_passwords.txt'
PATH_TO_WORST_THREE = './lists/common_passwords.txt'

def make_pattern(path):
    pattern = '('
    with open(path, 'r') as f:
        for line in f:
            pattern += (line.strip() + '|')
    pattern = pattern[:-1] + ')'
    return re.compile(pattern)

worst = make_pattern(PATH_TO_WORST_THREE)


t1 = time.time()
r = re.findall(worst, PATH_TO_SURNAMES)
t2 = time.time()
print(t2 - t1)

def get_list(path):
    output = []
    with open(path, 'r') as f:
        for line in f:
            name = line.strip()
            output.append(name)
    return output

names = get_list(PATH_TO_NAMES)
worst_one = get_list(PATH_TO_WORST_ONE)
worst_two = get_list(PATH_TO_WORST_TWO)
worst_three = get_list(PATH_TO_WORST_THREE)
worst = set(worst_one + worst_two + worst_three)
worst = {x for x in worst if len(re.findall(year, x)) == 0}

def contain(s, vocab):
    if type(s) == str:
        output = int(any(x.lower() in s.lower() for x in vocab))
    else:
        output = int(False)
    return output


def target_log(df):
    df['log_y'] = df['Times'].map(lambda x: np.log(x))
    return df

def get_features(df):
    df['has_name'] = df['Password'].map(lambda x: contain(x, names))
    df['key_dist'] = df['Password'].map(lambda x: sum(keyboard_dist(gram[0], gram[1]) for gram in zip(x[:-1], x[1:])) if type(x) == str else x)
    df['has_alpha'] = df['Password'].map(lambda x: int(any(token.isalpha() for token in x)) if type(x) == str else x)
    df['has_digit'] = df['Password'].map(lambda x: int(any(token.isdigit() for token in x)) if type(x) == str else x)
    df['has_upper'] = df['Password'].map(lambda x: int(any(token.isupper() for token in x)) if type(x) == str else x)
    df['has_lower'] = df['Password'].map(lambda x: int(any(token.islower() for token in x)) if type(x) == str else x)
    return df

def contain_year(df):
    df['has_year'] = df['Password'].str.contains(year)
    df['has_year'] = df['has_year'].map(lambda x: int(x) if type(x) != float else x)
    return df


def contain_worst(df):
    df['has_worst'] = df['Password'].map(lambda x: contain(x, worst))
    return df

train = pd.read_csv(PATH_TO_TRAIN)
test = pd.read_csv(PATH_TO_TEST)

train = train[train['Password'].isnull() == False]
train = target_log(train)
train = get_features(train)
train = contain_worst(train)
train.to_csv(PATH_TO_TRAIN, index=False)

test = get_features(test)
test = contain_year(test)
test = contain_worst(test)
test.to_csv(PATH_TO_TEST, index=False)

vocab = set(ch.lower() for x in train['Password'].values for ch in x)

cnt = Counter()
gram_cnt = 0
for _, row in train.iterrows():
    for gram in zip(row['Password'][:-1], row['Password'][1:]):
        cnt[gram] += 1
        gram_cnt += 1

for gram, val in cnt.items():
    cnt[gram] = np.log(val / gram_cnt)

def prob(s):
    smooth = np.log(1 / gram_cnt)
    if type(s) == str:
        output = sum(cnt.get(gram, smooth) for gram in zip(s[:-1], s[1:]))
    else:
        output = smooth
    return output

train['pass_prob'] = train['Password'].map(prob)
test['pass_prob'] = test['Password'].map(prob)



