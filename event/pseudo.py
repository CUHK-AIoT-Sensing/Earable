'''
Implement the method to get relative clean result
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from scipy.special import softmax

def pseduo_label(audio, vision, text, y, method='skewness'):
    total = len(y)
    cosine_audio = audio @ text.transpose()
    cosine_vision = vision @ text.transpose()
    zero_shot_audio = np.argmax(cosine_audio, axis=-1) == y
    zero_shot_vision = np.argmax(cosine_vision, axis=-1) == y
    print(sum(zero_shot_audio)/total, sum(zero_shot_vision)/total)
    cosine = (cosine_audio + cosine_vision) / 2
    zero_shot = np.argmax(cosine, axis=-1) == y
    print('zero-shot performance:', sum(zero_shot)/total)

    sort_cosine = np.sort(cosine, axis=-1)
    top_cos = sort_cosine[:, -1]
    top_ratio = sort_cosine[:, -1] / sort_cosine[:, -2]
    gap = np.mean(sort_cosine[:, -3:], axis=-1) - np.mean(sort_cosine[:, :-3], axis=-1)
    mean = np.mean(sort_cosine, axis=-1)
    var = np.var(sort_cosine, axis=-1)

    if method == 'threshold':
        above_threshold = top_cos > 0.2
    elif method == 'skewness':
        above_threshold = top_ratio > 1.1
    else:
        above_threshold = gap > 0.1
    check = np.argmax(cosine[above_threshold], axis=-1) == y[above_threshold]
    print('utilized percentage:', np.sum(check) / total, 'accuracy:', np.sum(check) / np.shape(check)[0])
    label = np.argmax(cosine, axis=-1)
    # label = cosine/np.sum(cosine, axis=1, keepdims=True)
    return above_threshold, label




