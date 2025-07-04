'''
minimal example of a dataset for FSD50K
'''
import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm

def extract_features(split='eval'):
    import torch
    from hear_mn import mn01_all_b_mel_avgs, mn40_all_b_mel_avgs
    wrapper = mn40_all_b_mel_avgs.load_model().cuda()
    dataset = FSD50KDataset(folder='../dataset/Audio/FSD50K', split=split, length=2, sample_rate=32000)
    output_folder = f'../dataset/Audio/FSD50K/FSD50K.{split}_feature'
    os.makedirs(output_folder, exist_ok=True)
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        audio = torch.from_numpy(data['audio'])[None, :].cuda()
        # embed, time_stamps = mn01_all_b_mel_avgs.get_timestamp_embeddings(audio, wrapper)
        embed = mn40_all_b_mel_avgs.get_scene_embeddings(audio, wrapper)
        filename = os.path.join(output_folder, f"{data['filename']}.npy")
        np.save(filename, embed.cpu().numpy())

class FSD50KDataset:
    def __init__(self, folder, split='all', length=5, sample_rate=16000, feature=False):
        self.folder = folder
        self.length = length
        self.sample_rate = sample_rate
        self.dataset = []
        self.split_folder = os.path.join(folder, f'FSD50K.{split}_audio')
        self.feature_folder = os.path.join(folder, f'FSD50K.{split}_feature')
        label_path = os.path.join(folder, 'FSD50K.ground_truth', f'{split}.csv')
        self.labels = pd.read_csv(label_path, header=0, delimiter=',')

        vocabulary_path = os.path.join(folder, 'FSD50K.ground_truth', 'vocabulary.csv')
        vocabulary = pd.read_csv(vocabulary_path, header=None)
        self.vocabulary = {}
        for (index, name, mid) in vocabulary.itertuples(index=False):
            self.vocabulary[mid] = index
        self.num_classes = len(self.vocabulary)
        print(f'{len(self.labels)} labels from {self.split_folder}')

        # add a new column label_idxs to self.labels
        self.labels['label_idxs'] = self.labels['mids'].apply(lambda mids: [self.vocabulary[mid] for mid in mids.split(',') if mid in self.vocabulary])
        self.feature = feature
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels.iloc[index]
        filename, label_names, mids, label_idxs = label['fname'], label['labels'], label['mids'], label['label_idxs']

        one_hot = np.zeros(self.num_classes, dtype=np.float32)
        for label_idx in label_idxs:
            one_hot[label_idx] = 1.0
        output_dict = {
            'filename': filename,
            'mids': mids.split(','),
            'label_names': label_names.split(','),
            'idx': label_idxs,
            'one_hot': one_hot,}
        if self.feature:
            feature_path = os.path.join(self.feature_folder, f"{filename}.npy")
            feature = np.load(feature_path)
            output_dict['feature'] = feature

        audio, sr = librosa.load(os.path.join(self.split_folder, str(filename) + '.wav'), sr=self.sample_rate, duration=self.length)
        if len(audio) < self.sample_rate * self.length:
            # pad the audio to the specified length
            audio = np.pad(audio, (0, self.sample_rate * self.length - len(audio)), mode='constant')
        else:
            # trim the audio to the specified length
            audio = audio[:self.sample_rate * self.length]
        output_dict['audio'] = audio
        return output_dict
    
class FSD50KDataset_class(FSD50KDataset):
    '''
    only keep one class from the original dataset
    '''
    def __init__(self, folder, split='all', length=5, sample_rate=16000, class_id=[0], feature=False):
        super().__init__(folder, split, length, sample_rate, feature)
        labels = []
        for label in self.labels.itertuples(index=False):
            for label_idx in label.label_idxs:
                if label_idx in class_id:
                    labels.append(label)
                    break
        self.labels = pd.DataFrame(labels)



if __name__ == '__main__':
    # extract_features(split='eval')
    dataset = FSD50KDataset(folder='../dataset/Audio/FSD50K', split='eval', length=2, sample_rate=16000, feature=True)
    features = []
    for i in range(100):
        dataset_i = FSD50KDataset_class(folder='../dataset/Audio/FSD50K', split='eval', length=2, sample_rate=16000, class_id=[i], feature=True)
        print(f'Class {i} has {len(dataset_i)} samples')
        features_i = []
        for j in range(20):
            data = dataset_i[j]
            feature = data['feature']
            features_i.append(feature)
        features.append(np.stack(features_i, axis=0))
    features = np.stack(features, axis=0).squeeze() # shape (4, 20, C)
    print(f'Features shape: {features.shape}')  # shape (4, 20, C)

    inter_class_similarity = []; intra_class_similarity = []
    for i in range(features.shape[0]):
        feature_i = features[i]  # shape (20, C)

        feature_other = np.delete(features, i, axis=0)  # shape (3, 20, C)
        feature_other = feature_other.reshape(-1, feature_other.shape[-1])  # shape

        cosine_similarity = feature_i @ feature_other.T / (np.linalg.norm(feature_i, axis=1, keepdims=True) * np.linalg.norm(feature_other, axis=1, keepdims=True).T)
        cosine_similarity = cosine_similarity.mean()
        inter_class_similarity.append(cosine_similarity)

        cosine_similarity = feature_i @ feature_i.T / (np.linalg.norm(feature_i, axis=1, keepdims=True) * np.linalg.norm(feature_i, axis=1, keepdims=True).T)
        cosine_similarity = cosine_similarity.mean()
        intra_class_similarity.append(cosine_similarity)
    print(inter_class_similarity)
    print(intra_class_similarity)

    # plot the similarities in histogram
    import matplotlib.pyplot as plt
    plt.hist(inter_class_similarity, bins=20, alpha=0.5, label='Same class', color='blue')
    plt.hist(intra_class_similarity, bins=20, alpha=0.5, label='Different class', color='orange')
    plt.legend()
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Inter vs Intra Class Similarity')
    plt.savefig('resources/sound_similarity_histogram.pdf')


    # import matplotlib.pyplot as plt
    # import sklearn.manifold
    # TSNE = sklearn.manifold.TSNE(n_components=2, random_state=42)
    # features_2d = TSNE.fit_transform(features.reshape(-1, features.shape[-1]))
    # features_2d = features_2d.reshape(features.shape[0], -1, 2)  # shape (4, 20, 2)

    # plt.figure(figsize=(10, 10))
    # for i in range(features_2d.shape[0]):
    #     plt.scatter(features_2d[i, :, 0], features_2d[i, :, 1], label=f'Class {i}')
    # plt.legend()
    # plt.title('t-SNE of FSD50K Features')
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')
    # plt.savefig('fsd50k_features_tsne.png')
