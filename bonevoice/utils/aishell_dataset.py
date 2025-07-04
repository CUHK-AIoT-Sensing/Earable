
import os
import librosa
import numpy as np


def enroll_audio(folder):
    output_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.wav') or file.endswith('.flac') or file.endswith('.WAV'):
                output_files.append(os.path.join(root, file))
    return output_files

class aishellDataset():
    def __init__(self, folder, split='all', length=5, sample_rate=16000):
        self.folder = folder
        self.split = f'aishell-{split}'
        self.length = length
        self.sample_rate = sample_rate
        self.dataset = []
        self.split_folder = os.path.join(folder, self.split)
        self.audio_files = enroll_audio(self.split_folder)
    def __len__(self):
        return len(self.audio_files)
    def __getitem__(self, index):
        file = self.audio_files[index]
        audio = librosa.load(file, sr=self.sample_rate, mono=True)[0]  # Load audio file
        if len(audio) <= self.length * self.sample_rate:
            pad = (self.length * self.sample_rate) - len(audio)
            audio = np.pad(audio, (0, pad), mode='constant')
        else:
            offset = np.random.randint(0, len(audio) - self.length * self.sample_rate)
            audio = audio[offset:offset + self.length * self.sample_rate]
        data = {
            'filename': file,
            'index': index,
            'audio': audio,
        }
        return data
    
if __name__ == '__main__':
    from resemblyzer import VoiceEncoder, preprocess_wav

    encoder = VoiceEncoder()

    dataset = aishellDataset(folder='../dataset/Audio/aishell', split='train', length=5, sample_rate=16000)
    embeddings = {}
    # for i in range(10):
    for i in np.random.randint(0, len(dataset), size=1000):
        data = dataset[i]
        filename = data['filename']
        SID = filename.split('/')[-2]
        embed = encoder.embed_utterance(data['audio'])
        if SID not in embeddings:
            embeddings[SID] = []
        embeddings[SID].append(embed)
    print(len(embeddings), 'speakers found')
    similarities = []
    for SID, embeds in embeddings.items():
        if len(embeds) < 2:
            continue
        # INTRA-SPEAKER SIMILARITY
        embeds = np.stack(embeds, axis=0)
        intra_similarity = embeds @ embeds.T / (np.linalg.norm(embeds, axis=1, keepdims=True) * np.linalg.norm(embeds, axis=1, keepdims=True).T)
        intra_similarity = intra_similarity.mean()

        # INTER-SPEAKER SIMILARITY
        other_embeds = []
        for other_SID, other_embeds_list in embeddings.items():
            if other_SID != SID:
                other_embeds.extend(other_embeds_list)
        other_embeds = np.stack(other_embeds, axis=0)
        inter_similarity = embeds @ other_embeds.T / (np.linalg.norm(embeds, axis=1, keepdims=True) * np.linalg.norm(other_embeds, axis=1, keepdims=True).T) 
        inter_similarity = inter_similarity.mean()

        similarities.append((intra_similarity, inter_similarity))
        print(f'Speaker {SID}: Intra-similarity: {intra_similarity:.4f}, Inter-similarity: {inter_similarity:.4f}')

    # plot the similarities in histogram
    import matplotlib.pyplot as plt
    intra_similarities = [s[0] for s in similarities]
    inter_similarities = [s[1] for s in similarities]
    plt.hist(intra_similarities, bins=50, alpha=0.5, label='Same speaker', color='blue')
    plt.hist(inter_similarities, bins=50, alpha=0.5, label='Different speaker', color='orange')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Intra vs Inter Speaker Similarity')
    plt.legend()
    plt.savefig('resources/speech_similarity_histogram.pdf')