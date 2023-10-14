import torch
import torchaudio
import os
from tqdm import tqdm
import numpy as np
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract', action="store_true", default=False, required=False)
    args = parser.parse_args()

    if args.extract:
        wav2mel = torch.jit.load("dvector/wav2mel.pt")
        dvector = torch.jit.load("dvector/dvector-step250000.pt").eval()
        folder = '../ABCS/'
        for split in ['train', 'test', 'dev']:
            data = os.path.join(folder, 'Audio', split)
            X_speaker = {}
            X = []
            Y = []
            with torch.no_grad():
                for i, speaker in enumerate(os.listdir(data)):
                    directory = os.path.join(data, speaker)
                    j = 0
                    print(speaker)
                    for file in tqdm(os.listdir(directory)):
                        file = os.path.join(directory, file)
                        try:
                            wav_tensor, sample_rate = torchaudio.load(file)
                            mel_tensor = wav2mel(wav_tensor, sample_rate)  # shape: (frames, mel_dim)
                            emb_tensor = dvector.embed_utterance(mel_tensor).numpy()  # shape: (emb_dim)        print(rec_result['text'], text)
                            X.append(emb_tensor)
                            Y.append(i)
                            j += 1
                        except:
                            continue
                    X_speaker[speaker] = np.mean(np.array(X[-j:]), axis=0)
            np.savez('json/ABCS_' + split + '.npz', **X_speaker)
        X = np.array(X)
        Y = np.array(Y)
        np.save('X.npy', X)
        np.save('Y.npy', Y)
    else:
        X = np.load('X.npy')
        Y = np.load('Y.npy')
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.model_selection import train_test_split
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
        clf.fit(X_train, y_train)
        clf.predict(X_test)
        print(clf.score(X_test, y_test))