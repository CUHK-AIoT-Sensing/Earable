import scipy

file = 'example.wav'
int16_file = 'example_int16.wav'
data = scipy.io.wavfile.read(file)[1]
data *= 2**15
data = data.astype('int16')
scipy.io.wavfile.write(int16_file, 16000, data)