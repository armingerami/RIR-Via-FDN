import scipy.io.wavfile as wavfile
import numpy as np
from scipy.signal import convolve

rir = abs(np.load('RIR.npy'))
rir /= max(abs(rir))
early_reflections_indices, scale_index = np.argsort(rir)[::-1][:43][:43], np.argsort(rir)[::-1][43]
early_reflections = rir[early_reflections_indices]
scale = rir[scale_index] # used to scale the synthesized tail

def convolve_wav_and_save(wav_file_path, signal, output_wav_file_path):
  """Reads a .wav file, convolves it with a signal, and saves the result."""

  sample_rate, audio_data = wavfile.read(wav_file_path)

  audio_data = audio_data.astype(np.float32)
  try:
    if(audio_data.shape[1] > 1):
      audio_data = audio_data[:,1]
  except:
    audio_data.flatten
  # print(audio_data.shape)
  convolved_audio = convolve(audio_data, signal, mode='full')

  convolved_audio = convolved_audio / np.max(np.abs(convolved_audio))

  wavfile.write(output_wav_file_path, sample_rate, convolved_audio)

# wav_file_path = 'input_instrument.wav'
# output_wav_file_path = 'Actual RIR - instrument.wav'
wav_file_path = 'speech.wav'
output_wav_file_path = 'Actual RIR - speech.wav'
convolve_wav_and_save(wav_file_path, rir, output_wav_file_path)


k = 16 # number of feedback loops in the feedback delay network (FDN). Total parameters = 2k; k decay rates and k scalars
n = len(rir) # duration of room impulse response in time-steps
alpha = 0.995 # decay rate of FDN
locs = np.array([0, 4, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 2560, 28106, 3072, 3584, 4096]).reshape(k,1)

coeffs = np.array([0.23319344, 0.22061893, 0.17574172, 0.12599885, 0.06956492, 0.033961847, 0.028124204, 0.06441578, 0.08327039, 0.07072372, 0.06388091, 0.122572176, 0.995, 0.08605025, 0.07184157, 0.07634564, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995])

times = np.arange(n, dtype = np.float32).reshape(n,1)
times = np.repeat(times,k,1).T
times -= (locs-1)
activation = np.zeros((k,n), dtype = np.float32)
for i in range(k):
  for j in range(n):
    times[i][j] = max(times[i][j],0)
    activation[i][j] = min(times[i][j], 1)
times -= 1
times = np.array(times, dtype = np.float32)
activation = np.array(activation, dtype = np.float32)


exp = np.pow(coeffs[k:].reshape(k,1),times)
our_rir = activation*exp*((coeffs[:k].reshape(k,1))**2)
our_rir = np.sum(our_rir,0)
our_rir *= scale/max(abs(our_rir))
for i in early_reflections_indices:
    our_rir[i] = rir[i]


# output_wav_file_path = 'Our Synthesized RIR - instrument.wav'
output_wav_file_path = 'Our Synthesized RIR3 - speech.wav'
convolve_wav_and_save(wav_file_path, our_rir, output_wav_file_path)