import time
import numpy as np
from rtlsdr import RtlSdr
import random, string, math
import scipy.signal as signal


def read_samples(sdr, freq):
    sdr.center_freq = freq
    time.sleep(0.04)
    return sdr.read_samples(sample_rate * 0.25)


def plot_samples(iq_samples):
    from scipy import signal
    from scipy.fftpack import fftshift
    import matplotlib.pyplot as plt

    f, Pxx = signal.welch(iq_samples, sample_rate, detrend=lambda x: x)
    f, Pxx = fftshift(f), fftshift(Pxx)

    plt.semilogy(f / 1e3, Pxx)
    plt.xlabel('f, kHz')
    plt.ylabel('PSD, Power/Hz')
    plt.grid()

    plt.xticks(np.linspace(-sample_rate / 2e3, sample_rate / 2e3, 7))
    plt.xlim(-sample_rate / 2e3, sample_rate / 2e3)

    # draw it
    plt.show()


def randomword(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


sdr = RtlSdr()
sdr.sample_rate = sample_rate = 2400000
dec_rate = 48
sdr.err_ppm = 56  # change it to yours
sdr.gain = 20     # change it to yours, it is better to obtain samples at different gain levels

# collect "other" class training data
# for example here we are scanning ether from 110M to 144M assuming there are no interesting signals (it's not true)
freq = 114000000
while freq <= 144000000:
    print('  reading at', freq)
    iq_samples = read_samples(sdr, freq)
    iq_samples = signal.decimate(iq_samples, dec_rate, zero_phase=None)
    filename = "training_data/other/samples-" + randomword(16) + ".npy"
    np.save(filename, iq_samples)
    freq += 50000

# collect "wfm" class traininig data
for i in range(0, 1000):
    iq_samples = read_samples(sdr, 95000000)  # put here frequency of your local WFM station
    iq_samples = signal.decimate(iq_samples, dec_rate, zero_phase=None)
    filename = "training_data/wfm/samples-" + randomword(16) + ".npy"
    np.save(filename, iq_samples)
    if not (i % 10): print(i / 10, "%")
    
# collect "dmr" training data
for i in range(0, 1000):
    iq_samples = read_samples(sdr, 147337500)  # put here your local DMR frequency
    iq_samples = signal.decimate(iq_samples, dec_rate, zero_phase=None)
    filename = "training_data/dmr/samples-" + randomword(16) + ".npy"
    np.save(filename, iq_samples)
    if not (i % 10): print(i / 10, "%")

# collect "Tv" training data
for i in range(0, 1000):
    # iq_samples = read_samples(sdr, 191260000)
    iq_samples = read_samples(sdr, 49250000)
    iq_samples = signal.decimate(iq_samples, dec_rate, zero_phase=None)
    filename = "training_data/tv/samples-" + randomword(16) + ".npy"
    np.save(filename, iq_samples)
    if not (i % 10): print(i / 10, "%")

# Now do not forget to move about 20% of samples to their corresponding folders /testing_data/CLASS_LABEL/
