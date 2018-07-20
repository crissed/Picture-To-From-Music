import matplotlib.pyplot as plt
import colorsys
from PIL import Image
from scipy.io import wavfile
import random as rnd
import numpy as np

def Fp(p):
    # Fp(f) converts pitch index into corresponding frequency
    freq = 440*2**((p-69)/12)
    return freq

def Pic2Sound_prototype(input, output, base_note, T_per_time_frame = 0.5, res_factor = 50, the_major_scale = [0, 0, 0, 2, 2, 4, 4, 4, 5, 5, 7, 7, 7, 9, 11]):
    img_file = Image.open(input)
    img = img_file.load()
    fs = float(5000)
    Ts = 1 / fs
    N_time_frame_index = int(T_per_time_frame / Ts)
    freq_res = fs / N_time_frame_index
    time_frame_domain = np.linspace(0, T_per_time_frame, N_time_frame_index)
    freq_domain = np.linspace(0, fs, N_time_frame_index)

    [xs, ys] = img_file.size
    [xs, ys] = [xs // res_factor, ys // res_factor]

    pic_pixel = [[[] for j in range(ys)] for i in range(xs)]
    pic_pixel_aligned = [[] for i in range(ys * xs)]

    # Mapping 2D space into an indexed 1D space
    for y in range(ys):
        for x in range(xs):
            # (4)  Get the RGB color of the pixel
            [r, g, b] = img[res_factor * x, res_factor * y]
            [r, g, b] = [float(r), float(g), float(b)]
            r = 10 ** ((40 + r / 255 * 40) / 20)
            g = 10 ** ((40 + g / 255 * 40) / 20)
            b = 10 ** ((40 + b / 255 * 40) / 20)

            pic_pixel[x][y] = [r, g, b]
            pic_pixel_aligned[x + xs * y] = [r, g, b]


    # base_note = 50  # C2

    the_sound = []
    temp_scaler = []
    freq_one_oct_noter = []
    freq_two_oct_noter = []
    freq_three_oct_noter = []

    one_oct_noter = []
    two_oct_noter = []
    three_oct_noter = []

    for n in range(len(pic_pixel_aligned)):
        # for n in range(1):
        temp_scale = the_major_scale[rnd.randrange(0, len(the_major_scale) - 1)]
        temp_scaler.append(temp_scale)

        one_oct_note = float(base_note + temp_scale)
        two_oct_note = float(base_note + 12 + temp_scale)
        three_oct_note = float(base_note + 12 * 2 + temp_scale)

        one_oct_noter.append(one_oct_note)
        two_oct_noter.append(two_oct_note)
        three_oct_noter.append(three_oct_note)

        freq_one_oct_note = Fp(one_oct_note)
        freq_two_oct_note = Fp(two_oct_note)
        freq_three_oct_note = Fp(three_oct_note)

        freq_one_oct_noter.append(freq_one_oct_note)
        freq_two_oct_noter.append(freq_two_oct_note)
        freq_three_oct_noter.append(freq_three_oct_note)

        freq_one_oct_note_index = int(freq_one_oct_note / freq_res)
        freq_two_oct_note_index = int(freq_two_oct_note / freq_res)
        freq_three_oct_note_index = int(freq_three_oct_note / freq_res)

        temp_freq_bin = [0 for i in range(N_time_frame_index)]

        temp_freq_bin[freq_one_oct_note_index] = r
        temp_freq_bin[freq_two_oct_note_index] = g
        temp_freq_bin[freq_three_oct_note_index] = b

        sig_bin = np.real(np.fft.ifft(temp_freq_bin))
        sig_bin = list(sig_bin)
        the_sound += sig_bin
        print(n, "/", len(pic_pixel_aligned))

    wavfile.write('/Users/user/Music/GarageBand/' + output + '.wav', fs, np.array(the_sound))
