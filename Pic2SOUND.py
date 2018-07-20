import numpy as np
import matplotlib.pyplot as plt
import colorsys
from PIL import Image
from scipy.io import wavfile
import random as rnd
from Functions import *


Pic2Sound_prototype("/Users/user/PycharmProjects/Pic2Sound/KakaoTalk_Photo_2018-07-08-00-23-40_65.jpeg", "Dansuh", 51, T_per_time_frame = 0.4, res_factor = 30, the_major_scale = [0, 1, 4, 6, 7, 9, 12])

# plt.figure(1)
# plt.plot(the_sound)
# plt.show()

# 0 C
# 1 C#/Db
# 2 D
# 3 D#/Eb
# 4 E
# 5 F
# 6 F#/Gb
# 7 G
# 8 G#/Ab
# 9 A
# 10 A#/Bb
# 11 B