import numpy as np
import matplotlib.pyplot as plt
import colorsys
from PIL import Image
from scipy.io import wavfile
import random as rnd


img_file = Image.open("/Users/user/PycharmProjects/Pic2Sound/KakaoTalk_Photo_2018-07-07-20-20-39_38.jpeg")
img = img_file.load()

# (2) Get image width & height in pixels
[xs, ys] = img_file.size
max_intensity = 100
hues = {}
pic_pixel = [[[] for j in range(ys)] for i in range(xs)]

# (3) Examine each pixel in the image file
for x in range(xs):
    for y in range(ys):
        # (4)  Get the RGB color of the pixel
        [r, g, b] = img[x, y]
        [r,g,b] = [float(r), float(g), float(b)]
        [r, g, b] = [10**((20*r/255) * (rnd.randrange(-500,500)/100)/20), 10**((20*g/255) * (rnd.randrange(-500,500)/100)/20), 10**((20*b/255) * ( rnd.randrange(-500,500)/100)/20)]
        pic_pixel[x][y] = [r, g, b]

freq_comp = [0 for i in range(xs*ys)]

for x in range(xs):
    for y in range(ys):
        freq_comp[y + x*ys] = pic_pixel[x][y][0] - pic_pixel[x][y][1] + 1j*(pic_pixel[x][y][1] + pic_pixel[x][y][2])

sound_from_image = np.fft.ifft(freq_comp)
len_sound = len(sound_from_image)
sound_from_image = np.array([np.real(x) for x in sound_from_image])

plt.figure(1)
plt.plot(sound_from_image)
plt.show()

freq_domain = np.linspace(0, 15000, xs*ys)
plt.figure(2)
plt.subplot(211)
plt.semilogy(freq_domain, np.abs(freq_comp))
plt.subplot(212)
plt.plot(freq_domain, np.angle(freq_comp))
plt.show()

wavfile.write('/Users/user/Music/GarageBand/' + 'YeonJu.wav', 15000, sound_from_image)