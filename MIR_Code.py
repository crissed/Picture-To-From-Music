import math
import numpy as np
from random import shuffle, randint
import matplotlib.pyplot as pyp
from matplotlib import colors
from scipy.io import wavfile


# Python code required for MIR_Study @ NAVER corp. Copyrights: Jisu Jeon @ 26/5/2018

def sgram(x, N, H, wtype, is_abs = True, is_angle = False):
    # sgram(x,H,N,wtype) returns spectogram of a signal x given the hop size of H, windowing length of N, and
    # the type of window. A window function is first defined
    
    # INPUT #
    # x: audio file of which spectogram is to be computed
    # N: length of window function
    # H: hop size
    # wtype: type of window function used. OPTIONS : "Bartlett", "Hann", "Hamming", "Blackman", "Rectangular"
    # is_abs: to check if sgram returns absolute values of the spectogram obtained. Default set to True.
    # is_angle: to check if sgram returns phase values of the spectogram obtained. Default set to False.
    
    # OUTPUT #
    # specto_abs: absolute values of the spectogram obtained.
    # specto_phase: phase values of the spectogram obtained
    if wtype == "Rectangular":
        w = [1 for n in range(N)]
    elif wtype == "Bartlett":
        w = [2 * n / N if 0 <= n <= math.floor(N / 2) else 2 - 2 * n / N for n in range(N)]
    elif wtype == "Hann":
        w = [0.5 - 0.5 * np.cos(2 * math.pi * n / N) for n in range(N)]
    elif wtype == "Hamming":
        w = [0.54 - 0.46 * np.cos(2 * math.pi * n / N) for n in range(N)]
    elif wtype == "Blackman":
        w = [0.42 - 0.5 * np.cos(2 * math.pi * n / N) + 0.08 * np.cos(4 * math.pi * n / N) for n in range(N)]
    else:
        print('Windowing function must either be "Bartlett", "Hann", "Hamming", "Blackman" or "Rectangular"')

    # There exists a last time frame with shorter length than N if the cascaded windowing functions, together with hops
    # , do not fit exactly to the overall signal

    # Defining domain for time frame taps, n = 0,1,2,...,M-1 for t = t[n]
    M = (len(x) - N) // H + 1
    # Any signal cut that arises when the last sampled signal does not coincide with the end of the signal will be discarded
    M2 = N//2  # Defining domain for frequency bin frame, k = 0,1,2,....,M2-1 for f = f[k]
    if is_abs == True:
        specto_abs = [[] for i in range(M)]
    if is_angle == True:
        specto_angle = [[] for i in range(M)]
    if is_abs == True:
        for n in range(M):
            val = np.fft.fft([e1 * e2 for e1, e2 in zip(x[n * H:N + n * H], w)])
            specto_abs[n] = [abs(2*val_arg/N) ** 2 for val_arg in val[0:M2]]
    if is_angle == True:
        for n in range(M):
            val = np.fft.fft([e1 * e2 for e1, e2 in zip(x[n * H:N + n * H], w)])
            specto_angle[n] = [np.angle(val_arg) for val_arg in val[0:M2]]
    #     specto = [list(x) for x in zip(*specto)] # transposing the matrix so that the column represents time and the row represents frequency

    if is_abs == True and is_angle == False:
        return specto_abs
    if is_abs == False and is_angle == True:
        return specto_angle
    if is_abs == True and is_angle == True:
        return specto_abs, specto_angle


def sgram_lf(sgram,f_range):
    # sgram_lf converts spectogram of frequency basis into that of a pitch index for p = [0:127] given a frequency domain of f_range. (spectogram -> log-frequency spectogram)
    p_list = [x for x in range(128)]
    f_max = f_range[len(f_range)-1]

    if Fp(127.5) <= f_max:
        p_max = 127
    else:
        for p in range(len(p_list)):
            if Fp(p-0.5) <= f_max < Fp(p+0.5):
                p_max = p
    t_max_index, f_max_index = np.shape(sgram)
    sgram_log = [[0]*128 for i in range(t_max_index)]
    for n in range(t_max_index):
        for p in range(p_max+1):
            sgram_log[n][p] = sum([sgram[n][k] for k in range(f_max_index) if Fp(p-0.5) <= f_range[k] < Fp(p+0.5)])
    return sgram_log

def cgram(sgram_lff, normalization = "off", epsilon = 0):
    # cgram converts a log-frequency spectrum of pitch index basis into that of a chroma index for c = [0:11] given a pitch index (log-frequency spectogram -> chromatogram)
    t_max_index, p_max = np.shape(sgram_lff)
    chroma_list = [x for x in range(12)]
    sgram_chroma = [[0]*len(chroma_list) for n in range(t_max_index)]
    for n in range(t_max_index):
        for c in range(len(chroma_list)):
            sgram_chroma[n][c] = sum([sgram_lff[n][p] for p in range(p_max) if p % 12 == c])
    if normalization == "off":
        return sgram_chroma
    elif normalization == "on":
        for n in range(t_max_index):
            norm = np.sqrt([x^2 for x in sgram_chroma[n]])
            for c in range(len(chroma_list)):
                if np.abs(sgram_chroma) > epsilon:
                    sgram_chroma[n][c] /= norm
                else:
                    sgram_chroma[n][c] = 1/np.sqrt(12)
        return sgram_chroma

def Fp(p):
    # Fp(f) converts pitch index into corresponding frequency
    freq = 440*2**((p-69)/12)
    return freq

def transpose(x):
    # Transpose a given matrix
    transposed = [list(i) for i in zip(*x)]
    return transposed

def c_dist(x,y):
    # Computes cosine distance between vector x and y
    x_abs = np.sqrt(sum([v**2 for v in x])) #Modulus of x
    y_abs = np.sqrt(sum([v**2 for v in y])) #Modulus of y
    xy_inner =  sum([u*v for u, v in zip(x,y)])
    cos_dist = 1 - xy_inner/(x_abs*y_abs)
    return cos_dist

def dist(x,y):
    # Computes L2 distance between vector x and y
    cart_dist = np.sqrt(sum([(u-v)**2 for u, v in zip(x, y)]))
    return cart_dist

def dtw(X, Y, cost_type):
    # X = [[x_1],[x_2],...,[x_N]] in R^N x R^C  and Y = [[y_1],[y_2], ..., [y_M]] in R^M x R^C with x_i and y_j in R^C for i = 1,2,...,N and j = 1,2,...,M
    # are the matrices between which optimal warping path is to be determined
    # cost_type: type of the cost function that measure the difference between X and Y. Choose btw "L2" and "cosine_dist"
    # Step conditions are assumed to be of order 1: (1,0), (0,1), (1,1)
    
    # If X and Y are column vectors,
    if np.shape(X)[0] == np.size(X):
        N = np.shape(X)[0]
        M = np.shape(Y)[0]
        if cost_type == "L2":
            cost_matrix = [[np.abs(X[n]- Y[m]) for m in range(M)] for n in range(N)]
        elif cost_type == "cosine_dist":
            cost_matrix = [[c_dist(X[n], Y[m]) for m in range(M)] for n in range(N)]
        else:
            print("cost_type must either be 'L2' for L2 norm or 'cosine_dist' for cosine distance")
    else:
    # If X and Y are matrices
        N, C_X = np.shape(X)
        M, C_Y = np.shape(Y)
        if C_X != C_Y:
            print("Feature vectors between X and Y must have the same length")
        else:
            if cost_type == "L2":
                cost_matrix = [[dist(X[n], Y[m]) for m in range(M)] for n in range(N)]
            elif cost_type == "cosine_dist":
                cost_matrix = [[c_dist(X[n], Y[m]) for m in range(M)] for n in range(N)]
            else:
                print("cost_type must either be 'L2' for L2 norm or 'cosine_dist' for cosine distance")

    # Now, the accumulated cost matrix is predefined
    acc_cost_matrix = [[0]*M for n in range(N)]
    # With initialization
    acc_cost_matrix[0][0] = cost_matrix[0][0]
    for m in range(0, M):
        acc_cost_matrix[0][m] = sum(cost_matrix[0][0:m+1])
    transposed_cost = transpose(cost_matrix)
    for n in range(0, N):
        acc_cost_matrix[n][0] = sum(transposed_cost[0][0:n+1])
    # Let the main algorithm begin finding DTW cost
    for n in range(1, N):
        for m in range(1, M):
            acc_cost_matrix[n][m] = cost_matrix[n][m] + np.min([acc_cost_matrix[n-1][m-1], acc_cost_matrix[n-1][m], acc_cost_matrix[n][m-1]])
    # The optimal warping path is determined by backtracking
    index = [N-1, M-1]
    optimal_path = [index]
    while index != [0, 0]:
        if index[0] == 0:
            index = [0, index[1]-1]
        if index[1] == 0:
            index = [index[0]-1, 0]
        if index[0] != 0 and index[1] != 0:
            val1 = acc_cost_matrix[index[0]-1][index[1]-1]
            val2 = acc_cost_matrix[index[0]][index[1]-1]
            val3 = acc_cost_matrix[index[0]-1][index[1]]
            if val1 == min([val1, val2, val3]):
                index = [index[0]-1, index[1]-1]
            elif val2 == min([val1, val2, val3]):
                index = [index[0], index[1]-1]
            elif val3 == min([val1, val2, val3]):
                index = [index[0]-1, index[1]]
        optimal_path.append(index)

    return cost_matrix, acc_cost_matrix, optimal_path

def sinusoid_wave(tf,freq,fs):
    t = np.arange(0, tf, 1/fs)
    sig = np.cos(2*np.pi*freq*t)
    wavfile.write('/Users/user/Music/GarageBand/' + 'sinusoid_' + str(freq) + 'Hz' + str(tf) + 's.wav', fs, sig)

def three_harmonic_sum(tf,freq,fs):
    t = np.arange(0, tf, 1/fs)
    sig = 1/3*np.sin(2*np.pi*freq*t) + 1/3*np.sin(2*np.pi*2*freq*t) + 1/3*np.sin(2*np.pi*3*freq*t)
    wavfile.write('/Users/user/Music/GarageBand/' + '3harmonic_' + str(freq) + 'Hz' + str(tf) + 's.wav', fs, sig)

def onethree_harmonic_sum(tf,freq,fs):
    t = np.arange(0, tf, 1/fs)
    sig = 1/2*np.sin(2*np.pi*freq*t) + 1/2*np.sin(2*np.pi*3*freq*t)
    wavfile.write('/Users/user/Music/GarageBand/' + 'onethreeharmonic_' + str(freq) + 'Hz' + str(tf) + 's.wav', fs, sig)

def twothree_harmonic_sum(tf,freq,fs):
    t = np.arange(0, tf, 1/fs)
    sig = 1/2*np.sin(2*np.pi*2*freq*t) + 1/2*np.sin(2*np.pi*3*freq*t)
    wavfile.write('/Users/user/Music/GarageBand/' + 'twothreeharmonic_' + str(freq) + 'Hz' + str(tf) + 's.wav', fs, sig)

def steering_two_harmonic(tf,freq,fs):
    t = np.arange(0, tf, 1/fs)
    steering_t = [(1.1+0.2*t_temp/tf) for t_temp in t]
    pre_multiplied_t = [e1*e2 for e1, e2 in zip(steering_t, t)]
    sig1 = [1/2*np.sin(2*np.pi*freq*t_temp) for t_temp in t]
    sig2 = [1/2*np.sin(2*np.pi*freq*pre_multiplied_t_temp) for pre_multiplied_t_temp in pre_multiplied_t]
    sig = np.array([e1+e2 for e1, e2 in zip(sig1, sig2)])
    wavfile.write('/Users/user/Music/GarageBand/' + 'steering_two_tones' + str(freq) + 'Hz' + str(tf) + 's.wav', fs, sig)

def close_two_harmonic(tf,freq1,freq2,fs):
    t = np.arange(0, tf, 1/fs)
    sig = 1/2*np.sin(2*np.pi*freq1*t) + 1/2*np.sin(2*np.pi*3*freq2*t)
    wavfile.write('/Users/user/Music/GarageBand/' + 'close_two_harmonics' + str(freq1) + 'Hz+' + str(freq2) + 'Hz' + str(tf) + 's.wav', fs, sig)

def quicksort(s):
    if s == []:
        return []
    pivot_index = randint(0, len(s)-1)
    pivot = s[pivot_index]
    lower = quicksort([x for x in s if x < pivot])
    equal = [x for x in s if x == pivot]
    higher = quicksort([x for x in s if x > pivot])
    return lower+equal+higher

def power_test(fs, type, freq1=100,fc=100, bw=100):
    # Loudness comparison test with tones of a given frequency, freq1, as a test signal or a narrowband signal with
    # center frequency of fc and bandwidth of bw
    # fs : sampling frequency
    # type : "narrowband" or "tone"
    # freq1 : frequency of the pure tone
    # fc : center frequency of narrowband signal
    # bw : bandwidth of the narrowband signal
    power_proportion_db = [-25, -25, -20, -20, -15, -15, -10, -10, -5, -5, 0, 0, 5, 5, 10, 10, 15, 15, 20, 20]
    shuffle(power_proportion_db)
    power_proportion = [10**(x/20) for x in power_proportion_db]

    if type == "tone":
        t = np.arange(0, 5, 1 / fs)
        total_sig = []
        sig1 = [np.sin(2 * np.pi * freq1 * t[i]) if t[i] <= 1 else 0 for i in range(len(t))]
        for j in range(len(power_proportion_db)):
            sig2 = [power_proportion[j] * np.sin(2 * np.pi * freq1 * t[i]) if 1.5 < t[i] <= 2.5 else 0 for i in
                    range(len(t))]
            sig = [e1 + e2 for e1, e2 in zip(sig1, sig2)]
            total_sig += sig
        total_sig = np.array(total_sig)
        t_total_sig = np.linspace(0,100,len(list(total_sig)))
        wavfile.write('/Users/user/Music/GarageBand/' + 'tone_loudness_test_fc=' + str(freq1) + 'Hz' + '.wav', fs, total_sig)

        return power_proportion_db, total_sig, t_total_sig

    elif type == "narrowband":
        t_duration = np.arange(0, 1, 1/fs)
        N = len(t_duration)
        f_domain = np.linspace(0, fs, N)
        f_u = 1/2*(bw + np.sqrt((bw)**2 + 4*fc**2))
        f_l = 1/2*(-bw + np.sqrt((bw)**2 + 4*fc**2))
        f_u_index = []
        f_l_index = []
        for i in range(len(f_domain)):
            if np.abs(f_domain[i]-f_u) < 1:
                f_u_index.append(i)
            elif np.abs(f_domain[i]-f_l) < 1:
                f_l_index.append(i)

        f_u_index = f_u_index[len(f_u_index)//2]
        f_l_index = f_l_index[len(f_l_index)//2]
        f_spectrum = np.zeros((N,), dtype=complex)
        f_spectrum[f_l_index:f_u_index] = 1/3/np.sqrt(f_u-f_l)*np.exp(1j*np.random.uniform(0, 2*np.pi, (f_u_index - f_l_index,)))
        narrow_band_sig = (np.fft.ifft(f_spectrum)*fs).real
        total_sig = []

        for j in range(len(power_proportion_db)):
            sig2 = [power_proportion[j] * narrow_band_sig[i] for i in range(len(narrow_band_sig))]
            zero_padding1 = [0 for x in range(int(0.5*np.floor(len(t_duration))))]
            zero_padding2 = [0 for x in range(int(4.5*np.floor(len(t_duration))))]
            total_sig += list(narrow_band_sig) + list(zero_padding1) + list(sig2) + list(zero_padding2)
        zero_padding3 = [0 for i in range(3*len(t_duration))]
        t_total_sig = np.linspace(0, 143, len(zero_padding3) + len(power_proportion_db)*(len(list(narrow_band_sig) + list(zero_padding1) + list(sig2) + list(zero_padding2))))
        total_sig = np.array(zero_padding3 + [x for x in total_sig])
        wavfile.write('/Users/user/Music/GarageBand/' + 'narrowband_loudness_test_1_fc=' +  str(fc) + 'Hz' + 'bw=' + str(bw) + '.wav', fs, total_sig)
        return power_proportion_db, f_domain, f_spectrum, total_sig, narrow_band_sig, sig2, zero_padding1, zero_padding2, t_total_sig

def regression(x,y):
    # returns linear regression of the data sat x and y
    # Input: x = independent variable, y = data set
    # Output: linear regression coefficient a, b which minimizes error y-(ax+b)
    vec_len = len(x)
    y = np.transpose(np.array(y))
    A = np.array([[x[i], 1] for i in range(vec_len)])
    ps_inv = np.matmul(np.linalg.inv(np.matmul(np.transpose(A), A)), np.transpose(A)) #Pseudo_inverse of A
    coeff = np.matmul(ps_inv, y)

    return coeff[0], coeff[1]

def rectify(x):
    # Returns only positive values of x while removing the nagative values
    rectified_x = (x+abs(x))/2
    return rectified_x

def wrap(x):
    # Returns principal value (-pi: pi] of phase x
    if abs(x) <= np.pi:
        return x
    elif abs(x) > np.pi:
        if x < 0:
            while abs(x) >= np.pi:
                x += 2*np.pi
        elif x > 0:
            while abs(x) > np.pi:
                x -= 2*np.pi
    return x

def tempo_novelty(data, M_window, H_window, wtype, novelty_type, gamma = 1, enhanced = True):
    # Returns tempo-related novelty functions.
    # INPUT #
    # data: audio file of which tempo is to be determined.
    # M_window: length of the window function
    # H_window: hop size for the window function
    # wtype: window type. Can be chosen between "Rectangular", "Bartlett", "Hann", "Hamming", "Blackman"
    # novelty_type: type of the novelty function to be used. Can be chosen between "energy", "energy_log", "spectral", "phase", "complex"
    # gamma: compression constant for log-compression of spectogram which is required to compute spectral novelty function. Default set to 1.
    # enhanced: check to enhance the spectral novelty function by suppressing undesired peaks. Default set to True.
    if wtype == "Rectangular":
        w_novelty = [1 for n in range(M_window)]
    elif wtype == "Bartlett":
        w = [2 * n / M_window if 0 <= n <= math.floor(M_window / 2) else 2 - 2 * n / M_window for n in range(M_window)]
    elif wtype == "Hann":
        w = [0.5 - 0.5 * np.cos(2 * math.pi * n / M_window) for n in range(M_window)]
    elif wtype == "Hamming":
        w = [0.54 - 0.46 * np.cos(2 * math.pi * n / M_window) for n in range(M_window)]
    elif wtype == "Blackman":
        w = [0.42 - 0.5 * np.cos(2 * math.pi * n / M_window) + 0.08 * np.cos(4 * math.pi * n / M_window) for n in range(M_window)]
    else:
        print('Windowing function must either be "Bartlett", "Hann", "Hamming", "Blackman" or "Rectangular"')


    half_M_window = M_window // 2
    half_zero_window = [0 for i in range(half_M_window)]
    data = half_zero_window + data
    M = (len(data) - M_window) // H_window + 1 # Number of taps in the novelty function


    if novelty_type == "energy":
        E_x = [sum([np.abs((x*y))**2 for x,y in zip(data[n*H_window: n*H_window + M_window], w)]) for n in range(M)]

        novelty_func = [rectify(E_x[n+1] - E_x[n]) for n in range(M-1)]
        max_novelty = max(novelty_func)
        novelty_func = [x/max_novelty for x in novelty_func]
        return M, novelty_func + [0]

    elif novelty_type == "energy_log":
        E_x = [sum([np.abs((x*y)**2) for x,y in zip(data[n*H_window: n*H_window + M_window], w)]) for n in range(M)]
        novelty_func = [rectify(np.log((E_x[n+1]+0.001)/(E_x[n]+0.001))) for n in range(M-1)]

        return M, novelty_func + [0]

    elif novelty_type == "spectral":
        specto_abs = sgram(data, M_window, H_window, wtype)
        time_index_max, freq_index_max = np.shape(specto_abs)
        specto_log = [[np.log(1+gamma*abs(specto_abs[n][k])) for k in range(freq_index_max)] for n in range(time_index_max)]
        novelty_func = [ sum([rectify(x - y) for x,y in zip(specto_log[n+1],specto_log[n])]) for n in range(time_index_max - 1) ]
        max_novelty = max(novelty_func)
        novelty_func = [x / max_novelty for x in novelty_func]
        if enhanced == True:
            window_size = int(np.floor(M_window/len(data) * time_index_max))
            half_window_size_lower = int(window_size//2)
            half_window_size_upper = int(window_size//2 + window_size%2)
            zero_padded_novelty_func = [0 for i in range(half_window_size_lower)] + novelty_func + [0 for i in range(half_window_size_upper)]
            norm_novelty = [ 1/window_size*sum([zero_padded_novelty_func[n + m] for m in range(window_size)]) for n in range(len(novelty_func))]
            enhanced_novelty_func = [rectify(novelty_func[n] - norm_novelty[n]) for n in range(time_index_max - 1)]
            max_enhanced_novelty = max(enhanced_novelty_func)
            enhanced_novelty_func = [x / max_enhanced_novelty for x in enhanced_novelty_func]

            return M, enhanced_novelty_func + [0]
        else:
            
            return M, novelty_func + [0]

    elif novelty_type == "phase":
        specto_phase = sgram(data, M_window, H_window, wtype, is_abs = False, is_angle = True)
        time_index_max, freq_index_max = np.shape(specto_phase)
        phase_diff = [[specto_phase[n+1][k] - 2 * specto_phase[n][k] + specto_phase[n-1][k] for k in range(freq_index_max)] for n in range(1, time_index_max-1)]
        novelty_phase = [ 1/np.pi*wrap(abs(sum(phase_diff[n]))) for n in range(time_index_max-2) ]
        max_novelty = max(novelty_phase)
        novelty_phase = [x / max_novelty for x in novelty_phase]
        return M, [0] + novelty_phase + [0]

    elif novelty_type == "complex":
        specto_abs, specto_phase = sgram(data, M_window, H_window, wtype, is_abs = True, is_angle = True)
        time_index_max, freq_index_max = np.shape(specto_phase)
        phase_diff = [[specto_phase[n][k] - specto_phase[n-1][k] if 0 < n < time_index_max else 0 for k in range(freq_index_max)] for n in range(time_index_max)]
        chi_hat = [[ specto_abs[n-1][k] * np.exp(2*np.pi*1j*(specto_phase[n-1][k] + phase_diff[n-1][k])) if 1 < n < time_index_max else 0 for k in range(freq_index_max)] for n in range(time_index_max)]
        chi_dot = [[abs(chi_hat[n][k] - specto_abs[n][k]) if 1 < n < time_index_max else 0 for k in range(freq_index_max)] for n in range(time_index_max)]
        chi_plus = [[0 for k in range(freq_index_max)] for n in range(time_index_max)]
        for n in range(time_index_max):
            if n < 2:
                for k in range(freq_index_max):
                    chi_plus[n][k] = 0
            else:
                for k in range(freq_index_max):
                    if specto_abs[n][k] > specto_abs[n - 1][k]:
                        chi_plus[n][k] = chi_dot[n][k]
                    else:
                        chi_plus[n][k] = 0
        complex_novelty = [sum(chi_plus[n]) for n in range(time_index_max)]
        max_novelty = max(complex_novelty)
        novelty_func = [x / max_novelty for x in complex_novelty]
        return M, novelty_func

