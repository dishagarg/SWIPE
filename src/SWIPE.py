
# coding: utf-8

# In[ ]:

# RECORD WORKING

import math
from os import listdir, getcwd
from pylab import *
import matplotlib.pyplot as plt
# from bqplot import pyplot as pl
import numpy as np
from numpy import matlib
from scipy.io import wavfile
from scipy import signal
from scipy import interpolate

WAVE_OUTPUT_FILENAME = 'Audio.wav'  # Path to an audio

#********************SWIPE-pitch extraction**************************


def swipep(x, fs, plim, dt, sTHR):
    if not plim:
        plim = [30, 5000]
    if not dt:
        dt = 0.01
    dlog2p = 1.0/96.0
    dERBs = 0.1
    if not sTHR:
        sTHR = -float('Inf')

    t = np.arange(0, len(x)/float(fs), dt)    # Times

    dc = 4   # Hop size (in cycles)

    K = 2   # Parameter k for Hann window
    # Define pitch candidates
    log2pc = np.arange(np.log2(plim[0]), np.log2(plim[len(plim)-1]), dlog2p)
    pc = np.power(2, log2pc)

    S = np.zeros(shape=(len(pc), len(t)))   # Pitch strength matrix

    # Determine P2-WSs
    logWs = np.round_(np.log2(np.multiply(4*K, (np.divide(float(fs), plim)))))
    ws = np.power(2, np.arange(logWs[1-1], logWs[2-1]-1, -1))   # P2-WSs
    pO = 4*K * np.divide(fs, ws)   # Optimal pitches for P2-WSs
    # Determine window sizes used by each pitch candidate
    d = 1 + log2pc - np.log2(np.multiply(4*K, (np.divide(fs, ws[1-1]))))
    # Create ERBs spaced frequencies (in Hertz)
    fERBs = erbs2hz(np.arange(hz2erbs(pc[1-1]/4), hz2erbs(fs/2), dERBs))

    for i in range(0, len(ws)):
        # for i in range(0, 1):
        dn = round(dc * fs / pO[i])  # Hop size (in samples)
        # Zero pad signal
        will = np.zeros((ws[i]/2, 1))
        learn = np.reshape(x, -1, order='F')[:, np.newaxis]
        mir = np.zeros((dn + ws[i]/2, 1))
        xzp = np.vstack((will, learn, mir))
        xk = np.reshape(xzp, len(xzp), order='F')
        # Compute spectrum
        w = np.hanning(ws[i])  # Hann window
        o = max(0, round(ws[i] - dn))  # Window overlap
        [X, f, ti, im] = plt.specgram(xk, NFFT=int(ws[i]), Fs=fs, window=w, noverlap=int(o))

        # Interpolate at equidistant ERBs steps
        f = np.array(f)
        X1 = np.transpose(X)

        ip = interpolate.interp1d(f, X1, kind='linear')(fERBs[:, np.newaxis])
        interpol = ip.transpose(2, 0, 1).reshape(-1, ip.shape[1])
        interpol1 = np.transpose(interpol)
        M = np.maximum(0, interpol1)  # Magnitude
        L = np.sqrt(M)  # Loudness
        # Select candidates that use this window size
        if i == (len(ws)-1):
            j = find(d - (i+1) > -1)
            k = find(d[j] - (i+1) < 0)
        elif i == 0:
            j = find(d - (i+1) < 1)
            k = find(d[j] - (i+1) > 0)
        else:
            j = find(abs(d - (i+1)) < 1)
            k1 = np.arange(0, len(j))  # transpose added by KG
            k = np.transpose(k1)
        Si = pitchStrengthAllCandidates(fERBs, L, pc[j])
        # Interpolate at desired times
        if Si.shape[1] > 1:
            tf = []
            tf = ti.tolist()
            tf.insert(0, 0)
            del tf[-1]
            ti = np.asarray(tf)
            Si = interpolate.interp1d(ti, Si, 'linear', fill_value=nan)(t)
        else:
            Si = matlib.repmat(float('NaN'), len(Si), len(t))
        lambda1 = d[j[k]] - (i+1)
        mu = ones(size(j))
        mu[k] = 1 - abs(lambda1)
        S[j, :] = S[j, :] + np.multiply(((np.kron(np.ones((Si.shape[1], 1)), mu)).transpose()), Si)

    # Fine-tune the pitch using parabolic interpolation
    p = np.empty((Si.shape[1],))
    p[:] = np.NAN
    s = np.empty((Si.shape[1],))
    s[:] = np.NAN
    for j in range(0, Si.shape[1]):
        s[j] = (S[:, j]).max(0)
        i = np.argmax(S[:, j])
        if s[j] < sTHR:
            continue
        if i == 0:
            p[j] = pc[0]
        elif i == len(pc)-1:
            p[j] = pc[0]
        else:
            I = np.arange(i-1, i+2)
            tc = np.divide(1, pc[I])
            # print "pc[I]", pc[I]
            # print "tc", tc
            ntc = ((tc/tc[1]) - 1) * 2*pi
            # print "S[I,j]: ", shape(S[I,j])
            # with warnings.catch_warnings():
            # warnings.filterwarnings('error')
            # try:
            c = polyfit(ntc, (S[I, j]), 2)
            # print "c: ", c
            ftc = np.divide(1, np.power(2, np.arange(np.log2(pc[I[0]]), np.log2(pc[I[2]]), 0.0013021)))
            nftc = ((ftc/tc[1]) - 1) * 2*pi
            s[j] = (polyval(c, nftc)).max(0)
            k = np.argmax(polyval(c, nftc))
            # except np.RankWarning:
            # print ("not enough data")
            p[j] = 2 ** (np.log2(pc[I[0]]) + (k-1)/768)
    p[np.isnan(s)-1] = float('NaN')  # added by KG for 0s
    return p, t, s

def pitchStrengthAllCandidates(f, L, pc):
    # Normalize loudness
    # warning off MATLAB:divideByZero
    hh = np.sum(np.multiply(L, L), axis=0)
    ff = (hh[:, np.newaxis]).transpose()
    sq = np.sqrt(ff)

    gh = matlib.repmat(sq, len(L), 1)
    L = np.divide(L, gh)
    S = np.zeros((len(pc), len(L[0])))
    for j in range(0, (len(pc))-1):
        S[j, :] = pitchStrengthOneCandidate(f, L, pc[j])
    return S

numArr = []

def is_prime(n):
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True
def primeArr(n):
    for num in range(1, n+2):
        if is_prime(num):
            numArr.append(num)
    jg = (np.expand_dims(numArr, axis=1)).transpose()
    return numArr

# Normalize the square root of spectrum "L" by applying normalized cosine kernal decaying as 1/sqrt(f)
def pitchStrengthOneCandidate(f, L, pc):
    n = fix(f[-1]/pc - 0.75)
    k = np.zeros(size(f))
    q = f / pc
    for i in (primeArr(int(n))):
        # print "i is:",i
        a = abs(q - i)
        p = a < .25
        k[find(p)] = np.cos(2*math.pi * q[find(p)])
        v = np.logical_and(.25 < a, a < .75)
        pl = np.cos(2*np.pi * q[find(v)]) / 2
        k[find(v)] = np.cos(2*np.pi * q[find(v)]) / 2

    ff = np.divide(1, f)

    k = (k*np.sqrt(ff))
    k = k / norm(k[k > 0.0])
    S = np.dot((k[:, np.newaxis]).transpose(), L)
    return S

def hz2erbs(hz):
    erbs = 21.4 * np.log10(1 + hz/229)
    return erbs

def erbs2hz(erbs):
    hz = (np.power(10, np.divide(erbs, 21.4)) - 1) * 229
    return hz

def swipe(audioPath):
    print("Swipe running", audioPath)
    fs, x = wavfile.read(audioPath)
    np.seterr(divide='ignore', invalid='ignore')
    p, t, s = swipep(x, fs, [100, 600], 0.001, 0.3)
    print("Pitches: ", p)
    fig = plt.figure()
    plt.plot(p)
    fig.savefig('hummed.png')
    plt.show()  # show in a window of contour on UI

#***************************************CALL function**************************
swipe(WAVE_OUTPUT_FILENAME)
