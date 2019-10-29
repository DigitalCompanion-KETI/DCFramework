## STFT; Short Time Fourier Transform

STFT 예제는 `*.wav`파일을 전송하고 결과값으로 STFT; Short Time Fourier Transform을 거친 영상을 결과값으로 받는 예제이다. 본 예제에서 사용할 음성 파일은 [링크](https://www.loc.gov/item/00694083/)에서 다운로드 할 수 있다.

함수의 구동 순서는 아래와 같다.

- Client
  - `*.wav`파일을 전송
- DCF
  - `*.wav`파일 수신
  - STFT 알고리즘 통과
  - STFT 결과 이미지를 반환

### Init

stft라는 함수를 생성한다

```bash
$ dcf-cli function init stft
>>>
Directory: stft is created.
Function handler created in directory: stft
Rewrite the function handler code in stft/src directory
Config file written: config.yaml
```

### Write function

#### Dockerfile

필요한 패키지 설치를 위해서 Dockerfile에서 apt-get으로 `libfreetype6-dev`, `libpng-dev` 패키지를 설치한다

```dockerfile
# 2-Stage
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

# Arguments for Nvidia-Docker
ARG CUDA_VERSION
ARG CUDNN_VERSION
ARG CUDA_VERSION_BACKUP

RUN echo "/usr/local/cuda-${CUDA_VERSION_BACKUP}/extras/CUPTI/lib64" > /etc/ld.so.conf.d/cupti.conf

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    tar \
    libgomp1 \
        python-setuptools \
        libgtk2.0-dev \
        libcudnn7=${CUDNN_VERSION}-1+cuda${CUDA_VERSION_BACKUP} \
        python${PYTHON_VERSION} \
        python3-dev \
        python3-numpy \
        python3-pip \
        python3-tk \
        cmake \
        unzip \
        pkg-config \
        libfreetype6-dev \
        libpng-dev \
    ${ADDITIONAL_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*
```

#### handler.py

아래와 같이 `handler.py` 파일을 작성한다

```python
# -*- coding: utf-8 -*-

import io
import os
import base64
import numpy as np
import matplotlib as m
from PIL import Image

# 함수에서는 디스플레이가 불가능하니, 디스플레이 옵션을 사용하지 않게 변경해준다
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    m.use('Agg')

import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks


""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)   
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)    

""" scale frequency axis logarithmically """    
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):        
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs

""" plot spectrogram"""
def plotstft(audiopath, binsize=2**10, plotpath=None, colormap="jet"):
    # 받은 audio raw bytes파일을 wav.read함수를 이용하여 load
    samplerate, samples = wav.read(io.BytesIO(audiopath))

    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)

    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    fig = plt.figure(figsize=(15, 7.5))
    plt.axis("off")
    plt.subplots_adjust(top = 1, bottom = 0, right=1, left=0, hspace=0, wspace=0)
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.colorbar()
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    image = np.fromstring(fig.canvas.tostring_rgb() , dtype=np.uint8)
    image = image.reshape((h, w, 3))
    fig.clear()
    m.pyplot.close()

    return image

class Handler:
    def __init__(self):
        pass

    def __call__(self, req):
        # 입력값을 받아, 그대로 plotstft 함수로 전달
        rawwav = req.input
        image = plotstft(rawwav)

        # numpy image를 PIL.Image객체로 변환
        im = Image.fromarray(image.astype("uint8"))
        rawBytes = io.BytesIO()

        # PIL Image를 PNG 포맷으로 변환
        im.save(rawBytes, "PNG")
        rawBytes.seek(0)  # return to the start of the file

        # PNG포맷으로 변환된 이미지를 base64 문자열로 인코딩
        base64Str = base64.b64encode(rawBytes.read()).decode("utf-8")
        return base64Str    
```

#### requirements.txt

필요한 패키지 파일을 requirements.txt에 명시한다

```bash
numpy
matplotlib
scipy
pillow
```

### Build function

작성한 함수를 빌드한다

```bash
$ dcf-cli function build -f config.yaml -v
>>>
Building function (stft) ...
Sending build context to Docker daemon  325.8MB
Step 1/47 : ARG ADDITIONAL_PACKAGE
Step 2/47 : ARG REGISTRY
...
```

### Test Function

빌드한 함수를 테스트한다.

```bash
$ cat 57007r.wav | dcf function call send-wav
>>
iVBORw......XqWY4HTbaj+3/xNf634UHr7xLvnzb2f/zL8Cdf6j14yamriQAAAABJRU5ErkJggg==
```

출력된 결과를 Base64 디코딩하여 이미지를 확인해보면 아래와 같은 이미지를 확인할 수 있다. Base64 디코딩은 [여기](https://codebeautify.org/base64-to-image-converter)에서 할 수 있다.

![stft](https://user-images.githubusercontent.com/13328380/48114112-171a1f00-e2a1-11e8-9347-fd31f7324577.png)
