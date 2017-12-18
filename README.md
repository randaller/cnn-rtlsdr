# CNN-rtlsdr
Deep learning signal classification using rtl-sdr dongle.

### TEST WITH PRETRAINED MODEL

Unpack software archive into some folder, e.g. C:\rtlsdr

Go to https://www.anaconda.com/download/ and choose Python 3.6 version, 64-Bit Graphical Installer
or download directly: https://repo.continuum.io/archive/Anaconda3-5.0.1-Windows-x86_64.exe

If you do not have modern NVIDIA graphics card, remove the following line from requirements.txt file: tensorflow-gpu==1.4.0

Run anaconda prompt, change dir to C:\rtlsdr, then run:
```
pip install -r requirements.txt
```

Only for CUDA version of Tensorflow, if you have installed CPU version, skip these steps:
	
	1. Download and install CUDA 8 Toolkit: https://developer.nvidia.com/cuda-80-ga2-download-archive

	2. Download CUDNN for Toolkit 8. https://developer.nvidia.com/cudnn
	Extract file cudnn64_6.dll from zip into C:\Windows folder.

Last step is to copy 2 files from x64!!! osmocom rtl-sdr drivers: https://osmocom.org/attachments/download/2242/RelWithDebInfo.zip

Copy these [rtl-sdr-release/x64/]: rtlsdr.dll & libusb-1.0.dll into C:\Windows folder.

Reboot your system.

Now open your anaconda prompt again, change folder to C:\rtlsdr and run:
```
python predict_scan.py
```
to scan entire band and predict signal types , or the full version scan:
```
python predict_scan.py --start 85000000 --stop 108000000 --step 50000 --gain 20 --ppm 56 --threshold 0.9955
```

Some help also available:
```
python predict_scan.py --help
```

### TRAIN YOUR OWN DATA

To train your own model, edit the file [prepare_data.py] to set own frequencies and ppm level (lines 40-41).

Then to obtain some samples run:
```
python prepare_data.py
```

Now do not forget to move about 20% of samples from /training_data/***/ folders to their corresponding folders in /testing_data/***/

Delete unnecessary folders under [/testing_data] and [/training_data] as they are responsible for classificator.
E.g., if you want to train only WFM and OTHER classes, delete everything, except of:
- /training_data/wfm/
- /training_data/other/
- /testing_data/wfm/
- /testing_data/other/

It is better to obtain different samples of signals at different frequencies, gain levels etc.
Instead of recording, you may prefer to download mine train_data.zip file with samples here.
Delete [/training_data] & [/testing_data] folders and unpack zip file.
	
Finally, we may now run training (of course, we are still inside anaconda prompt):
```
python train.py
```

Best decision is to stop the training [ctrl+c], when validation loss become 0.04 - 0.01 or below.