# End-to-end speech dereverberation using acoustic intensity based on deep neural networks

This is an implementation of my [master thesis](https://chungks603.github.io/assets/master-thesis.pdf).



## Feature Extraction

Because the code for the feature extraction is considered my lab's assets, it won't be uploaded.

The procedure can be explained in brief:

1. Transform the 32-channel room impulse responses (RIRs) into the **spherical harmonic domain (SHD) signals** with real spherical Fourier transform basis.
2. Calculate **reverberant SHD signals** from speech sources, the modified inverse of the rigid sphere modal strength $b^{-1}_n(kr)$, and the result of 1.
3. Calculate directional feature, instantaneous intensity vector (**IIV** in thesis, **IV** in python code).



## Model

The model is a modified version of the Wave-U-Net, https://github.com/f90/Wave-U-Net-Pytorch.

Refer to the `waveunet` directory.



## train_test.py

`train_test.py` is the code for model training or evaluation.



## Evaluation Metric

Source codes of PESQ, STOI, and fwSegSNR are in the `matlab_lib` directory.

SegSNR is implemented in the `audio_utils.py`
