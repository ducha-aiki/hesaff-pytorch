This is implementation of Hessian-Affine local feature detector. 
It is heavily based on Michal Perdoch C++ implementation https://github.com/perdoch/hesaff

pytaff - current implementation
hesamp - Michal Perdoch C++ one.

![average](/img/repeatability.png)

There are several differences:
    1) No SIFT description, the output is image patches. If one needs to, patches could be feed into [PyTorchSIFT](https://github.com/ducha-aiki/pytorch-sift)
    2) Subpixel precision is done via "center-of-responce-mass" inspired by [LIFT](https://arxiv.org/abs/1603.09114) paper, instead of original iterative quadratic fitting
    3) Instead of setting threshold to control number of detection, this implementation simply outputs top-K local extreme points. 
    
If you use this code for academic purposes, please cite the following paper:

    @article{tbd}