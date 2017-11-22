This is implementation of Hessian-Affine local feature detector. 
It is heavily based on Michal Perdoch C++ implementation https://github.com/perdoch/hesaff

pytaff - current implementation
hesamp - Michal Perdoch C++ one.

![average](/img/repeatability.png)

There are several differences:

1) No SIFT description, the output is image patches. If one needs to, patches could be feed into [PyTorchSIFT](https://github.com/ducha-aiki/pytorch-sift)
    
2) Subpixel precision is done via "center-of-responce-mass" inspired by [LIFT](https://arxiv.org/abs/1603.09114) paper, instead of original iterative quadratic fitting
    
3) Instead of setting threshold to control number of detection, this implementation simply outputs top-K local extreme points. 
    
You also might be interested in [HesAffNet](https://github.com/ducha-aiki/affnet), which gives significantly better results because of learned affine shape estimation procedure.

If you use this code for academic purposes, please cite the following paper:

```
@article{AffNet2017,
 author = {Dmytro Mishkin, Filip Radenovic, Jiri Matas},
    title = "{Learning Discriminative Affine Regions via Discriminability}",
     year = 2017,
    month = nov}
```
