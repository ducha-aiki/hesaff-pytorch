import torch
import torch.nn as nn
import numpy as np
import sys
import time
from PIL import Image
from torch.autograd import Variable

from SparseImgRepresenter import ScaleSpaceAffinePatchExtractor
from LAF import denormalizeLAFs, LAFs2ell
from Utils import line_prepender

USE_CUDA = False

try:
    input_img_fname = sys.argv[1]
    output_fname = sys.argv[2]
    nfeats = int(sys.argv[3])
except:
    print "Wrong input format. Try python extract_features_oxaff.py graf1.ppm out.txt 2000"
    sys.exit(1)

img = Image.open(input_img_fname).convert('RGB')
img = np.mean(np.array(img), axis = 2)

var_image = torch.autograd.Variable(torch.from_numpy(img.astype(np.float32)))
var_image = var_image.view(1, 1, var_image.size(0),var_image.size(1))
    
HA = ScaleSpaceAffinePatchExtractor(mrSize = 5.192, num_features = nfeats, border = 5, num_Baum_iters = 16)

if USE_CUDA:
    HA = HA.cuda()
    var_image = var_image.cuda()

LAFs, resp  = HA(var_image)
ells = LAFs2ell(LAFs.data.cpu().numpy())

np.savetxt(output_fname, ells, delimiter=' ', fmt='%10.10f')
line_prepender(output_fname, str(len(ells)))
line_prepender(output_fname, '1.0')