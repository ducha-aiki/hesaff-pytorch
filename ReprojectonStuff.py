import torch
from torch.autograd import Variable
import numpy as np
from LAF import rectifyAffineTransformationUpIsUp
from Utils import zeros_like
def distance_matrix_vector(anchor, positive):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    d1_sq = torch.sum(anchor * anchor, dim=1)
    d2_sq = torch.sum(positive * positive, dim=1)
    eps = 1e-6
    return torch.sqrt(torch.abs((d1_sq.expand(positive.size(0), anchor.size(0)) +
                       torch.t(d2_sq.expand(anchor.size(0), positive.size(0)))
                      - 2.0 * torch.bmm(positive.unsqueeze(0), torch.t(anchor).unsqueeze(0)).squeeze(0))+eps))

def LAFs_to_H_frames(aff_pts):
    H3_x = torch.Tensor([0, 0, 1 ]).unsqueeze(0).unsqueeze(0).expand_as(aff_pts[:,0:1,:]);
    if aff_pts.is_cuda:
        H3_x = H3_x.cuda()
    H3_x = torch.autograd.Variable(H3_x)
    return torch.cat([aff_pts, H3_x], dim = 1)


def linH(H, x, y):
    assert x.size(0) == y.size(0)
    A = torch.zeros(x.size(0),2,2)
    if x.is_cuda:
        A = A.cuda()
    A = Variable(A)
    den = x * H[2,0] + y * H[2,1] + H[2,2]
    num1_densq = (x*H[0,0] + y*H[0,1] + H[0,2]) / (den*den)
    num2_densq = (x*H[1,0] + y*H[1,1] + H[1,2]) / (den*den)
    A[:,0,0] = H[0,0]/den - num1_densq * H[2,0]
    A[:,0,1] = H[0,1]/den - num1_densq * H[2,1]
    A[:,1,0] = H[1,0]/den - num2_densq * H[2,0]
    A[:,1,1] = H[1,1]/den - num2_densq * H[2,1]
    return A

def reprojectLAFs(LAFs1, H1to2, return_LHFs = False):
    LHF1 = LAFs_to_H_frames(LAFs1)
    xy1 = torch.bmm(H1to2.expand(LHF1.size(0),3,3), LHF1[:,:,2:])
    xy1 = xy1 / xy1[:,2:,:].expand(xy1.size(0), 3, 1)
    As  = linH(H1to2, LAFs1[:,0,2], LAFs1[:,1,2])
    AF = torch.bmm(As, LHF1[:,0:2,0:2])
    if return_LHFs:
        return LAFs_to_H_frames(torch.cat([AF, xy1[:,:2,:]], dim = 2))
    return torch.cat([AF, xy1[:,:2,:]], dim = 2)
    
def inverseLHFs(LHFs):
    LHF1_inv =torch.zeros(LHFs.size())
    if LHFs.is_cuda:
        LHF1_inv = LHF1_inv.cuda()
    LHF1_inv = torch.autograd.Variable(LHF1_inv);
    for i in range(LHF1_inv.size(0)):
        LHF1_inv[i,:,:] = LHFs[i,:,:].inverse()
    return LHF1_inv

def reproject_to_canonical_Frob_batched(LHF1_inv, LHF2, batch_size = 2, skip_center = False):
    out = torch.zeros((LHF1_inv.size(0), LHF2.size(0)))
    eye1 = torch.eye(3)
    if LHF1_inv.is_cuda:
        out = out.cuda()
        eye1 = eye1.cuda()
    eye1 =  torch.autograd.Variable(eye1)
    out =  torch.autograd.Variable(out)
    len1 = LHF1_inv.size(0)
    len2 = LHF2.size(0)
    n_batches = int(np.floor(len1 / batch_size) + 1);
    for b_idx in range(n_batches):
        #print b_idx
        start = b_idx * batch_size;
        fin = min((b_idx+1) * batch_size, len1)
        current_bs = fin - start
        if current_bs == 0:
            break
        should_be_eyes = torch.bmm(LHF1_inv[start:fin, :, :].unsqueeze(0).expand(len2,current_bs, 3, 3).contiguous().view(-1,3,3),
                                   LHF2.unsqueeze(1).expand(len2,current_bs, 3,3).contiguous().view(-1,3,3))
        if skip_center:
            out[start:fin, :] = torch.sum(((should_be_eyes - eye1.unsqueeze(0).expand_as(should_be_eyes))**2)[:,:2,:2] , dim=1).sum(dim = 1).view(current_bs, len2)
        else:
            out[start:fin, :] = torch.sum((should_be_eyes - eye1.unsqueeze(0).expand_as(should_be_eyes))**2 , dim=1).sum(dim = 1).view(current_bs, len2)
    return out

def get_GT_correspondence_indexes(LAFs1, LAFs2, H1to2, dist_threshold = 4):    
    LHF2_in_1_pre = reprojectLAFs(LAFs2, torch.inverse(H1to2), True)
    just_centers1 = LAFs1[:,:,2];
    just_centers2_repr_to_1 = LHF2_in_1_pre[:,0:2,2];
    
    dist  = distance_matrix_vector(just_centers2_repr_to_1, just_centers1)
    min_dist, idxs_in_2 = torch.min(dist,1)
    plain_indxs_in1 = torch.autograd.Variable(torch.arange(0, idxs_in_2.size(0)),requires_grad = False)
    if LAFs1.is_cuda:
        plain_indxs_in1 = plain_indxs_in1.cuda()
    mask =  min_dist <= dist_threshold
    return min_dist[mask], plain_indxs_in1[mask], idxs_in_2[mask]

def get_GT_correspondence_indexes_Fro(LAFs1,LAFs2, H1to2, dist_threshold = 4,
                                      skip_center_in_Fro = False):
    LHF2_in_1_pre = reprojectLAFs(LAFs2, torch.inverse(H1to2), True)
    LHF1_inv = inverseLHFs(LAFs_to_H_frames(LAFs1))
    frob_norm_dist = reproject_to_canonical_Frob_batched(LHF1_inv, LHF2_in_1_pre, batch_size = 2, skip_center = skip_center_in_Fro)
    min_dist, idxs_in_2 = torch.min(frob_norm_dist,1)
    plain_indxs_in1 = torch.autograd.Variable(torch.arange(0, idxs_in_2.size(0)), requires_grad = False)
    if LAFs1.is_cuda:
        plain_indxs_in1 = plain_indxs_in1.cuda()
    #print min_dist.min(), min_dist.max(), min_dist.mean()
    mask =  min_dist <= dist_threshold
    return min_dist[mask], plain_indxs_in1[mask], idxs_in_2[mask]

def get_GT_correspondence_indexes_Fro_and_center(LAFs1,LAFs2, H1to2, dist_threshold = 4, center_dist_th = 2.0,
                                                 skip_center_in_Fro = False, do_up_is_up = False, return_LAF2_in_1 = False):
    LHF2_in_1_pre = reprojectLAFs(LAFs2, torch.inverse(H1to2), True)
    if do_up_is_up:
        sc = torch.sqrt(LHF2_in_1_pre[:,0,0] * LHF2_in_1_pre[:,1,1] - LHF2_in_1_pre[:,1,0] * LHF2_in_1_pre[:,0,1]).unsqueeze(-1).unsqueeze(-1).expand(LHF2_in_1_pre.size(0), 2,2)
        LHF2_in_1 = torch.zeros(LHF2_in_1_pre.size())
        if LHF2_in_1_pre.is_cuda:
            LHF2_in_1 = LHF2_in_1.cuda()
        LHF2_in_1 = Variable(LHF2_in_1)
        LHF2_in_1[:, :2,:2] = rectifyAffineTransformationUpIsUp(LHF2_in_1_pre[:, :2,:2]/sc) * sc
        LHF2_in_1[:,:, 2] = LHF2_in_1_pre[:,:,2]
    else:
        LHF2_in_1 = LHF2_in_1_pre
    LHF1_inv = inverseLHFs(LAFs_to_H_frames(LAFs1))
    frob_norm_dist = reproject_to_canonical_Frob_batched(LHF1_inv, LHF2_in_1, batch_size = 2, skip_center = skip_center_in_Fro)
    #### Center replated
    just_centers1 = LAFs1[:,:,2];
    just_centers2_repr_to_1 = LHF2_in_1[:,0:2,2];
    center_dist_mask  = distance_matrix_vector(just_centers2_repr_to_1, just_centers1) >= center_dist_th
    
    frob_norm_dist_masked = center_dist_mask.float() * 1000. + frob_norm_dist;
    
    min_dist, idxs_in_2 = torch.min(frob_norm_dist_masked,1)
    plain_indxs_in1 = torch.arange(0, idxs_in_2.size(0))
    if LAFs1.is_cuda:
        plain_indxs_in1 = plain_indxs_in1.cuda()
    plain_indxs_in1 = torch.autograd.Variable(plain_indxs_in1, requires_grad = False)
    #min_dist, idxs_in_2 = torch.min(dist,1)
    #print min_dist.min(), min_dist.max(), min_dist.mean()
    mask =  (min_dist <= dist_threshold )
    
    if return_LAF2_in_1:
        return min_dist[mask], plain_indxs_in1[mask], idxs_in_2[mask], LHF2_in_1[:,0:2,:]
    else:
        return min_dist[mask], plain_indxs_in1[mask], idxs_in_2[mask]



