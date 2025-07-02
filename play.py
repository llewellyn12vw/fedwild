import scipy.io
import torch
import numpy as np
def evaluate(qf, ql, gf, gl):
    query = qf.view(-1, 1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    
    index = np.argsort(score)[::-1]
    good_index = np.argwhere(gl == ql).flatten()
    junk_index = np.argwhere(gl == -1).flatten()
    
    ap, cmc = compute_mAP(index, good_index, junk_index)
    return ap, cmc

def compute_mAP(index, good_index, junk_index):
    ap = 0.0
    cmc = torch.IntTensor(len(index)).zero_()
    
    if len(good_index) == 0:
        cmc[0] = -1
        return ap, cmc

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask).flatten()
    
    if len(rows_good) > 0:
        cmc[rows_good[0]:] = 1
        
        for i in range(ngood):
            d_recall = 1.0 / ngood
            precision = (i + 1) / (rows_good[i] + 1)
            if rows_good[i] != 0:
                old_precision = i / rows_good[i]
            else:
                old_precision = 1.0
            ap += d_recall * (old_precision + precision) / 2

    return ap, cmc
result = scipy.io.loadmat("/home/wellvw12/local_fed/FedReID/model/ex2_110_kd/pytorch_result.mat")
query_feature = torch.FloatTensor(result['query_f']).cuda()
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f']).cuda()
gallery_label = result['gallery_label'][0]

CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
valid_queries = 0

for i in range(len(query_label)):
    ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], gallery_feature, gallery_label)
    if CMC_tmp[0] == -1:
        continue
    CMC += CMC_tmp
    ap += ap_tmp
    valid_queries += 1