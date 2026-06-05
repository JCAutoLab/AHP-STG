import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

# metric
def metric(pred, label, chunk_size=256):
    pred = np.asarray(pred)
    label = np.asarray(label)

    abs_sum = 0.0
    sq_sum = 0.0
    pct_sum = 0.0
    valid_count = 0

    with np.errstate(divide='ignore', invalid='ignore'):
        for start in range(0, pred.shape[0], chunk_size):
            end = min(start + chunk_size, pred.shape[0])
            pred_chunk = pred[start:end]
            label_chunk = label[start:end]
            mask = np.not_equal(label_chunk, 0)
            count = int(np.count_nonzero(mask))
            if count == 0:
                continue

            diff = np.subtract(pred_chunk, label_chunk)
            abs_err = np.abs(diff).astype(np.float32, copy=False)
            valid_abs = abs_err[mask]
            valid_label = label_chunk[mask]

            abs_sum += float(np.nan_to_num(valid_abs, copy=False).sum(dtype=np.float64))
            sq_sum += float(np.nan_to_num(np.square(valid_abs), copy=False).sum(dtype=np.float64))
            pct = np.divide(valid_abs, valid_label)
            pct_sum += float(np.nan_to_num(pct, copy=False).sum(dtype=np.float64))
            valid_count += count

    if valid_count == 0:
        return 0.0, 0.0, 0.0

    mae = abs_sum / valid_count
    rmse = np.sqrt(sq_sum / valid_count)
    mape = pct_sum / valid_count
    return mae, rmse, mape

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def _compute_loss(y_true, y_predicted):
        return masked_mae(y_predicted, y_true, 0.0)

def read_meta(path):
    meta = pd.read_csv(path)
    lat = meta['Lat'].values
    lng = meta['Lng'].values
    locations = np.stack([lat,lng], 0)
    return locations

def construct_adj(data):
    # construct the adj through the cosine similarity
    data_mean = np.mean([data[24*12*i: 24*12*(i+1)] for i in range(data.shape[0]//(24*12))], axis=0)
    data_mean = data_mean.squeeze().T
    tem_matrix = cosine_similarity(data_mean, data_mean)
    tem_matrix = np.exp((tem_matrix-tem_matrix.mean())/tem_matrix.std())
    return tem_matrix

def augmentAlign(dist_matrix, auglen):
    # find the most similar points in other leaf nodes
    sorted_idx = np.argsort(dist_matrix.reshape(-1)*-1)
    sorted_idx = sorted_idx % dist_matrix.shape[-1]
    augidx = []
    for idx in sorted_idx:
        if idx not in augidx:
            augidx.append(idx)
        if len(augidx) == auglen:
            break
    return np.array(augidx, dtype=int)

def reorderData(parts_idx, adj, sps):
    # parts_idx: segmented indices by kdtree
    # adj: pad similar points through the cos_sim adj
    # sps: spatial patch (small leaf nodes) size for padding
    ori_parts_idx = np.array([], dtype=int)
    reo_parts_idx = np.array([], dtype=int)
    reo_all_idx = np.array([], dtype=int)
    for i, part_idx in enumerate(parts_idx):
        part_dist = adj[part_idx, :].copy()
        part_dist[:, part_idx] = 0
        if sps-part_idx.shape[0] > 0:
            local_part_idx = augmentAlign(part_dist, sps-part_idx.shape[0])
            auged_part_idx = np.concatenate([part_idx, local_part_idx], 0)
        else:
            auged_part_idx = part_idx

        reo_parts_idx = np.concatenate([reo_parts_idx, np.arange(part_idx.shape[0])+sps*i])
        ori_parts_idx = np.concatenate([ori_parts_idx, part_idx])
        reo_all_idx = np.concatenate([reo_all_idx, auged_part_idx])

    return ori_parts_idx, reo_parts_idx, reo_all_idx

def kdTree(locations, times, axis):
    # locations: [2,N] contains lng and lat
    # times: depth of kdtree
    # axis: select lng or lat as hyperplane to split points
    sorted_idx = np.argsort(locations[axis])
    part1, part2 = np.sort(sorted_idx[:locations.shape[1]//2]), np.sort(sorted_idx[locations.shape[1]//2:])
    parts = []
    if times == 1:
        return [part1, part2]
    else:
        left_parts = kdTree(locations[:,part1], times-1, axis^1)
        right_parts = kdTree(locations[:,part2], times-1, axis^1)
        for part in left_parts:
            parts.append(part1[part])
        for part in right_parts:
            parts.append(part2[part])
    return parts

def loadSpatialManagedIndices(data, metapath, adjpath, recurtimes, sps, log):
    # load data
    trainData = data.data_x
    locations = read_meta(metapath)
    # load adj for padding
    if os.path.exists(adjpath):
        adj = np.load(adjpath)
    else:
        adj = construct_adj(trainData)
        np.save(adjpath, adj)
    # partition and pad data with new indices
    parts_idx = kdTree(locations, recurtimes, 0)
    ori_parts_idx, reo_parts_idx, reo_all_idx = reorderData(parts_idx, adj, sps)
    # log
    log_string(log, f'Padded Nodes: {reo_all_idx.shape[0]}')
    
    return ori_parts_idx, reo_parts_idx, reo_all_idx
    
