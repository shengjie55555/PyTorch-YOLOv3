import torch
import torch.nn as nn
import numpy as np


def iou_wh_without_center(box1, box2):
    box1_w, box1_h = box1[0], box1[1]
    box2_w, box2_h = box2[:, 0], box2[:, 1]
    w = torch.min(box1_w, box2_w)
    h = torch.min(box1_h, box2_h)
    inter_area = w * h
    union_area = box1_w * box1_h + box2_w * box2_h - inter_area
    return inter_area / union_area


def label_transform(detections, labels, dim, img_dim, device, batch_size):
    anchors = torch.from_numpy(np.array([[116, 90], [156, 198], [373, 326], [30, 61], [62, 45], [59, 119],
                                         [10, 13], [16, 30], [33, 23]])).to(device)
    labels_transformed = torch.zeros(labels.size(0), 5 + 10 + 1)
    # (x1, y1, x2, y2, p, p0, p1 ... p9, ind)
    labels_transformed[:, :4] = labels[:, 1:5]
    labels_transformed[:, 4] = 1
    labels_transformed[:, -1] = labels[:, 0]
    for i in range(labels.size(0)):
        labels_transformed[i, 5 + int(labels[i, 5])] = 1
    # change to dim * dim scale: (x1, y1, x2, y2, p, p0, p1, ... p9, ind)
    scaling_factor = torch.min(dim / img_dim)
    labels_transformed[:, :4] *= scaling_factor
    new_img_dim = img_dim * scaling_factor
    delta_h, delta_w = (dim - new_img_dim) // 2
    labels_transformed[:, [0, 2]] = labels_transformed[:, [0, 2]].clone() + delta_w
    labels_transformed[:, [1, 3]] = labels_transformed[:, [1, 3]].clone() + delta_h
    # dim * dim scale: (xc, yc, w, h, p, p0, p1, ... p9, ind)
    labels_transformed_center = torch.zeros_like(labels_transformed).to(device)
    labels_transformed_center[:, 0] = (labels_transformed[:, 0] + labels_transformed[:, 2]) / 2
    labels_transformed_center[:, 1] = (labels_transformed[:, 1] + labels_transformed[:, 3]) / 2
    labels_transformed_center[:, 2] = (labels_transformed[:, 2]) - labels_transformed[:, 0]
    labels_transformed_center[:, 3] = (labels_transformed[:, 3]) - labels_transformed[:, 1]
    labels_transformed_center[:, 4:] = labels_transformed[:, 4:]
    # find the true prediction index for each label
    # (σ(tx), σ(ty), tw, th, σ(p, p0, p1, ... p9))
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    y1 = torch.zeros(batch_size, 13, 13, 3, 15).to(device)
    y2 = torch.zeros(batch_size, 26, 26, 3, 15).to(device)
    y3 = torch.zeros(batch_size, 52, 52, 3, 15).to(device)
    ground_truth = [y1, y2, y3]
    mask_1_obj = torch.BoolTensor(batch_size, 13, 13, 3).fill_(0).to(device)
    mask_2_obj = torch.BoolTensor(batch_size, 26, 26, 3).fill_(0).to(device)
    mask_3_obj = torch.BoolTensor(batch_size, 52, 52, 3).fill_(0).to(device)
    masks_obj = [mask_1_obj, mask_2_obj, mask_3_obj]
    mask_1_noobj = torch.BoolTensor(batch_size, 13, 13, 3).fill_(1).to(device)
    mask_2_noobj = torch.BoolTensor(batch_size, 26, 26, 3).fill_(1).to(device)
    mask_3_noobj = torch.BoolTensor(batch_size, 52, 52, 3).fill_(1).to(device)
    masks_noobj = [mask_1_noobj, mask_2_noobj, mask_3_noobj]
    for i in range(labels_transformed_center.size(0)):
        ious = iou_wh_without_center(labels_transformed_center[i, 2:4], anchors)
        layer_num = (torch.max(ious, 0)[1] // 3).type(torch.LongTensor)
        anchor_num = (torch.max(ious, 0)[1] % 3).type(torch.LongTensor)
        image_ind = labels_transformed_center[i, -1].type(torch.LongTensor)
        grid = (labels_transformed_center[i, :2] // (32 / 2 ** layer_num)).type(torch.LongTensor).to(device)
        labels_transformed_center[i, :2] = labels_transformed_center[i, :2] / (32 / 2 ** layer_num) - grid
        labels_transformed_center[i, 2:4] = \
            torch.log(labels_transformed_center[i, 2:4] / anchors[layer_num * 3 + anchor_num, :])
        ground_truth[layer_num][image_ind, grid[1], grid[0], anchor_num, :] = labels_transformed_center[i, :15]
        masks_obj[layer_num][image_ind, grid[1], grid[0], anchor_num] = 1
        masks_noobj[layer_num][image_ind, grid[1], grid[0], anchor_num] = 0

    total_loss = 0
    loss_conf_obj, loss_x, loss_y, loss_w, loss_h, loss_cls = 0, 0, 0, 0, 0, 0
    for i in range(len(detections)):
        g_obj = ground_truth[i][masks_obj[i]]
        p_obj = detections[i][masks_obj[i]]
        g_noobj = ground_truth[i][masks_noobj[i]]
        p_noobj = detections[i][masks_noobj[i]]

        if g_obj.size(0) != 0:
            loss_x = mse_loss(p_obj[:, 0], g_obj[:, 0])
            loss_y = mse_loss(p_obj[:, 1], g_obj[:, 1])
            loss_w = mse_loss(p_obj[:, 2], g_obj[:, 2])
            loss_h = mse_loss(p_obj[:, 3], g_obj[:, 3])
            loss_conf_obj = bce_loss(p_obj[:, 4], g_obj[:, 4])
            loss_cls = bce_loss(p_obj[:, 5:], g_obj[:, 5:])
        loss_conf_noobj = bce_loss(p_noobj[:, 4], g_noobj[:, 4])

        layer_loss = 100 * loss_conf_noobj + loss_conf_obj + loss_x + loss_y + loss_w + loss_h + 10 * loss_cls
        loss_conf_obj, loss_x, loss_y, loss_w, loss_h, loss_cls = 0, 0, 0, 0, 0, 0

        total_loss += layer_loss
    return total_loss / batch_size


def prediction_transform(detections, batch_size, classes):
    y = []
    for i in range(len(detections)):
        grid_size = detections[i].size(2)
        y1 = detections[i].transpose(2, 1).transpose(3, 2).contiguous().view(batch_size, grid_size, grid_size, 3,
                                                                             len(classes) + 5)
        y1[:, :, :, :, :2] = torch.sigmoid(y1[:, :, :, :, :2].clone())
        y1[:, :, :, :, 4:] = torch.sigmoid(y1[:, :, :, :, 4:].clone())
        y.append(y1)
    return y
