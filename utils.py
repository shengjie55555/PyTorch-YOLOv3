import torch
import numpy as np
import cv2


def get_classes(classes_file):
    f = open(classes_file)
    lines = f.readlines()
    line = [line.rstrip('\n') for line in lines]
    return line


def load_one_layer_weights(weights, basic_block_seq, ptr):
    conv = basic_block_seq[0]
    if conv.bias is None:
        bn = basic_block_seq[1]
        num_bn_biases = bn.bias.numel()
        bn_biases = torch.from_numpy(weights[ptr: ptr + num_bn_biases]).view_as(bn.bias.data)
        ptr += num_bn_biases
        bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases]).view_as(bn.weight.data)
        ptr += num_bn_biases
        bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases]).view_as(bn.running_mean.data)
        ptr += num_bn_biases
        bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases]).view_as(bn.running_var.data)
        ptr += num_bn_biases

        bn.bias.data.copy_(bn_biases)
        bn.weight.data.copy_(bn_weights)
        bn.running_mean.data.copy_(bn_running_mean)
        bn.running_var.data.copy_(bn_running_var)
    else:
        num_biases = conv.bias.numel()
        conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases]).view_as(conv.bias.data)
        ptr += num_biases
        conv.bias.data.copy_(conv_biases)
    num_weights = conv.weight.numel()
    conv_weights = torch.from_numpy(weights[ptr: ptr + num_weights]).view_as(conv.weight.data)
    ptr += num_weights
    conv.weight.data.copy_(conv_weights)
    return ptr


def image_pre_processing(img, dim):
    # rescale
    h, w = img.shape[0:2]
    if h > w:
        new_h = dim
        new_w = w * dim / h
    else:
        new_w = dim
        new_h = h * dim / w
    new_h, new_w = int(new_h), int(new_w)
    img = cv2.resize(img, (new_w, new_h))
    # padding
    img_padded = np.ones((dim, dim, 3)) * 0.5
    delta_h, delta_w = (dim - new_h) // 2, (dim - new_w) // 2
    img_padded[delta_h: delta_h + new_h, delta_w: delta_w + new_w, :] = img / 255
    # BGR -> RGB
    img_padded = img_padded[:, :, ::-1]
    # numpy to tensor
    img_padded = torch.unsqueeze(torch.from_numpy(img_padded.transpose((2, 0, 1)).copy()), 0)
    return img_padded


def predict_transform(y, dim, device, classes):
    anchors = np.array([[[10, 13], [16, 30], [33, 23]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[116, 90], [156, 198], [373, 326]]])
    cat = False
    for i, detection in enumerate(y):
        batch_size = detection.size(0)
        grid_size = detection.size(2)
        stride = dim // grid_size
        # change the shape of the detection -> batch_size * grid_size * grid_size * 3 * 85
        detection = detection.transpose(2, 1).transpose(2, 3).contiguous().view(batch_size, grid_size, grid_size, 3, len(classes)+5)
        # sigmoid center offset, object confidence and class confidence
        detection[:, :, :, :, :2] = torch.sigmoid(detection[:, :, :, :, :2])
        detection[:, :, :, :, 4:] = torch.sigmoid(detection[:, :, :, :, 4:])
        # add center offset
        grid = np.arange(grid_size)
        a, b = np.meshgrid(grid, grid)
        a, b = torch.from_numpy(a).unsqueeze(2).to(device), torch.from_numpy(b).unsqueeze(2).to(device)
        grid = torch.cat((a, b), 2).unsqueeze(2).unsqueeze(0).repeat(1, 1, 1, 3, 1)
        detection[:, :, :, :, 0:2] += grid
        # calculate width and height
        anchor = torch.from_numpy(anchors[2 - i, :] / stride).to(device)
        anchor = anchor.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, grid_size, grid_size, 1, 1)
        detection[:, :, :, :, 2:4] = torch.exp(detection[:, :, :, :, 2:4]) * anchor
        # rescale
        detection[:, :, :, :, :4] *= stride
        # change the shape to batch_size * (grid_size * grid_size * 3) * 85 in order to cat easily
        detection = detection.view(batch_size, grid_size * grid_size * 3, len(classes)+5)
        if not cat:
            detections = detection
            cat = True
        else:
            detections = torch.cat((detections, detection), 1)
    return detections


def get_result(detections, confidence=0.5, nms_conf=0.4):
    # set the object confidence lower than 0.5 to 0
    conf_mask = (detections[:, :, 4] > confidence).float().unsqueeze(2)
    detections *= conf_mask
    # change (bx, by, bw, bh) to (left, top, right, bottom)
    box_corner = torch.zeros_like(detections)
    box_corner[:, :, 0] = detections[:, :, 0] - detections[:, :, 2] / 2
    box_corner[:, :, 1] = detections[:, :, 1] - detections[:, :, 3] / 2
    box_corner[:, :, 2] = detections[:, :, 0] + detections[:, :, 2] / 2
    box_corner[:, :, 3] = detections[:, :, 1] + detections[:, :, 3] / 2
    detections[:, :, :4] = box_corner[:, :, :4]
    # nms: each image, each class
    batch_size = detections.size(0)
    cat = False
    for ind in range(batch_size):
        image_det = detections[ind, :, :]

        # delete detections whose object confidence is lower than 0.5
        nonzero_ind = torch.nonzero(image_det[:, 4]).squeeze()
        # in case nonzero_ind has only one element and its size will become torch.Size([])
        # and image_det's size will become [85] not [1, 85]
        if nonzero_ind.size() == torch.Size([]):
            nonzero_ind = nonzero_ind.unsqueeze(0)
        image_det = image_det[nonzero_ind, :]
        if image_det.size(0) == 0:  # in case nonzero_ind is tensor([], size=(0, 1))
            image_detections = None
            print('There is no detections whose object confidence is lager than 0.5')
            continue

        # get class with the largest probability
        max_class_score, max_class_ind = torch.max(image_det[:, 5:], 1)
        max_class_score, max_class_ind = max_class_score.unsqueeze(1), max_class_ind.unsqueeze(1)
        image_det = torch.cat((image_det[:, :5], max_class_score, max_class_ind), 1)

        # get the various classes detected in the image
        image_det_cpu = image_det.cpu().detach().numpy()
        image_classes = np.unique(image_det_cpu[:, -1])

        # perform nms for each class
        for cls in image_classes:
            # pick up the detection with the same class
            class_mask = (image_det[:, -1] == cls).unsqueeze(1)
            image_det_class = image_det * class_mask
            nonzero_ind_class = torch.nonzero(image_det_class[:, 4]).squeeze()
            image_det_class = image_det_class[nonzero_ind_class, :].view(-1, 7)

            # sort by object score
            class_score_ind = torch.sort(image_det_class[:, 4], descending=True)[1].squeeze()
            # in case class_score_ind has only one element and its size will become torch.Size([])
            # and image_det_class's size will become [7] not [1, 7]
            if class_score_ind.size() == torch.Size([]):
                class_score_ind = class_score_ind.unsqueeze(0)
            image_det_class = image_det_class[class_score_ind, :]

            for i in range(image_det_class.size(0)):
                try:
                    # broadcast
                    ious = iou(image_det_class[i, :4], image_det_class[i + 1:, :4])
                except IndexError:
                    break
                iou_mask = (ious < nms_conf).unsqueeze(1)
                image_det_class[i + 1:] *= iou_mask
                nonzero_ind_iou = torch.nonzero(image_det_class[:, 4]).squeeze()
                image_det_class = image_det_class[nonzero_ind_iou, :].view(-1, 7)

            batch_ind = image_det_class.new(image_det_class.size(0), 1).fill_(ind)
            image_det_class = torch.cat((image_det_class, batch_ind), 1)
            if not cat:
                image_detections = image_det_class
                cat = True
            else:
                image_detections = torch.cat((image_detections, image_det_class), 0)

    return image_detections


def iou(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[0], box1[1], box1[2], box1[3]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    left_line = torch.max(box1_x1, box2_x1)
    top_line = torch.max(box1_y1, box2_y1)
    right_line = torch.min(box1_x2, box2_x2)
    bottom_line = torch.min(box1_y2, box2_y2)
    inter_area = torch.clamp(right_line - left_line, min=0) * torch.clamp(bottom_line - top_line, min=0)
    union_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1) + (box2_x2 - box2_x1) * (box2_y2 - box2_y1) - inter_area
    return inter_area / union_area


def show_result(detections, image, dim, classes):
    if detections is not None:
        # get class name
        f = open(classes, mode='r')
        lines = f.readlines()
        # change original point and rescale
        img_dim = torch.tensor(image.shape[:2])
        scaling_factor = torch.max(img_dim / dim)
        delta_h, delta_w = (dim - img_dim / scaling_factor) // 2
        detections[:, [0, 2]] -= delta_w
        detections[:, [1, 3]] -= delta_h
        detections[:, :4] *= scaling_factor
        for i in range(detections.size(0)):
            class_name = lines[int(detections[i, 6])].split('\n')[0]
            x_left = int(detections[i, 0])
            x_top = int(detections[i, 1])
            x_right = int(detections[i, 2])
            x_bottom = int(detections[i, 3])
            text_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)[0]
            cv2.rectangle(image, (x_left, x_top), (x_right, x_bottom), color=(0, 255, 0))
            # cv2.rectangle(image, (x_left, x_top), (x_left + text_size[0], x_top + text_size[1]), color=(128, 128, 128),
            #               thickness=-1)
            # cv2.putText(image, class_name, (x_left, x_top + text_size[1]),
            #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
            cv2.putText(image, class_name, (x_left, x_top),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=1)
        cv2.imshow("src", image)
    else:
        cv2.imshow('src', image)
