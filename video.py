import torch
import cv2
from utils import image_pre_processing, predict_transform, get_result, show_result, get_classes
from Darknet import darknet
import time


if __name__ == "__main__":
    dim = 416
    classes = get_classes('data/coco.names')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    start_time = time.time()
    net = darknet(classes)
    net.to(device)
    net.load_weights('yolov3.weights')
    print('Finished loading net in %f s' % (time.time() - start_time))

    cap = cv2.VideoCapture('test.mp4')
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if ret:
            img = image_pre_processing(frame, dim)
            y = net(img.type(torch.FloatTensor).to(device))
            detections = predict_transform(y, dim, device, classes)
            detections = get_result(detections, 0.5, 0.5)
            show_result(detections, frame, dim, 'data/coco.names')
            print('Finished transform and visualization. FPS %f' % (1 / (time.time() - start_time)))
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            break
