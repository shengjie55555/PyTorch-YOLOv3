import torch
import cv2
from utils import image_pre_processing, predict_transform, get_result, show_result, get_classes
from Darknet import darknet
import time


if __name__ == '__main__':
    dim = 416
    classes = get_classes('data/coco.names')

    stat_time = time.time()
    image = cv2.imread('dog-cycle-car.png')
    img = image_pre_processing(image, dim)
    print('Finished loading image in %f s' % (time.time() - stat_time))
    end_time = time.time()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        net = darknet(classes)
        net.to(device)
        net.load_weights('yolov3.weights')
        print('Finished loading net in %f s' % (time.time() - end_time))
        end_time = time.time()

        y = net(img.type(torch.FloatTensor).to(device))
        print('Finished detection in %f s' % (time.time() - end_time))
        end_time = time.time()

        detections = predict_transform(y, dim, device, classes)
        detections = get_result(detections, 0.5, 0.4)
        show_result(detections, image, dim, 'data/coco.names')
        print('Finished transform and visualization in %f s' % (time.time() - end_time))
        if cv2.waitKey(0) == ord('q'):
            exit()
