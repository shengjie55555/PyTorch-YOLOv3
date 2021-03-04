import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Darknet import darknet
from training_utils import *
from utils import *


class my_dataset(Dataset):
    def __init__(self, root, image_dir, label_dir, dim):
        self.root = root
        self.dim = dim
        self.image_dir = root + '/' + image_dir
        self.label_dir = root + '/' + label_dir

    def __len__(self):
        f = np.load(self.image_dir)
        return f.shape[0]

    def get_image(self, item):
        image = np.load(self.image_dir)[item]
        h, w = image.shape
        image = np.reshape(image, (h, w, 1))
        image = np.concatenate((image, image, image), axis=2)
        image = image_pre_processing(image, self.dim).squeeze()
        return image

    def get_raw_image(self, item):
        image = np.load(self.image_dir)[item]
        h, w = image.shape
        image = np.reshape(image, (h, w, 1))
        image = np.concatenate((image, image, image), axis=2)
        return image

    def get_label(self, item):
        f = open(self.label_dir, 'r')
        lines = f.readlines()
        line = lines[item]
        boxes = line.rstrip('\n').split('\t')[1:]
        labels = []
        for box in boxes:
            label = box.split(',')
            labels.append([int(x) for x in label])
        return torch.from_numpy(np.array(labels))

    def __getitem__(self, item):
        image = self.get_image(item)
        label = self.get_label(item)
        return image, label


def collect_fn(batch):
    images, labels = zip(*batch)
    cat = False
    for ind, label in enumerate(labels):
        try:
            label_ind = torch.ones(label.size(0), label.size(1) + 1)
            label_ind[:, 1:6] = label
            label_ind[:, 0] = ind
        except:
            label_ind = torch.zeros(0, 6)
        if not cat:
            labels_ind = label_ind
            cat = True
        else:
            labels_ind = torch.cat((labels_ind, label_ind), 0)
    return torch.stack(images, 0), labels_ind


if __name__ == '__main__':
    dim = 416
    classes = get_classes('m2nist/numbers.txt')

    train_dataset = my_dataset('m2nist', 'combined.npy', 'bbox.txt', 416)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collect_fn)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net = darknet(classes)
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1, last_epoch=-1)

    for epoch in range(4):
        net.train()
        for i, batch in enumerate(train_dataloader):
            images, targets = batch[0].type(torch.FloatTensor).to(device), batch[1]

            detections = net(images)

            detections = prediction_transform(detections, images.size(0), classes)

            loss = label_transform(detections, targets, 416, torch.tensor([64, 84]), device, images.size(0))

            net.zero_grad()
            loss.backward()
            optimizer.step()

            print(epoch, i, loss.item())
            if i % 100 == 99:
                torch.save(net.state_dict(), './test.pth')
                print('Finished training and saved the model!')
                print('Learning rate: %f' % optimizer.param_groups[0]['lr'])
        scheduler.step()

    net.load_state_dict(torch.load('./test.pth'))
    with torch.no_grad():
        for i in range(len(train_dataset)):
            raw_image = train_dataset.get_raw_image(i)
            image, label = train_dataset[i]
            image, label = image.type(torch.FloatTensor).to(device).unsqueeze(0), label.type(torch.FloatTensor).to(device)
            y = net(image)
            detections = predict_transform(y, dim, device, classes)
            detections = get_result(detections, 0.7, 0.3)
            print(detections)
            show_result(detections, raw_image, dim, 'm2nist/numbers.txt')
            if cv2.waitKey(0) == ord('q'):
                exit()
