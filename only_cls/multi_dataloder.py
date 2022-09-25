import os
import time
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class MyDataSet_multi(Dataset):
    def __init__(self, dataset_path,multi_cls, transform=None):
        super(MyDataSet_multi, self).__init__()
        self.dataset_path = dataset_path
        self.transform = transform
        self.file_list = os.listdir(self.dataset_path)
        self.multi_cls = multi_cls
        # print(file_list)
        self.img_list = []
        for file in self.file_list:
            if file.endswith(".xml"):
                continue
            else:
                self.img_list.append(file)

    def read_xml(self, file_path, img_name):
        tree = ET.parse(file_path)
        root = tree.getroot()
        bbox = []  # xyxy
        for child in root:
            if child.tag == "object":
                for info in child:
                    if info.tag == "bndbox":
                        for bb in info:
                            bbox.append(bb.text)
        cls = self.ana_cls(img_name)
        cls_num = self.multi_cls.index(cls)
        # print(cls_num)
        # return cls_num, bbox
        return cls_num, bbox

    def ana_cls(self, img_name):
        img_name = img_name.split('.')[0]
        # print(img_name)
        cls = ""
        for s in img_name:
            if ord(s) > 60 or ord(s) == ord("-"):
                # print(s)
                cls = cls + s
            else:
                break
        # print(cls)
        return cls


    def __getitem__(self, index):
        img_name = self.img_list[index]
        img_path = os.path.join(self.dataset_path, img_name)
        img = Image.open(img_path).convert("RGB")
        # print(img)
        xml_name = img_name.split('.')[0] + '.xml'
        xml_path = os.path.join(self.dataset_path, xml_name)
        cls, [x1, y1, x2, y2] = self.read_xml(xml_path, img_name)
        crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
        # print(crop)
        crop = self.transform(crop)
        return crop, cls

    def __len__(self):
        return len(self.img_list)

if __name__ == "__main__":
    classes = ['Recyclables', 'Other Waste', 'Harmful Waste', 'Kitchen Waste']
    multi_cls = ['Non-perishable_Items', 'Small_Items', 'Plastic', 'Paper', 'Expired_Drug', 'Food', 'FruitFruit',
                 'Insecticide', 'Others', 'Glass', 'Contaminated_Things', 'Paint_Bucket', 'Battery_Mercury', 'Fabric',
                 'Plant', 'Photosensitive_Film', 'Lamp_Cartridge', 'Expired_Food', 'Metal']
    path = "D:\\datasets\\Garbage_classification\\train_set"
    # path = "D:\\datasets\\Garbage_classification\\test_set"

    data_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])
    ])

    dataSet = MyDataSet_multi(dataset_path=path, multi_cls=multi_cls, transform=data_transform)
    # dataSet = MyDataSet(dataset_path=path, transform=data_transform)
    dataLoader = DataLoader(dataSet, batch_size=8, shuffle=True, num_workers=4)
    image_batch, label_batch = iter(dataLoader).next()
    # cls = {}
    # for i, data in enumerate(dataLoader):
    #     imgs, annos = data
    #     # print(annos)
    #     for anno in annos:
    #         if anno not in cls.keys():
    #             cls[anno] = "1"
    #             print(anno)
    # print(cls.keys())
    for i in range(image_batch.data.shape[0]):
        label = label_batch.data[i]
        print(label)
        # img = np.array(image_batch.data[i] * 255, np.uint8)
        # plt.imshow(np.transpose(img, [1, 2, 0]))
        # plt.show()
        












