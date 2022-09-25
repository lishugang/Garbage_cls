import os
import time
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class MyDataSet(Dataset):
    def __init__(self, dataset_path, classes, transform=None):
        super(MyDataSet, self).__init__()
        self.dataset_path = dataset_path
        self.transform = transform
        self.file_list = os.listdir(self.dataset_path)
        self.classes = classes
        # print(file_list)
        self.img_list = []
        for file in self.file_list:
            if file.endswith(".xml"):
                continue
            else:
                self.img_list.append(file)

    def read_xml(self, file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()
        bbox = []  # xyxy
        cls = ""
        for child in root:
            if child.tag == "object":
                for info in child:
                    if info.tag == "name":
                        cls = info.text
                    elif info.tag == "bndbox":
                        for bb in info:
                            bbox.append(bb.text)
        cls_num = self.classes.index(cls)
        # print(cls_num)
        return cls_num, bbox
    
    def pad_image(self, image, target_size, xml_name):
        iw, ih = image.size 
        w, h = target_size
        # print("original size: ",(iw,ih))
        # print("new size: ", (w, h))
        scale = min(w / iw, h / ih)  
        # 保证长或宽，至少一个符合目标图像的尺寸 0.5保证四舍五入
        nw = int(iw * scale+0.5)
        nh = int(ih * scale+0.5)
        image = image.resize((nw, nh), Image.BICUBIC)  # 更改图像尺寸
        new_image = Image.new('RGB', target_size, (0, 0, 0))  # 生成黑色图像
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 填充中间
        return new_image


    def __getitem__(self, index):
        img_name = self.img_list[index]
        img_path = os.path.join(self.dataset_path, img_name)
        img = Image.open(img_path).convert("RGB")
        # print(img)
        xml_name = img_name.split('.')[0] + '.xml'
        xml_path = os.path.join(self.dataset_path, xml_name)
        cls, [x1, y1, x2, y2] = self.read_xml(xml_path)
        crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
        # print(crop)
        crop = self.pad_image(crop, (500,500), xml_name)
        
        crop = self.transform(crop)
        return crop, cls

    def __len__(self):
        return len(self.img_list)

if __name__ == "__main__":
    classes = ['Recyclables', 'Other Waste', 'Harmful Waste', 'Kitchen Waste']
    # path = "D:\\datasets\\Garbage_classification\\train_set"
    path = "D:\\datasets\\Garbage_classification\\test_set"

    data_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])
    ])

    dataSet = MyDataSet(dataset_path=path, classes=classes, transform=data_transform)
    dataLoader = DataLoader(dataSet, batch_size=8, shuffle=True, num_workers=4)
    image_batch, label_batch = iter(dataLoader).next()
    for i in range(image_batch.data.shape[0]):
        label = label_batch.data[i]
        print(label)
        img = np.array(image_batch.data[i] * 255, np.uint8)
        plt.imshow(np.transpose(img, [1, 2, 0]))
        plt.show()
        












