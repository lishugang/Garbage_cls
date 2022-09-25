import os
import time
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from dataloder import MyDataSet
import torch
import argparse

class MyDataSet(Dataset):
    def __init__(self, res_json_path, dataset_path, test_json, classes, transform=None):
        super(MyDataSet, self).__init__()
        self.res_json_path = res_json_path
        self.dataset_path = dataset_path
        self.test_json = test_json
        self.transform = transform
        self.classes = classes
        self.file_list = list(filter(lambda x: x.endswith('jpg'),os.listdir(self.dataset_path)))
        # print(len(self.file_list))
        self.annFile = COCO(self.test_json)
        self.dets = self.annFile.loadRes(self.res_json_path)
        self.img_id_list = []
        self.tmpt_img_id_list = self.dets.getImgIds()
        for i in range(len(self.tmpt_img_id_list)):
            img_id = self.tmpt_img_id_list[i]
            img_info = self.annFile.loadImgs(img_id)[0]
            img_name = img_info['file_name']
            if img_name in self.file_list:
                self.img_id_list.append(img_id)
        # print(len(self.img_id_list))

    def read_xml(self, file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()
        cls = ""
        for child in root:
            if child.tag == "object":
                for info in child:
                    if info.tag == "name":
                        cls = info.text
        cls_num = self.classes.index(cls)
        return cls_num
    
    def pad_image(self, image, target_size, xml_name):
        iw, ih = image.size 
        w, h = target_size
        # print("original size: ",(iw,ih))
        # print("new size: ", (w, h))
        # print(xml_name)
        scale = min(w / iw, h / ih)  
        # 保证长或宽，至少一个符合目标图像的尺寸 0.5保证四舍五入
        nw = int(iw * scale+0.5)
        nh = int(ih * scale+0.5)
        image = image.resize((nw, nh), Image.BICUBIC)  # 更改图像尺寸
        new_image = Image.new('RGB', target_size, (0, 0, 0))  # 生成黑色图像
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 填充中间
        return new_image

    def __getitem__(self, index):
        img_id = self.img_id_list[index]
        img_info = self.annFile.loadImgs(img_id)[0]
        img_name = img_info['file_name']
        anns_id = self.dets.getAnnIds(img_id)
        bboxes = self.dets.loadAnns(anns_id)
        bbox = [int(bboxes[0]['bbox'][0]), int(bboxes[0]['bbox'][1]), int(bboxes[0]['bbox'][2]), int(bboxes[0]['bbox'][3])]
        for i in range(len(bbox)):
            bbox[i] = 0 if bbox[i] < 0 else bbox[i]
        bbox = (bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3])
        img_path = os.path.join(self.dataset_path, img_name)
        img = Image.open(img_path).convert("RGB")
        # print(img)
        xml_name = img_name.split('.')[0] + '.xml'
        xml_path = os.path.join(self.dataset_path, xml_name)
        cls = self.read_xml(xml_path)

        # print("img : ",img.size)
        crop = img.crop(bbox)
        # print(crop.size)
        crop = self.pad_image(crop, (500,500), xml_name)
        # print(crop)
        
        crop = self.transform(crop)
        return crop, cls

    def __len__(self):
        return len(self.file_list)

# if __name__ == "__main__":
#     classes = ['Recyclables', 'Other Waste', 'Harmful Waste', 'Kitchen Waste']
#     path = "/data1/TL/data/garbage_dataset/test_set"
#     res_json_path = "/data1/lsg/res_cocotype_json/deformable_detr_origin_garbage.json"
#     test_json = "/data1/TL/data/garbage_dataset/test.json"

#     data_transform = transforms.Compose([
#         transforms.Resize((224,224)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(15),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])
#     ])

#     dataSet = MyDataSet(dataset_path=path, res_json_path=res_json_path, test_json=test_json, classes=classes, transform=data_transform)
#     dataLoader = DataLoader(dataSet, batch_size=8, shuffle=True, num_workers=4)
#     image_batch, label_batch = iter(dataLoader).next()
    # for i in range(image_batch.data.shape[0]):
    #     label = label_batch.data[i]
    #     print(label)
    #     img = np.array(image_batch.data[i] * 255, np.uint8)
    #     plt.imshow(np.transpose(img, [1, 2, 0]))
    #     plt.show()



parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=20, type=int)
parser.add_argument("--num_workers", default=10, type=int)
parser.add_argument("--test_time_norm", default=False, type=bool)
parser.add_argument("--test_ColorJitter", default=False, type=str)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval(net, test_loader):
    correct = 0
    test_total = 0
    for i, data in enumerate(test_loader):
        print(i)
        imgs, annos = data
        # plt.imshow(np.transpose(imgs[0], [1, 2, 0]))
        # plt.show()
        imgs, annos = imgs.to(device), annos.to(device)
        outputs = net(imgs)
        _, predicted = torch.max(outputs.data, 1)
        test_total += annos.size(0)
        correct += (predicted == annos.data).sum()
    eval_acc = 100 * correct // test_total
    return eval_acc

if __name__ == "__main__":
    classes = ['Recyclables', 'Other Waste', 'Harmful Waste', 'Kitchen Waste']
    testset_path = "/data1/TL/data/garbage_dataset/test_set"
    res_json_path = "/data1/lsg/res_cocotype_json/deformable_detr_origin_garbage.json"
    test_json = "/data1/TL/data/garbage_dataset/test.json"

    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        ])
    if args.test_time_norm:
        test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225]),
        ])
    if args.test_ColorJitter:
        test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.ToTensor(),
        ])
    if args.test_ColorJitter and args.test_time_norm:
        test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225]),
        ])


    test_dataset = MyDataSet(dataset_path=testset_path, res_json_path=res_json_path, test_json=test_json, classes=classes, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    net = torch.load('best72.pth')
    net.to(device)

    net.eval()
    eval_acc = eval(net, test_loader)
    print(eval_acc)