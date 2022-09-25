from dataloder import MyDataSet
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch
import torch.optim as optim
import argparse

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


    test_dataset = MyDataSet(dataset_path=testset_path, classes=classes, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    net = torch.load('best72.pth')
    net.to(device)

    net.eval()
    eval_acc = eval(net, test_loader)
    print(eval_acc)