from dataloder import MyDataSet
from multi_dataloder import MyDataSet_multi
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--batch_size", default=100, type=int)
parser.add_argument("--num_workers", default=10, type=int)
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--pretrain", default=False, type=bool)
parser.add_argument("--test_time_norm", default=False, type=bool)
parser.add_argument("--test_batch", default=False, type=bool)
parser.add_argument("--test_batch_num", default=30, type=int)
parser.add_argument("--train_rotation", default=False, type=bool)
parser.add_argument("--train_ColorJitter", default=False, type=str)
parser.add_argument("--test_ColorJitter", default=False, type=str)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(epoch, net, train_loader, cirterion, optimizer, test_loader, best_eval_acc, eval_dict):
    train_correct = 0
    train_total = 0
    loss = 0
    for i, data in enumerate(train_loader):
        imgs, annos = data
        imgs, annos = imgs.to(device), annos.to(device)

        optimizer.zero_grad()
        outputs = net(imgs)
        _, train_predicted = torch.max(outputs.data, 1)

        train_correct += (train_predicted == annos.data).sum()
        loss = cirterion(outputs, annos)
        loss.backward()
        optimizer.step()

        train_total += annos.size(0)
        if args.test_batch:
            if i % args.test_batch_num == 0:
                net.eval()
                best_eval_acc = eval(net, test_loader, best_eval_acc, eval_dict)
                print('epoch: %d batch: %d loss: %.3f best eval acc: %.3f' % (epoch+1, i, loss, best_eval_acc))
                net.train()
    print('epoch: %d acc: %.3f ' % (epoch+1, 100 * train_correct // train_total))
    return best_eval_acc


def eval(net, test_loader, best_eval_acc, eval_dict):
    correct = 0
    test_total = 0
    for i, data in enumerate(test_loader):
        imgs, annos = data
        imgs, annos = imgs.to(device), annos.to(device)

        outputs = net(imgs)
        _, predicted = torch.max(outputs.data, 1)
        coarse_cls = predicted.cpu().numpy().tolist()
        for i in range(len(coarse_cls)):
            coarse_cls[i] = eval_dict[str(coarse_cls[i])]
        test_total += annos.size(0)
        correct += (predicted == annos.data).sum()
    eval_acc = 100 * correct // test_total
    if eval_acc > best_eval_acc:
        best_eval_acc = eval_acc
        torch.save(net, "multi_best.pth")
    # print('eval acc: %.3f best acc: %.3f' % (eval_acc, best_eval_acc))
    return best_eval_acc

if __name__ == "__main__":
    classes = ['Recyclables', 'Other Waste', 'Harmful Waste', 'Kitchen Waste']
    trainset_path = "/data1/TL/data/garbage_dataset/train_set"
    testset_path = "/data1/TL/data/garbage_dataset/test_set"
    multi_cls = ['Non-perishable_Items', 'Small_Items', 'Plastic', 'Paper', 'Expired_Drug', 'Food', 'FruitFruit',
                 'Insecticide', 'Others', 'Glass', 'Contaminated_Things', 'Paint_Bucket', 'Battery_Mercury', 'Fabric',
                 'Plant', 'Photosensitive_Film', 'Lamp_Cartridge', 'Expired_Food', 'Metal']
    eval_dict = {"0":1, "1":1, "2":0, "3":0, "4":2, "5":3, "6":3, "7":2, "8":0, "9":0, "10":1, "11":2, "12":3, "13":0, "14":3, "15":2, "16":2, "17":3, "18":0 }

    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])
        ])
    if args.train_rotation:
        train_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])
        ])
    if args.train_ColorJitter:
        train_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])
        ])
    if args.train_rotation and args.train_ColorJitter:
        train_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])
            ])

    train_dataset = MyDataSet_multi(dataset_path=trainset_path, multi_cls=multi_cls, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

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

    net = models.resnet50(pretrained=args.pretrain)
    num_cls_ori = net.fc.in_features
    net.fc = nn.Linear(num_cls_ori, len(multi_cls))
    net.to(device)

    cirterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    best_eval_acc = 0
    method = 'multi'
    print('********************************************************************/n')
    print('********************************************************************/n')
    print('********************************************************************/n')   
    print("method: {0} lr: {1} pretrain: {2} test_time_norm: {3} test_batch: {4} train_rotation: {5} train_ColorJitter: {6} test_ColorJitter: {7}".format(method, args.lr, args.pretrain, args.test_time_norm, args.test_batch, args.train_rotation, args.train_ColorJitter, args.test_ColorJitter)) 
    print('********************************************************************/n')
    print('********************************************************************/n')
    print('********************************************************************/n')
    for epoch in range(args.epochs):
        net.train()
        best_eval_acc = train_one_epoch(epoch, net, train_loader, cirterion, optimizer, test_loader, best_eval_acc, eval_dict)
        net.eval()
        best_eval_acc = eval(net, test_loader, best_eval_acc, eval_dict)












