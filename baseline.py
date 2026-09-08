'''Baseline model and data utilities.

Defines a MobileNetV2-based baseline for CIFAR-10, data loaders,
training and evaluation helper functions used by the compression
pipeline.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def baseline():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    num_classes = 10
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.cuda()
    print(model.classifier)
    return model


def prepare_data(batch_size=64, num_workers=2):
    train_tf = T.Compose([
        T.Resize(224), T.RandomCrop(224, padding=8), T.RandomHorizontalFlip(),
        T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_tf = T.Compose([
        T.Resize(224), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                                num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total


def train(model, train_loader, test_loader, num_epochs=20, lr=0.01, checkpoint_path="baseline_best.pt"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_acc = 100. * correct / total
        test_acc = evaluate(model, test_loader)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {running_loss/total:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), checkpoint_path)

    print(f"Best test accuracy: {best_acc:.2f}%")
    return model, best_acc


if __name__ == "__main__":
    model = baseline()
    train_loader, test_loader = prepare_data()

    checkpoint_path = "baseline_best.pt"

    model, best_acc = train(model, train_loader, test_loader, num_epochs=20, lr=0.01,
                             checkpoint_path=checkpoint_path)