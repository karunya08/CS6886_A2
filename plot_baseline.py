import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as T
import matplotlib.pyplot as plt

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def baseline():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
    return model.cuda()


def prepare_data(batch_size=64, num_workers=2):
    test_tf = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_set = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_tf
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return test_loader


def evaluate(model, loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.cuda(), labels.cuda()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_sum += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    loss = loss_sum / total
    acc = 100.0 * correct / total

    return loss, acc


if __name__ == "__main__":
    checkpoint_path = "baseline_best.pt"

    model = baseline()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cuda"
    )

    model.load_state_dict(checkpoint)

    test_loader = prepare_data()

    test_loss, test_acc = evaluate(model, test_loader)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Final Test Top-1 Accuracy: {test_acc:.2f}%")