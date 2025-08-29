import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import model, criterion


model.load_state_dict(torch.load("cnn_mnist.pth"))
model.eval()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)


loss, correct, total = 0.0, 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        loss += criterion(outputs, labels).item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

print(f"Test Loss: {loss/len(test_loader):.4f}, Test Accuracy: {100*correct/total:.2f}%")
