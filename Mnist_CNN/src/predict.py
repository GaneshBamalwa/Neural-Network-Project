import torch
import torchvision.transforms as transforms
from PIL import Image
from model import model


model.load_state_dict(torch.load("cnn_mnist.pth"))
model.eval()


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_image(image_path):
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

if __name__ == "__main__":
    sample = "sample_digit.png"  #(replace with you own image)
    prediction = predict_image(sample)
    print(f"Predicted Digit: {prediction}")
