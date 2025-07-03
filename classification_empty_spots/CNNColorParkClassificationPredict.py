import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from CNNColorParkClassificationTrain import SimpleColorCNN

'''
Evaluate model for the classification of type of empty parking slots 
'''

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5
BATCH_SIZE = 16
TEST_DIR = "../datasets/dataset_emptyslotsCNN/test"

class_names = ['normal', 'pay', 'disabled', 'unknown', 'pregnant']

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = SimpleColorCNN(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load("../models/model_parking_color_cnn.pth", map_location=DEVICE))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy on test set: {accuracy:.2f}%")