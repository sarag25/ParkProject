import matplotlib.pyplot as plt

'''
Create graph with the training results of the ResNet model,
values taken from ResnetTrain 
'''

epochs = list(range(1, 21))
train_loss = [
    0.2036, 0.0397, 0.0381, 0.0167, 0.0186, 0.0218, 0.0087, 0.0022, 0.0042, 0.0067,
    0.0042, 0.0079, 0.0216, 0.0139, 0.0085, 0.0129, 0.0070, 0.0008, 0.0008, 0.0005
]

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Train Loss', marker='o', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.grid(True)
plt.tight_layout()
plt.show()