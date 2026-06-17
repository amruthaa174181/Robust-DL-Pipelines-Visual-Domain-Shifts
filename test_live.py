import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2  # Run 'pip install opencv-python' in terminal if you don't have it

# 1. Load your trained robust network model weights
# (Assuming you saved your model weights after training)
device = torch.device("cpu")
model = torch.load("robust_resnet18.pt", map_location=device, weights_only=False)
model.eval()

# 2. Define the real-world corruption (Simulating a smudge/fog on a lens)
blur_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.GaussianBlur(kernel_size=5, sigma=2.0), # Live Blur Attack!
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 3. Capture a real picture using your webcam
cam = cv2.VideoCapture(0)
print("Look at your webcam and press SPACEBAR to snap a test photo!")
while True:
    ret, frame = cam.read()
    cv2.imshow('Live Camera Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord(' '): # Press Space to take picture
        cv2.imwrite('real_world_test.jpg', frame)
        break
cam.release()
cv2.destroyAllWindows()

# 4. Run the live image through your robust model
live_image = Image.open('real_world_test.jpg')
input_tensor = blur_transform(live_image).unsqueeze(0) # Format tensor array

with torch.no_grad():
    prediction = model(input_tensor)
    predicted_class = torch.argmax(prediction, dim=1).item()

# CIFAR-10 category labels mapping
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print(f"\n🎯 [Prediction Result]: Even through heavy blur, the robust AI identifies this object as a: {classes[predicted_class].upper()}!")