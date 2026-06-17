# train.py
import torch
import torch.nn as nn
import torchvision.models as models
from dataset import get_data_loaders

def evaluate_model(model, data_loader, device):
    """
    Evaluates model accuracy on a given dataloader.
    """
    model.eval() # Put the model into evaluation mode
    correct = 0
    total = 0
    
    with torch.no_grad(): # Disable gradient calculations to save memory
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) # Find index of highest logit score
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    return accuracy

def run_experiment(use_defense=False, epochs=5):
    # Set execution device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Running Experiment | Data Augmentation Defense: {use_defense} ---")
    print(f"Using device: {device}")

    # 1. Fetch data arrays from our data engine
    train_loader, test_clean_loader, test_corrupt_loader = get_data_loaders(batch_size=64, use_augmentation=use_defense)

    # 2. Load and structurally modify the ResNet-18 model
    model = models.resnet18(pretrained=False) # Fresh, un-trained network
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 10) # Swap final layer to match CIFAR-10's 10 outputs
    model = model.to(device)

    # 3. Configure Loss criteria and Optimizer parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 4. THE CORE 4-STEP TRAINING LOOP
    for epoch in range(epochs):
        model.train() # Active training mode
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # --- THE 4 MANDATORY STEPS ---
            outputs = model(images)           # Step 1: Forward Pass (Take a guess)
            loss = criterion(outputs, labels) # Step 2: Compute Loss (Evaluate error)
            
            optimizer.zero_grad()             # Reset previous step's gradient memory
            loss.backward()                   # Step 3: Backward Pass (Backpropagate gradients)
            optimizer.step()                  # Step 4: Optimizer Step (Update model weights)
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] complete. Average Operational Loss: {epoch_loss:.4f}")

    # 5. TEST STRESS TEST EVALUATION
    clean_acc = evaluate_model(model, test_clean_loader, device)
    corrupt_acc = evaluate_model(model, test_corrupt_loader, device)

    print(f"\n>> Final Performance Benchmarks:")
    print(f"Accuracy on Clean Test Images (Source Domain): {clean_acc:.2f}%")
    print(f"Accuracy on Corrupted Test Images (Target Domain): {corrupt_acc:.2f}%")
    print(f"Quantified Domain Gap: {clean_acc - corrupt_acc:.2f}%")
    if use_defense:
        torch.save(model, "robust_resnet18.pt")

if __name__ == '__main__':
    # Experiment A: Baseline training (No defenses against noise)
    run_experiment(use_defense=False, epochs=5)
    
    # Experiment B: Strategic training (With our Data Augmentation Pipeline defense enabled)
    run_experiment(use_defense=True, epochs=5)
    