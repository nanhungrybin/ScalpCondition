import torch
import numpy as np
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score


def Multi_test(model, test_dataloader, device, loss_function, threshold=0.3):
    model.eval()  # Set the model to evaluation mode
    total_hamming_accuracy = 0  # To calculate Hamming accuracy
    total_exact_match = 0  # To calculate Subset Accuracy
    total_samples = 0  # Total number of samples
    
    total_loss = 0  # To calculate total loss

    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient calculation
        for X, y in test_dataloader:
            X = X.to(device)
            y = y.to(device)

            # Get model predictions
            output = model(X)
            
            # Compute loss
            loss = loss_function(output, y.float())  # Make sure labels are in float for BCEWithLogitsLoss
            total_loss += loss.item() * X.size(0)
            
            output = torch.sigmoid(output)  # Apply sigmoid to convert to probabilities

            # Apply threshold to determine predicted labels
            preds = (output > threshold).int()
            preds = preds.view_as(y)
            ######## swim model => preds가 4차원
            # preds = preds.float() 
            # preds = torch.mean(preds, dim=(2, 3))
            ##################################

            # Store predictions and labels for F1 calculation
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            # Calculate Hamming Accuracy
            total_hamming_accuracy += (preds == y).float().mean().item()

            # Calculate Exact Match (Subset Accuracy)
            total_exact_match += (preds == y).all(dim=1).float().mean().item()

            total_samples += y.size(0)

    # Calculate average metrics
    avg_hamming_accuracy = total_hamming_accuracy / total_samples
    avg_exact_match = total_exact_match / total_samples
    
    avg_loss = total_loss / total_samples
    
    f1_samples = f1_score(all_labels, all_preds, average='samples', zero_division=1)  # Calculate sample-wise F1 score for multi-label classification
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=1)  # Weighted F1 score across all labels
    
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=1)  # F1 score for each class
    
    jaccard = jaccard_score(all_labels, all_preds, average='samples', zero_division=1)  # Sample-wise Jaccard score
    precision = precision_score(all_labels, all_preds, average='samples', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='samples', zero_division=1)
    
    return all_preds, all_labels, avg_hamming_accuracy, avg_exact_match, avg_loss, f1_samples, f1_weighted, f1_per_class, jaccard, precision, recall


