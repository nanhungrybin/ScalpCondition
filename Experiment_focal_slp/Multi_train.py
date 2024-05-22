import torch
from sklearn.metrics import f1_score, jaccard_score
import numpy as np
import torch.nn.functional as F

# def Multi_train(model, train_dataloader, epoch, num_epochs, optimizer, loss_function, device, save_path):
#     model.train()  # Set the model to training mode

#     total_loss = 0  # Total loss for the epoch
#     total_hamming_accuracy = 0  # To calculate Hamming accuracy
#     total_exact_match = 0  # To calculate Subset Accuracy
    
#     num_batches = len(train_dataloader)  # Number of batches
    
#     all_preds = []
#     all_labels = []

#     for X, y in train_dataloader:
#         X = X.to(device) #torch.Size([32, 3, 224, 224])
#         y = y.to(device) #torch.Size([32, 7])

#         optimizer.zero_grad()  # Zero the gradients

#         # Forward pass
#         output = model(X) #torch.Size([32, 7, 7, 7]) 모델의 출력은 (batch_size, num_classes) 모양이어야함

#         # Compute loss
#         loss = loss_function(output, y.float())  # 손실함수에서 이미 output, y.float()의 차원 크기 맞춤
#         total_loss += loss.item()

#         # Backward pass and optimize
#         loss.backward()
#         optimizer.step()

#         # Convert output probabilities to binary output
#         predictions = (torch.sigmoid(output) > 0.5).int() #torch.Size([32, 7, 7, 7])
#         predictions = predictions.float() 
#         predictions = torch.mean(predictions, dim=(2, 3))
        
#         # Store predictions and labels for later metric calculation
#         all_preds.append(predictions.detach().cpu().numpy())
#         all_labels.append(y.detach().cpu().numpy())

#         # Calculate Hamming Accuracy
#         # print(y.shape, predictions.shape)
  
#         total_hamming_accuracy += (predictions == y).float().mean().item()

#         # Calculate Exact Match (Subset Accuracy)
#         total_exact_match += (predictions == y).all(dim=1).float().mean().item()

#     avg_loss = total_loss / num_batches  # Compute average loss
#     avg_hamming_accuracy = total_hamming_accuracy / num_batches
#     avg_exact_match = total_exact_match / num_batches
    
#     # Combine all predictions and labels
#     all_preds = np.vstack(all_preds)
#     all_labels = np.vstack(all_labels)

#     # Calculate F1 and Jaccard scores
#     f1_samples = f1_score(all_labels, all_preds, average='samples')
#     f1_weighted = f1_score(all_labels, all_preds, average='weighted')
#     jaccard = jaccard_score(all_labels, all_preds, average='samples')

#     # Print training status
#     print(f"[Epoch: {epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f}, Hamming Accuracy: {avg_hamming_accuracy:.4f}, Exact Match: {avg_exact_match:.4f}, F1 (Samples): {f1_samples:.4f}, F1 (Weighted): {f1_weighted:.4f}, Jaccard: {jaccard:.4f}")

#     # Save model weights
#     if save_path is not None:
#         torch.save(model.state_dict(), save_path)

#     return model, avg_loss, avg_hamming_accuracy, avg_exact_match, f1_samples, f1_weighted, jaccard


#VIT
def Multi_train(model, train_dataloader, epoch, num_epochs, optimizer, loss_function, device, save_path):
    model.train()  # Set the model to training mode

    total_loss = 0  # Total loss for the epoch
    total_hamming_accuracy = 0  # To calculate Hamming accuracy
    total_exact_match = 0  # To calculate Subset Accuracy
    
    num_batches = len(train_dataloader)  # Number of batches
    
    all_preds = []
    all_labels = []

    for X, y in train_dataloader:
        X = X.to(device) #torch.Size([32, 3, 224, 224])
        y = y.to(device) #torch.Size([32, 7])

        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        output = model(X) #torch.Size([32, 7, 7, 7]) 모델의 출력은 (batch_size, num_classes) 모양이어야함

        # Compute loss
        loss = loss_function(output, y.float())  # 손실함수에서 이미 output, y.float()의 차원 크기 맞춤
        total_loss += loss.item()

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Convert output probabilities to binary output
        predictions = (torch.sigmoid(output) > 0.5).int() #torch.Size([32, 7, 7, 7])
        # predictions = predictions.float() 
        # predictions = torch.mean(predictions, dim=(2, 3))
        predictions = predictions.view_as(y)

        
        # Store predictions and labels for later metric calculation
        all_preds.append(predictions.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

        # Calculate Hamming Accuracy
        # print(y.shape, predictions.shape)
  
        total_hamming_accuracy += (predictions == y).float().mean().item()

        # Calculate Exact Match (Subset Accuracy)
        total_exact_match += (predictions == y).all(dim=1).float().mean().item()

    avg_loss = total_loss / num_batches  # Compute average loss
    avg_hamming_accuracy = total_hamming_accuracy / num_batches
    avg_exact_match = total_exact_match / num_batches
    
    # Combine all predictions and labels
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Calculate F1 and Jaccard scores
    f1_samples = f1_score(all_labels, all_preds, average='samples')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    jaccard = jaccard_score(all_labels, all_preds, average='samples')

    # Print training status
    print(f"[Epoch: {epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f}, Hamming Accuracy: {avg_hamming_accuracy:.4f}, Exact Match: {avg_exact_match:.4f}, F1 (Samples): {f1_samples:.4f}, F1 (Weighted): {f1_weighted:.4f}, Jaccard: {jaccard:.4f}")

    # Save model weights
    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    return model, avg_loss, avg_hamming_accuracy, avg_exact_match, f1_samples, f1_weighted, jaccard