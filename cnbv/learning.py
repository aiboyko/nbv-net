# Validation Function

def validation(model, testloader, criterion, device='cpu'):
    test_loss = 0
    accuracy = 0
    N = 0
    N_equal = 0
    for X, y in testloader:
        N += X.shape[0]
        # get sample data: images and ground truth keypoints
        labels = y.T[0].to(device)
        output = model.forward(X.to(device).view(-1, 1, 32, 32, 32))
        test_loss += criterion(output, labels)

        # for log.  ps = torch.exp(output)
        N_equal += sum((output.max(dim=1)[1] == labels))
    
    accuracy = N_equal / N
    return test_loss, accuracy