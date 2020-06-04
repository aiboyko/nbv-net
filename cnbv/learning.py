# Validation Function
import torch

def validation(model, testloader, criterion, device='cpu'):
    test_loss = 0
    accuracy = 0
    N = 0
    N_equal = 0
    for X, y in testloader:
        N += X.shape[0]
        # get sample data: images and ground truth keypoints
        labels = y.T[0].to(device)
        model.to(device)
        output = model.forward(X.to(device).view(-1, 1, 32, 32, 32))
        test_loss += criterion(output, labels)

        # for log.  ps = torch.exp(output)
        N_equal += sum((output.max(dim=1)[1] == labels))
    
    accuracy = N_equal / N
    return test_loss, accuracy

def train(model, optimizer, train_dataloader,  test_dataloader, criterion, device='cpu', calculate_eval=False, epochs=400):

    # Training loop
    running_loss = 0
    save_after = 100

    history_epoch = []
    history_train_loss = []
    history_validation_loss = []
    history_train_accuracy = []
    history_validation_accuracy = []

    import time
    for e in range(epochs):
        model.train()
        tic = time.time()
        cumulative_loss = 0
        for i, (X, y) in enumerate(train_dataloader):
            # print(X.shape)
            optimizer.zero_grad()
            model.to(device)
            output = model(X.to(device).view(-1, 1, 32, 32, 32))
            loss = criterion(output, y.to(device).T[0])
            loss.backward()
            cumulative_loss += loss
            optimizer.step()

        history_train_loss.append(cumulative_loss.detach().cpu().numpy())
        history_epoch.append(e)


        if calculate_eval == True:
            model.eval()
            with torch.no_grad():
                train_loss, train_accuracy = validation(model, train_dataloader, criterion, device)
                val_loss, val_accuracy = validation(model, test_dataloader, criterion,device)
                train_accuracy = train_accuracy / len(train_dataloader)
                val_accuracy = val_accuracy / len(test_dataloader)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                    "Training Loss: {:.3f}.. ".format(train_loss),
                    "Val. Loss: {:.3f}.. ".format(val_loss),
                    "Train Accuracy: {:.3f}".format(train_accuracy),
                    "Val. Accuracy: {:.3f}".format(val_accuracy))
            
            history_validation_loss.append(val_loss)
            history_train_accuracy.append(train_accuracy)
            history_validation_accuracy.append(val_accuracy)

        print(time.time() - tic)
        print('---')
