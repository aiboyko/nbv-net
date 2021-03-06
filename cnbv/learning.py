# Training and testing functions
import torch
import time
import sys
import os

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


def test(model, testloader, criterion):
    device = next(model.parameters()).device
    test_loss = 0
    accuracy = 0
    for (X, y) in testloader:
        
        # get sample data: images and ground truth keypoints
        X = X.to(device)
        y = y.to(device).flatten()
                       
        output = model.forward(X)
        test_loss += criterion(output, y).item()
        
        # for log.  ps = torch.exp(output)
        correct_labels = (nbvs.data == output.max(dim=1)[1])
        accuracy += correct_labels.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy



def train(model, optimizer, train_dataloader, criterion,
          epochs=400, name_of_experiment="", log_dir='/log', test_dataloader=None):
    
    device = next(model.parameters()).device
    running_loss = 0
    save_after = 1
    
    history_epoch = []
    history_train_loss = []
    history_test_loss = []
    history_train_accuracy = []
    history_test_accuracy = []
    
    path_to_log = os.path.join(log_dir)
    
    for e in range(epochs):
        tic = time.time()
        for i, (X, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = model(X.to(device))
            loss = criterion(output, y.flatten().to(device))
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            train_loss, train_accuracy = test(model, train_dataloader, criterion)
            train_accuracy = train_accuracy / len(train_dataloader)
            
            history_train_loss.append(train_loss)
            history_train_accuracy.append(train_accuracy)
            
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(train_loss),
                  "Training Accuracy: {:.3f}.. ".format(train_accuracy))
                
            if test_dataloader is not None:
                test_loss, test_accuracy = test(model, test_dataloader, criterion)
                test_accuracy = test_accuracy / len(test_dataloader)
                              
                history_test_loss.append(test_loss)
                history_test_accuracy.append(test_accuracy)
                                      
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                        "Test Loss: {:.3f}.. ".format(val_loss),
                        "Test Accuracy: {:.3f}".format(val_accuracy))
                  
                if (e % save_after) == 0:
                    np.save(path_to_log + 'train_loss' + name_of_experiment, history_train_loss)
                    np.save(path_to_log + 'train_accuracy' + name_of_experiment, history_train_accuracy)
                    torch.save(net.state_dict(), path_to_log + 'weights' + name_of_experiment + '.pth')
                                                  
                    if calculate_eval==True:
                        np.save(path_to_log + 'test_loss' + name_of_experiment, history_test_loss)
                        np.save(path_to_log + 'test_accuracy' + name_of_experiment, history_test_accuracy)
                                                              
        toc = time.time()
        print('time per epoch = ', toc - tic)
        print('-----')












