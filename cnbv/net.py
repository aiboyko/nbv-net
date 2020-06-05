import torch.nn as nn
import torch.nn.functional as F

class NBV_Net_Main(nn.Module):
    
    def __init__(self, dropout_prob):
        
        super(NBV_Net_Main, self).__init__()
        
        # Three 3D convolutional layers
        self.conv1 = nn.Conv3d(1, 10, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(2,2,2), stride = (2,2,2))
        
        self.conv2 = nn.Conv3d(10, 12, 3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2,2,2), stride = (2,2,2))
        
        self.conv3 = nn.Conv3d(12, 8, 3, stride=1, padding=1)
        self.conv3_drop = nn.Dropout(dropout_prob)
        self.pool3 = nn.MaxPool3d(kernel_size=(2,2,2), stride = (2,2,2))
    
    
    def forward(self, x):
        ## feedforward behavior of NBV-net
        x = self.pool1(F.relu(self.conv1(x)))
        
        x = self.pool2(F.relu(self.conv2(x)))
        
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Aplanar
        x = x.view(x.size(0), -1)
        
        return x


class NBV_Net_1FC(NBV_Net_Main):
    def __init__(self, dropout_prob):
        NBV_Net_Main.__init__(self, dropout_prob)
        self.fc1 = nn.Linear(512, 14)
    
    def forward(self, x):
        x = NBV_Net_Main.forward(self, x)
        x = F.relu(self.fc1(x))
        x = F.softmax(x, dim=1)
        return x


class NBV_Net_2FC(NBV_Net_Main):
    def __init__(self, dropout_prob):
        NBV_Net_Main.__init__(self, dropout_prob)
        self.fc1 = nn.Linear(512, 300)
        self.fc1_drop = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(300, 14)
    
    def forward(self, x):
        x = NBV_Net_Main.forward(self, x)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = F.softmax(x, dim=1)
        return x


class NBV_Net_3FC(NBV_Net_Main):
    def __init__(self, dropout_prob):
        NBV_Net_Main.__init__(self, dropout_prob)
        self.fc1 = nn.Linear(512, 300)
        self.fc1_drop = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(300, 100)
        self.fc2_drop = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(100, 14)
    
    def forward(self, x):
        x = NBV_Net_Main.forward(self, x)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = F.relu(self.fc3(x))
        x = F.softmax(x, dim=1)
        return x

class NBV_Net_4FC(NBV_Net_Main):
    def __init__(self, dropout_prob):
        NBV_Net_Main.__init__(self, dropout_prob)
        self.fc1 = nn.Linear(512, 1200)
        self.fc1_drop = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(1200, 500)
        self.fc2_drop = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(500, 100)
        self.fc3_drop = nn.Dropout(dropout_prob)
        self.fc4 = nn.Linear(100, 14)
    
    def forward(self, x):
        x = NBV_Net_Main.forward(self, x)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = F.relu(self.fc3(x))
        x = self.fc3_drop(x)
        x = F.relu(self.fc4(x))
        x = F.softmax(x, dim=1)
        return x


class NBV_Net_5FC(NBV_Net_Main):
    def __init__(self, dropout_prob):
        NBV_Net_Main.__init__(self, dropout_prob)
        self.fc1 = nn.Linear(512, 1200)
        self.fc1_drop = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(1200, 500)
        self.fc2_drop = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(500, 100)
        self.fc3_drop = nn.Dropout(dropout_prob)
        self.fc4 = nn.Linear(100, 50)
        self.fc4_drop = nn.Dropout(dropout_prob)
        self.fc5 = nn.Linear(50, 14)
    
    def forward(self, x):
        x = NBV_Net_Main.forward(self, x)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = F.relu(self.fc3(x))
        x = self.fc3_drop(x)
        x = F.relu(self.fc4(x))
        x = self.fc4_drop(x)
        x = F.relu(self.fc5(x))
        
        x = F.softmax(x, dim=1)
        return x
