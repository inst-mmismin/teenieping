import torch.nn as nn 

class teeniefier(nn.Module):
    def __init__(self, num_teenieping): 
        super().__init__()
        self.num_teenieping = num_teenieping 
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), 
            nn.ReLU(), 
            nn.Conv2d(16, 16, 3, 1, 1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 64, 3, 1, 1), 
            nn.ReLU(), 
            nn.Conv2d(64, 64, 3, 1, 1), 
            nn.ReLU(), 
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.layer3 = nn.Sequential(
            nn.Linear(64, 128),
            nn.Dropout(0,1), 
            nn.ReLU(), 
            nn.Linear(128, 512), 
            nn.Dropout(0,1), 
            nn.ReLU(), 
            nn.Linear(512, num_teenieping)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.squeeze(-1)
        x = x.squeeze(-1)
        x = self.layer3(x)
        
        return x 