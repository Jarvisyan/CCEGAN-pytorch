import torch
import lightly
from torch import nn
from lightly.models.modules.heads import SimSiamProjectionHead
from lightly.models.modules.heads import SimSiamPredictionHead


class Generator(nn.Module):
    def __init__(self, class_num, z_dim):
        super().__init__()
        self.input_height = 28
        self.input_width = 28
        self.input_dim = z_dim + class_num 
        self.output_dim = 1
        

        #FC_block: 72 -> 1024 -> 128*7*7
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.BatchNorm1d(128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.ReLU(),
        )
        #deconv_block: 128*7*7 -> 64*14*14 -> 1*28*28
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, X):
        X = self.fc(X).view(-1, 128, (self.input_height // 4), (self.input_width // 4))
        return self.deconv(X)     

class Discriminator(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.class_num = class_num
        
        #conv_block: 1*28*28 -> 64*14*14 -> 128*7*7 -> 256*3*3
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        #FC_block: 256*3*3 -> 1024 -> 11
        self.fc = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.class_num + 1), 
        )
        
    def forward(self, X):
        X = self.conv(X).view(-1, 256 * 3 * 3)
        return self.fc(X) 

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        #conv_block: 1*28*28 -> 64*14*14 -> 128*7*7 -> 256*3*3
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
    def forward(self, X):
        return self.conv(X).view(-1, 256 * 3 * 3)

class extended_SimSiam(nn.Module):
    def __init__(
        self, backbone, num_ftrs, proj_hidden_dim, pred_hidden_dim, out_dim
    ):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimSiamProjectionHead(
            num_ftrs, proj_hidden_dim, out_dim
        )
        self.prediction_head = SimSiamPredictionHead(
            out_dim, pred_hidden_dim, out_dim
        )

    def forward(self, x):
        # get representations
        f = self.backbone(x).flatten(start_dim=1)
        # get projections
        z = self.projection_head(f)

        z = nn.Softmax(dim=1)(z)
        H1 = (-1 * z * z.log()).sum() / z.shape[0] #average entropy
        emperical_z = z.mean(0)
        H2 = (-1 * emperical_z * emperical_z.log()).sum()
        # get predictions
        p = self.prediction_head(z)
        # stop gradient
        z = z.detach()
        return z, p, H1, H2

class Finetune_net(nn.Module):
    def __init__(self, backbone, class_num):
        super().__init__()
        self.features = backbone
        self.output_new = nn.Sequential(
                nn.Linear(256 * 3 * 3, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.2),
                nn.Linear(1024, class_num),
        )
    def forward(self, X):
        X = self.features(X).view(-1, 256 * 3 * 3)
        return self.output_new(X)

class E_net(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.class_num = class_num
        
        #conv_block: 1*28*28 -> 64*14*14 -> 128*7*7 -> 256*3*3
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        #FC_block: 256*3*3 -> 1024 -> 11
        self.fc = nn.Sequential(
            # nn.Linear(128 * 7 * 7, 1024),
            nn.Linear(256 * 3 * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.class_num), 
        )
        
    def forward(self, X):
        # X = self.conv(X).view(-1, 128 * 7 * 7)
        X = self.conv(X).view(-1, 256 * 3 * 3)
        return self.fc(X)











