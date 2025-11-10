import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder_AE_nobn(nn.Module):
    def __init__(self, opt, device):
        super(Decoder_AE_nobn, self).__init__()

        self.dense = nn.Linear(opt["latent_dim"], 1024)
        #S
        self.conv1 = nn.ConvTranspose2d(64, 64, 9, stride=2, padding=0)
        self.conv2 = nn.ConvTranspose2d(64, 32, 9, stride=2, padding=0)
        self.conv3 = nn.ConvTranspose2d(32, 16, 7, stride=2, padding=0)
        self.conv4 = nn.ConvTranspose2d(16, 1, 4, stride=1, padding=1)
#         self.train()

         #TODO inits?
#         if opt["initialization"]=="xavier":
#             self.apply(init_xavier)
#         elif opt["initialization"]=="adaptive":
#             self.apply(init_adaptive)
#         elif opt["initialization"]=="base":
#             self.apply(init_base)

#         relu_gain = nn.init.calculate_gain('relu')
#         self.conv1.weight.data.mul_(relu_gain)
#         self.conv2.weight.data.mul_(relu_gain)
#         self.conv3.weight.data.mul_(relu_gain)
#         self.conv4.weight.data.mul_(relu_gain)



    def forward(self, z):

        z = self.dense(z)

        z = z.view(z.size(0), 64, 4, 4)
        x = F.relu(self.conv1(z))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)

        return x
