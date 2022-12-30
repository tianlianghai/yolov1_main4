from torchvision import models
from torch import nn
class ResnetYolo(nn.Module):
    def __init__(self, feature_size=7, num_bboxes=2, num_classes=20) -> None:
        super().__init__()
        resnet = models.resnet18(weights= models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_size = feature_size
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.convs = self._make_conv_layers(True)
        self.fcs = self._make_fc_layers()

        self.net = nn.Sequential(
            self.features,
            self.convs,
            self.fcs
        )
    
    def forward(self, x):
        S = self.feature_size
        B = self.num_bboxes
        C = self.num_classes
        return self.net(x).view(-1, S, S, 5*B+C)
    
    def _make_conv_layers(self, bn):
        if bn:
            net = nn.Sequential(
                nn.LeakyReLU(0.1,),
                nn.Conv2d(512, 1024, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
                nn.LeakyReLU(0.1),

                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True)
            )

        else:
            net = nn.Sequential(
                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1),

                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1, inplace=True)
            )

        return net

    def _make_fc_layers(self):
        S, B, C = self.feature_size, self.num_bboxes, self.num_classes

        net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5, inplace=False), # is it okay to use Dropout with BatchNorm?
            nn.Linear(4096, S * S * (5 * B + C)),
            nn.Sigmoid()
        )

        return net