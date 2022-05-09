import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from .backbones.resnet import AIBNResNet, TNormResNet


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(
            m.weight.data, a=0,
            mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)



class ft_net_intra_TNorm(nn.Module):
    def __init__(self, num_classes, stride=1, init_weight=0.1):
        super(ft_net_intra_TNorm, self).__init__()
        model_ft = TNormResNet(domain_number=len(num_classes),
                               last_stride=stride,
                               layers=[3, 4, 6, 3],
                               init_weight=init_weight)

        self.model = model_ft
        self.classifier = nn.ModuleList([
            nn.Sequential(nn.BatchNorm1d(2048), nn.Linear(2048,
                                                          num,
                                                          bias=False))
            for num in num_classes
        ])
        for classifier_one in self.classifier:
            init.normal_(classifier_one[1].weight.data, std=0.001)
            init.constant_(classifier_one[0].weight.data, 1.0)
            init.constant_(classifier_one[0].bias.data, 0.0)
            classifier_one[0].bias.requires_grad_(False)

    def backbone_forward(self, x, domain_index=None, convert=False):
        x = self.model(x, domain_index=domain_index, convert=convert)
        return x

    def forward(self, x, k=0, convert=False):
        x = self.backbone_forward(x, domain_index=k, convert=convert)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier[k](x)
        return x


class ft_net_inter_TNorm(nn.Module):
    def __init__(self, num_classes, domain_number, stride=1, init_weight=0.1):
        super(ft_net_inter_TNorm, self).__init__()
        # domain number only for param initialization has no meaning
        model_ft = TNormResNet(domain_number,
                               last_stride=stride,
                               layers=[3, 4, 6, 3],
                               init_weight=init_weight
                               )

        self.model = model_ft
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(2048), nn.Linear(2048, num_classes, bias=False))
        init.normal_(self.classifier[1].weight.data, std=0.001)
        init.constant_(self.classifier[0].weight.data, 1.0)
        init.constant_(self.classifier[0].bias.data, 0.0)
        self.classifier[0].bias.requires_grad_(False)

    def backbone_forward(self, x, domain_index=None, convert=False):
        x = self.model(x, domain_index=domain_index, convert=convert)
        return x

    def forward(self, x, domain_index=None, convert=False):
        x = self.backbone_forward(x, domain_index=domain_index, convert=convert)
        x = x.view(x.size(0), x.size(1))
        prob = self.classifier(x)
        return prob, x


class ft_net_intra(nn.Module):
    def __init__(self, num_classes, stride=1, init_weight=0.1, with_permute_adain=False):
        super(ft_net_intra, self).__init__()
        model_ft = AIBNResNet(last_stride=stride,
                              layers=[3, 4, 6, 3],
                              init_weight=init_weight,
                              with_permute_adain=with_permute_adain)

        self.model = model_ft
        self.classifier = nn.ModuleList([
            nn.Sequential(nn.BatchNorm1d(2048), nn.Linear(2048,
                                                          num,
                                                          bias=False))
            for num in num_classes
        ])
        for classifier_one in self.classifier:
            init.normal_(classifier_one[1].weight.data, std=0.001)
            init.constant_(classifier_one[0].weight.data, 1.0)
            init.constant_(classifier_one[0].bias.data, 0.0)
            classifier_one[0].bias.requires_grad_(False)

    def backbone_forward(self, x):
        x = self.model(x)
        return x

    def forward(self, x, k=0):
        x = self.backbone_forward(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier[k](x)
        return x


class ft_net_inter(nn.Module):
    def __init__(self, num_classes, stride=1, init_weight=0.1, with_permute_adain=False):
        super(ft_net_inter, self).__init__()
        model_ft = AIBNResNet(last_stride=stride,
                              layers=[3, 4, 6, 3],
                              init_weight=init_weight,
                              with_permute_adain=with_permute_adain)

        self.model = model_ft
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(2048), nn.Linear(2048, num_classes, bias=False))
        init.normal_(self.classifier[1].weight.data, std=0.001)
        init.constant_(self.classifier[0].weight.data, 1.0)
        init.constant_(self.classifier[0].bias.data, 0.0)
        self.classifier[0].bias.requires_grad_(False)

    def backbone_forward(self, x):
        x = self.model(x)
        return x

    def forward(self, x):
        x = self.backbone_forward(x)
        x = x.view(x.size(0), x.size(1))
        prob = self.classifier(x)
        return prob, x


class ft_net_both(nn.Module):
    def __init__(self, cam_num_classes, global_num_classes, stride=1):
        super(ft_net_both, self).__init__()
        model_ft = AIBNResNet(last_stride=stride, layers=[3, 4, 6, 3])

        self.model = model_ft
        self.bn_neck = nn.BatchNorm1d(2048)
        self.global_classifier = nn.Linear(2048,
                                           global_num_classes,
                                           bias=False)
        self.classifier = nn.ModuleList([
            nn.Linear(2048, cam_num_classes[i], bias=False)
            for i in range(len(cam_num_classes))
        ])
        self.intra_loss = nn.CrossEntropyLoss()
        init.normal_(self.global_classifier.weight.data, std=0.001)

        init.constant_(self.bn_neck.weight.data, 1.0)
        init.constant_(self.bn_neck.bias.data, 0.0)
        for cam in self.classifier:
            init.normal_(cam.weight.data, std=0.001)
        self.bn_neck.bias.requires_grad_(False)

    def backbone_forward(self, x):
        x = self.model(x)
        return x

    def forward(self, x, targets, domain_targets, camid):
        x = self.backbone_forward(x)
        x = x.view(x.size(0), x.size(1))
        prob = self.bn_neck(x)
        global_prob = self.global_classifier(prob)
        unique_camids = torch.unique(camid)
        intra_loss = 0
        for index, current in enumerate(unique_camids):
            current_camid = (camid == current).nonzero().view(-1)
            data = torch.index_select(prob, index=current_camid, dim=0)
            pids = torch.index_select(domain_targets,
                                      index=current_camid,
                                      dim=0)
            intra_out = self.classifier[current](data)
            intra_loss = intra_loss + self.intra_loss(intra_out, pids)
        intra_loss /= len(unique_camids)
        return global_prob, x, intra_loss


class ft_net_intra_specific(nn.Module):
    def __init__(self, domain_number, num_classes, stride=1):
        super(ft_net_intra_specific, self).__init__()
        model_ft = CameraAIBNResNet(domain_number=domain_number,
                                    last_stride=stride,
                                    layers=[3, 4, 6, 3])

        self.model = model_ft
        self.bn_neck = CameraBNorm1d(2048, domain_number)
        self.classifier = nn.ModuleList(
            [nn.Linear(2048, num, bias=False) for num in num_classes])
        for classifier_one in self.classifier:
            init.normal_(classifier_one.weight.data, std=0.001)
        init.constant_(self.bn_neck.weight.data, 1.0)
        init.constant_(self.bn_neck.bias.data, 0.0)
        self.bn_neck.bias.requires_grad_(False)

    def backbone_forward(self, x, domain_index, using_running=False):
        x = self.model(x, domain_index, using_running)
        return x

    def forward(self, x, k=0, using_running=False):
        x = self.backbone_forward(x, k, using_running)
        x = x.view(x.size(0), x.size(1))
        x = self.bn_neck(x, k, using_running)
        x = self.classifier[k](x)
        return x


class ft_net_inter_specific(nn.Module):
    def __init__(self, domain_number, num_classes, stride=1):
        super(ft_net_inter_specific, self).__init__()
        model_ft = CameraAIBNResNet(domain_number=domain_number,
                                    last_stride=stride,
                                    layers=[3, 4, 6, 3])
        self.model = model_ft
        self.bn_neck = CameraBNorm1d(2048, domain_number)
        self.classifier = nn.Linear(2048, num_classes, bias=False)
        init.normal_(self.classifier.weight.data, std=0.001)
        init.constant_(self.bn_neck.weight.data, 1.0)
        init.constant_(self.bn_neck.bias.data, 0.0)
        self.bn_neck.bias.requires_grad_(False)

    def backbone_forward(self, x, domain_index, using_running=True):
        x = self.model(x, domain_index, using_running=using_running)
        return x

    def forward(self, x, domain_index, targets, using_running=True):
        unique_camids = torch.unique(domain_index)
        success = 0
        for index, current in enumerate(unique_camids):
            current_camid = (domain_index == current).nonzero().view(-1)
            if current_camid.size(0) > 1:
                data = torch.index_select(x, index=current_camid, dim=0)
                pids = torch.index_select(targets, index=current_camid, dim=0)
                out = self.backbone_forward(data, current, False)
                out = out.view(out.size(0), out.size(1))
                out = self.bn_neck(out, current, using_running=False)
                if success == 0:
                    out_features = out
                    out_targets = pids
                else:
                    out_features = torch.cat((out_features, out), dim=0)
                    out_targets = torch.cat((out_targets, pids), dim=0)
                success += 1
        prob = self.classifier(out_features)
        return prob, out_features, out_targets


class ft_net_test(nn.Module):
    def __init__(self, domain_number, stride=1):
        super(ft_net_test, self).__init__()
        model_ft = CameraAIBNResNet(domain_number=domain_number,
                                    last_stride=stride,
                                    layers=[3, 4, 6, 3])

        self.model = model_ft

    def backbone_forward(self, x, domain_index, using_running=False):
        x = self.model(x, domain_index, using_running)
        return x

    def forward(self, x, k=0, using_running=False):
        x = self.backbone_forward(x, k, using_running)
        x = x.view(x.size(0), x.size(1))
        return x
