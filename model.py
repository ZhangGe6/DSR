import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
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
        init.constant_(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock_IV(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=False, num_bottleneck=512):
        super(ClassBlock_IV, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)] 
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        if dropout>0:
            classifier+= [nn.Dropout(p=dropout)]
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        f = x
        f_norm = f.norm(p=2, dim=1, keepdim=True) + 1e-8
        f = f.div(f_norm)
        x = self.classifier(x)
        return x,f

class ClassBlock_PCB(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock_PCB, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x

# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, mode):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        # 注意目前base、各种aug是在dropout=0.5下进行的，虽然没用到这一层但时确实有这一层，test时也应改回0.5,否则load model会出错. 
        self.classifier = ClassBlock_IV(2048, class_num, dropout=0, relu=False)
        
        # V2: last_conv_stride 2->1
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        self.mode = mode
        
        self.AvgPool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.AvgPool3 = nn.AvgPool2d(kernel_size=3, stride=3, padding=0)
        # 这个后面可能会删除
        self.fc = nn.Linear(2048, 751)
        
        # dropout - 性能会降#
        #self.dropout = nn.Dropout(p=0.2)

        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)  
        
        if self.mode == 'train_IV':
            x = self.model.avgpool(x)
            x = x.view(x.size(0), x.size(1))
            x, f = self.classifier(x)
            return x,f
            
        if self.mode == 'test_market':
            x = self.model.avgpool(x)
            x = x.view(x.size(0), x.size(1))
            x, f = self.classifier(x)
            return x
            
        # augmantation效果用    
        elif self.mode == 'train_CNN':
            x = self.model.avgpool(x)
            #x = self.dropout(x)      # 试一下dropout
            #print(x.size()) #128 2048 1 1
            x = x.view(x.size(0), -1)
            #print(x.size())    # 128 2048
            x = self.fc(x)
            return x
        elif self.mode == 'multi_DSR_dist':
            x2 = self.AvgPool2(x)
            x3 = self.AvgPool2(x)
            spatial_feat1 = x.view(x.size(0), x.size(1), x.size(2) * x.size(3))
            spatial_feat2 = x2.view(x2.size(0), x2.size(1), x2.size(2) * x2.size(3))
            #spatial_feat3 = x3.view(x3.size(0), x3.size(1), x3.size(2) * x3.size(3))
 
            x = F.avg_pool2d(x, x.size()[2:]) # Global feature
            # print(tmp.size(0), tmp.size(1)) # 1 2048 
            global_feat = x.view(x.size(0), -1)
            #print(global_feat.size())
            return global_feat, spatial_feat1, spatial_feat2
            
        elif self.mode == 'train_SFR+IV':
            x1 = x
            x1 = self.model.avgpool(x1)
            x1 = x1.view(x1.size(0), x1.size(1))
            pred, f = self.classifier(x1)
            
            spatial_feat = x.view(x.size(0), x.size(1), x.size(2) * x.size(3))
            x = F.avg_pool2d(x, x.size()[2:]) # Global feature
            # print(tmp.size(0), tmp.size(1)) # 1 2048 
            global_feat = x.view(x.size(0), -1)
            return pred, global_feat, spatial_feat 
            
        elif self.mode == 'test' or self.mode == 'train_SFR':
            ##
            '''
            weighted_feats = [] 
            for raw_feat in x:
                print(raw_feat.size())
                pixel_weight = feat.sum(dim=0).cpu().numpy()
                print(pixel_weight.size())
                weighted_feat = torch.mul(raw_feat, pixel_weight)
                weighted_feats.append(weighted_feat)   
            '''
            ##
            
            #print(x.size())
            ''' rank1-47% 不变
            heatmap = x.squeeze().sum(dim=0).cpu()
            heatmap = heatmap / torch.sum(heatmap)
            '''
            
            '''rank1-24.6% 
            heatmap = x.squeeze().sum(dim=0).cpu()
            w,h= heatmap.size()
            heatmap = heatmap.view(1,-1)
            #print(heatmap.size())
            heatmap = F.softmax(heatmap,dim=1)
            #print(heatmap)
            heatmap = heatmap.view(w,h)
            #print(heatmap.size())
            #print(heatmap.size())
            weighted_feat = torch.mul(x, heatmap)
            #print(weighted_feat.size())
            
            spatial_feat = weighted_feat.view(weighted_feat.size(0), weighted_feat.size(1), weighted_feat.size(2) * weighted_feat.size(3))
            '''
            spatial_feat = x.view(x.size(0), x.size(1), x.size(2) * x.size(3))
            x = F.avg_pool2d(x, x.size()[2:]) # Global feature
            # print(tmp.size(0), tmp.size(1)) # 1 2048 
            global_feat = x.view(x.size(0), -1)
            return global_feat,spatial_feat 

# Define a 2048 to 2 Model
class verif_net(nn.Module):
    def __init__(self):
        super(verif_net, self).__init__()
        #self.classifier = ClassBlock(512, 2, dropout=0.75, relu=False)
        # V2: dropout 0.75->0.5
        self.classifier = ClassBlock_IV(512, 2, dropout=0, relu=False)
    def forward(self, x):
        x = self.classifier.classifier(x)
        return x

# Define the DenseNet121-based Model
class ft_net_dense(nn.Module):

    def __init__(self, class_num ):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 1024 
        self.classifier = ClassBlock_IV(1024, class_num)

    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x
    
# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).
class ft_net_middle(nn.Module):

    def __init__(self, class_num ):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock_IV(2048+1024, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # x0  n*1024*1*1
        x0 = self.model.avgpool(x)
        x = self.model.layer4(x)
        # x1  n*2048*1*1
        x1 = self.model.avgpool(x)
        x = torch.cat((x0,x1),1)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x


# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num, mode):
        super(PCB, self).__init__()

        self.part = 6 # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        self.mode = mode
        
        #if self.mode == 'PCB':
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        self.maxpool = nn.AdaptiveMaxPool2d((self.part,1))
        #self.share_classifier = ClassBlock_PCB(2048, class_num, droprate=0, relu=False, bnorm=True, num_bottleneck=256)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock_PCB(2048, class_num, droprate=0, relu=False, bnorm=True, num_bottleneck=256))   # droprate=0.5改为0
      
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        if self.mode == "train_SFR+PCB":
            x1 = x
            spatial_feat = x1.view(x1.size(0), x1.size(1), x1.size(2) * x1.size(3))
            x1 = F.avg_pool2d(x1, x1.size()[2:]) # Global feature
            # print(tmp.size(0), tmp.size(1)) # 1 2048 
            global_feat = x1.view(x1.size(0), -1)
            
            x2 = self.avgpool(x)  
            part = {}
            predict = {}
            # get six part feature batchsize*2048*6
            for i in range(self.part):
                part[i] = torch.squeeze(x2[:,:,i])
                name = 'classifier'+str(i)
                c = getattr(self,name)
                predict[i] = c(part[i])

            # sum prediction
            #y = predict[0]
            #for i in range(self.part-1):
            #    y += predict[i+1]
            y = []
            for i in range(self.part):
                y.append(predict[i])
            
            return y, global_feat, spatial_feat
        
        elif self.mode == 'PCB':
            x_a = self.avgpool(x)  
            #x_m= self.maxpool(x)
            #x = x_a + x_m
            x = x_a
            #print(x.size())   # 128 2048 6 1
            #print(x[:,:,0].size()) # 128 2048 1
            # x = self.dropout(x)   # 先不dropout
            part = {}
            predict = {}
            # get six part feature batchsize*2048*6
            for i in range(self.part):
                part[i] = torch.squeeze(x[:,:,i])
                name = 'classifier'+str(i)
                c = getattr(self,name)
                predict[i] = c(part[i])

            # sum prediction
            #y = predict[0]
            #for i in range(self.part-1):
            #    y += predict[i+1]
            y = []
            for i in range(self.part):
                y.append(predict[i])
            return y
            
        elif self.mode == 'PCB_share_fc':
            x_a = self.avgpool(x)  
            #x_m= self.maxpool(x)
            #x = x_a + x_m
            x = x_a
            #print(x.size())   # 128 2048 6 1
            #print(x[:,:,0].size()) # 128 2048 1
            # x = self.dropout(x)   # 先不dropout
            part = {}
            predict = {}
            # get six part feature batchsize*2048*6
            for i in range(self.part):
                part[i] = torch.squeeze(x[:,:,i])
                # name = 'classifier'+str(i)
                # c = getattr(self,name)
                predict[i] = self.share_classifier(part[i])

            # sum prediction
            #y = predict[0]
            #for i in range(self.part-1):
            #    y += predict[i+1]
            y = []
            for i in range(self.part):
                y.append(predict[i])
            return y     
            
        elif self.mode == 'test':
            spatial_feat = x.view(x.size(0), x.size(1), x.size(2) * x.size(3))     
            x = F.avg_pool2d(x, x.size()[2:]) # Global feature
            # print(x.size(0), x.size(1)) # 1 2048 
            x = x.view(x.size(0), -1)
            return x, spatial_feat

# Part Model proposed in Yifan Sun etal. (2018)
class PPCB(nn.Module):
    def __init__(self, class_num, mode):
        super(PPCB, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        self.mode = mode
        
        self.avgpool1 = nn.AdaptiveAvgPool2d((1,1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((3,1))
        self.avgpool3 = nn.AdaptiveAvgPool2d((4,1))
        self.avgpool4 = nn.AdaptiveAvgPool2d((6,1))
        #self.avgpool4 = nn.AdaptiveAvgPool2d((8,1))
        # define 6 classifiers
        for i in range(1+3+6):
        #for i in range(6):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock_PCB(2048, class_num, droprate=0, relu=False, bnorm=True, num_bottleneck=256))   # droprate=0.5改为0
            
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
                        
        if self.mode == 'PPCB':
            x1 = self.avgpool1(x)   #1
            x2 = self.avgpool2(x)   #2
            x3 = self.avgpool3(x)   #4
            x4 = self.avgpool4(x)   #8
            x_t = [x1, x2, x4]
           

            part = {}
            predict = {}
            
            bins = [1,3,6]; bin_id = 0 #指拼接起来后的ID
            for scale_id in range(len(bins)):
                bin_num = bins[scale_id]
                for bd in range(bin_num):  #bd指bin的内部ID
                   part[bin_id] = torch.squeeze(x_t[scale_id][:,:,bd]) 
                   name = 'classifier'+str(bin_id); c = getattr(self,name)
                   predict[bin_id] = c(part[bin_id])
                   bin_id += 1
                #print(bin_id)
            #print(bin_id)
                  
            y = []
            for i in range(1+3+6):
                y.append(predict[i])
            return y 
     
        elif self.mode == 'test':
            spatial_feat = x.view(x.size(0), x.size(1), x.size(2) * x.size(3))     
            x = F.avg_pool2d(x, x.size()[2:]) # Global feature
            # print(x.size(0), x.size(1)) # 1 2048 
            x = x.view(x.size(0), -1)
            return x, spatial_feat
