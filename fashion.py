import torch.autograd
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
#from tensorboardX import SummaryWriter

#writer = SummaryWriter()

datapath = '/content/drive/MyDrive/'

train1 = pd.read_csv(datapath+'fashion-mnist_train.csv')
#train = train.values()
train = train1.values
Xtrain = train[:,1:]
Ytrain = train[:,0].reshape(len(train),1)

test1 = pd.read_csv('fashion-mnist_test_data.csv')
test = test1.values
#test = test.values()
Xtest = test[:,1:]
Ytest = test[:,0].reshape(len(test),1)


a = np.concatenate((Xtrain,Xtest))
ma = a.mean()
sa = a.std()
Xtrain = (Xtrain-ma)/sa
Xtest = (Xtest-ma)/sa
#device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.l1 = nn.Linear(784,32*16*8)
        self.t1 = nn.Tanh()

        self.cin = nn.Conv2d(in_channels=1,out_channels=20,kernel_size=5,padding=1)
        self.c2 = nn.Conv2d(in_channels=20,out_channels=10,kernel_size=5,padding=1)
        self.c3 = nn.Conv2d(in_channels=10,out_channels=1,kernel_size=3,padding=1)
        self.c1 = nn.Conv2d(in_channels=16,kernel_size=3,out_channels=4)
        self.f1 = nn.Flatten()
        self.l2 = nn.Linear(121,16)
        self.l3 = nn.Linear(16,10)
        self.mp = nn.MaxPool2d(2)



    def forward(self,x):
        x = x.view(-1,1,28,28)
        x = self.cin(x)
        x = self.mp(x)
        #print(x.shape)
        x = self.c2(x)
        x = self.c3(x)
        x = self.f1(x)
        x = self.l2(x)
        x = self.l3(x)
        return nn.functional.log_softmax(x,dim=1)

model = Net()

if torch.cuda.is_available():

    model = model.cuda()

    criterion = nn.NLLLoss()
    m_optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    num_epoch = 2

    for epoch in range(num_epoch):
        #model.train()
        for i in range(len(Xtrain)):

            img = torch.tensor(Xtrain[i],dtype=torch.float32,requires_grad=True).cuda()
            #label = Variable(torch.from_numpy(Ytrain[i],dtype = torch.float32)).cuda()
            label = torch.tensor(Ytrain[i]).cuda()
            result = model(img)
            #这里为什么会出现输出是【4，1】的情况，好奇怪
            #print(result)
            loss = criterion(result,label)

            m_optimizer.zero_grad()
            loss.backward()
            m_optimizer.step()

            if i%1000==0:
                print("Train Epoch: {} [{}/{}] \t Loss: {:.6f} " .format(epoch,i,len(Xtrain),loss.item() ))
                #writer.add_scalar('loss',loss.item(),global_step=(epoch*len(Xtrain)+i)/1000)

ans = []
idx = ['{}.jpg'.format(i) for i in range(len(Xtest))]
with torch.no_grad():
    for i in range(len(Xtest)):
        img = torch.tensor(Xtest[i],dtype=torch.float32,requires_grad=True).cuda()
        #ans.append(model(img))
        #temp = torch.mean(img)
        temp = model(img)
        temp = temp.cpu()
        temp = temp.numpy()
        #print("this is the origin",temp)
        temp = np.argmax(temp)
        #print("choose",temp)
        ans.append(temp)
res = pd.DataFrame(ans,index=idx)
res.to_csv('ans1.csv',header=False)