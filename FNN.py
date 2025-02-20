import torch
import torch.nn as nn
import torch.nn.functional as F # we can get functions like F.softmax(...,dim=...)
import torch.utils.data as data

import matplotlib.pyplot as plt
import seaborn as sns

class simple_NN(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim, depth):
        super().__init__() # this is used to call the __init__ of the parent class
        self.input_dim=input_dim
        self.out_dim=out_dim
        self.hidden_dim=hidden_dim
        self.depth=depth
        self.layers=[nn.Linear(self.input_dim,self.hidden_dim), nn.ReLU()] # adding activation function after the layer
        for i in torch.arange(self.depth):
            self.layers=self.layers+[nn.Linear(self.hidden_dim,self.hidden_dim), nn.ReLU()]
        self.layers=self.layers+[nn.Linear(self.hidden_dim,self.out_dim)]
        self.all_layers=nn.Sequential(*self.layers) # * is used to unpack the list of layers
    
    def forward(self, x):
        return self.all_layers(x)

model=simple_NN(2,1,6,2)    

for name, pram in model.named_parameters(): # initialization following Xavier
    if "bias" in name:
        pram.data.fill_(0)
    elif "weight" in name:
        pram.data.normal_(std=2**(1/2)/(pram.shape[0]+pram.shape[1])**(1/2))

class sum_data(data.Dataset):
    def __init__(self, datasize):
        super().__init__()
        self.datasize=datasize
        self.data=torch.randn((self.datasize, 2),dtype=torch.float32) # prepare the data to have mean close to zero and std close to 1
        self.label=torch.sum(self.data, dim=1)

    def __len__(self):
        return self.datasize
    
    def __getitem__(self,idx):
        return self.data[idx], self.label[idx]

dataset=sum_data(1000)
dataloader=data.DataLoader(dataset, batch_size=10,shuffle=True)

loss_function=nn.MSELoss()
# optimizer=torch.optim.SGD(model.parameters(), lr=0.1) 

class optimizer_SGD: # custom optimizer
    def __init__(self,pram, lr):
        self.pram=pram
        self.lr=lr
    def zero_grad(self):
        for pram in self.pram:
            if pram.grad is not None:
                pram.grad.data.detach_() # detach the grad from the computational graph
                pram.grad.data.zero_()
    def step(self):
        for pram in self.pram:
            if pram.grad is not None:
                pram.data.add_(-self.lr*pram.grad) # inplace update

optimizer=optimizer_SGD(model.parameters(), lr=0.1)


def calculate_gradients(model, dataloader, loss_function):
    model=model.to("mps")
    model.train()
    model.zero_grad()
    small_sample=next(iter(dataloader))
    data, label=small_sample
    data=data.to("mps")
    label=label.to("mps")
    output=model(data).view(-1,1)
    loss_val=loss_function(output, label)
    loss_val.backward()
    grad_profile={name: pram.grad.data.view(-1).cpu().numpy() for name, pram in model.named_parameters() if "weight" in name} # flattening out the grad
    return grad_profile

grads=calculate_gradients(model, dataloader, loss_function)
columns = len(grads)
fig, ax = plt.subplots(1, columns, figsize=(columns*3.5, 2.5))
fig_index = 0
for key in grads:
    key_ax = ax[fig_index%columns]
    sns.histplot(data=grads[key], bins=30, ax=key_ax, kde=True)
    key_ax.set_title(str(key))
    key_ax.set_xlabel("Grad magnitude")
    fig_index += 1
fig.suptitle(f"Gradient magnitude distribution", fontsize=14, y=1.05)
plt.savefig('gradient_distribution.png', dpi=300, bbox_inches='tight')

def calculate_activations(model, dataloader):
    model=model.to("mps")
    model.eval()
    small_sample=next(iter(dataloader))
    data, label=small_sample
    data=data.to("mps")
    label=label.to("mps")
    activation_profile={}
    with torch.no_grad():
        for layer_index, layer in enumerate(model.layers):
            data=layer(data)
            if isinstance(layer, nn.ReLU):
                activation_profile[f"{layer_index}"]=data.view(-1).to("cpu").numpy() # detach() is used to prevent backpropagation, here we are using torch.nograd so we don't have to use detach
    return activation_profile

activations=calculate_activations(model, dataloader)
columns = len(activations)
fig, ax = plt.subplots(1, columns, figsize=(columns*3.5, 2.5))
fig_index = 0
for key in activations:
    key_ax = ax[fig_index%columns]
    sns.histplot(data=activations[key], bins=30, ax=key_ax, kde=True)
    key_ax.set_title(str(key))
    key_ax.set_xlabel("Activation magnitude")
    fig_index += 1
fig.suptitle(f"Activation magnitude distribution", fontsize=14, y=1.05)
plt.savefig('activation_distribution.png', dpi=300, bbox_inches='tight')

def train(model, dataloader, loss_function, optimizer, num_epoch=2):
        model=model.to("mps")
        model.train()
        for epoch in torch.arange(num_epoch):
            epoch_loss = 0
            for data, label in dataloader:
                data=data.to("mps")
                label=label.to("mps")
                optimizer.zero_grad() # forget the previous gradients
                output=model(data).view(-1, 1)  # Reshape to (batch_size, 1)
                loss_val=loss_function(output, label)
                loss_val.backward() # backpropagation to calculate the gradients
                max_grad_norm=10.0
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) # gradient clipping
                optimizer.step() # update the weights
                epoch_loss += loss_val.item()
            print(f"Epoch {epoch+1} Loss: {epoch_loss/len(dataloader)}")
            #print("dataloader size: ",len(dataloader))

train(model, dataloader, loss_function, optimizer, num_epoch=10)

test_dataset=sum_data(10)
test_dataloader=data.DataLoader(test_dataset, batch_size=10, shuffle=False)

def val_loss(model, test_dataloader,loss_function):
    model=model.to("mps")
    model.eval() # set the model to evaluation mode
    loss_val=0
    with torch.no_grad():
        for data, label in test_dataloader:
            data=data.to("mps")
            label=label.to("mps")
            output=model(data).view(-1,1)
            loss_val=loss_val+loss_function(output,label)
    return loss_val

print("val_loss: ",val_loss(model, test_dataloader, loss_function))