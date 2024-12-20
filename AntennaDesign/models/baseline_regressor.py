import torch
import torch.nn as nn

class baseline_forward_model(nn.Module):
    """
    model designed to find the regression between 12 geometric parameters as input and the spectrum parameters as output.
    for now spectrum parameters are 1001x2 = 2002 parameters.
    """
    def __init__(self,weight_range=0.1):
        super(baseline_forward_model,self).__init__()
        self.name = "baseline forward model"
        self.linear1 = nn.Linear(12, 30)
        self.linear2 = nn.Linear(30, 60)
        self.linear3 = nn.Linear(60, 120)
        self.linear4 = nn.Linear(120, 240)
        self.linear5 = nn.Linear(240, 240)
        self.linear6 = nn.Linear(240, 480)
        self.linear7 = nn.Linear(480, 480)
        self.linear8 = nn.Linear(480, 960)
        self.linear9 = nn.Linear(960, 960)
        self.linear10 = nn.Linear(960, 2002)
        self.relu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        #self.init_weights(weight_range)
    def init_weights(self,init_range):
        for p in self.parameters():
            p.data.uniform_(-init_range, init_range)
    def forward(self,input): # input is the geometric parameters
        scaled_params = input
        output = self.relu(self.linear1(scaled_params))
        output = self.relu(self.linear2(output))
        output = self.relu(self.linear3(output))
        output = self.relu(self.linear4(output))
        output = self.relu(self.linear5(output))
        output = self.relu(self.linear6(output))
        output = self.relu(self.linear7(output))
        output = self.relu(self.linear8(output))
        output = self.relu(self.linear9(output))
        output = self.linear10(output)
        output = torch.cat((self.sigmoid(output[:,:output.shape[1]//2]),output[:,output.shape[1]//2:]),dim=1) # first half is sigmoided because this is magnitude
        return output

class small_baseline_forward_model(nn.Module):
    """
    model designed to find the regression between 12 geometric parameters as input and the spectrum parameters as output.
    for now spectrum parameters are 251x2 = 502 parameters.
    """
    def __init__(self,weight_range=0.1):
        super(small_baseline_forward_model,self).__init__()
        self.name = "small baseline forward model"
        self.linear1 = nn.Linear(12, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 128)
        self.linear4 = nn.Linear(128, 256)
        self.linear5 = nn.Linear(256, 502)
        self.relu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        #self.init_weights(weight_range)
    def init_weights(self,init_range):
        for p in self.parameters():
            p.data.uniform_(-init_range, init_range)
    def forward(self,input): # input is the geometric parameters
        scaled_params = input
        output = self.relu(self.linear1(scaled_params))
        output = self.relu(self.linear2(output))
        output = self.relu(self.linear3(output))
        output = self.relu(self.linear4(output))
        output = self.linear5(output)
        output = torch.cat((self.sigmoid(output[:,:output.shape[1]//2]),output[:,output.shape[1]//2:]),dim=1) #magnitude is between 0 and 1
        return output

class small_deeper_baseline_forward_model(nn.Module):
    """
    Model designed to find the regression between 12 geometric parameters as input and the spectrum parameters as output.
    For now, spectrum parameters are 251x2 = 502 parameters.
    """
    def __init__(self, weight_range=0.1, p_dropout=0.25):
        super(small_deeper_baseline_forward_model, self).__init__()
        self.name = "small baseline forward model with added 2 layers"

        # Layers
        self.input_layer = None  # Initialize lazily later
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)  # Batch normalization for fc2
        self.fc3 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(512)  # Batch normalization for fc3
        self.fc4 = nn.Linear(512, 502)

        # Activations, Dropout, and Constants
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=p_dropout)
        self.eps = 1e-1

    def forward(self, input):  # input is the geometric parameters
        if self.input_layer is None:
            self.input_layer = nn.Sequential(
                nn.Linear(input.shape[1], 1024),
                nn.BatchNorm1d(1024),  # Batch normalization for input layer
                self.elu
            ).to(input.device)

        # Forward pass
        x = self.input_layer(input)
        x = self.dropout(self.elu(self.bn2(self.fc2(x))))  # FC2 + BN + Activation + Dropout
        x = self.dropout(self.elu(self.bn3(self.fc3(x))))  # FC3 + BN + Activation + Dropout
        x = self.fc4(x)

        # Output processing
        mag = self.sigmoid(x[:, :x.shape[1] // 2])
        mag = torch.clamp(mag, self.eps, 1)
        phase = self.sigmoid(x[:, x.shape[1] // 2:]) * 2 * torch.pi - torch.pi  # phase is between -pi and pi
        output = torch.cat((mag, phase), dim=1)

        return output



class baseline_inverse_model(nn.Module):
    """
    model designed to find the regression between 2002 spectrum parameters as input and the 12 geometric parameters as output.
    """
    def __init__(self,weight_range=0.1,p_dropout=0.25):
        super(baseline_inverse_model,self).__init__()
        self.linear1 = nn.Linear(2002, 960)
        self.linear2 = nn.Linear(960, 960)
        self.linear3 = nn.Linear(960, 480)
        self.linear4 = nn.Linear(480, 480)
        self.linear5 = nn.Linear(480, 240)
        self.linear6 = nn.Linear(240, 240)
        self.linear7 = nn.Linear(240, 120)
        self.addlayer0 = nn.Linear(120, 120)
        self.linear8 = nn.Linear(120, 60)
        self.addlayer1 = nn.Linear(60, 60)
        self.linear9 = nn.Linear(60, 30)
        self.addlayer2 = nn.Linear(30, 30)
        self.linear10 = nn.Linear(30, 12)
        self.dropout = nn.Dropout(p=p_dropout)
        self.relu = nn.ELU()
        self.init_weights(weight_range)
    def init_weights(self,init_range):
        for p in self.parameters():
            p.data.uniform_(-init_range, init_range)
    def forward(self,inputs): # input is the spectrum parameters
        gamma, radiation = inputs
        output = self.relu(self.linear1(gamma))
        output = self.dropout(output)
        output = self.relu(self.linear2(output))
        output = self.relu(self.linear3(output))
        # output = self.dropout(output)
        output = self.relu(self.linear4(output))
        output = self.relu(self.linear5(output))
        # output = self.dropout(output)
        output = self.relu(self.linear6(output))
        output = self.dropout(output)
        output = self.relu(self.linear7(output))
        output = self.relu(self.addlayer0(output))
        output = self.relu(self.linear8(output))
        # output = self.relu(self.addlayer1(output))
        output = self.dropout(output)
        output = self.relu(self.linear9(output))
        # output = self.relu(self.addlayer2(output))
        output = self.linear10(output) # no activation function at the end, that is the geometric parameters
        return output
class baseline_inverse_forward_model(nn.Module):
    """
    model designed to return the input spectrum parameters as output spectrum parameters.
    """
    def __init__(self):
        super(baseline_inverse_forward_model,self).__init__()
        self.inverse_part = baseline_inverse_model()
        self.forward_part = baseline_forward_model()
    def forward(self,input):
        geometric_parameters = self.inverse_part(input)
        spectrum_parameters = self.forward_part(geometric_parameters)
        return geometric_parameters,spectrum_parameters

if __name__ == "__main__":
    print("print number of parameters in the model")
    model = small_deeper_baseline_forward_model()
    inp = torch.randn(1,2032)
    out = model(inp)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
