import torch
from torch import nn
from torch.nn import functional as F

class MyLinearLayer(nn.Module):
    def __init__(self, size_in, size_out, eps_init = 0, bias = True, regularization = True):
        super(MyLinearLayer, self).__init__()
        self.size_in, self.size_out = size_in, size_out

        self.regularization = regularization

        self.weight = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(self.size_out, self.size_in), a=5**0.5))  # nn.Parameter is a Tensor that's a module parameter.

        if bias == True:
            self.bias = nn.Parameter(nn.init.uniform_(torch.empty(self.size_out), a = -(1/size_in)**0.5 ,b = (1/size_in)**0.5))
        else:
            self.bias = self.register_parameter('bias', None)

        
        self.eps = nn.Parameter(eps_init * torch.ones(self.size_out))

        self.ch_score = nn.Parameter(torch.ones(self.size_out), requires_grad=False)

        self.norm_factor = nn.Parameter(torch.ones(self.size_out))

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        if self.regularization:
            with torch.no_grad():
                disturbed_ws = torch.bitwise_and(torch.tensor(-8388608, dtype = torch.int32 ), self.weight.view(torch.int32)).view(torch.float32)
                if self.bias != None:
                    disturbed_bs = torch.bitwise_and(torch.tensor(-8388608, dtype = torch.int32 ), self.bias.view(torch.int32)).view(torch.float32)
                else:
                    disturbed_bs = None
                y_d = F.linear(x, disturbed_ws, disturbed_bs)

                # ratio = torch.nan_to_num(torch.div(y,y_d) , nan=10.0, posinf=10.0, neginf = -10.0) + torch.nan_to_num(torch.div(y_d,y) , nan=10.0, posinf=10.0, neginf = -10.0)
                ratio = torch.nan_to_num(torch.div(y,y_d) , nan=10.0, posinf=10.0, neginf = -10.0)
                ratio = torch.abs(ratio) 
                if len(ratio.shape) == 4:
                    ratio = torch.sum(ratio, [1,2])[:,None, None,:]
                else:
                    ratio = torch.sum(ratio, [0])[None,:] / (ratio.shape[0])
                ratio = ratio / (ratio.mean())

            scr = torch.sigmoid((-ratio - self.eps + 1.0)) + torch.sigmoid((ratio - self.eps - 1.0)) 
            self.ch_score.data = scr[0] * 0.5 + self.ch_score * 0.5
            scr = scr/torch.min(scr)

            y = y * scr


        return y

class MyConvLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_shape, stride = (1,1),
                  padding = 'valid', bias = True, dilation = 1, groups = 1, eps_init = 0.25, regularization = True):
      super(MyConvLayer, self).__init__()
      
      self.kernel_h, self.kernel_w = kernel_shape
      self.inp_ch = input_channels
      self.out_ch = output_channels
      self.stride = stride
      self.padding = padding
      self.dilation = dilation
      self.groups = groups
      self.regularization = regularization

      self.weight = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.out_ch, self.inp_ch // self.groups, self.kernel_h, self.kernel_w)))
      if bias == True:
          self.bias = nn.Parameter(torch.zeros(self.out_ch))
      else:
          self.bias = self.register_parameter('bias', None)

      self.eps = nn.Parameter(eps_init * torch.ones(self.out_ch, 1, 1))

      self.ch_score = nn.Parameter(torch.ones(self.out_ch, 1, 1), requires_grad=False)


    def forward(self, x):
        y = F.conv2d(x, self.weight , self.bias , self.stride, self.padding, self.dilation, self.groups)
        if self.regularization:
            with torch.no_grad():
                disturbed_ws = torch.bitwise_and(torch.tensor(-8388608, dtype = torch.int32 ), self.weight.view(torch.int32)).view(torch.float32)
                if self.bias != None:
                    disturbed_bs = torch.bitwise_and(torch.tensor(-8388608, dtype = torch.int32 ), self.bias.view(torch.int32)).view(torch.float32)
                else:
                    disturbed_bs = None
                y_d = F.conv2d(x, disturbed_ws , disturbed_bs , self.stride, self.padding, self.dilation, self.groups)
                ratio = torch.nan_to_num(torch.div(y,y_d) , nan=10.0, posinf=10.0, neginf = -10.0)
                ratio = torch.abs(ratio)
                ratio = torch.sum(ratio, [0,2,3])[None,:,None, None]/(ratio.shape[0] * ratio.shape[2] * ratio.shape[3]) 
                ratio = ratio/(ratio.mean())

            scr = torch.sigmoid(-ratio - self.eps + 1.0) + torch.sigmoid(ratio - self.eps - 1.0)
            self.ch_score.data = scr[0] * 0.5 + self.ch_score * 0.5
            scr = scr / torch.min(scr)
            y = y * scr
        
        return y




class MyConvLayerBN(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_shape, stride = (1,1),
                  padding = 'valid', bias = True, dilation = 1, groups = 1, eps_init = 0.5, regularization = True):
      super(MyConvLayerBN, self).__init__()

      self.kernel_h, self.kernel_w = kernel_shape
      self.inp_ch = input_channels
      self.out_ch = output_channels
      self.stride = stride
      self.padding = padding
      self.dilation = dilation
      self.groups = groups
      self.regularization = regularization

      self.weight = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.out_ch, self.inp_ch // self.groups, self.kernel_h, self.kernel_w)))
      if bias == True:
          self.bias = nn.Parameter(torch.zeros(self.out_ch))
      else:
          self.bias = self.register_parameter('bias', None)

      self.eps = nn.Parameter(eps_init * torch.ones(self.out_ch, 1, 1))

      self.ch_score = nn.Parameter(torch.ones(self.out_ch, 1, 1), requires_grad=False)

      self.bn_weight = nn.Parameter(torch.ones(self.out_ch)) 
      self.bn_bias = nn.Parameter(torch.zeros(self.out_ch)) 

      self.pow = nn.Parameter(2 * torch.ones(1)) 

      self.running_mean = nn.Parameter(torch.zeros(self.out_ch), requires_grad=False) 
      self.running_var = nn.Parameter(torch.ones(self.out_ch), requires_grad=False)

      self.running_mean_ds = nn.Parameter(torch.zeros(self.out_ch), requires_grad=False) 
      self.running_var_ds = nn.Parameter(torch.ones(self.out_ch), requires_grad=False)

    def forward(self, x):
        y = F.conv2d(x, self.weight , self.bias , self.stride, self.padding, self.dilation, self.groups)
       
        if self.regularization:
            with torch.no_grad():
                disturbed_ws = torch.bitwise_and(torch.tensor(-8388608, dtype = torch.int32 ), self.weight.view(torch.int32)).view(torch.float32)
                if self.bias != None:
                    disturbed_bs = torch.bitwise_and(torch.tensor(-8388608, dtype = torch.int32 ), self.bias.view(torch.int32)).view(torch.float32)
                else:
                    disturbed_bs = None
                y_d = F.conv2d(x, disturbed_ws , disturbed_bs , self.stride, self.padding, self.dilation, self.groups)
                ratio = torch.nan_to_num(torch.div(y,y_d) , nan=10.0, posinf=10.0, neginf = -10.0)
                ratio = torch.abs(ratio)
                ratio = torch.sum(ratio, [0,2,3])[None,:,None, None]/(ratio.shape[0] * ratio.shape[2] * ratio.shape[3]) 
                ratio = ratio/(ratio.mean())

            scr = torch.sigmoid((-ratio - self.eps + 1.0)) + torch.sigmoid((ratio - self.eps - 1.0))
            self.ch_score.data = scr[0] * 0.5 + self.ch_score * 0.5
            scr = scr*torch.min(scr)
            scr = scr ** self.pow
            scr = scr[0,:,0,0] * self.bn_weight  
            y = F.batch_norm(y, self.running_mean, self.running_var, weight=scr, bias=self.bn_bias, training=True) 
        
        else:
            y = F.batch_norm(y, self.running_mean, self.running_var, weight=self.bn_weight, bias=self.bn_bias, training=True) 
        
        
        return y

