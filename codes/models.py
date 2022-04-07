from selfonn import SelfONN1DLayer
import torch
import torch.nn as nn




def downsample_selfONN(in_ch, out_ch, f_size, q, samp_fact=2):
 
    result = nn.Sequential(
        SelfONN1DLayer(in_channels=in_ch,out_channels=out_ch,kernel_size=f_size,q=q,pad=-1,sampling_factor=samp_fact),
        torch.nn.Tanh()
    )

    return result

def upsample_selfONN(in_ch, out_ch, f_size, q, samp_fact=-2):
 
    result = nn.Sequential(
        SelfONN1DLayer(in_channels=in_ch,out_channels=out_ch,kernel_size=f_size,q=q,pad=-1,sampling_factor=samp_fact),
        torch.nn.Tanh()
    )

    return result

class UNet_selfONN(nn.Module):
    def __init__(self, q, n_channels = 1):
        super(UNet_selfONN, self).__init__()
        self.n_channels = n_channels


        # model 2 [4-->4] [16] [9 7  5 3]
        self.down1 = downsample_selfONN(in_ch = 1, out_ch =  16, f_size = 9, q = q, samp_fact=4)
        self.down2 = downsample_selfONN(in_ch = 16, out_ch = 16, f_size = 7, q = q, samp_fact=4)
        self.down3 = downsample_selfONN(in_ch = 16, out_ch = 16, f_size = 5, q = q, samp_fact=4)

        self.up1 = upsample_selfONN(in_ch = 16, out_ch = 16, f_size = 5, q = q, samp_fact=-4)
        self.up2 = upsample_selfONN(in_ch = 32, out_ch = 16, f_size = 7, q = q, samp_fact=-4)
        self.up3 = upsample_selfONN(in_ch = 32, out_ch = 16, f_size = 9, q = q, samp_fact=-4)

        self.last = nn.Sequential(
                        SelfONN1DLayer(in_channels=16,out_channels=1,kernel_size=1,q=q,pad=0,sampling_factor = 1),
                        torch.nn.Sigmoid())

    def forward(self, x):

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        u1 = self.up1(d3)
        u2 = self.up2(torch.cat([u1,d2],1))
        u3 = self.up3(torch.cat([u2,d1],1))
        out = self.last(u3)
        #print(u3.size())

        return out


if __name__ == "__main__":
    pass









    