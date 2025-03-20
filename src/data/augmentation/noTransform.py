from torch import nn


class NoTransform(nn.Module):
    def __init__(self):
        super(NoTransform, self).__init__()

    def forward(self, x):
        return x


"""
    Takes an image and shifts it by scale into negative space
"""
class ShiftTransform(nn.Module):

    def __init__(self, scale, multiplication_scale=1.0):
        super(ShiftTransform, self).__init__()
        self.scale = scale
        self.m_scale = multiplication_scale
    def forward(self, x):
        return self.m_scale*(x-self.scale)

    def reverse(self,x):
        return x/self.m_scale+self.scale