#code from
#TODO citation


import torch

class NoisyTeacherForcing():
    def __init__(self, A_size, noise_prob=0.):
        self.noise_prob = torch.Tensor([noise_prob])
        self.A_size = A_size
      #  self.device = 'cpu'

        if torch.cuda.is_available():
           # self.device = 'cuda'
            self.noise_prob = self.noise_prob.cuda()

    def __call__(self, x,unpadded_text_len):

        noise = torch.randint(low=3, high=self.A_size, size=x.shape,device=x.get_device())
        prob = torch.rand(size=x.shape,device=x.get_device())
        self.noise_prob = self.noise_prob.to(x.get_device())
        prob[:,0] = 1
        i = 0
        for EOS_TOKEN_PLACE in unpadded_text_len:
            prob[i,EOS_TOKEN_PLACE+1:] = 1
            i = i+1

        return torch.where(prob>self.noise_prob,x,noise)

if __name__ == "__main__":
    NTF = NoisyTeacherForcing(A_size=89, noise_prob=0.8)
    x = torch.LongTensor([[0,5,6,7,78,5,6,7,2,3,3,3,3,3,3,3], [0,5,6,7,78,5,6,7,2,3,3,3,3,3,3,3]]).cuda()
    print(x-NTF(x))
