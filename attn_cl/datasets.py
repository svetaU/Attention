import torch
import torch.utils.data as data
from einops import repeat


class FakeDataset(data.Dataset):
    def __init__(self, num_seqs, num_tokens, token_size, class_fraction):
        super().__init__()
        self.num_seqs = num_seqs
        self.num_tokens = num_tokens
        self.token_size = token_size
        self.num_elements = int(num_seqs*class_fraction)
        self._generate_data()

    def _generate_data(self):

        assert self.num_tokens > 2
        self.labels = torch.zeros(self.num_seqs,dtype=torch.float)
        mean = repeat(torch.arange(0.,self.num_tokens).unsqueeze(-1).expand(-1,self.token_size),
                      't i -> s t i',s=self.num_seqs) #each token has the mean of its index
        self.data = torch.normal(mean=mean)
        #generate a fraction of abnormal data
        class_inds = torch.randperm(self.num_seqs)[:self.num_elements]
        self.labels[class_inds] = 1.
        #self.data[class_inds,0,:] = torch.empty(self.token_size).normal_(mean=4.)
        #self.data[class_inds,2,:] = torch.empty(self.token_size).normal_(mean=10.)
        self.data[class_inds,0,:] = torch.empty(self.token_size).normal_(mean=1.)
        self.data[class_inds,2,:] = torch.empty(self.token_size).normal_(mean=0.)

    def __len__(self):
        return self.num_seqs

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]