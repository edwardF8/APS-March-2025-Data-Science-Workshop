import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
 
def check_for_nans(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"NaN or Inf detected in {name}")
        return True
    return False
 
class DiffractionDataset(Dataset):
    def __init__(self,classes, background, data, labels=None, unsupervised=False, categorical='Bravais Lattice'):
        print("Attempting to load data")
        assert categorical=='Bravais Lattice' or categorical=='Space Group', "The key word argument categorical should be Bravais Lattice or Space Group, not {}".format(categorical)
        self.data= data
        self.unsupervised= unsupervised
        if labels==None and not unsupervised:
            data=torch.load(data)
            self.data = data['X'] 
            self.labels= data['Y']
        elif not labels == None:
            self.labels = labels
        elif not self.unsupervised:
            self.labels = labels
            self.data = self.data 
        self.categorical=categorical
        self.data = self.data.add(background)
        self.mapping=torch.load('mapping.pt')[categorical]

    def __getitem__(self, index):
        if self.unsupervised:
            return self.data[index].unsqueeze(0)
        return self.data[index].unsqueeze(0), self.labels[index]

    def __len__(self):
        return len(self.data)

    def batch_u(self, batch_size):
        idx = torch.randint(0,len(self.data),(batch_size,))
        return self.data[idx]

    def plot(self, idx=0, norm=False):
        assert idx<len(self.data), "Index out of range, please input an index between 0 and {}".format(len(self))
        data=self.data[idx].cpu().numpy().flatten()
        if(norm):
            data = data-20
            data = data/data.max() + 0.001
        xrange=np.arange(3, data.shape[-1]*0.05+3, 0.05)
        fig = plt.plot(xrange,data)
        plt.xlabel("2\u03B8\xb0")
        plt.ylabel("Normalized Intensity")
        str_label=self.mapping[int(self.labels[idx].item())]
        plt.title("{}: {}".format(self.categorical,str_label))
        return fig

    def _unencode_labels(self, labels):
        return np.asarray([self.mapping[int(i.item())] for i in labels])

    def compare(self, pred1, pred2=None, heading=[]):
        pred1=self._unencode_labels(pred1)
        labels=self._unencode_labels(self.labels)

        if pred2 is not None:
            pred2=self._unencode_labels(pred2)
            print("{: >3}{: >20}{: >20}{: >20}".format("Index", "True Label", heading[0], heading[1]))
            for i in range(len(pred1)):
                print("{: >3} {: >20} {: >20} {: >20}".format(i, labels[i], pred1[i], pred2[i]))

        else:            
            print("{: >3}{: >20}{: >20}".format("Index", "True Label", "Prediction"))
            for i in range(len(pred1)):
                print("{: >3} {: >20} {: >20}".format(i, labels[i], pred1[i]))

 