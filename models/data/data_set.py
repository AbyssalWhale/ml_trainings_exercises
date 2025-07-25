from torch.utils.data import Dataset
import torch

class DataSet(Dataset):
    def __init__(self, x_df, y_df, device):
        # Convert PyTorch DataFrame tensors and move them to the specified device
        self.xs = torch.tensor(x_df).float().to(device)
        self.ys = torch.tensor(y_df).to(device)

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return x, y

    def __len__(self):
        return len(self.xs)
