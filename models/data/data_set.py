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


class DataSetL3(Dataset):
    def __init__(
            self,
            base_df,
            device,
            img_chs=1,
            img_height=28,
            img_width=28,
    ):
        x_df = base_df.copy()
        y_df = x_df.pop('label')
        x_df = x_df.values / 255
        x_df = x_df.reshape(-1, img_chs, img_height, img_width)
        self.xs = torch.tensor(x_df).float().to(device)
        self.ys = torch.tensor(y_df).to(device)

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return x, y

    def __len__(self):
        return len(self.xs)
