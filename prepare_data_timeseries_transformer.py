import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


# Load the CSV file
csv_path = "farm_data.csv"  # Ensure this is the correct path
df = pd.read_csv(csv_path)

# Convert 'Date' to datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Sort by Field_ID and Date
df = df.sort_values(by=["Field_ID", "Date"])

# Get the maximum sequence length
MAX_SEQ_LENGTH = df.groupby("Field_ID").size().max()

# Define image transformation
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class TimeSeriesImageDataset(Dataset):
    def __init__(self, df=None, max_seq_length=MAX_SEQ_LENGTH, transform=None):
        # If no dataframe is provided, load the CSV automatically.
        if df is None:
            csv_path = "farm_data.csv"  # Adjust the path if needed.
            df = pd.read_csv(csv_path)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values(by=["Field_ID", "Date"])
        self.df = df
        self.max_seq_length = max_seq_length
        self.transform = transform if transform else image_transform
        self.field_ids = df["Field_ID"].unique()

    def __len__(self):
        return len(self.field_ids)

    def __getitem__(self, idx):
        field_id = self.field_ids[idx]
        field_data = self.df[self.df["Field_ID"] == field_id].sort_values("Date")

        sequences = []
        labels = []

        for i in range(len(field_data)):
            cig_path = field_data.iloc[i]["CIG_Path"]
            evi_path = field_data.iloc[i]["EVI_Path"]
            ndvi_path = field_data.iloc[i]["NDVI_Path"]
            label = field_data.iloc[i]["Infestation"]

            # Load images
            cig_img = self.load_image(cig_path)   # shape: [3, 224, 224]
            evi_img = self.load_image(evi_path)   # shape: [3, 224, 224]
            ndvi_img = self.load_image(ndvi_path)   # shape: [3, 224, 224]

            # Stack into one tensor
            img_tensor = torch.cat([cig_img, evi_img, ndvi_img], dim=0)  # New shape: [9, 224, 224]
            sequences.append(img_tensor)
            labels.append(label)

        # Convert to tensor: each sequence now has shape [T, 9, 224, 224]
        sequences = torch.stack(sequences) if len(sequences) > 0 else torch.zeros(1, 9, 224, 224)

        # Pad sequences to the max length if needed
        if sequences.shape[0] < self.max_seq_length:
            padding = torch.zeros(self.max_seq_length - sequences.shape[0], 9, 224, 224)
            sequences = torch.cat([sequences, padding], dim=0)

        return sequences, torch.tensor(labels[-1], dtype=torch.float32)  # Use last label for classification

    def load_image(self, path):
        if path == "":
            return torch.zeros(3, 224, 224)  # Placeholder image
        return self.transform(Image.open(path).convert("RGB"))

# Create dataset
time_series_dataset = TimeSeriesImageDataset(df)

# Create DataLoader
time_series_dataloader = DataLoader(time_series_dataset, batch_size=8, shuffle=True)

print("✅ Time-Series Transformer dataset is ready and padded!")
