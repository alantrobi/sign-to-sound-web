import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ISLDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.labels = sorted(os.listdir(data_path))
        self.samples = []
        
        print("--- Scanning Dataset Folders ---")
        for idx, label in enumerate(self.labels):
            class_path = os.path.join(data_path, label)
            if not os.path.isdir(class_path): continue
            
            # Go into subfolders (0, 1, 2... 119)
            subfolders = os.listdir(class_path)
            for sub in subfolders:
                sub_path = os.path.join(class_path, sub)
                
                # If this is a folder, we need the .npy files inside it
                if os.path.isdir(sub_path):
                    # Usually, these folders contain 30 .npy files
                    # We only need the path to the folder to load the sequence
                    self.samples.append((sub_path, idx))
        
        print(f"Total sequences found: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder_path, label = self.samples[idx]
        
        # Load all 30 .npy files in that subfolder and stack them
        # This creates the (30, 126) shape the LSTM needs
        sequence = []
        # Sort files to ensure they are in order (frame0, frame1... frame29)
        file_names = sorted(os.listdir(folder_path))[:30] 
        
        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)
            frame_data = np.load(file_path) # Each frame is (126,)
            sequence.append(frame_data)
            
        data = np.array(sequence) # Shape: (30, 126)
        
        # --- NORMALIZATION ---
        data = data.reshape(30, 2, 21, 3)
        for frame in range(30):
            for hand in range(2):
                wrist = data[frame, hand, 0, :].copy()
                data[frame, hand, :, :] -= wrist
        
        data = data.reshape(30, 126)
        
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
