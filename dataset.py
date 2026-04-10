import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, transform=None):
        """
        Args:
            root_dir (str): Path to dataset directory (should contain 'violence' and 'non_violence' folders).
            num_frames (int): fixed number of frames to sample per video.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.classes = ['non_violence', 'violence']  # 0: non_violence, 1: violence
        self.videos = []
        self.labels = []
        self.transform = transform
        
        # Default transforms if none provided (Resize and Normalize)
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
            ])

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} does not exist.")
                continue
                
            for file in os.listdir(class_dir):
                if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    self.videos.append(os.path.join(class_dir, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        label = self.labels[idx]
        
        frames = self._extract_frames(video_path)
        
        # Apply transforms and stack
        processed_frames = []
        for frame in frames:
            if self.transform:
                frame = self.transform(frame)
            processed_frames.append(frame)
            
        # Stack into tensor: shape will be (C, T, H, W) for 3D CNNs
        video_tensor = torch.stack(processed_frames, dim=1)
        
        return video_tensor, label

    def _extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            return [np.zeros((112, 112, 3), dtype=np.uint8)] * self.num_frames

        # Sample indices uniformly
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        idx_count = 0
        frame_count = 0
        while cap.isOpened() and idx_count < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count == indices[idx_count]:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                idx_count += 1
                
            frame_count += 1

        cap.release()
        
        # Pad with last frame if needed
        while len(frames) < self.num_frames:
            if len(frames) > 0:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((112, 112, 3), dtype=np.uint8))
                
        return frames
