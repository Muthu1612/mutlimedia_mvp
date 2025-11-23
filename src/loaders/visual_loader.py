import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import mediapipe as mp

def extract_face(frame):
    """Safe face extractor: creates Mediapipe instance on each call."""
    with mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    ) as fd:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = fd.process(rgb)

        if not results.detections:
            return None

        det = results.detections[0]
        bbox = det.location_data.relative_bounding_box

        h, w, _ = frame.shape
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)

        return frame[y1:y2, x1:x2]


class CelebDFVisualDataset(Dataset):
    def __init__(self, root_dir, frames_per_video=8):
        self.root_dir = Path(root_dir)
        self.frames_per_video = frames_per_video
        
        self.video_paths = []
        self.labels = []

        # Correct label mapping
        folders = {
            "Celeb-real": 0,
            "YouTube-real": 0,
            "Celeb-synthesis": 1
        }

        for folder, lbl in folders.items():
            folder_path = self.root_dir / folder
            videos = list(folder_path.glob("*.mp4"))
            for v in videos:
                self.video_paths.append(str(v))
                self.labels.append(lbl)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.video_paths)

    def _read_video_frames(self, path):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < 1:
            cap.release()
            return torch.zeros(self.frames_per_video, 3, 224, 224)

        # Time-uniform sampling
        idxs = torch.linspace(0, total_frames - 1, self.frames_per_video).long()
        frames = []

        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i.item()))
            ret, frame = cap.read()
            if not ret:
                continue

            face = extract_face(frame)
            if face is None:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            img = self.transform(face)

            if img.shape[0] != 3:
                img = img[:3, :, :]

            frames.append(img)

        cap.release()

        # If NO frames extracted (no faces) — fill black frames
        if len(frames) == 0:
            return torch.zeros((self.frames_per_video, 3, 224, 224))

        # Pad short videos
        while len(frames) < self.frames_per_video:
            frames.append(torch.zeros((3, 224, 224)))

        return torch.stack(frames)


    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frames = self._read_video_frames(video_path)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return frames, label


# # Test
# dataset = CelebDFVisualDataset("../../data/celeb_df", frames_per_video=8)
# loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

# frames, label = next(iter(loader))
# print("Frames:", frames.shape)
# print("Label:", label)
