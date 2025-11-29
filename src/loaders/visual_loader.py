import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import mediapipe as mp
import numpy as np


# -----------------------------
# SAFE FACE VALIDATION
# -----------------------------
def _is_valid_face(face):
    return (
        face is not None
        and isinstance(face, np.ndarray)
        and face.size > 0
        and face.shape[0] > 10    # height
        and face.shape[1] > 10    # width
    )


# -----------------------------
# SAFE FACE EXTRACTOR
# -----------------------------
def extract_face(frame):
    """Safe face extractor: mediapipe instance each call."""
    with mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    ) as fd:

        if frame is None or frame.size == 0:
            return None

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

        face = frame[y1:y2, x1:x2]

        if not _is_valid_face(face):
            return None

        return face



# ===========================================================
#                CELEB-DF VISUAL DATASET
# ===========================================================
class CelebDFVisualDataset(Dataset):
    def __init__(self, root_dir, frames_per_video=8, video_list= None):
        self.root_dir = Path(root_dir)
        self.frames_per_video = frames_per_video
        
        self.video_paths = []
        self.labels = []
        if video_list is None:
            # label mapping
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
        else:
            for item in video_list:
                self.video_paths.append(item["path"])
                self.labels.append(item["label"])

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.video_paths)

    # -----------------------------
    # SAFE VIDEO FRAME READING
    # -----------------------------
    def _read_video_frames(self, path):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < 1:
            cap.release()
            return torch.zeros(self.frames_per_video, 3, 224, 224)

        indices = torch.linspace(0, total_frames - 1, self.frames_per_video).long()
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx.item()))
            ret, frame = cap.read()

            if not ret or frame is None:
                continue

            face = extract_face(frame)

            # -----------------------------
            # SAFETY CHECK
            # -----------------------------
            if not _is_valid_face(face):
                continue

            # now safe to convert
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            img = self.transform(face)

            if img.shape[0] != 3:
                img = img[:3, :, :]

            frames.append(img)

        cap.release()

        # if nothing extracted → fallback
        if len(frames) == 0:
            return torch.zeros((self.frames_per_video, 3, 224, 224))

        # pad to fixed length
        while len(frames) < self.frames_per_video:
            frames.append(frames[-1].clone())

        return torch.stack(frames)


    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frames = self._read_video_frames(video_path)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return frames, label
