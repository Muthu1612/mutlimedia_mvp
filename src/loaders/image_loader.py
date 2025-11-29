import os
import cv2
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
import mediapipe as mp


def extract_face(frame):
    """Standalone SAFE face extractor with bounding-box clipping."""
    if frame is None or frame.size == 0:
        return None

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

        # SAFE clipped box
        x1 = max(0, int(bbox.xmin * w))
        y1 = max(0, int(bbox.ymin * h))
        x2 = min(w, int((bbox.xmin + bbox.width) * w))
        y2 = min(h, int((bbox.ymin + bbox.height) * h))

        crop = frame[y1:y2, x1:x2]

        # Make sure crop is valid
        if crop is None or crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            return None

        return crop



class CelebDFImageDataset(Dataset):
    """
    Loads ONE face image per video (middle frame).
    Useful for multimodal fusion: image + video + audio.
    """
    def __init__(self, root_dir, video_list=None):
        self.root_dir = Path(root_dir)

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


    def _extract_middle_frame_face(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < 1:
            cap.release()
            return torch.zeros(3, 224, 224)

        mid_idx = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
        ret, frame = cap.read()

        cap.release()

        if not ret or frame is None or frame.size == 0:
            return torch.zeros(3, 224, 224)

        face = extract_face(frame)

        # Face not found OR invalid crop
        if face is None or face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
            return torch.zeros(3, 224, 224)

        # Safe convert
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        return self.transform(face)


    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        img_tensor = self._extract_middle_frame_face(video_path)

        return img_tensor, torch.tensor(label, dtype=torch.long)
