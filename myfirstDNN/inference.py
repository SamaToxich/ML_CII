import cv2
import torch
import numpy as np
from pathlib import Path
from app.ml.model import FaceEncoder

# Детектор лиц (OpenCV DNN)
class FaceDetector:
    def __init__(self):
        detector_dir = Path(__file__).parent / 'face_detector'
        prototxt = str(detector_dir / 'deploy.prototxt')
        model = str(detector_dir / 'res10_300x300_ssd_iter_140000.caffemodel')
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )

    def detect(self, image_bytes):
        """image_bytes: байты JPEG/PNG → выровненное лицо 112×112 или None"""
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return self._align(img)

    def _align(self, img):
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
        self.net.setInput(blob)
        detections = self.net.forward()

        best = None
        for i in range(detections.shape[2]):
            if detections[0, 0, i, 2] > 0.5:
                box = (detections[0, 0, i, 3:7] * [w, h, w, h]).astype(int)
                best = box
                break
        if best is None:
            return None

        x, y, x2, y2 = best
        x, y = max(0, x), max(0, y)
        bw, bh = min(w, x2) - x, min(h, y2) - y
        if bw <= 0 or bh <= 0:
            return None

        # Расширяем на 20%
        mx = int(bw * 0.2)
        my = int(bh * 0.2)
        x, y = max(0, x - mx), max(0, y - my)
        bw = min(w - x, bw + 2 * mx)
        bh = min(h - y, bh + 2 * my)
        face = img[y:y+bh, x:x+bw]

        # Выравнивание по глазам
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(20, 20))
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])
            le = eyes[0][:2] + eyes[0][2:] / 2
            re = eyes[-1][:2] + eyes[-1][2:] / 2
            angle = np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0]))
            mid = ((le[0] + re[0]) / 2, (le[1] + re[1]) / 2)
            M = cv2.getRotationMatrix2D(mid, angle, 1.0)
            face = cv2.warpAffine(face, M, (face.shape[1], face.shape[0]))

        return cv2.resize(face, (112, 112))


# Сервис распознавания
class FaceRecognizer:
    def __init__(self, model_path='app/ml/face_encoder.pt'):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FaceEncoder(embedding_size=512).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.detector = FaceDetector()

    def get_embedding(self, image_bytes):
        """image_bytes → эмбеддинг [512] или None"""
        face = self.detector.detect(image_bytes)
        if face is None:
            return None

        tensor = torch.from_numpy(face).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model(tensor)
        return emb[0].cpu().numpy()

    def compare(self, emb1, emb2):
        """Косинусное сходство двух эмбеддингов [0, 1]"""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))