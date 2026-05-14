"""
Тестирование модели на паре фотографий.
Выводит косинусное сходство, межклассовое/внутриклассовое сходство и угол.
"""
import torch
import cv2
import numpy as np
from pathlib import Path
import sys
import urllib.request

sys.path.append(str(Path(__file__).parent.parent.parent))

from app.ml.model import FaceEncoder

# --- Детектор ---
DETECTOR_DIR = Path('app/ml/face_detector')
DETECTOR_DIR.mkdir(parents=True, exist_ok=True)

prototxt_path = DETECTOR_DIR / 'deploy.prototxt'
model_path = DETECTOR_DIR / 'res10_300x300_ssd_iter_140000.caffemodel'

if not prototxt_path.exists():
    print("Скачиваем конфиг детектора...")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    urllib.request.urlretrieve(url, str(prototxt_path))

if not model_path.exists():
    print("Скачиваем веса детектора...")
    url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    urllib.request.urlretrieve(url, str(model_path))

face_net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))


def detect_face(image):
    """Находит лицо на фото и выравнивает его"""
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
    face_net.setInput(blob)
    detections = face_net.forward()

    best_conf = 0.5
    best_box = None
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > best_conf:
            best_conf = confidence
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            best_box = box.astype(int)

    if best_box is None:
        return None

    x, y, x2, y2 = best_box
    x, y = max(0, x), max(0, y)
    x2, y2 = min(w, x2), min(h, y2)
    bw, bh = x2 - x, y2 - y
    if bw <= 0 or bh <= 0:
        return None

    margin = 0.2
    x = max(0, int(x - margin * bw))
    y = max(0, int(y - margin * bh))
    bw = min(w - x, int(bw * (1 + 2 * margin)))
    bh = min(h - y, int(bh * (1 + 2 * margin)))
    face_crop = image[y:y+bh, x:x+bw]
    if face_crop.size == 0:
        return None

    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(20, 20))

    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0])
        left_eye = np.array([eyes[0][0] + eyes[0][2]/2, eyes[0][1] + eyes[0][3]/2])
        right_eye = np.array([eyes[-1][0] + eyes[-1][2]/2, eyes[-1][1] + eyes[-1][3]/2])
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        eyes_mid = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        M = cv2.getRotationMatrix2D(eyes_mid, angle, 1.0)
        face_crop = cv2.warpAffine(face_crop, M, (face_crop.shape[1], face_crop.shape[0]))

    return cv2.resize(face_crop, (112, 112))


def get_embedding(model, face_img, device):
    """Получает эмбеддинг из подготовленного лица"""
    face_tensor = torch.from_numpy(face_img).float() / 255.0
    face_tensor = (face_tensor - 0.5) / 0.5
    face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(face_tensor)
    return embedding[0].cpu().numpy()


def cosine_similarity(emb1, emb2):
    """Косинусное сходство между двумя эмбеддингами"""
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))


def compute_metrics(embeddings):
    """
    Вычисляет inter, intra, θ° для набора эмбеддингов.
    embeddings: список из 3 массивов [emb1, emb2, emb3]
    """
    e1, e2, e3 = embeddings

    # Сходства
    sim_same = cosine_similarity(e1, e2)       # intra (один человек)
    sim_diff1 = cosine_similarity(e1, e3)      # inter (разные люди)
    sim_diff2 = cosine_similarity(e2, e3)      # inter (разные люди)
    inter_sim = (sim_diff1 + sim_diff2) / 2    # среднее

    # Угол между своими (в градусах)
    theta_same = np.degrees(np.arccos(np.clip(sim_same, -1.0, 1.0)))

    # Угол между чужими (средний)
    theta_diff1 = np.degrees(np.arccos(np.clip(sim_diff1, -1.0, 1.0)))
    theta_diff2 = np.degrees(np.arccos(np.clip(sim_diff2, -1.0, 1.0)))
    theta_diff = (theta_diff1 + theta_diff2) / 2

    return {
        'intra': sim_same,
        'inter': inter_sim,
        'θ_same': theta_same,
        'θ_diff': theta_diff
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    # Загружаем модель
    model = FaceEncoder(embedding_size=512).to(device)
    path = 'app/ml/checkpoints/face-v3-epoch=30-loss=5.449.ckpt'
    checkpoint = torch.load(path, map_location=device)
    state_dict = {k.replace('encoder.', ''): v for k, v in checkpoint['state_dict'].items()
                  if k.startswith('encoder.')}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Модель {path} загружена\n")

    # Тест 1: Два фото ОДНОГО человека
    print("=" * 60)
    print("ТЕСТ 1: ОДИН И ТОТ ЖЕ ЧЕЛОВЕК (должно быть > 0.3)")
    print("=" * 60)

    photo1_path = input("Путь к первому фото одного человека: ").strip().strip('"')
    photo2_path = input("Путь ко второму фото ТОГО ЖЕ человека: ").strip().strip('"')

    img1 = cv2.imread(photo1_path)
    img2 = cv2.imread(photo2_path)

    if img1 is None or img2 is None:
        print("Ошибка загрузки фото.")
        return

    face1 = detect_face(img1)
    face2 = detect_face(img2)

    if face1 is None or face2 is None:
        print("Не удалось найти лицо на одном из фото.")
        return

    emb1 = get_embedding(model, face1, device)
    emb2 = get_embedding(model, face2, device)

    sim_same = cosine_similarity(emb1, emb2)
    print(f"Косинусное сходство: {sim_same:.4f}")
    if sim_same > 0.3:
        print("✅ ОТЛИЧНО! Модель узнала одного человека.")
    else:
        print("⚠️ Низкое сходство.")

    # Тест 2: Два фото РАЗНЫХ людей
    print("\n" + "=" * 60)
    print("ТЕСТ 2: РАЗНЫЕ ЛЮДИ (должно быть < 0.2)")
    print("=" * 60)

    photo3_path = input("Путь к фото ДРУГОГО человека: ").strip().strip('"')

    img3 = cv2.imread(photo3_path)
    if img3 is None:
        print("Ошибка загрузки фото.")
        return

    face3 = detect_face(img3)
    if face3 is None:
        print("Не удалось найти лицо.")
        return

    emb3 = get_embedding(model, face3, device)
    sim_diff = cosine_similarity(emb1, emb3)

    print(f"Косинусное сходство (разные люди): {sim_diff:.4f}")
    if sim_diff < 0.2:
        print("✅ ОТЛИЧНО! Модель различает разных людей.")
    else:
        print("⚠️ Высокое сходство.")

    # --- Метрики ---
    metrics = compute_metrics([emb1, emb2, emb3])

    print("\n" + "=" * 60)
    print("МЕТРИКИ (как в обучении)")
    print("=" * 60)
    print(f"intra (свои):  {metrics['intra']:.4f}")
    print(f"inter (чужие): {metrics['inter']:.4f}")
    print(f"θ° свои:       {metrics['θ_same']:.1f}°")
    print(f"θ° чужие:      {metrics['θ_diff']:.1f}°")
    print(f"Разница:       {metrics['intra'] - metrics['inter']:.4f}")

    # Итог
    print("\n" + "=" * 60)
    print("ИТОГ")
    print("=" * 60)
    print(f"Косинусное сходство:")
    print(f"  Свой человек:  {sim_same:.4f}  (хорошо > 0.3)")
    print(f"  Чужой человек: {sim_diff:.4f}  (хорошо < 0.2)")
    print(f"  Разница:       {sim_same - sim_diff:.4f}  (хорошо > 0.15)")
    print()
    print(f"Углы:")
    print(f"  θ° свои:       {metrics['θ_same']:.1f}°  (хорошо < 50°)")
    print(f"  θ° чужие:      {metrics['θ_diff']:.1f}°  (хорошо > 75°)")

    if sim_same > 0.3 and sim_diff < 0.2 and (sim_same - sim_diff) > 0.15:
        print("\n🔥 Модель работает корректно!")
    else:
        print("\n⚠️ Нужно ещё поучить или подобрать порог.")


if __name__ == "__main__":
    main()