import os
import cv2
import gc
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
import h5py
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LinearRegression
from collections import defaultdict
import joblib
import xml.etree.ElementTree as ET
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def feature_collate(batch):
    # batch: list of (img_id, crops)
    return batch

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

# ----------------------------- Config -----------------------------
DEVICE = (
    torch.device("xpu") if hasattr(torch, "xpu") and torch.xpu.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"[DEVICE] {DEVICE}")

VOC_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor"
]
CLASS_TO_IDX = {c:i for i,c in enumerate(VOC_CLASSES)}

DATA_ROOT    = "./VOC2007"
JPEG_DIR     = os.path.join(DATA_ROOT, "JPEGImages")
ANNOT_DIR    = os.path.join(DATA_ROOT, "Annotations")
CACHE_DIR    = "./RCNNDataCache"
FEATURE_DIR  = os.path.join(CACHE_DIR, "features")
MODEL_DIR    = os.path.join(CACHE_DIR, "models")
RESULT_DIR   = os.path.join(CACHE_DIR, "results")

os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULT_DIR,  exist_ok=True)

PROPOSAL_PATH = os.path.join(CACHE_DIR, "proposals.pkl")
INDEX_FILE    = os.path.join(CACHE_DIR, "rcnn_index.txt")
ALEXNET_PATH  = os.path.join(MODEL_DIR, "custom_alexnet.pth")

H5_PATH = os.path.join(CACHE_DIR, "features.h5")
_h5file = None

def get_h5():
    global _h5file
    if _h5file is None:
        _h5file = h5py.File(H5_PATH, "r")
    return _h5file

# ----------------------------- Selective Search -----------------------------
from cv2.ximgproc import segmentation
cv2.setNumThreads(1)
_ss = None

def init_ss():
    global _ss
    _ss = segmentation.createSelectiveSearchSegmentation()

def compute_proposals(img_id):
    img = cv2.imread(os.path.join(JPEG_DIR, f"{img_id}.jpg"))
    if img is None:
        return img_id, []
    _ss.setBaseImage(img)
    _ss.switchToSelectiveSearchFast()
    rects = _ss.process()[:500]  # 최대 2000개
    filtered = [r for r in rects if r[2]>=20 and r[3]>=20]
    return img_id, filtered

def selective_search_all_images():
    ids = [f[:-4] for f in os.listdir(JPEG_DIR) if f.endswith(".jpg")]
    props = {}
    cores = max(1, mp.cpu_count()-2)
    with mp.Pool(cores, initializer=init_ss) as pool:
        for img_id, boxes in tqdm(pool.imap(compute_proposals, ids),
                                  total=len(ids),
                                  desc="Selective Search"):
            if boxes:
                props[img_id] = boxes
    return props

# ----------------------------- Index 생성 -----------------------------

def compute_iou(a, b):
    xa, ya, xa2, ya2 = a; xb, yb, xb2, yb2 = b
    ix1, iy1 = max(xa, xb), max(ya, yb)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw = max(ix2-ix1+1,0); ih = max(iy2-iy1+1,0)
    inter = iw*ih
    union = (xa2-xa+1)*(ya2-ya+1)+(xb2-xb+1)*(yb2-yb+1)-inter
    return inter/union if union>0 else 0

def generate_index_file(props, path):
    with open(path, "w") as f:
        for img_id, boxes in tqdm(props.items(), desc="Generate Index"):
            tree = ET.parse(os.path.join(ANNOT_DIR, f"{img_id}.xml"))
            gts = []
            for obj in tree.findall("object"):
                cls = obj.find("name").text
                bb = obj.find("bndbox")
                xmin = int(bb.find("xmin").text)
                ymin = int(bb.find("ymin").text)
                xmax = int(bb.find("xmax").text)
                ymax = int(bb.find("ymax").text)
                gts.append((CLASS_TO_IDX[cls], (xmin, ymin, xmax, ymax)))
            for idx, (x,y,w,h) in enumerate(boxes):
                prop = (x, y, x+w, y+h)
                best_i, best_lbl = 0, 20
                for cls_i, gt_bb in gts:
                    iou = compute_iou(prop, gt_bb)
                    if iou > best_i:
                        best_i, best_lbl = iou, cls_i
                if best_i >= 0.5:
                    lbl = best_lbl
                elif best_i < 0.3:
                    lbl = 20
                else:
                    continue
                f.write(f"{img_id},{idx},{lbl},{x},{y},{w},{h}\n")

# ----------------------------- CustomAlexNet 정의 -----------------------------
class CustomAlexNet(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,11,4,2), nn.ReLU(), nn.MaxPool2d(3,2),
            nn.Conv2d(64,192,5,padding=2), nn.ReLU(), nn.MaxPool2d(3,2),
            nn.Conv2d(192,384,3,padding=1), nn.ReLU(),
            nn.Conv2d(384,256,3,padding=1), nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(3,2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        EMBED_DIM = 1024
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, EMBED_DIM), nn.ReLU(),      # fc6 → EMBED_DIM
            nn.Dropout(),
            nn.Linear(EMBED_DIM, EMBED_DIM), nn.ReLU(),    # fc7 → EMBED_DIM
            nn.Linear(EMBED_DIM, num_classes)              # 최종 분류기
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        return self.classifier(x)

    def extract_fc7(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        # fc6 & fc7
        return self.classifier[:4](x)

# ----------------------------- PatchDataset -----------------------------

crop_transform = transforms.Compose([
    transforms.Resize((227,227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

class PatchDataset(Dataset):
    def __init__(self, index_file):
        self.entries = []
        with open(index_file) as f:
            for line in f:
                img_id, idx, lbl, x, y, w, h = line.strip().split(",")
                self.entries.append((img_id, int(lbl), int(x), int(y), int(w), int(h)))

        # Normalize 트랜스폼 미리 생성
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        img_id, lbl, x, y, w, h = self.entries[i]
        img = cv2.imread(os.path.join(JPEG_DIR, f"{img_id}.jpg"))
        roi = img[y:y+h, x:x+w]
        # BGR→RGB
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        # OpenCV로 바로 리사이즈
        small = cv2.resize(roi, (227, 227), interpolation=cv2.INTER_LINEAR)
        # NumPy→Tensor, [H,W,3]→[3,H,W]
        tensor = torch.from_numpy(small.astype(np.float32) / 255.0).permute(2, 0, 1)
        # 정규화
        tensor = self.normalize(tensor)
        return tensor, lbl

class FeatureExtractDataset(Dataset):
    def __init__(self, props):
        # props: dict[img_id] = list of (x,y,w,h)
        self.items = list(props.items())
        # 전처리 트랜스폼: numpy→Tensor, 정규화
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_id, boxes = self.items[idx]
        img = cv2.imread(os.path.join(JPEG_DIR, f"{img_id}.jpg"))
        crops = []
        for i, (x, y, w, h) in enumerate(boxes):
            roi = img[y:y+h, x:x+w]
            if roi.shape[0]<20 or roi.shape[1]<20:
                continue
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            small = cv2.resize(roi, (227,227), interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(small.astype(np.float32)/255.0).permute(2,0,1)
            t = self.normalize(t)
            crops.append((i, t))
        return img_id, crops

def extract_with_dataloader(model, props, h5_path, batch_imgs=4, num_workers=4):
    ds = FeatureExtractDataset(props)
    loader = DataLoader(ds, batch_size=batch_imgs, num_workers=num_workers,
                         shuffle=False, collate_fn=feature_collate)

    # HDF5 파일을 쓰려면 'w' 로 열고, 한 번만 그룹 생성
    with h5py.File(h5_path, "w") as hf:
        model.eval()
        for batch in tqdm(loader, desc="Feat Extract (HDF5)"):
            for img_id, crops in batch:
                if not crops: continue
                idxs, tensors = zip(*crops)
                x = torch.stack(tensors,0).to(DEVICE)
                with torch.no_grad():
                    feats = model.extract_fc7(x).cpu().numpy()  # (N,4096)

                grp = hf.create_group(img_id)
                grp.create_dataset("features", data=feats, 
                                   compression="lzf")
                grp.create_dataset("indices",  data=np.array(idxs,dtype=np.int32),
                                   compression="lzf")

# ----------------------------- train_model -----------------------------

def train_model():
    ds = PatchDataset(INDEX_FILE)
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)
    model = CustomAlexNet().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(5):
        model.train()
        total, correct = 0, 0
        for imgs, labels in tqdm(loader, desc=f"Epoch {ep+1}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss_fn(out, labels).backward()
            optimizer.step()
            correct += (out.argmax(1)==labels).sum().item()
            total += labels.size(0)
        print(f"[Epoch {ep+1}] Acc: {correct/total*100:.2f}%")
    torch.save(model.state_dict(), ALEXNET_PATH)
    return model

class FeatureDataset(Dataset):
    def __init__(self, index_file):
        self.items = []
        with open(index_file) as f:
            for line in f:
                img_id, prop_idx, label, *_ = line.strip().split(",")
                self.items.append((img_id, int(prop_idx), int(label)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img_id, prop_idx, label = self.items[i]
        hf = get_h5()
        grp = hf[img_id]
        indices = grp["indices"][()]
        feats   = grp["features"][()]
        pos = np.where(indices == prop_idx)[0][0]
        vec = feats[pos]
        return vec, label

# ----------------------------- train_svms -----------------------------

def train_svms():
    # 1) index_file 에서 (img_id, prop_idx, label) 리스트 만들기
    items = []
    with open(INDEX_FILE) as f:
        for line in f:
            img_id, prop_idx, label, *_ = line.strip().split(",")
            items.append((img_id, int(prop_idx), int(label)))

    # 2) 이미지별로 묶기
    entries_by_img = defaultdict(list)
    for img_id, prop_idx, label in items:
        entries_by_img[img_id].append((prop_idx, label))

    # 3) 클래스별 pos/neg 버킷
    X_by_cls = {c:([],[]) for c in range(len(VOC_CLASSES))}

    # 4) HDF5 에서 피처 한 번만 로드
    hf = get_h5()
    print("[SVM] HDF5에서 피처 로드 중…")
    for img_id, plist in tqdm(entries_by_img.items(), desc="Images"):
        grp     = hf[img_id]
        feats   = grp["features"][()]
        indices = grp["indices"][()]
        for prop_idx, label in plist:
            pos = np.where(indices == prop_idx)[0][0]
            vec = feats[pos]
            if label < 20:
                X_by_cls[label][0].append(vec)
            else:
                for c in range(len(VOC_CLASSES)):
                    X_by_cls[c][1].append(vec)

    # 5) 클래스별 SVM 학습/로드
    print("\n[SVM] 클래스별 학습 또는 로드 시작")
    for c, cls in enumerate(VOC_CLASSES):
        pkl_path = os.path.join(MODEL_DIR, f"svm_{cls}.pkl")
        pos, neg = X_by_cls[c]
        total = len(pos) + len(neg)
        pct_pos = len(pos) / total * 100 if total else 0
        ratio = f"{len(pos)}:{len(neg)}"

        # 로깅: 개수, 퍼센트, 비율
        print(
            f"[SVM] [{cls:12}] Pos={len(pos):5d}, Neg={len(neg):7d}  "
            f"({pct_pos:5.2f}% positive, ratio {ratio}) → ",
            end=""
        )

        if os.path.exists(pkl_path):
            # 이미 학습된 모델이 있으면 로드만
            clf = joblib.load(pkl_path)
            print("Loaded")
            continue

        # 학습에 필요한 샘플이 충분하지 않으면 건너뛰기
        if not pos or not neg:
            print("Skip (no samples)")
            continue

        X = np.vstack([pos, neg])
        y = np.hstack([np.ones(len(pos)), -np.ones(len(neg))])
        clf = LinearSVC(C=0.01, max_iter=10000)
        clf.fit(X, y)
        joblib.dump(clf, pkl_path)
        print("Trained & Saved")
        # --- 디버깅: 학습된 SVM 점수 분포 확인 ---
        sample_idx = np.random.choice(len(X), size=min(1000, len(X)), replace=False)
        sample_scores = clf.decision_function(X[sample_idx])
        print(f"    [DEBUG] {cls} score range: {sample_scores.min():.3f} ~ {sample_scores.max():.3f}")

# ----------------------------- train_bbox_regressors -----------------------------

def train_bbox_regressors():
    # 1) index_file 로부터 (img_id, prop_idx)→(label, x,y,w,h) 딕셔너리 생성
    idx_dict = {}
    with open(INDEX_FILE) as f:
        for line in f:
            img_id, prop_idx, lbl, x, y, w, h = line.strip().split(",")
            idx_dict[(img_id, int(prop_idx))] = (
                int(lbl), int(x), int(y), int(w), int(h)
            )

    # 2) 클래스별 positive 엔트리만 모아두기
    samples_by_cls = {c: [] for c in range(len(VOC_CLASSES))}
    for (img_id, prop_idx), (lbl, x, y, w, h) in idx_dict.items():
        if lbl < len(VOC_CLASSES):
            samples_by_cls[lbl].append((img_id, prop_idx, (x, y, w, h)))

    hf = get_h5()
    print("\n[Reg] BBox 회귀용 데이터 준비 완료")

    # 3) 클래스별 학습
    for c, cls in enumerate(tqdm(VOC_CLASSES, desc="Training BBox Regressors")):
        pkl_path = os.path.join(MODEL_DIR, f"reg_{cls}.pkl")

        # 이미 학습된 모델은 로드
        if os.path.exists(pkl_path):
            print(f"  → [{cls}] Load existing regressor")
            continue

        entries = samples_by_cls[c]
        print(f"  → [{cls}] Samples: {len(entries):5d}", end=" … ")

        if not entries:
            print("Skip (No positive samples)")
            continue

        X_list, Y_list = [], []
        for img_id, prop_idx, (x, y, w, h) in entries:
            # HDF5에서 한 번만 로드
            grp     = hf[img_id]
            feats   = grp["features"][()]
            indices = grp["indices"][()]
            pos     = np.where(indices == prop_idx)[0][0]
            vec     = feats[pos]

            # GT 박스 로드
            tree = ET.parse(os.path.join(ANNOT_DIR, f"{img_id}.xml"))
            for obj in tree.findall("object"):
                if obj.find("name").text != cls:
                    continue
                bb  = obj.find("bndbox")
                gx1 = int(bb.find("xmin").text)
                gy1 = int(bb.find("ymin").text)
                gx2 = int(bb.find("xmax").text)
                gy2 = int(bb.find("ymax").text)
                # 타겟 계산
                pw, ph = w, h
                tx = (gx1 - x) / pw
                ty = (gy1 - y) / ph
                tw = np.log((gx2 - gx1 + 1) / pw)
                th = np.log((gy2 - gy1 + 1) / ph)

                X_list.append(vec)
                Y_list.append([tx, ty, tw, th])
                break  # 첫 번째 GT만 사용

        # 회귀 모델 학습 및 저장
        reg = LinearRegression()
        reg.fit(np.vstack(X_list), np.vstack(Y_list))
        joblib.dump(reg, pkl_path)
        print("Done")

# ----------------------------- inference, nms, evaluate_map -----------------------------

def apply_bbox_regression(box, offs):
    x,y,w,h = box; tx,ty,tw,th = offs
    cx, cy = x+0.5*w, y+0.5*h
    cx_p,cy_p = tx*w+cx, ty*h+cy
    w_p, h_p = np.exp(tw)*w, np.exp(th)*h
    return [int(cx_p-0.5*w_p),int(cy_p-0.5*h_p),int(cx_p+0.5*w_p),int(cy_p+0.5*h_p)]

def inference():
    # 1) SVM·회귀 모델 로드
    svms, regs = {}, {}
    for i, cls in enumerate(VOC_CLASSES):
        sp = os.path.join(MODEL_DIR, f"svm_{cls}.pkl")
        rp = os.path.join(MODEL_DIR, f"reg_{cls}.pkl")
        if os.path.exists(sp): svms[i] = joblib.load(sp)
        if os.path.exists(rp): regs[i] = joblib.load(rp)

    # 2) index_map 준비
    lines = open(INDEX_FILE).read().splitlines()
    index_map = defaultdict(list)
    for L in lines:
        img_id, prop_idx, *_ = L.split(",")
        x, y, w, h = map(int, L.split(",")[-4:])
        index_map[img_id].append((int(prop_idx), (x, y, w, h)))

    hf = get_h5()
    dets = []
    print(f"[7-1] Inference on {len(hf.keys())} images")
    for img_id in tqdm(hf.keys(), desc="Inference (HDF5)"):
        grp     = hf[img_id]
        feats   = grp["features"][()]
        indices = grp["indices"][()]

        for feat_vec, prop_idx in zip(feats, indices):
            # 원본 박스
            box = next((b for pid,b in index_map[img_id] if pid == prop_idx), None)
            if box is None:
                continue

            # 각 클래스별로 score 계산
            for c, clf in svms.items():
                score = clf.decision_function(feat_vec.reshape(1,-1))[0]
                if score <= 0:
                    continue

                # bbox regression
                offs = regs[c].predict(feat_vec.reshape(1,-1))[0]
                coords = apply_bbox_regression(box, offs)

                # coords 가 (x1,y1,x2,y2) 외에 더 길다면 [:4] 로 자르기
                x1, y1, x2, y2 = map(int, coords[:4])

                dets.append((img_id, c, score, x1, y1, x2, y2))

    print(f"[7-1] Raw detections: {len(dets)}")
    return dets

def nms(dets, iou_thr=0.3):
    """
    dets: list of (img_id, cls, score, x1,y1,x2,y2)
    """
    print("[7-2] Applying NMS (vectorized)")
    keep = []

    # 1) NumPy 구조화 배열로 변환
    dt = np.dtype([
        ("img",    "U16"),
        ("cls",    "i4"),
        ("score",  "f4"),
        ("x1",     "f4"),
        ("y1",     "f4"),
        ("x2",     "f4"),
        ("y2",     "f4"),
    ])
    arr = np.array(dets, dtype=dt)

    # 2) 이미지별 · 클래스별로 나눠서 처리
    for img_id in np.unique(arr["img"]):
        mask_img = (arr["img"] == img_id)
        for cls in np.unique(arr["cls"][mask_img]):
            mask = mask_img & (arr["cls"] == cls)
            sub = arr[mask]
            if sub.size == 0:
                continue

            # 3) score 내림차순 정렬
            order = np.argsort(sub["score"])[::-1]
            boxes = np.vstack([sub["x1"], sub["y1"], sub["x2"], sub["y2"]]).T

            while order.size:
                i = order[0]
                keep.append((
                    sub["img"][i], sub["cls"][i], float(sub["score"][i]),
                    float(sub["x1"][i]), float(sub["y1"][i]),
                    float(sub["x2"][i]), float(sub["y2"][i]),
                ))

                # 남은 박스들과의 IoU 계산
                xx1 = np.maximum(boxes[i,0], boxes[order[1:],0])
                yy1 = np.maximum(boxes[i,1], boxes[order[1:],1])
                xx2 = np.minimum(boxes[i,2], boxes[order[1:],2])
                yy2 = np.minimum(boxes[i,3], boxes[order[1:],3])

                w = np.maximum(0, xx2 - xx1 + 1)
                h = np.maximum(0, yy2 - yy1 + 1)
                inter = w * h

                area_i = (boxes[i,2] - boxes[i,0] + 1) * (boxes[i,3] - boxes[i,1] + 1)
                area_rest = (boxes[order[1:],2] - boxes[order[1:],0] + 1) * \
                            (boxes[order[1:],3] - boxes[order[1:],1] + 1)
                union = area_i + area_rest - inter
                iou = inter / union

                # IoU <= threshold 인 것만 남긴다
                keep_idxs = np.where(iou <= iou_thr)[0]
                order = order[keep_idxs + 1]

    print(f"[7-2] After NMS: {len(keep)}")
    return keep

def evaluate_map(dets):
    print("[7-3] Computing mAP")
    gt = defaultdict(list)
    for f in os.listdir(ANNOT_DIR):
        img_id = f[:-4]
        tree   = ET.parse(os.path.join(ANNOT_DIR, f))
        for obj in tree.findall("object"):
            bb = obj.find("bndbox")
            gt[img_id].append([
                int(bb.find("xmin").text), int(bb.find("ymin").text),
                int(bb.find("xmax").text), int(bb.find("ymax").text)
            ])

    by_cls = defaultdict(list)
    for img_id, c, score, x1, y1, x2, y2 in dets:
        by_cls[c].append((img_id, score, [x1,y1,x2,y2]))

    aps = []
    for c in tqdm(by_cls.keys(), desc="mAP per-class"):
        preds = sorted(by_cls[c], key=lambda x: -x[1])
        tp, fp, matched = [], [], set()
        total = sum(len(v) for v in gt.values())

        for img_id, _, bb in preds:
            is_tp = False
            for i, gt_bb in enumerate(gt[img_id]):
                if compute_iou(bb, gt_bb) > 0.5 and (img_id,i) not in matched:
                    tp.append(1); fp.append(0); matched.add((img_id,i))
                    is_tp = True
                    break
            if not is_tp:
                tp.append(0); fp.append(1)

        tp   = np.cumsum(tp)
        fp   = np.cumsum(fp)
        rec  = tp/total if total else tp*0
        prec = tp/np.maximum(tp+fp, np.finfo(float).eps)

        ap = sum((np.max(prec[rec>=t]) if np.any(rec>=t) else 0)/11.0
                 for t in np.arange(0,1.1,0.1))
        aps.append(ap)

    mAP = np.mean(aps)
    print(f"[7-3] Final mAP: {mAP:.4f}")

def visualize_detections(raw, max_show=5):
    for img_id, cls, score, x1, y1, x2, y2 in raw[:max_show]:
        img = cv2.imread(os.path.join(JPEG_DIR, f"{img_id}.jpg"))[..., ::-1]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        ax.set_title(f"{VOC_CLASSES[cls]} ({score:.2f})")
        ax.axis('off')
        plt.show()

# ----------------------------- Main -----------------------------

if __name__ == "__main__":
    if os.path.exists(PROPOSAL_PATH):
        print("[1] 로드: proposals")
        with open(PROPOSAL_PATH, "rb") as f:
            props = pickle.load(f)
    else:
        print("[1] 생성: proposals")
        props = selective_search_all_images()
        with open(PROPOSAL_PATH, "wb") as f:
            pickle.dump(props, f)

    if not os.path.exists(INDEX_FILE):
        print("[2] 생성: index")
        generate_index_file(props, INDEX_FILE)
    else:
        print("[2] 로드: index")

    model = CustomAlexNet().to(DEVICE)

    if os.path.exists(ALEXNET_PATH):
        print("[3] 로드: Conv+Pool 계층만 전이, FC 계층 랜덤 초기화")
        ckpt = torch.load(ALEXNET_PATH, map_location=DEVICE)
        own_state = model.state_dict()
        for name, param in ckpt.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
        model.load_state_dict(own_state)
    else:
        print("[3] 학습: CustomAlexNet from scratch")
        model = train_model()

    if not os.path.exists(H5_PATH):
       print("[4] 생성: features → HDF5")
       extract_with_dataloader(model, props,
                              H5_PATH,
                              batch_imgs=4, num_workers=4)
    else:
        print("[4] 로드: features.h5")

    print("[5] SVM 학습")
    train_svms()

    print("[6] BBox 회귀 학습")
    train_bbox_regressors()

    print("[7] 검증: inference → NMS → mAP")
    raw   = inference()
    visualize_detections(raw, max_show=5)
    final = nms(raw)
    evaluate_map(final)
