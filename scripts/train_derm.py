"""
DermAI — Pipeline d'entraînement EfficientNet-B4
=================================================
Datasets : ISIC 2020 + HAM10000 + BCN20000 + Fitzpatrick17k.
Classes  : 25 pathologies dermatologiques.
Export   : ONNX Runtime CPU.

Usage :
    python scripts/train_derm.py --dataset data/derm --epochs 50
    python scripts/train_derm.py --list-datasets
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

CLASSES = [
    "Mélanome","Carcinome basocellulaire","Carcinome épidermoïde","Kératose actinique","Carcinome de Merkel",
    "Psoriasis","Eczéma atopique","Dermatite de contact","Rosacée","Lichen plan",
    "Teigne","Candidose cutanée","Impétigo","Gale","Herpès","Zona","Verrues virales",
    "Vitiligo","Mélasman","Hyperpigmentation post-inflammatoire",
    "Lupus cutané","Sclérodermie","Dermatomyosite",
    "Nævus bénin","Kératose séborrhéique",
]

DATASETS = {
    "ISIC Archive 2020": {"images":33_126,"classes":"Mélanome + bénins","license":"CC-BY-NC",
                          "download":"kaggle competitions download -c siim-isic-melanoma-classification"},
    "HAM10000":          {"images":10_015,"classes":"7 (mélanome, BCC, kératose, nævus…)","license":"CC-BY-NC",
                          "download":"kaggle datasets download -d kmader/skin-lesion-analysis-toward-melanoma-detection"},
    "BCN20000":          {"images":19_424,"classes":"7 classes dermoscopiques","license":"CC-BY-NC",
                          "download":"https://challenge.isic-archive.com/"},
    "Fitzpatrick17k":    {"images":16_577,"classes":"114 conditions + Fitzpatrick phototype","license":"CC-BY-NC",
                          "download":"https://github.com/mattgroh/fitzpatrick17k"},
    "PH2 Dataset":       {"images":200,"classes":"Mélanome/atypique/bénin","license":"Libre",
                          "download":"https://www.fc.up.pt/addi/ph2 database.html"},
    "SD-198":            {"images":6_584,"classes":"198 conditions dermatologiques","license":"Recherche",
                          "download":"https://xiaoxiaoshen.com/research/SD-198"},
}

TRAINING_SCRIPT = '''"""
DermAI — Entraînement EfficientNet-B4
"""
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.metrics import roc_auc_score
import json, time

CLASSES = {classes}
NUM_CLASSES = len(CLASSES)
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "deep_learning"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class SkinDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df; self.img_dir = Path(img_dir); self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.array(Image.open(self.img_dir/row["filename"]).convert("RGB"))
        if self.transform: img = self.transform(image=img)["image"]
        return img, torch.LongTensor([row.get("label",0)])[0]

class DermModel(nn.Module):
    def __init__(self, n_classes=25):
        super().__init__()
        from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
        self.backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        in_f = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(in_f, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, n_classes))
    def forward(self, x): return self.backbone(x)

def get_transforms():
    train = A.Compose([
        A.Resize(380,380), A.RandomCrop(380,380),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3),
        A.Rotate(limit=45,p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.6),
        A.GaussNoise(var_limit=(10,40),p=0.3),
        A.CLAHE(clip_limit=2.0,p=0.4),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()])
    val = A.Compose([
        A.Resize(380,380),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()])
    return train, val

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train(); loss_sum = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            out = model(imgs); loss = criterion(out, labels)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        loss_sum += loss.item()
    return loss_sum / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval(); loss_sum=0; all_p=[]; all_l=[]
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with autocast(): out=model(imgs); loss=criterion(out,labels)
            loss_sum += loss.item()
            all_p.append(torch.softmax(out,dim=1).cpu().numpy())
            all_l.append(labels.cpu().numpy())
    preds=np.vstack(all_p); labs=np.hstack(all_l)
    from sklearn.preprocessing import label_binarize
    labs_bin = label_binarize(labs, classes=list(range(NUM_CLASSES)))
    try: auc = roc_auc_score(labs_bin, preds, average="macro", multi_class="ovr")
    except: auc = 0.0
    return loss_sum/len(loader), auc

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {{device}} | Classes: {{NUM_CLASSES}}")
    train_tf, val_tf = get_transforms()
    dp = Path(args.dataset)
    import pandas as pd
    train_df = pd.read_csv(dp/"train.csv"); val_df = pd.read_csv(dp/"val.csv")
    train_ds = SkinDataset(train_df, dp/"images", train_tf)
    val_ds   = SkinDataset(val_df,   dp/"images", val_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)
    model = DermModel(NUM_CLASSES).to(device)
    optimizer = optim.AdamW([
        {{"params":model.backbone.features.parameters(),"lr":1e-5}},
        {{"params":model.backbone.classifier.parameters(),"lr":1e-3}},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()
    best_auc = 0
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        tl = train_one_epoch(model,train_loader,optimizer,criterion,scaler,device)
        vl, vauc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"Epoch {{epoch:3d}} | loss {{tl:.4f}} | val_auc {{vauc:.4f}} | {{time.time()-t0:.1f}}s")
        if vauc > best_auc:
            best_auc = vauc
            torch.save({{"state_dict":model.state_dict(),"classes":CLASSES,"auc":vauc}},
                       MODEL_DIR/"derm_best.pth")
    # Export ONNX
    ckpt = torch.load(MODEL_DIR/"derm_best.pth", map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    dummy = torch.randn(1,3,380,380)
    torch.onnx.export(model, dummy, str(MODEL_DIR/"derm_model.onnx"),
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={{"input":{{0:"batch"}},"output":{{0:"batch"}}}},
                      opset_version=17, do_constant_folding=True)
    meta = {{"classes":CLASSES,"architecture":"EfficientNet-B4","best_auc":best_auc,"input_size":380}}
    with open(MODEL_DIR/"derm_model.json","w") as f: json.dump(meta,f,indent=2,ensure_ascii=False)
    print(f"ONNX exporté — AUC best: {{best_auc:.4f}}")

if __name__ == "__main__": main()
'''.format(classes=repr(CLASSES))


def main():
    parser = argparse.ArgumentParser(description="DermAI Training Pipeline")
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--generate-script", action="store_true")
    parser.add_argument("--export-config", action="store_true")
    args = parser.parse_args()
    ROOT = Path(__file__).resolve().parent.parent

    if args.list_datasets:
        print("\n=== Datasets DermAI ===\n")
        for name, info in DATASETS.items():
            print(f"{'─'*55}")
            print(f"  {name}")
            print(f"  Images   : {info['images']:,}")
            print(f"  Classes  : {info['classes']}")
            print(f"  Licence  : {info['license']}")
            print(f"  Download : {info['download']}")
        print()
    elif args.generate_script:
        out = ROOT/"scripts"/"train_derm_runner.py"
        out.write_text(TRAINING_SCRIPT, encoding="utf-8")
        print(f"Script généré : {out}")
    elif args.export_config:
        cfg = {"module":"DermAI v2.0","classes":CLASSES,"datasets":DATASETS,
               "architecture":"EfficientNet-B4","input_size":380,
               "scores":["ABCDE","Breslow","Clark","TNM AJCC 8e","PASI","SCORAD","EASI","GAGS","VASI"]}
        out = ROOT/"scripts"/"derm_config.json"
        with open(out,"w",encoding="utf-8") as f: json.dump(cfg,f,indent=2,ensure_ascii=False)
        print(f"Config exportée : {out}")
    else:
        print(f"DermAI Training | Classes: {len(CLASSES)} | Datasets: {len(DATASETS)}")
        print("Options: --list-datasets | --generate-script | --export-config")

if __name__ == "__main__": main()
