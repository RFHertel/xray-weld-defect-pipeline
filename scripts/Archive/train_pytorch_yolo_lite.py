# scripts/train_pytorch_yolo_fixed.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import json
from pathlib import Path
import logging
from datetime import datetime
import time
from tqdm import tqdm
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class YOLOLiteFixed(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.num_classes = num_classes
        
        # Simplified backbone with tracked dimensions
        self.layer1 = ConvBlock(3, 32, stride=1)      # 640 -> 640
        self.layer2 = ConvBlock(32, 64, stride=2)     # 640 -> 320
        self.layer3 = ConvBlock(64, 128, stride=2)    # 320 -> 160  
        self.layer4 = ConvBlock(128, 256, stride=2)   # 160 -> 80
        
        # Detection head (simplified - single scale)
        self.detect = nn.Sequential(
            ConvBlock(256, 512),
            ConvBlock(512, 256),
            nn.Conv2d(256, (5 + num_classes), kernel_size=1)  # x,y,w,h,obj + classes
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x) 
        x = self.layer3(x)
        x = self.layer4(x)
        return self.detect(x)

class WeldDefectDataset(Dataset):
    def __init__(self, data_dir, split='train', img_size=640):
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / split / 'images'
        self.label_dir = self.data_dir / split / 'labels'
        self.img_size = img_size
        
        self.images = list(self.img_dir.glob('*.jpg'))
        
        # Class weights based on your distribution
        self.class_weights = torch.tensor([
            1.0,   # porosity (well represented)
            1.0,   # inclusion (well represented)  
            1.0,   # crack (well represented)
            1.5,   # undercut (needs boost)
            2.0,   # lack_of_fusion (needs major boost)
            3.0,   # lack_of_penetration (worst performance)
        ])
        
        logging.info(f"Loaded {len(self.images)} images for {split}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.transpose(2, 0, 1) / 255.0  # HWC -> CHW, normalize
        
        # Load labels
        label_path = self.label_dir / (img_path.stem + '.txt')
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        labels.append([float(x) for x in parts])
        
        return torch.FloatTensor(img), torch.FloatTensor(labels) if labels else torch.zeros((0, 5))

    @staticmethod
    def collate_fn(batch):
        imgs, labels = zip(*batch)
        imgs = torch.stack(imgs, 0)
        
        max_labels = max(len(l) for l in labels)
        batch_labels = torch.zeros(len(labels), max_labels, 5)
        
        for i, l in enumerate(labels):
            if len(l) > 0:
                batch_labels[i, :len(l)] = l
        
        return imgs, batch_labels

class WeightedYOLOLoss(nn.Module):
    def __init__(self, num_classes=6, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights if class_weights is not None else torch.ones(num_classes)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')
        
    def forward(self, pred, target):
        batch_size = pred.size(0)
        pred = pred.view(batch_size, -1, 5 + self.num_classes)
        
        # Simple loss (for demonstration - real YOLO loss is more complex)
        obj_mask = target[..., 4] > 0  # Objects present
        
        # Objectness loss
        obj_loss = self.bce(pred[..., 4], target[..., 4]).mean()
        
        # Class loss with weights
        if obj_mask.any():
            class_targets = target[obj_mask.unsqueeze(-1).expand_as(target)].view(-1, 5)[:, 0].long()
            class_preds = pred[obj_mask].view(-1, 5 + self.num_classes)[:, 5:]
            
            # Apply class weights
            weights = self.class_weights[class_targets].to(pred.device)
            class_loss = (nn.functional.cross_entropy(class_preds, class_targets, reduction='none') * weights).mean()
        else:
            class_loss = torch.tensor(0.0).to(pred.device)
        
        # Box loss (simplified)
        if obj_mask.any():
            box_loss = self.mse(pred[..., :4][obj_mask], target[..., :4][obj_mask]).mean()
        else:
            box_loss = torch.tensor(0.0).to(pred.device)
        
        total_loss = obj_loss + class_loss + 5.0 * box_loss
        
        return total_loss, {'obj': obj_loss.item(), 'cls': class_loss.item(), 'box': box_loss.item()}

def train_model(data_path, epochs=50, batch_size=8, lr=1e-3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model
    model = YOLOLiteFixed(num_classes=6).to(device)
    
    # Datasets
    train_dataset = WeldDefectDataset(data_path, 'train')
    val_dataset = WeldDefectDataset(data_path, 'val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            collate_fn=WeldDefectDataset.collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          collate_fn=WeldDefectDataset.collate_fn, num_workers=0)
    
    # Loss with class weights
    criterion = WeightedYOLOLoss(num_classes=6, class_weights=train_dataset.class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training
    output_dir = Path('models_pytorch') / f"yolo_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {'epoch': [], 'train_loss': [], 'val_loss': [], 'lr': []}
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss, _ = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                loss, _ = criterion(outputs, labels)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['lr'].append(scheduler.get_last_lr()[0])
        
        # Save metrics
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, output_dir / 'best_model.pth')
        
        scheduler.step()
        
        logging.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return str(output_dir / 'best_model.pth')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='processed_balanced_final')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=8)
    
    args = parser.parse_args()
    
    model_path = train_model(
        data_path=Path(r'C:\AWrk\SWRD_YOLO_Project') / args.data,
        epochs=args.epochs,
        batch_size=args.batch
    )
    
    print(f"Model saved to: {model_path}")