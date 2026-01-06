import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
import numpy as np
import medmnist
from medmnist import INFO
from .utils import set_seed


def get_medmnist_dataset(key, split, download=True, as_rgb=True, size=64):
  key = key.lower(); info = INFO[key]; DataClass = getattr(medmnist, info['python_class'])
  tf = [T.Resize((size,size)), T.ToTensor()]
  if as_rgb: tf.append(T.Lambda(lambda x: x.repeat(3,1,1) if x.shape[0]==1 else x))
  transform = T.Compose(tf); return DataClass(split=split, transform=transform, download=download)

def get_loaders(key,batch_size=128,num_workers=2,label_frac=1.0,seed=42):
  set_seed(seed)
  ds_train=get_medmnist_dataset(key,'train'); ds_val=get_medmnist_dataset(key,'val'); ds_test=get_medmnist_dataset(key,'test')
  n_classes=len(INFO[key]['label'])

  if 0 < label_frac < 1.0:
    y = np.array([int(t[1]) for t in ds_train]); idxs = []

    for c in np.unique(y):
      cls_idx = np.where(y==c)[0]; k=max(1, int(len(cls_idx)*label_frac))
      idxs.extend(np.random.choice(cls_idx, size=k, replace=False))
    ds_train = Subset(ds_train, sorted(idxs))

  tl = DataLoader(ds_train,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=True)
  vl = DataLoader(ds_val,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=True)
  te = DataLoader(ds_test,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=True)

  return tl, vl, te, n_classes
