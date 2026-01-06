import argparse, os, json, numpy as np, torch, torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from .utils import set_seed, ensure_dir
from .data import get_loaders
from .models import SmallCNN, make_resnet18
from .metrics import compute_metrics, reliability_diagram


def main():
  ap=argparse.ArgumentParser()
  ap.add_argument('--dataset', type=str, default='pneumoniamnist')
  ap.add_argument('--model', type=str, default='smallcnn', choices=['smallcnn','resnet18'])
  ap.add_argument('--epochs', type=int, default=5)
  ap.add_argument('--batch_size', type=int, default=128)
  ap.add_argument('--lr', type=float, default=3e-4)
  ap.add_argument('--weight_decay', type=float, default=1e-4)
  ap.add_argument('--seed', type=int, default=42)
  ap.add_argument('--finetune', type=str, default='head', choices=['head','all'])
  ap.add_argument('--label_fracs', type=float, nargs='*', default=[1.0])
  ap.add_argument('--ece_bins', type=int, default=10)
  ap.add_argument('--outdir', type=str, default='runs')
  args=ap.parse_args()
  set_seed(args.seed); device='cuda' if torch.cuda.is_available() else 'cpu'

  for frac in args.label_fracs:
    run = f"{args.dataset}_{args.model}_frac{frac}_seed{args.seed}"; out = os.path.join(args.outdir, run); ensure_dir(out)
    tl, vl, te, nc = get_loaders(args.dataset, batch_size=args.batch_size, label_frac=frac, seed=args.seed)

    if args.model=='smallcnn':
      model = SmallCNN(in_ch=3,n_classes=nc); params = model.parameters()
    else:
      model = make_resnet18(n_classes=nc,in_ch=3,pretrained=True)

      if args.finetune=='head':
        for p in model.parameters(): p.requires_grad = False
        for p in model.fc.parameters(): p.requires_grad = True
        params = model.fc.parameters()
      else:
        params = model.parameters()

    model.to(device); crit = nn.CrossEntropyLoss(); opt = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sch = CosineAnnealingLR(opt, T_max=args.epochs)
    best = -1.0; best_state = None

    for ep in range(1, args.epochs+1):
      model.train(); ls = 0.0; n = 0

      for x, y in tl:
        x = x.to(device); y = y.long().to(device)
        logit = model(x); loss = crit(logit, y)
        opt.zero_grad(); loss.backward(); opt.step()
        ls += float(loss.item()) * x.size(0); n += x.size(0)
      # val
      model.eval(); import numpy as np
      probs = []; labels = []
      with torch.no_grad():
        for x, y in vl:
          x = x.to(device); p = torch.softmax(model(x), dim=1).cpu().numpy(); probs.append(p); labels.append(y.numpy())
      pv = np.concatenate(probs, 0); yv = np.concatenate(labels, 0)
      from sklearn.metrics import roc_auc_score, accuracy_score
      acc = accuracy_score(yv, np.argmax(pv, 1))

      try:
        auroc = roc_auc_score(yv, pv[:, 1])
      except Exception:
        auroc = float('nan')
        
      score = auroc if auroc == auroc else acc
      print(f"[{ep:02d}] loss={ls/max(1,n):.4f} | val acc={acc:.4f} auroc={auroc:.4f}")

      with open(os.path.join(out,'val_log.jsonl'),'a') as f:
        f.write(json.dumps({'epoch':ep,'loss':ls/max(1,n),'val_acc':float(acc),'val_auroc':float(auroc)})+'\n')

      sch.step()
      if score > best:
        best = score; best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        
    if best_state is not None: model.load_state_dict(best_state, strict=True)
    # test
    model.eval(); probs = []; labels = []
    with torch.no_grad():
      for x, y in te:
        x = x.to(device); p = torch.softmax(model(x), dim=1).cpu().numpy(); probs.append(p); labels.append(y.numpy())
    pt = np.concatenate(probs, 0); yt = np.concatenate(labels, 0)
    mt = compute_metrics(yt, pt, nc)
    ece = reliability_diagram(yt, pt, n_bins=args.ece_bins, save_path=os.path.join(out,'reliability.png'))

    with open(os.path.join(out,'test_metrics.json'),'w') as f:
      json.dump({'test':mt,'ece':ece,'n_classes':int(nc),'frac':float(frac)}, f, indent=2)
      
    torch.save(model.state_dict(), os.path.join(out,'best.pt'))
    print(f"TEST | acc={mt['acc']:.4f} auroc={mt['auroc']:.4f} ece={ece:.3f} -> {out}")
    
if __name__=='__main__':
  main()
