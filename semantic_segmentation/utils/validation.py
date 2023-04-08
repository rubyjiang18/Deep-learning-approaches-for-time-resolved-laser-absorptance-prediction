import torch
from tqdm import tqdm

def validation(model, device, val_loader, optimizer, criterion, scheduler, epoch, epochs=300, amp=True):
  model.eval()
  total_loss = 0
  with tqdm(total=len(val_loader), desc=f'Validation: Epoch {epoch}/{epochs}', unit='img') as pbar:
    for batch in val_loader:
      images = batch['image']
      true_masks = batch['mask']

      images = images.float().to(device)
      true_masks = true_masks.float().to(device)

      with torch.cuda.amp.autocast(enabled=amp):
        masks_pred = model(images)
        loss = criterion(masks_pred.float(), true_masks.float())

      optimizer.zero_grad(set_to_none=True)

      pbar.update(images.shape[0])
      total_loss += loss.item()

      pbar.set_postfix(**{'loss (batch)': loss.item()})
  
  total_loss = float(total_loss) / len(val_loader)
  scheduler.step(total_loss)
  
  return total_loss