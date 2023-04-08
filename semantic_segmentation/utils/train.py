import torch
from tqdm import tqdm


def train(model, device, train_loader, optimizer, criterion, scheduler, grad_scaler, epoch, epochs=300, amp=True):
  model.train()
  epoch_loss = 0
  with tqdm(total=len(train_loader), desc=f'Train: Epoch {epoch}/{epochs}', unit='img') as pbar:
    for batch in train_loader:
      images = batch['image']
      true_masks = batch['mask']

      # assert images.shape[1] == model.n_channels, \
      #             f'Network has been defined with {model.n_channels} input channels, ' \
      #             f'but loaded images have {images.shape[1]} channels. Please check that ' \
      #             'the images are loaded correctly.'

      images = images.float().to(device)
      true_masks = true_masks.float().to(device)
      #masks_pred = (model(images) > 0.5).type(torch.float64) # mask predicted?

      with torch.cuda.amp.autocast(enabled=amp):
        masks_pred = model(images)
        loss = criterion(masks_pred.float(), true_masks.float())

      optimizer.zero_grad(set_to_none=True)
      grad_scaler.scale(loss).backward()
      grad_scaler.step(optimizer)
      grad_scaler.update()

      pbar.update(images.shape[0])
      epoch_loss += loss.item()

      pbar.set_postfix(**{'loss (batch)': loss.item()})
  
  return float(epoch_loss) / len(train_loader)
