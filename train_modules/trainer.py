import torch
from tqdm import tqdm

def train_one_epoch(
    model, dataloader, criterion, optimizer, device, 
    gpu_transform, amp_dtype, use_bf16, scaler, 
    dora_magnitude_params, accumulation_steps=1
):
    model.train()
    running_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0, device=device)
    total = 0

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(dataloader, desc="Training", leave=False)

    trainable_params = [p for group in optimizer.param_groups for p in group['params']]

    for i, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).view(-1).long()

        with torch.no_grad():
            images = gpu_transform(images)
        
        images = images.to(memory_format=torch.channels_last)

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_scaled = loss / accumulation_steps

        if use_bf16:
            loss_scaled.backward()
        else:
            scaler.scale(loss_scaled).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            if use_bf16:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
            else:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            for p in dora_magnitude_params:
                p.data.clamp_(min=1e-6)

            optimizer.zero_grad(set_to_none=True)

        batch_size = images.size(0)
        running_loss += loss.detach() * batch_size
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        total += batch_size

        if i % 10 == 0:
            pbar.set_postfix({
                "loss": f"{(running_loss / total).item():.4f}",
                "acc": f"{(correct.float() / total).item():.4f}"
            })

    epoch_loss = (running_loss / total).item()
    epoch_acc = (correct.float() / total).item()
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device, gpu_transform, desc="Validation"):
    model.eval()
    running_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0, device=device)
    total = 0

    all_targets, all_preds, all_probs = [], [], []

    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    pbar = tqdm(dataloader, desc=desc, leave=False)

    with torch.no_grad():
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).view(-1).long()

            images = gpu_transform(images)
            images = images.to(memory_format=torch.channels_last)

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(images)
                loss = criterion(outputs, labels)

            probs = torch.softmax(outputs.float(), dim=1)
            _, preds = outputs.max(1)

            batch_size = images.size(0)
            running_loss += loss.detach() * batch_size
            correct += preds.eq(labels).sum()
            total += batch_size

            all_targets.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

            if i % 5 == 0 or (i + 1) == len(dataloader):
                pbar.set_postfix({
                    "loss": f"{(running_loss / total).item():.4f}",
                    "acc": f"{(correct.float() / total).item():.4f}"
                })

    epoch_loss = (running_loss / total).item()
    epoch_acc = (correct.float() / total).item()
    
    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_probs = torch.cat(all_probs).numpy()

    return epoch_loss, epoch_acc, (y_true, y_pred, y_probs)