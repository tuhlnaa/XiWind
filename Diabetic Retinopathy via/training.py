import torch
import random
import numpy as np
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from pathlib import Path
from torchvision.models import MobileNet_V2_Weights
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_curve, auc, classification_report, confusion_matrix

from model import DRModel
from dataloader import prepare_data


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch_idx, (images, target) in enumerate(train_loader):
        images, target = images.to(device), target.to(device)

        optimizer.zero_grad()             # Reset gradients
        output = model(images)            # Forward pass
        loss = criterion(output, target)  # Compute loss
        loss.backward()                   # Backward pass
        optimizer.step()                  # Update parameters
        total_loss += loss.item()

    return total_loss / len(train_loader)


# Validation function
def evaluate_performance(model, device, loader):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_targets, all_outputs = [], []
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            total_loss += F.cross_entropy(outputs,
                                          targets, reduction='sum').item()

            _, preds = torch.max(outputs, 1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)

            all_targets.append(targets.cpu().numpy())
            all_outputs.append(outputs.softmax(dim=1).cpu().numpy())
    return total_loss / total_samples, 100.*total_correct/total_samples, total_correct, np.concatenate(all_targets), np.concatenate(all_outputs)


def validate(model, device, loader, epoch, logger):
    test_loss, accuracy, correct, targets, outputs = evaluate_performance(
        model, device, loader)
    binary_predictions = outputs.argmax(axis=1)

    report = classification_report(
        targets,
        binary_predictions,
        target_names=['No Diabetic Retinopathy', 'Diabetic Retinopathy'],
        output_dict=True,
        zero_division=0)

    logger.log_report_metrics(
        epoch,
        report,
        ['No Diabetic Retinopathy', 'Diabetic Retinopathy',
            'macro avg', 'weighted avg'],
        ['precision', 'recall', 'f1-score'])

    logger.plot_precision_recall_curve(targets, outputs, epoch)
    logger.plot_confusion_matrix(targets, binary_predictions, epoch)
    return test_loss, correct, accuracy


class Logger:
    def __init__(self, log_dir, checkpoint_dir):
        self.writer = SummaryWriter(log_dir)
        self.checkpoint_dir = checkpoint_dir
        # Ensure the directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def log_report_metrics(self, epoch, report, category_names, metric_names, prefix='Metrics'):
        for metric in metric_names:
            self.writer.add_scalars(f'{prefix}/{metric}',
                                    {category: report[category][metric] for category in category_names}, epoch)

    def log_scalars(self, tag, values, epoch):
        self.writer.add_scalars(tag, values, epoch)

    def plot_precision_recall_curve(self, targets, predictions, epoch):
        precision, recall, _ = precision_recall_curve(
            targets, predictions[:, 1])
        auc_score = auc(recall, precision)
        fig, ax = plt.subplots(dpi=150, layout='tight')
        ax.plot(recall, precision, marker='.', label=f'AUC = {auc_score:.2f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        self.writer.add_figure('Precision-Recall Curve', fig, epoch)

    def plot_confusion_matrix(self, targets, predictions, epoch):
        cm = confusion_matrix(targets, predictions)
        fig, ax = plt.subplots(dpi=150, layout='tight')
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No Diabetic Retinopathy",
                                 "Diabetic Retinopathy"],
                    yticklabels=["No Diabetic Retinopathy", "Diabetic Retinopathy"])
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('Actual Label')
        ax.set_title('Confusion Matrix')
        self.writer.add_figure('Confusion Matrix', fig, epoch)

    def close(self):
        self.writer.close()

    def save_checkpoint(self, epoch, model, optimizer, scheduler, loss, frequency=5):
        if (epoch + 1) % frequency == 0:
            checkpoint_path = self.checkpoint_dir / \
                f'model_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'last_loss': loss,
            }, checkpoint_path)
            print(f"Checkpoint saved at '{checkpoint_path}'")

    def load_checkpoint(self, model, optimizer, scheduler):
        checkpoints = list(self.checkpoint_dir.glob('model_epoch_*.pth'))
        if checkpoints:
            # Extract epochs from file names and find the max epoch
            latest_checkpoint = max(
                checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
            checkpoint = torch.load(latest_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['last_loss']
            print(
                f"Loaded checkpoint from '{latest_checkpoint}' at epoch {epoch} with loss {loss}")
            return epoch, loss
        else:
            print("No checkpoints found.")
            return 0, None  # Return defaults if no checkpoint found


def main():
    # Hyperparameters
    lr = 0.0001        # Learning rate
    batch_size = 64    # Batch Size
    num_epochs = 70    # Number of epochs
    target_size = 224  # Image Size

    root_dir = Path(__file__).parent
    train_data_dir = root_dir / "dataset" / "train"
    valid_data_dir = root_dir / "dataset" / "valid"
    checkpoint_dir = root_dir / "checkpoint"

    # Logger setup
    logger = Logger('runs/experiment2', checkpoint_dir)
    device = ("cuda"
              if torch.cuda.is_available()
              else "mps"
              if torch.backends.mps.is_available()
              else "cpu")

    # Data preparation
    train_loader, test_loader = prepare_data(
        train_data_dir, valid_data_dir, batch_size, target_size)

    # Model, Optimizer, Scheduler, Loss function initialization
    model = DRModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', 0.5, 20)
    criterion = torch.nn.CrossEntropyLoss()

    # Load checkpoint if exists
    start_epoch, _ = logger.load_checkpoint(model, optimizer, scheduler)

    for epoch in range(start_epoch, num_epochs):
        train_loss = train(model, device, train_loader, optimizer, criterion)
        get_last_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Learning Rate: {get_last_lr}')

        validation_loss, correct, val_accuracy = validate(
            model, device, test_loader, epoch, logger)
        print(
            f'Validation Loss: {validation_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({val_accuracy:.2f}%)\n')
        scheduler.step(validation_loss)

        logger.log_scalars(
            'Loss', {'Training': train_loss, 'Validation': validation_loss}, epoch)
        logger.log_scalars('Accuracy', {'Validation': val_accuracy}, epoch)
        logger.save_checkpoint(epoch, model, optimizer, scheduler, train_loss)
    logger.close()


if __name__ == '__main__':
    main()
