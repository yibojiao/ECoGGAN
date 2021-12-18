import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

class classNet(nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # conv blocks
        convs = []
        convs.append(nn.Conv1d(in_channels=1, out_channels=64, padding=1, kernel_size=3, stride=1))
        convs.append(nn.ReLU())
        convs.append(nn.Conv1d(in_channels=64, out_channels=64, padding=1, kernel_size=3, stride=1))
        convs.append(nn.ReLU())
        convs.append(nn.MaxPool1d(kernel_size=3, stride=2))

        # block 2
        convs.append(nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        convs.append(nn.ReLU())
        convs.append(nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        convs.append(nn.ReLU())
        convs.append(nn.MaxPool1d(kernel_size=3, stride=2))

        # block 3
        convs.append(nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1))
        convs.append(nn.ReLU())
        convs.append(nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))
        convs.append(nn.ReLU())
        convs.append(nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))
        convs.append(nn.ReLU())
        convs.append(nn.MaxPool1d(kernel_size=3, stride=2))

        # block 4
        convs.append(nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1))
        convs.append(nn.ReLU())
        convs.append(nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        convs.append(nn.ReLU())
        convs.append(nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        convs.append(nn.ReLU())
        convs.append(nn.MaxPool1d(kernel_size=3, stride=2))

        # block 5
        convs.append(nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        convs.append(nn.ReLU())
        convs.append(nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        convs.append(nn.ReLU())
        convs.append(nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        convs.append(nn.ReLU())
        convs.append(nn.MaxPool1d(kernel_size=3, stride=2))
        
        # add net into class property
        self.conv_layer = nn.Sequential(*convs)

        # define an empty container for Linear operations
        linear = []
        linear.append(nn.Linear(in_features=1536, out_features=4096))
        linear.append(nn.ReLU())
        linear.append(nn.Dropout(p=0.15))
        linear.append(nn.Linear(in_features=4096, out_features=4096))
        linear.append(nn.ReLU())
        linear.append(nn.Dropout(p=0.15))
        linear.append(nn.Linear(in_features=4096, out_features=self.num_classes))

        # add classifier into class property
        self.fc = nn.Sequential(*linear)
        
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    
    def get_loss(self, pred, label):
        """Compute loss by comparing prediction and labels.

        Args:
            pred (array): BxD, probability distribution over D classes.
            label (array): B, category label.
        Returns:
            loss (tensor): scalar, cross entropy loss for classfication.
        """
        loss = F.nll_loss(pred, label)
        return loss

    def get_acc(self, pred, label):
        """Compute the acccuracy."""
        pred_choice = pred.max(dim=1)[1]
        acc = (pred_choice == label).float().mean()
        return acc
    
    
class Network(object):
    
    def __init__(self, model, config):
        """Initialize configuration."""
        self.config = config
        self.model = model
        self.optimizer = torch.optim.SGD(
            model.parameters(), lr=config.lr, momentum=0.9
        )
        if self.config.use_cuda:
            self.model.cuda()
        self._init_aux()
            
    def train(self, loader_tr, loader_va):
        """Training pipeline."""
        # Switch model into train mode.
        self.model.train()
        best_va_acc = 0  # Record the best validation accuracy.

        for epoch in range(15):
            losses = []
            for data in loader_tr:
                # Transfer data from CPU to GPU.
                if self.config.use_cuda:
                    for key in data.keys():
                        data[key] = data[key].cuda()
                
                pred = self.model(data["ecog"])
                loss = self.model.get_loss(pred, data["label"])
                losses += [loss]

                # Calculate the gradient.
                loss.backward()
                # Update the parameters according to the gradient.
                self.optimizer.step()
                # Zero the parameter gradients in the optimizer
                self.optimizer.zero_grad()

            loss_avg = torch.mean(torch.stack(losses))

            # Save model every epoch.
            #self._save(self.checkpts_file)
            acc = self.test(loader_va, mode="valid")

            # Early stopping strategy.
            if acc > best_va_acc:
                # Save model with the best accuracy on validation set.
                best_va_acc = acc
                self._save(self.bestmodel_file)
            self.log_func(
                "Epoch: %3d, loss_avg: %.5f, val OA: %.5f, best val OA: %.5f"
                % (epoch, loss_avg, acc, best_va_acc)
            )

            # Recording training losses and validation performance.
            self.train_losses += [loss_avg]
            self.valid_oas += [acc]
            self.idx_steps += [epoch]
            
    def _init_aux(self):
        """Intialize aux functions, features."""
        # Define func for logging.
        self.log_func = print

        # Define directory wghere we save states such as trained model.
        self.log_dir = self.config.log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # File that we save trained model into.
        self.checkpts_file = os.path.join(self.log_dir, "checkpoint.pth")

        # We save model that achieves the best performance: early stoppoing strategy.
        self.bestmodel_file = os.path.join(self.log_dir, "best_model.pth")

        # Recording the training losses and validation performance.
        self.train_losses = []
        self.valid_oas = []
        self.idx_steps = []

    def plot_log(self):
        """Plot training logs (better with tensorboard, but we will try matplotlib this time!)."""

        # Draw plots for the training and validation results, as shown in
        # the example results. Use matplotlib's subplots.
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle("Visualization of training logs")

        ax1.plot(self.idx_steps, self.train_losses)
        ax2.plot(self.idx_steps, self.valid_oas)
        ax1.set_title("Training loss curve")
        ax2.set_title("Validation accuracy curve")
        plt.tight_layout()
        plt.show()
        plt.close()
            
    def _save(self, pt_file):
        """Saving trained model."""

        # Save the trained model.
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            pt_file,
        )
        
    def _restore(self, pt_file):
        """Restoring trained model."""
        print(f"restoring {pt_file}")

        # Read checkpoint file.
        load_res = torch.load(pt_file)
        # Loading model.
        self.model.load_state_dict(load_res["model"])
        # Loading optimizer.
        self.optimizer.load_state_dict(load_res["optimizer"])
            
    def test(self, loader_te, mode="test"):
        """Estimating the performance of model on the given dataset."""
        # Choose which model to evaluate.
        if mode == "test":
            self._restore(self.bestmodel_file)
        # Switch the model into eval mode.
        self.model.eval()

        accs = []
        num_samples = 0
        for data in loader_te:
            if self.config.use_cuda:
                for key in data.keys():
                    data[key] = data[key].cuda()
            batch_size = len(data["ecog"])
            pred = self.model(data["ecog"])
            acc = self.model.get_acc(pred, data["label"])
            accs += [acc * batch_size]
            num_samples += batch_size

        avg_acc = torch.stack(accs).sum() / num_samples

        # Switch the model into training mode
        self.model.train()
        return avg_acc
            
if __name__ == "__main__":
    import sys
    sys.path.insert(1, '../')
    from get_config import get_config
    from get_dataloader import get_dataloader

    config = get_config()
    acc_logs = []
    for i in range(128):
        model = classNet(70)
        net = Network(model, config)
        dataloader_tr, dataloader_va = get_dataloader(config, i)

        net.train(dataloader_tr, dataloader_va)
        acc_logs.append(np.max(net.valid_oas))
        #net.plot_log()
    print(np.array(acc_logs), np.argmax(np.array(acc_logs)))
            
            
            
            
            
            
            
            
            
            
            
            
            
            
