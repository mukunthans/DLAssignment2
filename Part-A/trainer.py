
import torch
import torch.nn as nn
import wandb

class Trainer:
    def __init__(self, model_class, h_params, training_data):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model_class(h_params)
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        self.h_params = h_params
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=h_params["learning_rate"])
        self.train_loader = training_data['train_loader']
        self.val_loader = training_data['val_loader']
        self.test_loader = training_data['test_loader']
        self.train_len = training_data['train_len']
        self.val_len = training_data['val_len']
        self.test_len = training_data['test_len']

    def fit(self):
        for epoch in range(self.h_params["epochs"]):
            train_loss, train_acc = self._train_one_epoch(epoch)
            val_loss, val_acc = self._validate(epoch)
            print(f"epoch: {epoch} train accuracy: {train_acc:.4f} train loss: {train_loss:.4f} "
                  f"val accuracy: {val_acc:.4f} val loss: {val_loss:.4f}")
            wandb.log({
                "train_accuracy": train_acc,
                "train_loss": train_loss,
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "epoch": epoch
            })
        test_loss, test_acc = self._test()
        wandb.log({
            "test_accuracy": test_acc,
            "test_loss": test_loss
        })
        print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')
        print('Finished Training')
        torch.save(self.model.state_dict(), './bestmodel.pth')

    def _train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        for i, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            if i % 10 == 0:
                batch_acc = (predicted == labels).float().mean().item()
                print(f"epoch {epoch} batch {i} accuracy {batch_acc:.4f} loss {loss.item():.4f}")
        avg_loss = running_loss / len(self.train_loader)
        accuracy = correct / self.train_len
        return avg_loss, accuracy

    def _validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
        avg_loss = running_loss / len(self.val_loader)
        accuracy = correct / self.val_len
        return avg_loss, accuracy

    def _test(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
        avg_loss = running_loss / len(self.test_loader)
        accuracy = correct / self.test_len
        return avg_loss, accuracy
