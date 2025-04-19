import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Subset
import wandb

#  CONFIGURATION: Set your config here 
BEST_CONFIG = {
    "epochs": 15,
    "learning_rate": 0.0005,
    "batch_size": 32,
    "last_unfreeze_layers": 3,
    "image_size": 224,
    "num_classes": 10,
    "num_workers": 2,
    "seed": 42,
    "dropout": 0.4
}

TRAIN_DATA_DIR = "/kaggle/input/assignment2dataset/inaturalist_12K/train"
VAL_DATA_DIR = "/kaggle/input/assignment2dataset/inaturalist_12K/val"
WANDB_KEY = ""  
WANDB_PROJECT = "DL Assignment 2B"

# - Data Preparation Class -
class DataPreparer:
    def __init__(self, h_params, image_size, train_dir, val_dir):
        self.h_params = h_params
        self.image_size = image_size
        self.train_dir = train_dir
        self.val_dir = val_dir

    def get_train_transform(self):
        size = (self.image_size, self.image_size)
        return transforms.Compose([
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_test_transform(self):
        size = (self.image_size, self.image_size)
        return transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def stratified_split(self, dataset, ratio):
        train_idx, val_idx = [], []
        class_bounds = [
            (0, 999), (1000, 1999), (2000, 2999), (3000, 3999), (4000, 4998),
            (4999, 5998), (5999, 6998), (6999, 7998), (7999, 8998), (8999, 9998)
        ]
        for start, end in class_bounds:
            indices = list(range(start, end + 1))
            split_at = int(len(indices) * ratio)
            train_idx.extend(indices[:split_at])
            val_idx.extend(indices[split_at:])
        return Subset(dataset, train_idx), Subset(dataset, val_idx)

    def get_datasets(self):
        train_transform = self.get_train_transform()
        test_transform = self.get_test_transform()
        full_train = datasets.ImageFolder(self.train_dir, transform=train_transform)
        train_set, val_set = self.stratified_split(full_train, 0.8)
        test_set = datasets.ImageFolder(self.val_dir, transform=test_transform)
        return train_set, val_set, test_set

    def get_loaders(self):
        train_set, val_set, test_set = self.get_datasets()
        batch = self.h_params["batch_size"]
        return {
            "train_loader": DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=self.h_params["num_workers"]),
            "val_loader": DataLoader(val_set, batch_size=batch, shuffle=False, num_workers=self.h_params["num_workers"]),
            "test_loader": DataLoader(test_set, batch_size=batch, shuffle=False, num_workers=self.h_params["num_workers"]),
            "train_len": len(train_set),
            "val_len": len(val_set),
            "test_len": len(test_set)
        }

#  EfficientNetV2-S Model with Custom Dropout 
def efficientnetv2_model(h_params):
    model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
    num_ftrs = model.classifier[1].in_features
    # Replace the classifier with custom dropout rate
    model.classifier = nn.Sequential(
        nn.Dropout(p=h_params.get("dropout", 0.2), inplace=True),
        nn.Linear(num_ftrs, h_params["num_classes"])
    )
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze the last k layers of the classifier explicitly
    k = h_params["last_unfreeze_layers"]
    if k > 0:
        classifier_layers = list(model.classifier.children())
        for layer in classifier_layers[-k:]:
            for param in layer.parameters():
                param.requires_grad = True
    return model

#  Trainer Class 
class Trainer:
    def __init__(self, model_class, h_params, training_data):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model_class(h_params)
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        self.h_params = h_params
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=h_params["learning_rate"]
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
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
            self.scheduler.step()
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

#  Main Entry Point -
def main():
    torch.manual_seed(BEST_CONFIG["seed"])
    wandb.login(key=WANDB_KEY)
    data_preparer = DataPreparer(BEST_CONFIG, BEST_CONFIG["image_size"], TRAIN_DATA_DIR, VAL_DATA_DIR)
    training_data = data_preparer.get_loaders()
    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"efficientnetv2_ep_{BEST_CONFIG['epochs']}_bs_{BEST_CONFIG['batch_size']}_lr_{BEST_CONFIG['learning_rate']}_last_unfreeze_layers_{BEST_CONFIG['last_unfreeze_layers']}_dropout_{BEST_CONFIG['dropout']}",
        config=BEST_CONFIG
    )
    trainer = Trainer(efficientnetv2_model, BEST_CONFIG, training_data)
    trainer.fit()

if __name__ == "__main__":
    main()
