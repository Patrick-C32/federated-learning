import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):

    def __init__(self, in_channels=3):
        super(Net, self).__init__()
        # 32x32 CIFAR-10 -> 16*5*5, 28x28 MNIST -> 16*4*4
        self.flat_size = 16 * 5 * 5
        if in_channels == 1:
            self.flat_size = 16 * 4 * 4
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(self.flat_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.flat_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset
fds_key = None

dataset_ids = {"cifar10": "uoft-cs/cifar10", "mnist": "ylecun/mnist"}

# Scale every channel from [0, 1] to [-1, 1] 
pytorch_transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

# Apply transforms on partition from FederatedDataset
def apply_transforms(batch):
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    dataset: str = "cifar10",
    alpha: float = 0.0,
):
    # Only initialize FederatedDataset once per dataset, client count and alpha
    global fds, fds_key
    if fds is None or fds_key != (dataset, num_partitions, alpha):
        # T-3 Dirichlet partition over labels (alpha > 0), else IID
        if alpha > 0:
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=alpha,
                seed=42,
            )
        else:
            partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset=dataset_ids[dataset],
            partitioners={"train": partitioner},
        )
        fds_key = (dataset, num_partitions, alpha)
    partition = fds.load_partition(partition_id)
    # Make MNIST compatible with generic caller
    if dataset == "mnist":
        partition = partition.rename_column("image", "img")
    # Divide data 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def load_centralized_dataset(dataset="cifar10"):
    # Load entire test set
    test_dataset = load_dataset(dataset_ids[dataset], split="test")
    if dataset == "mnist":
        test_dataset = test_dataset.rename_column("image", "img")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)


def train(net, trainloader, epochs, lr, device):
    net.to(device)  # move model to GPU if possible
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test(net, testloader, device):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
