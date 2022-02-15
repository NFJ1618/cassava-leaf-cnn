import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import tensorboard
from tqdm import tqdm


def starting_train(dataset, model, hyperparameters, n_eval, device):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """

    # Get keyword arguments
    batch_size, epochs, test_ratio = hyperparameters["batch_size"], hyperparameters["epochs"], hyperparameters["test_ratio"]

    train_loader, val_loader = dataset.split_test_train_data(dataset, batch_size, test_ratio)
    # Initialize dataloaders
    print("Data loaded")

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    tb_summary = tensorboard.SummaryWriter()
    loss_arr = []
    accuracy_arr = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        step = 0
        total_loss = 0
        # Loop over each batch in the dataset
        for batch in tqdm(train_loader):
            model.train()
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1
            # Periodically evaluate our model + log to Tensorboard
            
            if step >= n_eval and step % n_eval == 0:
                # TODO:
                # Compute training loss and accuracy.
                # Log thxe results to Tensorboard.
                tb_summary.add_scalar('Loss (Training)', loss.item(), epoch)
                print('Training Loss: ', total_loss/(step*32))
                print('Total Loss: ', total_loss)
                # tb_summary.add_scalar('Accuracy (Training)', train_accuracy, epoch)

                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.

        avg_loss = total_loss/(step*32)
        loss_arr.append(avg_loss)
        print('Training Loss: ', avg_loss)
        print('Total Loss: ', total_loss)
        accuracy = evaluate(val_loader, model, loss_fn, device)
        accuracy_arr.append(accuracy)
    
    return loss_arr, accuracy_arr


# def compute_accuracy(outputs, labels):
#    """
#    Computes the accuracy of a model's predictions.
#
#    Example input:
#        outputs: [0.7, 0.9, 0.3, 0.2]
#        labels:  [1, 1, 0, 1]
#
#    Example output:
#        0.75
#    """
#
#    n_correct = (torch.round(outputs) == labels).sum().item()
#    n_total = len(outputs)
#    return n_correct / n_total


def evaluate(val_loader, model, loss_fn, device):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    TODO!
    """
    model.eval()
    correct, val_loss = 0, 0
    total = len(val_loader.dataset)

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            images = torch.reshape(images, (-1, 3, 224, 224))
            output = model(images)
            predictions = torch.argmax(output, dim = 1)
            correct += (labels == predictions).int().sum().item()
            val_loss += loss_fn(output, labels).item()

    print('Validation Accuracy:', (correct/total))
    print('Validation Loss: ', val_loss/len(val_loader))

    return correct/total
