import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval, device):
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
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )
    print("Data loaded")

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    #tb_summary = torch.utils.tensorboard.SummaryWriter(summary_path)

    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for batch in tqdm(train_loader):
            # TODO: Backpropagation and gradient descent
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Periodically evaluate our model + log to Tensorboard
            
            if step > n_eval and step % n_eval == 0:
                # TODO:
                # Compute training loss and accuracy.
                # Log the results to Tensorboard.
                # tb_summary.add_scalar('Loss (Training)', loss, epoch)
                # tb_summary.add_scalar('Accuracy (Training)', train_accuracy, epoch)

                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.
                # Don't forget to turn off gradient calculations!
                evaluate(val_loader, model, loss_fn, device)

            step += 1

        print('Epoch:', epoch, 'Loss:', loss.item())


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """

    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn, device):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    TODO!
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            images = torch.reshape(images, (-1, 3, 224, 224))
            output = model(images)
            predictions = torch.argmax(output, dim = 1)

            correct += (labels == predictions).int().sum()
            total += len(predictions)
    print('Accuracy:', (correct/total).item())
