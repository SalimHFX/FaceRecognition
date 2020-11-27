import torch


def train_net(train_loader, valid_loader, net, device, optimizer, criterion, model_path):

    train_losses = []
    val_losses = []

    best = 1000
    unsaved = 0
    for epoch in range(25):  # loop over the dataset multiple times

        running_loss = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # CPU version
            # inputs, labels = data
            # GPU version
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            epoch_loss += loss.item()
            running_loss += loss.item()
            if i % 200 == 199:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.5f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        # calculate epoch validation loss
        val_loss = 0.0
        with torch.no_grad():
            for data in valid_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss/len(valid_loader))
        train_losses.append(epoch_loss/len(train_loader))

        # save the model with the best validation loss
        if val_loss/len(valid_loader) < best:
            best = val_loss/len(valid_loader)
            path = model_path
            print("saved with", val_loss/len(valid_loader))
            torch.save(net.state_dict(), path)
            unsaved = 0
        else:
            unsaved += 1

        # if the model hasnt improved in more than 3 epochs, stop the training
        if unsaved > 3:
            break

    print('Finished Training')

    return net, train_losses, val_losses
