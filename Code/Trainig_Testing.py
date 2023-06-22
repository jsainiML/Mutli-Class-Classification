def accuracy(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct/len(y_pred))*100
  return acc

torch.manual_seed(62)
epochs = 500

for epoch in range(epochs):
  # Training the model
  model0.train()

  yb_logit = model0(Xb_train)
  yb_preds = torch.softmax(yb_logit, dim=1).argmax(dim=1)

  loss_train = lossfn(yb_logit, yb_train)
  acc = accuracy(y_true = yb_train, y_pred = yb_preds)

  optimizer.zero_grad()
  loss_train.backward()
  optimizer.step()


  model0.eval()

  with torch.inference_mode():
    result_logit = model0(Xb_test)
    result_preds = torch.softmax(result_logit, dim=1).argmax(dim=1)
    res_acc = accuracy(y_true = yb_test, y_pred = result_preds)

    loss_test = lossfn(result_logit, yb_test)

    if epoch % 30 == 0:
      print (f"Epoch:{epoch} | Train loss: {loss_train} | Acc: {acc} | Test loss {loss_test} | Test Acc: {res_acc}")
