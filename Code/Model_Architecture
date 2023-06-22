#Creating base model architecture
class classifier(nn.Module):
  def __init__(self):
    super().__init__()

    self.task1 = nn.Linear(in_features=2, out_features=8)
    self.task2 = nn.Linear(in_features=8, out_features=8)
    self.task3 = nn.Linear(in_features=8, out_features=6)

# Forwad pass function
  def forward(self, x):
    return self.task3(self.task2(self.task1(x)))


# Loading model to a instance to later train and test
model0 = classifier()
model0.state_dict()
