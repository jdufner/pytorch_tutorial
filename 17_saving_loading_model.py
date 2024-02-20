import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


FILE = "model.pth"

# 1) Save model
model = Model(n_input_features=6)
# train your model
learning_rate = .01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

checkpoint = {
    "epoch": 90,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict()
}
torch.save(checkpoint, "checkpoint.pth")

# for param in model.parameters():
#     print(param)
# torch.save(model.state_dict(), FILE)

# 2) Load model
# loaded_model = Model(n_input_features=6)
# loaded_model.load_state_dict(torch.load(FILE))
# model.eval()
# for param in loaded_model.parameters():
#     print(param)

loaded_checkpoint = torch.load("checkpoint.pth")
epoch = loaded_checkpoint["epoch"]
model = Model(n_input_features=6)
model.load_state_dict(loaded_checkpoint["model_state"])
optimizer = torch.optim.SGD(model.parameters(), lr=0)
optimizer.load_state_dict(loaded_checkpoint["optimizer_state"])
