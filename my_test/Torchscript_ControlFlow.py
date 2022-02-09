import torch


class ControlFlowTestModule():
    def forward(self, x, y, a, b):
        if a == b:
            x = x - y
        else:
            x = x + y
        return torch.add(x, y)
    # def forward(self, x, y):
    #     for i in range(1, 5):
    #         # for i in range(1, 5):
    #         # while(True):
    #         x = x - y
    #     return torch.add(x, y)
    # multi-output
    # def forward(self, x, y):
    #     while(x[0] > y[0]):
    #         x = x - y
    #     return torch.add(x, y)

    # # multi-output
    # def forward(self, x, y):
    #     z = x + y
    #     if x == y:
    #         x = x - y
    #         z = x + y
    #     else:
    #         x = x + y
    #         z = x - y
    #     return torch.add(x, z)


if __name__ == '__main__':
    test_module = ControlFlowTestModule()
    x = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    y = torch.tensor([[0.5, 0.6], [0.7, 0.8]])
    test_module.forward(x, y, 3, 2)
