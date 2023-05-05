import torch
import torch.nn as nn
# local modules
from custom_neural_ode import ODEF

class HysteresisODEF(ODEF):
    def __init__(self, in_dim, hid_dim, time_invariant=False):
        super(HysteresisODEF, self).__init__()
        self.time_invariant = time_invariant

        if time_invariant:
            self.lin1 = nn.Linear(in_dim, hid_dim)
        else:
            self.lin1 = nn.Linear(in_dim+1, hid_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim)
        self.lin3 = nn.Linear(hid_dim, in_dim)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x, t):
        if not self.time_invariant:
            x = torch.cat((x, t), dim=-1)

        h = self.elu(self.lin1(x))
        h = self.elu(self.lin2(h))
        out = self.lin3(h)
        return out

class simpleRNN(nn.Module):
    def __init__(self, in_dim, hid_dim, timesteps_count=2):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hid_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim)
        self.lin3 = nn.Linear(hid_dim, in_dim)
        self.elu = nn.ELU(inplace=True)
        self.timesteps_count = timesteps_count

    def forward(self, x):
        x_in = x
        for i in range(self.timesteps_count):
            x = self.elu(self.lin1(x_in))
            x = self.elu(self.lin2(x))
            x_in = self.lin3(x) + x_in
        return x_in


class ODEFunc(nn.Module):
    """
        usage : 
        func = ODEFunc()
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0, batch_t, method=config.solver_type, adjoint_method=config.solver_type).to(device)
    """

    def __init__(self, in_dim=2, hid_dim=2):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, in_dim),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, x):
        return self.net(x)
    