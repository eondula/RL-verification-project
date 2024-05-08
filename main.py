import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from z3 import *
import pandas as pd


def verify(model):

    s = Solver()

    # input conditions
    community_risk, curr_infected = Reals('community_risk curr_infected')
    cr_min = 0.7
    cr_max = 1.0
    ci_min = 40.
    ci_max = 100.
    s.add(And([community_risk >= cr_min, community_risk <= cr_max, curr_infected >= ci_min, curr_infected <= ci_max]))

    # model conditions
    layers = [param.detach().numpy() for param in model.parameters()]
    def Relu(x):
        return np.vectorize(lambda y: If(y >= 0, y, RealVal(0)))(x)
    def Net(x):
        x = np.array([x[0], x[1] / 100.]) * 2. - 1.
        for i in range(0,len(layers),2):
            w_i, b_i = layers[i:i+2]
            x = Relu(w_i @ x + b_i)
        return x
    x = np.array([community_risk, curr_infected])
    y = Net(x)

    # output conditions
    val_0, val_50, val_100 = y
    # s.add(Or(val_50 > val_100, val_0 > val_100))
    # s.add(val_0 < max(val_50, val_100))
    # For low infection danger
    # max_val_50_val_100 = If(val_50 > val_100, val_50, val_100)
    # s.add(val_0 < max_val_50_val_100)

    # for high infection danger
    # Calculate the maximum of val_0 and val_50 using Z3's If statement
    max_val_0_val_50 = If(val_0 > val_50, val_0, val_50)

    # Ensure val_100 is not the highest by asserting it's less than the maximum of the other two
    s.add(val_100 < max_val_0_val_50)

    set_option(verbose=10)
    res = s.check()
    print(res)
    if res == sat:
        m = s.model()
        input = torch.tensor([float(m[d].numerator_as_long()) / float(m[d].denominator_as_long()) for d in m]).reshape(1,2)
        print(input)
        print(model(input))

def get_infected_students_apprx_sir(num_infected, allowed_per_course, community_risk, const_1, const_2, const_r):

    susceptible = 100 - num_infected
    prop_allowed = allowed_per_course / 100
    prop_infected = num_infected / 100

    # Calculate the number of newly infected students
    outside_risk = const_2 * community_risk
    susceptible_out_class = susceptible * (1. - prop_allowed)
    infected_out_class = outside_risk * susceptible_out_class

    susceptible_in_class = susceptible * prop_allowed
    class_risk = const_1 * (prop_infected * prop_allowed) ** .5
    total_risk = torch.minimum(outside_risk + class_risk, torch.ones_like(class_risk))
    infected_in_class = total_risk * susceptible_in_class

    new_infected = infected_in_class + infected_out_class

    # Calculate the number of recovered students
    recovered = const_r * num_infected

    # Update the current infected count by adding new infections and subtracting recoveries
    infected = num_infected - recovered + new_infected

    return infected

def get_reward(allowed, d_infected, alpha: float):
    
    p_allowed = allowed / 100
    p_d_infected = d_infected / 100
    p_d_infected, s_d_infected = torch.abs(p_d_infected), torch.sign(p_d_infected)

    r_allowed = alpha * (p_allowed ** .5)
    r_infected = (1. - alpha) * (p_d_infected ** .5) * s_d_infected

    reward = r_allowed + r_infected
    return (reward + 1.) / 2. * 100.

allowed = [0,50,100]
def get_label(num_infected, community_risk, const_1, const_2, const_3, alpha):
    
    label = []
    for a in allowed:
        new_infected = get_infected_students_apprx_sir(num_infected, a, community_risk, const_1, const_2, const_3)
        d_infected = num_infected - new_infected
        reward = get_reward(a, d_infected, alpha)
        label.append(reward)
    label = torch.stack(label,dim=-1)
    label = torch.eye(3)[label.argmax(dim=-1).long()] + 1.
    return label

class Model(nn.Module):
    def __init__(self,input_sz=2,hsz=64,output_sz=3):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_sz,hsz),
            nn.ReLU(inplace=True),
            # nn.Linear(hsz, hsz),
            # nn.ReLU(inplace=True),
            nn.Linear(hsz,output_sz),
            nn.Softmax(dim=-1)
        )

    def forward(self,x):
        input = torch.stack((x[:,0] * 2. - 1, x[:,1] / 100 * 2. - 1),dim=-1)
        output = self.actor(input)
        return output

# community_risk, num_infected
def sample(batch_size):
    return torch.stack([torch.randint(100,size=(batch_size,)) / 100., torch.randint(100,size=(batch_size,))],dim=-1).float()

def select_action(output,label,exp_rate):
    probs = torch.rand(len(output))

    # action argmax
    action_argmax = output.argmax(dim=-1)

    # action random
    # action_random = torch.randint(output.shape[-1],size=(len(output),))
    cdf = torch.cumsum(output,dim=-1)
    val = torch.rand(len(output)).unsqueeze(-1)
    action_random = (val < cdf).int().argmax(dim=-1)
    
    action = torch.where(probs <= exp_rate, action_random, action_argmax).long()
    sample = torch.arange(len(output)).long()

    return output[sample,action], label[sample,action]

def loss_fn(output,label):
    prob_penalty = -torch.log(torch.maximum(output,torch.ones_like(output) * 1e-6))
    reward = label
    loss = prob_penalty * reward
    return loss.mean()

alpha = .8
const_1 = 2.
const_2 = .8
const_3 = .7 # recovery

def train():
    model = Model()
    opt = torch.optim.Adam(model.parameters(),lr=1e-4)

    N = 100000
    batch_size = 1000
    pbar = tqdm(total=N)
    
    exp_rate = 1.
    exp_gamma = .99997

    for i in range(N):

        batch = sample(batch_size)

        output = model(batch)
        label = get_label(batch[:,1], batch[:,0], const_1, const_2, const_3, alpha) 

        output,label = select_action(output,label,exp_rate)
        exp_rate *= exp_gamma

        loss = loss_fn(output,label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        pbar.update(1)
        pbar.set_description(f"Loss {loss.detach().numpy()} Exp {exp_rate}")

    DIM = 500
    y,x = torch.tensor((np.mgrid[0:DIM,0:DIM].reshape(2,-1) + .5)).float() / DIM 
    current_infected = (1 - y) * 100
    community_risk = x
    input = torch.stack([community_risk,current_infected],dim=-1)

    output = model(input).round()
    label = get_label(input[:,1], input[:,0], const_1, const_2, const_3, alpha) 

    def normalize(img):
        img = torch.eye(3)[img.argmax(axis=-1).long()].cpu()
        return img

    model_name = f"model-{alpha}"
    img0 = torch.clamp(normalize(output), 0., 1.).detach().numpy().reshape(DIM,DIM,3)
    plt.imsave(f"{model_name}.png", img0)

    img1 = normalize(label).reshape(DIM,DIM,3).numpy()
    plt.imsave(f"label_{model_name}.png", img1)


    torch.save(model.state_dict(),model_name)

def visualize_model_behavior(model, model_name):
    # Create a meshgrid for the input space
    community_risks = np.linspace(0, 0.1, 20)
    curr_infecteds = np.linspace(0, 10, 20)
    CR, CI = np.meshgrid(community_risks, curr_infecteds)
    inputs = torch.tensor(np.stack([CR.ravel(), CI.ravel()], axis=1), dtype=torch.float32)
    print("Inputs prepared, computing model outputs...")
    outputs = model(inputs).detach().numpy()
    print("Model outputs computed.")
    # Calculate mean probabilities for each policy
    # Calculate mean probabilities for each policy
    mean_probabilities = outputs.mean(axis=0)
    print("Mean probabilities calculated.")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    policies = ['Policy 0', 'Policy 50', 'Policy 100']
    for i, ax in enumerate(axes):
        policy_map = outputs[:, i].reshape(CR.shape)
        # Check if policy_map has non-zero values
        if np.any(policy_map > 0):
            norm = colors.LogNorm(vmin=policy_map[policy_map > 0].min(), vmax=policy_map.max())
        else:
            norm = colors.Normalize(vmin=0, vmax=1)  # Default normalization when all values are zero or close to zero
        cp = ax.contourf(CR, CI, policy_map, levels=50, cmap='viridis', norm=norm)
        fig.colorbar(cp, ax=ax)
        ax.set_title(f'{policies[i]} Probability')
        ax.set_xlabel('Community Risk')
        ax.set_ylabel('Current Infected')

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{model_name}_behavior.png")
    # # Create and print a DataFrame for mean probabilities
    df = pd.DataFrame(mean_probabilities.reshape(1, -1), columns=policies)
    print("Mean Probabilities for Each Policy:")
    print(df)


if __name__ == "__main__":
    # train()
    model = Model()
    model_name = f"model-{alpha}"
    model.load_state_dict(torch.load(model_name))

    verify(model)
    # # Assuming model is already loaded and trained
    visualize_model_behavior(model, model_name)
