#This file implements local FL system where all clients run on the same computer using OpenFL library,
#and their templates

#Paillier and CKKS encryption times were evaluated in this file, as well as gradient noising and compression
#effects on model accuracy


import copy
import time

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import numpy as np
import tenseal as ts
from phe import paillier

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 3
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
np.random.seed(random_seed)

mnist_train = torchvision.datasets.MNIST(
    "tmp/files/",
    train=True,
    download=False,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)

mnist_test = torchvision.datasets.MNIST(
    "tmp/files/",
    train=False,
    download=False,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)
cifar100_train = torchvision.datasets.CIFAR10(
    root="tmp/files/",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)     # CIFAR-100 training set std (per channel)
        ),
    ]),
)

cifar100_test = torchvision.datasets.CIFAR10(
    root="tmp/files/",
    train=False,
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=(0.5071, 0.4865, 0.4409),
            std=(0.2673, 0.2564, 0.2762)
        ),
    ]),
)

class Net(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(Net, self).__init__()
        act = nn.ReLU
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def inference(network,test_loader):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
      for data, target in test_loader:
        output = network(data)
        test_loss += F.cross_entropy(output, target, reduction='sum')
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))
    accuracy = float(correct / len(test_loader.dataset))
    return accuracy

_context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=4096,
            coeff_mod_bit_sizes=[30,30,30,19]
        )
_context.global_scale = 2 ** 40
_context.generate_galois_keys()
_context = _context.serialize(save_secret_key=True)



def get_context():
    global _context
    return ts.context_from(_context)

from copy import deepcopy

from openfl.experimental.workflow.interface import FLSpec, Aggregator, Collaborator
from openfl.experimental.workflow.runtime import LocalRuntime
from openfl.experimental.workflow.placement import aggregator, collaborator

#unencrypted fedavg
def FedAvg(models, weights=None):
    new_model = models[0]
    state_dicts = [model.state_dict() for model in models]
    state_dict = new_model.state_dict()
    for key in models[1].state_dict():
        state_dict[key] = torch.from_numpy(np.average([state[key].numpy() for state in state_dicts],
                                                      axis=0,
                                                      weights=weights))
    new_model.load_state_dict(state_dict)
    return new_model

#paillier encryted fed avg
def EncryptedFedAvg(encrypted_models, weights=None):
    aggregated_encrypted_model = {}
    for key in encrypted_models[0]:
        agg = encrypted_models[0][key]
        for model_enc in encrypted_models[1:]:
            agg = np.add(agg, model_enc[key])
        n = len(encrypted_models)
        agg = np.vectorize(lambda x: x * (1/n))(agg)
        aggregated_encrypted_model[key] = agg
    return aggregated_encrypted_model

def CKKSFedAvg(encrypted_states, weights=None):
    num_models = len(encrypted_states)
    if num_models == 0:
        raise ValueError("At least one model state must be provided for aggregation.")

    aggregated_state = {}
    for key in encrypted_states[0].keys():
        shape = encrypted_states[0][key]["shape"]
        agg_vector = ts.ckks_vector_from(get_context(), encrypted_states[0][key]["enc_tensor"])

        # Homomorphically add the corresponding vectors from the remaining models.
        for state in encrypted_states[1:]:
            vector = ts.ckks_vector_from(get_context(), state[key]["enc_tensor"])
            agg_vector += vector

        agg_vector = agg_vector * (1.0 / num_models)

        aggregated_state[key] = {
            "enc_tensor": agg_vector.serialize(),
            "shape": shape
        }
    return aggregated_state


def encrypt_model(model, public_key, scale=1e5):
    encrypted_model = {}
    state_dict = model.state_dict()
    print(len(state_dict.items()))
    i = 1
    for key, tensor in state_dict.items():
        print(i)
        arr = (tensor.cpu().numpy() * scale).astype(np.int64)
        vectorized_encrypt = np.vectorize(lambda x: public_key.encrypt(int(x)))
        encrypted_model[key] = vectorized_encrypt(arr)
        i += 1
    return encrypted_model

def encrypt_model_ckks(model):
    state_dict = model.state_dict()
    encrypted_state = {}
    for key, param in state_dict.items():
        param_np = param.cpu().numpy().astype(np.float64)
        shape = param_np.shape
        param_np = param_np.flatten()
        enc_tensor = ts.ckks_vector(get_context(), param_np).serialize()
        encrypted_state[key] = {
            "enc_tensor": enc_tensor,
            "shape": shape
        }
    return encrypted_state

def decrypt_model(encrypted_model, private_key, scale=1e5):
    decrypted_model = {}
    for key, encrypted_arr in encrypted_model.items():

        vectorized_decrypt = np.vectorize(lambda x: private_key.decrypt(x))
        arr_int = vectorized_decrypt(encrypted_arr)

        arr_float = arr_int.astype(np.float32) / scale

        decrypted_model[key] = torch.tensor(arr_float)
    return decrypted_model

def decrypt_model_ckks(encrypted_state):
    decrypted_state = {}
    for key, value in encrypted_state.items():
        serialized_enc_tensor = value["enc_tensor"]
        original_shape = value["shape"]
        enc_vector = ts.ckks_vector_from(get_context(), serialized_enc_tensor)
        flat_dec_np = enc_vector.decrypt()
        dec_np = np.array(flat_dec_np).reshape(original_shape)
        decrypted_state[key] = torch.tensor(dec_np, dtype=torch.float32)
    return decrypted_state


class FederatedFlow(FLSpec):

    def __init__(self, model=None, optimizer=None, rounds=3, **kwargs):
        super().__init__(**kwargs)
        self.public_key = None
        self.private_key = None
        if model is not None:
            self.model = model
            self.encrypted_model = None
            self.aggregated_encrypted_model = None
            self.optimizer = optimizer
        else:
            self.model = Net()
            total_params = sum(p.numel() for p in self.model.parameters())
            print("Total parameters:", total_params)
            self.encrypted_model = None
            self.aggregated_encrypted_model = None
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate,
                                       momentum=momentum)
        self.rounds = rounds

    @aggregator
    def start(self):
        print(f'Performing initialization for model')
        self.collaborators = self.runtime.collaborators
        self.private = 10
        self.current_round = 0
        self.next(self.aggregated_model_validation, foreach='collaborators', exclude=['private'])

    @collaborator
    def aggregated_model_validation(self):
        if self.aggregated_encrypted_model is not None:
            self.model.load_state_dict(decrypt_model_ckks(self.aggregated_encrypted_model))
            #self.model.load_state_dict(decrypt_model(self.aggregated_encrypted_model, self.private_key))
        print(f'Performing aggregated model validation for collaborator {self.input}')
        self.agg_validation_score = inference(self.model, self.test_loader)
        print(f'{self.input} value of {self.agg_validation_score}')
        self.next(self.train)

    @collaborator
    def train(self):
        self.model.train()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate,
                                   momentum=momentum)
        initial_state = copy.deepcopy(self.model.state_dict())
        train_losses = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            #gradient compression
            #threshold = 0.01
            #for param in self.model.parameters():
            #    if param.grad is not None:
            #        param.grad[param.grad.abs() < threshold] = 0
            #end
            self.optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: 1 [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))
                self.loss = loss.item()
                torch.save(self.model.state_dict(), 'model.pth')
                torch.save(self.optimizer.state_dict(), 'optimizer.pth')
        final_state = self.model.state_dict()
        aggregated_update = {}
        for key in final_state:
            aggregated_update[key] = final_state[key] - initial_state[key]
        # noise_var = 0.16  # Define noise level
        # for key in final_state:
        #    noise = torch.randn_like(final_state[key]) * noise_var
        #    final_state[key] += noise

        # Prune the smallest x% of updates (by magnitude) instead of using a fixed threshold
        prune_percent = 0.85  # Prune 85% of the smallest updates
        all_updates = torch.cat([((param.data - initial_state[name]).abs().flatten())
                                 for name, param in self.model.named_parameters() if param.requires_grad])
        k = int(prune_percent * all_updates.numel())
        threshold = torch.topk(all_updates, k, largest=False).values.max()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                update = param.data - initial_state[name]
                mask = update.abs() < threshold
                param.data[mask] = initial_state[name][mask]
        ratio = (all_updates < threshold).sum().item() / all_updates.numel()
        print("Ratio of compression params: ", ratio)
        self.training_completed = True
        self.next(self.local_model_validation)

    @collaborator
    def local_model_validation(self):
        self.local_validation_score = inference(self.model, self.test_loader)
        print(
            f'Doing local model validation for collaborator {self.input}: {self.local_validation_score}')
        start_time = time.time()
        #self.encrypted_model = encrypt_model(self.model, self.public_key)
        try:
            self.encrypted_model = encrypt_model_ckks(self.model)
        except:
            print("WHOPS")
        print("Time for encrypting:" , time.time() - start_time)
        self.next(self.join, exclude=['training_completed', 'model'])

    @aggregator
    def join(self, inputs):
        self.average_loss = sum(input.loss for input in inputs) / len(inputs)
        self.aggregated_model_accuracy = sum(
            input.agg_validation_score for input in inputs) / len(inputs)
        self.local_model_accuracy = sum(
            input.local_validation_score for input in inputs) / len(inputs)
        print(f'Average aggregated model validation values = {self.aggregated_model_accuracy}')
        print(f'Average training loss = {self.average_loss}')
        print(f'Average local model validation values = {self.local_model_accuracy}')
        self.aggregated_encrypted_model  = CKKSFedAvg([input.encrypted_model for input in inputs])
        self.optimizer = [input.optimizer for input in inputs][0]
        self.current_round += 1
        if self.current_round < self.rounds:
            self.next(self.aggregated_model_validation,
                      foreach='collaborators', exclude=['private'])
        else:
            self.next(self.end)

    @aggregator
    def end(self):
        print(f'This is the end of the flow')




aggregator = Aggregator()
aggregator.private_attributes = {}

# Generate public private key for all colaborators for AHE
public_key, private_key = paillier.generate_paillier_keypair(n_length=512)


# Setup collaborators
collaborator_names = ['Portland', 'Seattle', 'Chandler','Bangalore']
collaborators = [Collaborator(name=name) for name in collaborator_names]
print("CIFAR LENGHT:", len(cifar100_train.data))
for idx, collaborator in enumerate(collaborators):
    local_train = deepcopy(cifar100_train)
    local_test = deepcopy(cifar100_test)
    local_train.data = cifar100_train.data[idx::len(collaborators)]
    local_train.targets = cifar100_train.targets[idx::len(collaborators)]
    local_test.data = cifar100_test.data[idx::len(collaborators)]
    local_test.targets = cifar100_test.targets[idx::len(collaborators)]
    collaborator.private_attributes = {
            'train_loader': torch.utils.data.DataLoader(local_train,batch_size=batch_size_train, shuffle=True),
            'test_loader': torch.utils.data.DataLoader(local_test,batch_size=batch_size_train, shuffle=True),
            'public_key': public_key,
            'private_key': private_key

    }

local_runtime = LocalRuntime(aggregator=aggregator, collaborators=collaborators, backend='single_process')
print(f'Local runtime collaborators = {local_runtime.collaborators}')


model = None
best_model = None
optimizer = None
flflow = FederatedFlow(model, optimizer, rounds=3, checkpoint=True, )
flflow.runtime = local_runtime
flflow.run()
