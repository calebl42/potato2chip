import torch
from torch.utils.data import DataLoader  # easier dataset management, helps create mini batches
import torchvision.datasets as datasets  # standard datasets
import torchvision.transforms as transforms
import sys
import bitstring

data = torch.load('state_dict')

print(data)

sys.set_int_max_str_digits(0)
torch.set_printoptions(profile="full", sci_mode=False, precision=2)

fc1f = data['fc1._packed_params._packed_params'][0].T.dequantize()
fc2f = data['fc2._packed_params._packed_params'][0].T.dequantize()

fc1w = data['fc1._packed_params._packed_params'][0].T.int_repr()
fc2w = data['fc2._packed_params._packed_params'][0].T.int_repr()

#print('fc1w', fc1w)

#print(','.join('[' + ','.join(str(x) for x in r.tolist()) + ']' for r in fc1w))

def run_normal(x: torch.Tensor) -> torch.Tensor:

    x = x @ fc1f
    x = torch.where(x < 0, 0, x)
    x = x @ fc2f
    return x

def run_mod(x: torch.Tensor):
    #assert x.dtype == torch.uint8, x.dtype

    x = torch.quantize_per_tensor(x, data['quant.scale'], data['quant.zero_point'], torch.qint8).int_repr()

    arr = bitstring.BitArray()
    for i in x.tolist()[0]:
        arr.append(bitstring.Bits(uint=i, length=3).tobitarray())

    #print(x.dtype, len(b''.join(i.to_bytes(3,signed=True) for i in )))
    #print(arr.uint)
    x = ((x-3) @ fc1w) >> 6
    #x = torch.nn.functional.linear(x, fc1w)
    x = torch.where(x < 0, 0, x)
    #x = torch.nn.functional.linear(x, fc2w)
    x = x @ fc2w

    #x = x.dequantize()

    #x = x / data['fc1.scale']
    #x = x / data['quant.scale']

    return x


batch_size = 64

train_dataset = datasets.MNIST(root='../dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# print(w[0])
# print(w.int_repr())

def check_accuracy(loader):
    num_correct = 0
    num_samples = 0

    for x, y in loader:
        x = x.reshape(x.shape[0], -1)

        #print(torch.quantize_per_tensor(x, data['quant.scale'], data['quant.zero_point'], torch.qint8).int_repr().tolist())

        # print(x.shape)

        scores = run_mod(x)
        scores2 = run_normal(x)
        _, predictions = scores.max(1)

        #print(scores, scores2)

        num_correct += (predictions == y).sum()
        num_samples += predictions.size(0)

        break

    print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct) / float(num_samples) * 100: .2f}')

check_accuracy(train_loader)