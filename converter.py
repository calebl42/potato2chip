import torch

def gen_pyrtl(model):
    f = open('output.py', 'w')
    f.write('import pyrtl\n')
    f.write('import pyrtl.rtllib.matrix as mx\n\n')

    weights = [p for p in model.parameters()]
    w_index = 0

    f.write(f'x = pyrtl.Input({len(weights[0].T) * 4}, \'x\')\n')
    f.write(f'input_layer = mx.Matrix(1, {len(weights[0].T)}, 4, value=x)\n\n')

    edge_mx = 'input_layer'
    operations = []
    for name, module in model.named_children():
        operations.append(module.original_name)
    for op in operations:
        if (op == 'Linear'):
            f.write('#Linear layer\n')
            cur_w = weights[w_index].T
            scale = (float(torch.max(cur_w)) - float(torch.min(cur_w)))/14

            cur_w = torch.quantize_per_tensor(cur_w, scale, 0, torch.qint8)
            cur_w = torch.quantize_per_tensor_dynamic(cur_w, scale, 0, torch.qint8)
            cur_w = cur_w.int_repr().tolist()
            w_string = str(cur_w).replace(' ', '')
            w_name = 'w_' + str(w_index)
            w_out = 'layer_' + str(w_index)

            f.write(f'{w_name} = mx.Matrix({len(cur_w)}, {len(cur_w[0])}, 4, value={w_string})\n')
            f.write(f'{w_out} = {edge_mx} @ {w_name}\n\n')
            edge_mx = w_out
            w_index += 1
        elif (op == 'ReLU'):
            f.write('#ReLU layer\n')
            f.write(f'layer_relu{w_index} = mx.Matrix({edge_mx}.rows, {edge_mx}.columns, 4)\n\n')
            f.write(f'for r in range({edge_mx}.rows):\n' +
                    f'\tfor c in range({edge_mx}.columns):\n' +
                    f'\t\tlayer_relu{w_index}[r, c] = pyrtl.select({edge_mx}[r, c] < 0, 0, {edge_mx}[r, c])\n\n')
            edge_mx = 'layer_relu' + str(w_index)
        elif op == 'QuantStub' or op == 'DeQuantStub':
            pass
        else:
            print(f"unknown layer operation {op}")
            return
        
    f.write('y = pyrtl.Output(4, \'y\')\n')
    f.write(f'y <<= mx.argmax({edge_mx})')
    f.close()
