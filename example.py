import torch
import converter

model = torch.jit.load('model_scripted.pt')
model.eval()

converter.gen_pyrtl(model)


