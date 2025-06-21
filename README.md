# Introducting Potato2Chip: A Tool for Automated convertion from PyTorch models to PyRTL 
How to use potato2chip:
  1. Save your PyTorch model using TorchScript to your working directory (recommended method):
     ```python
     model_scripted = torch.jit.script(model) # Export to TorchScript
     model_scripted.save('model_scripted.pt') # Save
     ```
  2. Create a script that loads your saved model, and inside it, import `converter.py`:
     ```python
     import converter
     ...
     model = torch.jit.load('model_scripted.pt')
     ```
  3. Now add the line 
     ```python
     converter.gen_pyrtl(model)
     ```
  4. Execute your script
  5. You will see the outputed pyrtl code in 'output.py'
