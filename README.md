# Converting pytorch models to pyrtl 
How to use:
  1. Save your model using TorchScript to your working directory:
     ```python
     model_scripted = torch.jit.script(model) # Export to TorchScript
     model_scripted.save('model_scripted.pt') # Save
     ```
  2. Import converter into your file and load your saved model:
     ```python
     import converter
     ...
     model = torch.jit.load('model_scripted.pt')
     ```
  3. Include ```python converter.gen_pyrtl(model)``` and then run the program
  4. You will see the outputed pyrtl code in 'output.py'
