# PyTorch_ActivationVisualization

Module to visualize activations and weights from convolutional layers in PyTorch models

1. Load the trained PyTorch CNN model
```python3
  model = load_pytorch_model()
```

2. Define the class activation map object using
```python3
  layer_names = ['conv1', conv2']
  layers_to_hook = [model.conv1, model.conv2]
  cam = ClassActivationMaps(model, layer_names, layers_to_hook)
```

3. After inferencing through test samples, call get_activations() function which returns a dict containing layer names as keys and the activations as values as type <torch.Tensor>
```python3
  activation = cam.get_activations()
```

4. Use other utility functions provided to either save the activations or display them using matplotlib
