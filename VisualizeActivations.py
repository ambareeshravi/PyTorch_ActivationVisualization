'''
Author: Ambareesh Ravi
Description: To visualize conv 2d activations for CNNs in pytorch
Date: Jul 10, 2020
'''
import torch
import torch.nn
import matplotlib.pyplot as plt

from conv_models import Model_v1

class ClassActivationMaps:
    def __init__(self, model):
        '''
        initializes the class variables
        
        Args:
            model - loaded pytorch model for which visualization is to be performed
        Returns:
            -
        Exception:
            -
        '''
        self.model = model
        
        # this stores the activations for each run batch_size x map_channels x map_height x map_width
        self.activations = dict()
        
        # give some names to the layer & change according to your model configuration
        self.layer_names = ['conv1', 'bn1', 'lr1', 'conv2', 'bn2', 'lr2', 'conv3', 'bn3', 'lr3', 'conv4', 'bn4', 'lr4', 'conv5', 'output']
        # dict containing layer name and corresponding configuration
        self.layers = dict([(layer_name, layer) for layer_name, layer in zip(self.layer_names, list(self.model.main))])
        
        self.make_hooks()
    
    def get_activation(self, name):
        '''
        Creates and registers forward hooks to get activation for every forward pass
        
        Args:
            name - name of the layer as <str>
        Returns:
            PyTorch Layer hook
        Exception:
            -
        '''
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def make_hooks(self):
        '''
        Registers a hook for each of the layers in the network
        
        Args:
            -
        Returns:
            -
        Exception:
            -
        '''
        for idx, (layer_name, layer) in enumerate(zip(self.layer_names, list(self.model.main))):
            self.model.main[idx].register_forward_hook(self.get_activation(layer_name))
    
    def get_activations(self):
        '''
        Returns activations for every layer as a dict
        
        Args:
            -
        Returns:
            activations as <dict> layer name as key and layer activation <torch.Tensor> as value
        '''
        return self.activations

def visualize_activations(image_samples):
    '''
    Visualizes the activation for a list of image samples
    '''
    for image_sample in image_samples:
        output = model(image_sample.reshape(1,3,224,224))
        activations = cam.activations.copy()
        for layer, activation in activations.items():
            if layer not in layers_to_visualize:
                continue
            try:
                print(layer)
                activation = activation.squeeze().detach().numpy()
                c, w, h = activation.shape
                grid_size = int(np.sqrt(c)), int(np.ceil(np.sqrt(c)))
                fig, ax = plt.subplots(nrows = grid_size[0], ncols = grid_size[1], figsize=(15,15))
                idx = 0
                for row in range(grid_size[0]):
                    for col in range(grid_size[1]):
                        try:
                            ax[row][col].imshow(activation[idx, :, :], cmap = 'jet')
                            ax[row][col].axis('off')
                            idx += 1
                        except:
                            ax[row][col].axis('off')
                            continue
                plt.show()
            except Exception as e:
                print(e)
                continue

def visualize_weights(model):
    '''
    visualizes the weights in the network
    '''
    for layer in list(model.main): # change to model.main to layer name / sequential block name
        try:
            weights = layer.weight
            if len(weights.shape) < 3: continue
            print(layer)
            weights = torch.sum(weights, axis = 1).detach().numpy() # sum for channels
            c, w, h = weights.shape
            grid_size = int(np.sqrt(c)), int(np.ceil(np.sqrt(c)))
            fig, ax = plt.subplots(nrows = grid_size[0], ncols = grid_size[1], figsize=(15,15))
            idx = 0
            for row in range(grid_size[0]):
                for col in range(grid_size[1]):
                    try:
                        ax[row][col].imshow(weights[idx, :, :], cmap = 'jet')
                        ax[row][col].axis('off')
                        idx += 1
                    except Exception as e:
                        ax[row][col].axis('off')
                        continue
            plt.show()
        except Exception as e:
            continue
            
if __name__ == '__main__':
    # Load the model/ use state dict or torch.load here:
    model = Model_v1()
    cam = ClassActivationMaps(model)
    layers_to_visualize = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
    
    # visualize weights
    visualize_weights(model)
    
    # image_samples loaded and transformed as torch.Tensors
    visualize_activations(image_samples)