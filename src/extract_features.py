# global imports
import torch
import numpy as np
from torchvision import models

# local imports that we created
from preprocess import resnet_preprocess
from load_data import load_cifar10

"""
Function that loads the pre-trained ResNet18 model
and uses its default weights for accurate predictions.
Removes the last layer since CIFAR10 is not on ImageNet.
Set model to evaluation mode (not training mode) and return it.
The model converts an image to 512 numerical features.
"""
def load_resnet18_feature_extractor():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) # use pretrained weights
    model = torch.nn.Sequential(*list(model.children())[:-1])  # remove final layer
    model.eval() # set model to evaluation mode
    # return model which is now a feature extractor for 512 numbers (vector)
    return model

"""
Extracts features for each (image, label) pair in a dataset.
Returns numpy arrays x (features) and y (labels).
"""
def extract_features(model, dataset, device = "cpu"):
    
    # empty lists to store values
    features = []
    labels = []

    for img, label in dataset:
        # apply the preproccesing formula we created to each image
        x = resnet_preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            # pass image through resnet18 model
            feat = model(x).squeeze().cpu().numpy() # squeeze to remove unnecessary dimensions

        features.append(feat) # feature vectors for desired set
        labels.append(label) # list of labels

    return np.array(features), np.array(labels)

if __name__ == "__main__":

    # load both datasets of cifar10
    train_data, test_data = load_cifar10()

    # create resnet18 model
    model = load_resnet18_feature_extractor()

    # get feature vectors for both data sets
    x_train, y_train = extract_features(model, train_data)
    x_test, y_test = extract_features(model, test_data)

    # save to disk
    np.save("G:/My Drive/COMP472/project/features/x_train.npy", x_train)
    np.save("G:/My Drive/COMP472/project/features/y_train.npy", y_train)
    np.save("G:/My Drive/COMP472/project/features/x_test.npy", x_test)
    np.save("G:/My Drive/COMP472/project/features/y_test.npy", y_test)

    print("Saved features!")