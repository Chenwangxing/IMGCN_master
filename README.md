# IMGCN_master
The code of IMGCN: Interpretable Masked Graph Convolution Network for Pedestrian Trajectory Prediction

# IMGCN
The IMGCN utilizes interpretable information such as the pedestrian view area, distance, and motion direction to intelligently mask interaction features, resulting in more precise modeling of social interaction and movement factors. Specifically, we design a spatial and a temporal branch to model pedestrians' social interaction and movement factors, respectively. Within the spatial branch, the view-distance mask module masks pedestrian social interaction by determining whether the pedestrian is within a certain distance and view area to achieve more accurate interaction modeling. In the temporal branch, the motion offset mask module masks pedestrian temporal interaction according to the offset degree of their motion direction to achieve accurate modeling of movement factors. Ultimately, the 2D Gaussian distribution parameters of future trajectory points are predicted by the temporal convolution networks for multi-modal trajectory prediction.

![Figure 3 - 副2](https://github.com/user-attachments/assets/a42753aa-362a-42d8-b59b-8ba4e7fe0aaf)

## Code Structure
checkpoint folder: contains the trained models

dataset folder: contains ETH and UCY datasets

model.py: the code of IMGCN

train.py: for training the code

test.py: for testing the code

utils.py: general utils used by the code

metrics.py: Measuring tools used by the code

## Model Evaluation
You can easily run the model！ To use the pretrained models at checkpoint/ and evaluate the models performance run:  test.py

## Acknowledgement
Some codes are borrowed from Social-STGCNN and SGCN. We gratefully acknowledge the authors for posting their code.


## Cite this article:
Chen W, Sang H, Wang J, et al. IMGCN: interpretable masked graph convolution network for pedestrian trajectory prediction[J]. Transportmetrica B: Transport Dynamics, 2024, 12(1): 2389896. https://doi.org/10.1080/21680566.2024.2389896
