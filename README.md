# IMGCN_master
The code of IMGCN: Interpretable Masked Graph Convolution Network for Pedestrian Trajectory Prediction

The Paper: https://www.tandfonline.com/doi/abs/10.1080/21680566.2024.2389896

# IMGCN
The IMGCN utilizes interpretable information such as the pedestrian view area, distance, and motion direction to intelligently mask interaction features, resulting in more precise modeling of social interaction and movement factors. Specifically, we design a spatial and a temporal branch to model pedestrians' social interaction and movement factors, respectively. Within the spatial branch, the view-distance mask module masks pedestrian social interaction by determining whether the pedestrian is within a certain distance and view area to achieve more accurate interaction modeling. In the temporal branch, the motion offset mask module masks pedestrian temporal interaction according to the offset degree of their motion direction to achieve accurate modeling of movement factors. Ultimately, the 2D Gaussian distribution parameters of future trajectory points are predicted by the temporal convolution networks for multi-modal trajectory prediction.

![Figure 3 - 副2](https://github.com/user-attachments/assets/a42753aa-362a-42d8-b59b-8ba4e7fe0aaf)

## Code Structure
checkpoint folder: contains the trained models

dataset folder: contains ETH and UCY datasets

ETHUCYkeshihua folder: contains H.txt files and scene images for each scene of the ETH and UCY datasets

model.py: the code of IMGCN

train.py: for training the code

test.py: for testing the code

utils.py: general utils used by the code

metrics.py: Measuring tools used by the code


## Model Evaluation
You can easily run the model！ To use the pretrained models at checkpoint/ and evaluate the models performance run:  test.py

## Visualization
Visualize the model prediction trajectory, please run: visualization.py
It is worth noting that the H.txt file and scene pictures need to be adjusted for different scenes.


- For ETH dataset (eth, hotel), employ the code:

trajnrely, trajnrelx = world2image(V_y_rel_to_abs, H_inv)

trajnprey, trajnprex = world2image(V_pred_rel_to_abs, H_inv)

trajnobsy, trajnobsx = world2image(V_x_rel_to_abs, H_inv)


- For UCY dataset (univ, zara1, zara2), employ the code: 
            
trajnrelx, trajnrely = world2image(V_y_rel_to_abs, H_inv)

trajnprex, trajnprey = world2image(V_pred_rel_to_abs, H_inv)

trajnobsx, trajnobsy = world2image(V_x_rel_to_abs, H_inv)





## Trajectory prediction update
Different from previous random sampling (MC), we introduce Latin hypercube sampling (LHS) in pedestrian trajectory prediction to mitigate the long-tail effect. Compared with quasi-Monte Carlo sampling (QMC), Latin hypercube sampling is more suitable for trajectory prediction and can more accurately describe the diversity of pedestrian motion. It is worth noting that random sampling, quasi-Monte Carlo sampling, and Latin hypercube sampling are plug-and-play and do not require training. （For details, please refer to the paper: DSTIGCN: Deformable Spatial-Temporal Interaction Graph Convolution Network for Pedestrian Trajectory Prediction）

Prediction diagram of each sampling method. The top is a twodimensional scatter plot of 20 points using MC, QMC and LHS, respectively.
The asterisks represent the coordinates of the true destination in the sampling
space; the bottom is 20 random trajectories predicted by each method.
<img width="955" alt="不同采样方法的示意图 - 修改1" src="https://github.com/user-attachments/assets/cb0bd0ef-e9b2-4646-9d05-4417ca399b01" />

You can easily run the model! To use QMC sampling please run:  test-Qmc.py

You can easily run the model! To use LHS sampling please run:  test-Lhs.py

The prediction errors of different sampling methods are shown in the following table：
| IMGCN  | ETH | HOTEL| UNIV| ZARA1 | ZARA2 | AVG |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| MC  | 0.61/0.82 | 0.31/0.45| 0.37/0.67| 0.29/0.51 | 0.24/0.42 | 0.36/0.57 |
| QMC  | 0.59/1.09 | 0.22/0.34| 0.31/0.58| 0.25/0.48 | 0.22/0.41 | 0.32/0.58 |
| LHS  | 0.54/1.03 | 0.23/0.45| 0.26/0.47| 0.21/0.39 | 0.18/0.34 | 0.28/0.54 |


## Acknowledgement
Some codes are borrowed from Social-STGCNN and SGCN. We gratefully acknowledge the authors for posting their code.


## Cite this article:
Chen W, Sang H, Wang J, et al. IMGCN: interpretable masked graph convolution network for pedestrian trajectory prediction[J]. Transportmetrica B: Transport Dynamics, 2024, 12(1): 2389896. https://doi.org/10.1080/21680566.2024.2389896
