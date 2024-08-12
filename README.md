# IMGCN_master
The code of IMGCN: Interpretable Masked Graph Convolution Network for Pedestrian Trajectory Prediction

The code will coming soon!

# IMGCN
The IMGCN utilizes interpretable information such as the pedestrian view area, distance, and motion direction to intelligently mask interaction features, resulting in more precise modeling of social interaction and movement factors. Specifically, we design a spatial and a temporal branch to model pedestrians' social interaction and movement factors, respectively. Within the spatial branch, the view-distance mask module masks pedestrian social interaction by determining whether the pedestrian is within a certain distance and view area to achieve more accurate interaction modeling. In the temporal branch, the motion offset mask module masks pedestrian temporal interaction according to the offset degree of their motion direction to achieve accurate modeling of movement factors. Ultimately, the 2D Gaussian distribution parameters of future trajectory points are predicted by the temporal convolution networks for multi-modal trajectory prediction.

![Figure 3 - 副2](https://github.com/user-attachments/assets/a42753aa-362a-42d8-b59b-8ba4e7fe0aaf)
