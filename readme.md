
# MovieLens Recommender System
![PyTorch](https://img.shields.io/badge/Powered%20by-PyTorch-red.svg)
## Author
Noskov Nikita, B20-RO-01

---

## Overview
This project presents a Graph Neural Network-based recommender system, leveraging the MovieLens 100K dataset to predict user movie preferences. The system utilizes user demographics, movie ratings, and movie information to generate personalized recommendations.

## Strategy
The model employs a sophisticated Graph Attention Network (GATv2) to encode user-item interactions and an Edge Decoder for predicting ratings. It efficiently captures complex relationships within the data, ensuring accurate and relevant recommendations.

## Results
The recommender system demonstrates strong predictive performance with the following metrics:
- **Test RMSE (Root Mean Squared Error)**: 1.0582
- **Test MAE (Mean Absolute Error)**: 0.8563

These results indicate a high level of accuracy in predicting user ratings, showcasing the effectiveness of the GNN approach in handling recommendation tasks.



