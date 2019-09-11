# common_functions
functions to use to help keep code concise as well as a recommender system.

To install, use: ```pip install git+https://github.com/aaronengland/common_functions.git```

The ```recomendations``` function measures the strength of association of each item with a target item or list of target items. Association is determined using 3 metrics:
- Support
- Confidence
- Lift

Metric Definitions:
- Support: overall probability of an item being prescribed

<img src="https://latex.codecogs.com/gif.latex?Support&space;=&space;\frac{Prescriptions{_{Item}}}{Prescriptions{_{Total}}}" title="Support = \frac{Prescriptions{_{Item}}}{Prescriptions{_{Total}}}" />

- Confidence: probability of an item being prescribed, given that the ```target_item``` has been prescribed

<img src="https://latex.codecogs.com/gif.latex?Confidence&space;=&space;\frac{Prescriptions{_{Item}}}{Prescriptions{_{TargetItem}}}" title="Confidence = \frac{Prescriptions{_{Item}}}{Prescriptions{_{TargetItem}}}" />

- Lift: ratio of confidence to support

<img src="https://latex.codecogs.com/gif.latex?Lift&space;=&space;\frac{Confidence}{Support}" title="Lift = \frac{Confidence}{Support}" />

How to use the function:

The user passes equal length arrays for prescription [ID], product name, and modality. Note: Each value of these arrays pertains to one prescribed item. 
