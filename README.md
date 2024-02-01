Human Action Recognition (HAR) is an important form of application in computer vision that aims to accurately
describe human activities and their interactions by collecting sensors from never-before-seen data sets. It is able to serve
as an intermediary that can read human actions that are a reflection of a state or situation that may require further
handling or analysis so that the situation can be responded to or mitigated in the future. This research aims to detect 4
simple action clues that are commonly detected in research, namely standing, walking, running, or falling. This can be
done by developing a smart model that is then trained based on the dataset that needs to be created. The YOLOv8 model is
a model that can process a dataset of video frames that will then be able to perform object recognition, labeling, and
bounding boxes that limit the object. This model will be implemented in a web-based human action recognition system.
The system is then implemented using python programming language with Flask microframework. In the end, this
research produces a web-based application that is able to recognise human actions in 4 action classes, namely standing,
walking, running, and falling with the best values of mAP, precision, recall and F1-score of 97%, 99%, 100%, and 93%
