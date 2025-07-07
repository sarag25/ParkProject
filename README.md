## Parking Spot Detection and Classification: Assessing Legality and Slot Type Using Deep Learning and Rule-Based Approaches

***From the Abstract*** : Starting from an image of a parking area, first we employ detection to identify single parking spots, then classification to differentiate between occupied and empty spaces; additionally, if a car is present, we classify each slot as either correct or incorrect parking, otherwise we classify the type of empty space like normal, paid, disabled or pregnant parking. We employed mainly neural networks for object detection and classification, but also rule-based approaches.
<p></p>

Full paper of this project available here:
[Paper.pdf](https://github.com/user-attachments/files/21104356/Park_Project_Paper_compressed.pdf)


<p align="center">
  <img src="https://github.com/user-attachments/assets/9c3a7c16-429b-4965-af89-4e26f267624c" width='500'>
</p>
<p align="center"> <i>Pipeline</i> </p>

### How to execute the pipeline
The datasets are not available on this repository but are downloadable here: https://drive.google.com/drive/folders/1RHxNd7ZpMgi6An2oUwFgzz-q7toDPgbm?usp=drive_link
The models used can be found into the project 'models' folder.

The 'ParkProject.py' file can be used to try the full pipeline on a single image; example images can be found in the 'datasets' folder. The outputs generated will be saved in the 'results' folder.

Files structure:
<ul>
  <li>'preprocessing': inside this folder there are the files to apply filtering to the datasets used to train the models and to input images in general.</li>
  <li>'object_detection_and_classification': this contains the methods used to detect bounding boxes and classify each spot as empty/occupied. The output of the YOLO model is collected in the 'runs' subdirectory. It's also included the file that crops the single parking slots after detection.</li>
  <li>'classification_occupied_spots': this directory contains the files related to the ResNet model training and execution.</li>
  <li>'classification_empty_spots': it collects the methods proposed to classify empty parking slots according to the color of their lines.</li>
</ul>
