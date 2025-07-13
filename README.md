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

Output using some of the example images (original image -> YOLO output -> illegal parkings -> empty parkings count):

<p align='center'>
<img src="https://github.com/user-attachments/assets/b94a2ebd-8360-428c-b609-05e59e60eeda" width='200'>
<img src="https://github.com/user-attachments/assets/476ad931-7f78-4a47-8e57-6e5822d5e6d6" width='200'>
<img src="https://github.com/user-attachments/assets/f405f79b-205b-4588-9987-d67bef3a5f9e" width='200'>
<img src="https://github.com/user-attachments/assets/3549826d-17d2-49ba-ae88-5084cc59a1d1" width='300'>
</p>

<p align='center'>
<img src="https://github.com/user-attachments/assets/4d844192-5860-4e7b-ae45-b07c8c8c34d8" width='200'>
<img src="https://github.com/user-attachments/assets/5e33f35f-09b0-495a-a040-2695f11c5800" width='200'>
<img src="https://github.com/user-attachments/assets/30468e5a-145f-4e09-b4c3-a628cb63fc0b" width='200'>
<img src="https://github.com/user-attachments/assets/b79bd66d-8615-4257-b487-29edf961c279" width='300'>
</p>

