# Archeological Burial Mounds CNN Classifier 
------ 

The goal of this project was to build a machine learning system that could identify potential burial mounds in satellite images to a high degree of accuracy. The satellite images used come from the Tundzha Region in Bulgaria where many ancient mounds are found across a rugged terrain of mountains and open land.

To accomplish this goal, the project builds a CNN classifier model which takes in cropped tiles of a satellite image and predicts whether the area within is likely to contain a mound or not. The image's projected coordinates can then be used to determine the exact location of the mound. While the project focused on this one region, the greater interest lay in whether machine learning tools could be used to assist archeological teams in their field work, by identifying areas of interest before teams have to be sent out. 

## The Repository 

In this repository, we are investigating two avenues: first, the segmentation of large satellite images into smaller tiles which can be prepared for a CNN model. Second, the classification of these multi-band tile images as either a 'mound' or 'not a mound'. 
