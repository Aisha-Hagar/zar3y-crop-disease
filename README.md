# Zar3y
Zar3y (“my crop”) — a phone-camera app that takes a photo of a tomato, potato, pepper, or corn leaf and tells the farmer in plain language what disease (if any) it has, with a confidence score and a Grad-CAM overlay showing what the model looked at. 

## Requirement 1
Per-class count (Train Dataset)
| Class Name                    | Samples Count |
|-------------------------------|---------------|
| Tomato__Late__blight          | 1336          |
| Potato___Early_blight         | 700           |
| Pepper,_bell___healthy        | 1034          |
| Potato___healthy              | 106           |
| Corn_(maize)___Common_rust_   | 834           |
| Potato___Late_blight          | 700           |
| Tomato___Early_blight         | 700           |
| Pepper,_bell___Bacterial_spot | 697           |
| Tomato___healthy              | 1113          |
| Tomato___Leaf_Mold            | 666           |

<img width="560" height="638" alt="classcounts" src="https://github.com/user-attachments/assets/68f16536-5d16-4efe-b690-88f1c04dadd9" />
<img width="534" height="638" alt="classweights" src="https://github.com/user-attachments/assets/35d727ef-d1d5-4825-b2bf-1d3ea0616b8e" />
