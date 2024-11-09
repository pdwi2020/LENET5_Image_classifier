# LENET5_Image_classifier
Image classification using Parallel approach (CUDA)
# IMAGE CLASSIFIER WITH LENET MODEL USING PARALLEL APPROACH

**Digital Assignment- 2**  
**COMPUTER ARCHITECTURE AND ORGANIZATION**  
**MCSE503L**  
**FALL SEMESTER 2024-25**  
**SLOT: C2+TC2**  

### PRESENTED BY:
- PARITOSH DWIVEDI – 24MAI0013  
- C. DEEPTHI CHOWDARY – 24MAI0036  
- ADITYA HIREMATH – 24MAI0117  

### Under the guidance of  
**PROF. SAIRABANU J**  
School of Computing Science and Engineering  
VIT University  

---

## ABSTRACT
This project aims to classify land use in geospatial images using a modified LeNet-5 architecture, focusing on images from the EuroSAT dataset. EuroSAT, based on RGB images from the Sentinel-2 satellite, includes 10 distinct classes representing different land cover types such as annual crops, forests, highways, and residential areas. Each image is 64x64 pixels, capturing a ground sampling distance of 10 meters. The objective is to predict the top two land uses in each image to enhance decision-making in geospatial analysis. By leveraging a modified LeNet-5 CNN, the project achieves efficient and accurate classification of land use.

**Keywords**: Image Classification, LeNet-5, CNN, EuroSAT Dataset, Geospatial Images, Land Use, Remote Sensing, Sentinel-2, Deep Learning

---

## INTRODUCTION
Geospatial image classification enables land use monitoring, aiding researchers and policymakers in understanding land utilization. This project employs the EuroSAT dataset and a CNN based on the LeNet-5 architecture, modified for diverse land use categories. The goal is to accurately predict the top two land use categories in each image, aiding applications in urban development, environmental conservation, and resource allocation. The modified LeNet-5 model proves efficient for geospatial analysis, capturing important spatial patterns within the EuroSAT dataset.

---

## CONTRIBUTION OF THE WORK
1. **Adaptation of Lightweight CNN Architecture**:  
   Customized LeNet-5 to classify satellite images in EuroSAT effectively.
   
2. **Optimization of Data Pipeline**:  
   Utilized caching, prefetching, and parallel data processing to maximize hardware resource utilization.

3. **Parallel Processing and GPU Utilization**:  
   Employed batch processing, parallel matrix operations, and GPU acceleration for scalable satellite data handling.

4. **Data Augmentation for Generalization**:  
   Used rotation and flipping techniques to improve model robustness, making it suitable for remote sensing applications.

5. **Evaluation and Benchmarking**:  
   Benchmarked results on EuroSAT to enable comparisons with other methods, such as transfer learning.

6. **Educational and Practical Framework**:  
   This research provides a structured workflow for satellite image classification, offering an accessible framework for further studies.

---

## EXISTING SYSTEM DESCRIPTION
### Core Model
LeNet-5, initially designed for digit recognition, has been adapted to various fields including traffic sign recognition and medical imaging, demonstrating the adaptability of the architecture across complex classification tasks.

### EuroSAT Dataset Description
- **Purpose**: For remote sensing and land cover classification.
- **Contents**: 27,000 images across 10 land cover classes, each 64x64 pixels.
- **Download**: [EuroSAT Dataset on Kaggle](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset/data)

---

## PROPOSED SYSTEM ARCHITECTURE
1. **Input Layer**: Accepts 64x64 RGB images.
2. **C1 - Convolution Layer**: 6 filters of 5x5 kernel with Tanh/ReLU activation.
3. **S2 - Pooling Layer**: Average pooling with 2x2 filter.
4. **C3 - Convolution Layer**: 16 filters of 5x5 kernel.
5. **S4 - Pooling Layer**: Average pooling with 2x2 filter.
6. **C5 - Fully Connected Layer**: 120 neurons.
7. **Flatten Layer**
8. **F6 - Fully Connected Layer**: 84 neurons.
9. **Output Layer**: Softmax activation for 10 land cover classes.

### Parallel Processing & Optimization
- **Parallel Portion**: Each image processed independently to enable parallelization.
- **Technique**: Utilized GPU for batch processing of images.
- **Additional Optimizations**: Data augmentation and caching.

---

## IMPLEMENTATION DETAILS WITH GITHUB LINK
For complete code and implementation details, please refer to the [GitHub repository](#).

---

## RESULTS
- **Training Accuracy**: 72.85%
- **Test Accuracy**: 72.79%
- **Loss**: 0.7672

---

## CONCLUSION
The modified LeNet-5 architecture, alongside optimized data pipelines and parallel processing techniques, demonstrated effective satellite image classification with high accuracy. This approach highlights the potential of lightweight CNN models in remote sensing, showing that accurate results can be achieved with minimal resources.

### Future Directions
1. **More Complex Architectures**: Explore ResNet or Vision Transformers.
2. **Incorporating Temporal Data**: Investigate CNN-LSTM for time-series satellite images.
3. **Expanding Dataset**: Include diverse datasets to improve generalizability.

---

## ACKNOWLEDGEMENTS
We thank the EuroSAT team for the dataset, TensorFlow and Keras developers, Kaggle community contributors, and all mentors and peers who supported this research.

---

## REFERENCES
1. Basu et al., 2015, DeepSat: A Learning Framework for Satellite Imagery.
2. Castelluccio et al., 2015, Land Use Classification in Remote Sensing Images.
3. Mnih and Hinton, 2016, Learning to Detect Roads in High-Resolution Aerial Images.
4. Zhou et al., 2017, Pattern Recognition in Remote Sensing with CNNs.
5. Helber et al., 2019, EuroSAT: A Benchmark for Land Use Classification.

*And more...*
