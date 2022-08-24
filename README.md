# BiNet
This code is used to implement algorithms designed in Paper XX, including an end-to-end identification algorithm (BiNet) for common pathogens and transfer learning for the identification of uncommon pathogens 

## The description of each source code:

### BiNet.py:
The architecture of BiNet was implemented as a class BiNet.

### load_data.py:
load training and testing data

### train.py:
Implementing model training

### test.py:
Implement model testing

### classify_common_pathogens.py:
Integrate model training and testing to identify common pathogens

### classify_uncommon_pathogens.py:
Integrates model training and testing to identify uncommon pathogens. In this file, the classifier is redesigned and the backbone is fine-tuned as the manuscript descripts. For transfer learning, the weights file trained with common pathogens needs to be loaded (it should be placed in the model folder)


### draw_result.py:
Visualization of the classification results


requirements.txt：
The necessary packages were listed.

### Contact:

Dr. Chenglong Tao: chengltao@126.com

Prof. Dr. Bingliang Hu: hbl@opt.ac.cn

Prof. Dr. Zhoufeng Zhang: zhangzhoufeng@opt.ac.cn
