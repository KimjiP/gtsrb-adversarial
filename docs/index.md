## Thesis Title: Analyzing the Robustness of Deep Learning Classifiers for Traffic Sign Recognition Systems

[Download the paper here.](https://bit.ly/3h1VvDA)

Find the [source code on GitHub](https://github.com/KimjiP/gtsrb-adversarial-attack-grad-cam/blob/main/model2_gtsrb.py) for this project.

The [German Traffic Sign Recognition Benchmark (GTSRB) Dataset](https://benchmark.ini.rub.de/gtsrb_news.html) was used in training a CNN built using TensorFlow.

Sample GTSRB Images:
![Sample GTSRB](https://raw.githubusercontent.com/KimjiP/gtsrb-adversarial-attack-grad-cam/main/docs/gtsrb.png)

### Trying Different Explainable AI techniques for the GTSRB images

![Explainable AI](https://raw.github.com/KimjiP/gtsrb-adversarial-attack-grad-cam/main/docs/xai.PNG)

Grad-CAM offers the best visualization as it successfully highlights parts of the images that contributed to the final classification decision. LIME, Occlusion sensitivity, and SmoothGRAD failed at this task.

### Building and Training a Convolutional Neural Network (CNN) for Traffic Sign Recognition (TSR)

The standard CNN is the same architecture used in Princeton University INSPIRE group's [research about adversarial attacks](https://arxiv.org/pdf/1802.06430.pdf) . The table below shows its layers.

![standard CNN](https://raw.github.com/KimjiP/gtsrb-adversarial-attack-grad-cam/main/docs/standard%20cnn.PNG)

### Results

#### Grad-CAM Explainers for [Carlini-Wagner Attack](https://arxiv.org/pdf/1608.04644.pdf)

![standard CNN cw attack](https://raw.github.com/KimjiP/gtsrb-adversarial-attack-grad-cam/main/docs/cw.png)

#### Grad-CAM Explainers for [JSMA Attack](https://arxiv.org/pdf/1511.07528.pdf)

![standard CNN jsma attack](https://raw.github.com/KimjiP/gtsrb-adversarial-attack-grad-cam/main/docs/jsma.png)


#### Grad-CAM Explainers for [DeepFool Attack](https://arxiv.org/pdf/1511.04599.pdf)
![standard CNN deepfool attack](https://raw.github.com/KimjiP/gtsrb-adversarial-attack-grad-cam/main/docs/deepfool.png)

### Results Summary
![results summary](https://raw.github.com/KimjiP/gtsrb-adversarial-attack-grad-cam/main/docs/summary.png)
![transferability](https://raw.github.com/KimjiP/gtsrb-adversarial-attack-grad-cam/main/docs/transferability2.PNG)
