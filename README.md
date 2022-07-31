# lrp-grad_cam-chexpert

Repository for the ['best student paper award'](https://2022.cbms-conference.org/awards/) winning paper at the [35th IEEE 35th International Symposium on Computer Based Medical Systems (CBMS 2022)](https://2022.cbms-conference.org/),

**Exploring LRP and Grad-CAM visualization to interpret multi-label-multi-class pathology prediction using chest radiography**, Mahbub Ul Alam, Jón Rúnar Baldvinsson and Yuxia Wang.

## Abstract

The area of interpretable deep neural networks has received increased attention in recent years due to the need for transparency in various fields, including medicine, healthcare, stock market analysis, compliance with legislation, and law. Layer-wise Relevance Propagation (LRP) and Gradient-weighted Class Activation Mapping (Grad-CAM) are two widely used algorithms to interpret deep neural networks. In this work, we investigated the applicability of these two algorithms in the sensitive application area of interpreting chest radiography images. In order to get a more nuanced and balanced outcome, we use a multi-label classification-based dataset and analyze the model prediction by visualizing the outcome of LRP and Grad-CAM on the chest radiography images. The results show that LRP provides more granular heatmaps than Grad-CAM when applied to the CheXpert dataset classification model. We posit that this is due to the inherent construction difference of these algorithms (LRP is layer-wise accumulation, whereas Grad-CAM focuses primarily on the final sections in the model's architecture). Both can be useful for understanding the classification from a micro or macro level to get a superior and interpretable clinical decision support system.

### An excerpt of the paper

![An excerpt of the paper](http://dash.blogs.dsv.su.se/files/2022/07/Screenshot-2022-07-27-at-11.30.12-AM.png)

## Citation

Please acknowledge the following work in papers or derivative software:

Mahbub Ul Alam, Jón Rúnar Baldvinsson and Yuxia Wang, "Exploring LRP and Grad-CAM visualization to interpret multi-label-multi-class pathology prediction using chest radiography," 2022 IEEE 35th International Symposium on Computer-Based Medical Systems (CBMS), 2022


