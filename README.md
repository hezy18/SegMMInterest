# Short Video Segment-level User Interests Modeling in Personalized Recommendation

**The paper has been accepted by SIGIR 2025.**

arxiv link: https://arxiv.org/abs/2504.04237

acm link: https://dl.acm.org/doi/abs/10.1145/3726302.3730083

## Dataset

SegMM_inter_sample.csv is an example of **SegMM**'s interaction data. Due to the space limitation, we sampled 10000 lines from the preprocessed data. The full data will be released by Clouds with interaction (multi-behavior), video segment-level embeddings, and the sampled raw videos. 
Here's the [data details](https://github.com/hezy18/SegMMInterest/blob/main/SegMM.md).

## Codes

* MMinterest is the code for segment-level user dynamic interest modeling and baselines in video-skip prediction
* SegRec is the code for segment-integrated video recommendation
* SkipPredBaseline is the code for recommender baselines in video-skip prediction
* data_process is for the preparation of data in all tasks

## Citation

If you use our dataset(SegMM), methods, and find our contributions useful in your work, please cite our paper as:

```bib
@inproceedings{10.1145/3726302.3730083,
author = {He, Zhiyu and Ling, Zhixin and Li, Jiayu and Guo, Zhiqiang and Ma, Weizhi and Luo, Xinchen and Zhang, Min and Zhou, Guorui},
title = {Short Video Segment-level User Dynamic Interests Modeling in Personalized Recommendation},
year = {2025},
isbn = {9798400715921},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3726302.3730083},
doi = {10.1145/3726302.3730083},
booktitle = {Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {1880–1890},
numpages = {11},
keywords = {segment-level interest, user modeling, video recommendation},
location = {Padua, Italy},
series = {SIGIR '25}
}
```

If you have any questions, please get in touch with us at hezy22@mails.tsinghua.edu.cn.
