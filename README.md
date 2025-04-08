# Short Video Segment-level User Interests Modeling in Personalized Recommendation

**The paper has been accepted by SIGIR 2025.**

paper links: https://github.com/hezy18/SegMMInterest/blob/main/SIGIR_2025_Zhiyu.pdf

## Dataset

SegMM_inter_sample.csv is an example of **SegMM**'s interaction data. Due to the space limitation, we sampled 10000 lines from the preprocessed data. The full data will be released by Clouds with interaction (multi-behavior), video segment-level embeddings, and the sampled raw videos. 
Here's the [data details](https://github.com/hezy18/SegMMInterest/blob/main/SegMM.md).

## Codes

* MMinterest is the code for segment-level user dynamic interest modeling and baselines in video-skip prediction
* SegRec is the code for segment-integrated video recommendation
* SkipPredBaseline is the code for recommender baselines in video-skip prediction
* data_process is for preparation of data in all tasks

## Citation

If you use our dataset(SegMM), methods, find our contributions useful in your work, please cite our paper as:

```bib
@ARTICLE{2025arXiv250404237H,
       author = {{He}, Zhiyu and {Ling}, Zhixin and {Li}, Jiayu and {Guo}, Zhiqiang and {Ma}, Weizhi and {Luo}, Xinchen and {Zhang}, Min and {Zhou}, Guorui},
        title = "{Short Video Segment-level User Dynamic Interests Modeling in Personalized Recommendation}",
      journal = {arXiv e-prints},
     keywords = {Information Retrieval},
         year = 2025,
        month = apr,
          eid = {arXiv:2504.04237},
        pages = {arXiv:2504.04237},
archivePrefix = {arXiv},
       eprint = {2504.04237},
 primaryClass = {cs.IR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv250404237H},
}
'''


If you have any question, please contact us at hezy22@mails.tsinghua.edu.cn.
