# kaggle_inclusive_image_competition
Inclusive Image Competition hosted by Google Research on Kaggle

https://www.kaggle.com/c/inclusive-images-challenge

Although I have some course projects experience in computer vision with deep learning, this is the first time that I finish a complete computer vision project from EDA to final testing on my own. 

The dataset for this competition is very large, 500+GB images, 7000+ classes, 8.4 million human labels (7000+ unique), 15.3 million machine labels (4000+ unique) and 14.6 million bounding boxes (500+ unique labels). What makes the problem more challenging is the labels are heavily imbalanced. Some labels have 1+ million samples, such as face, tree, clothes, etc; Some labels only have a few samples. 

## My Strategies

Failure trials:

- Since this host forbids to use pretrained model, I decided to pretrain my model lightly on the bounding boxes since they are single-class labeled and human verified. I spent 1/3 of my cloud credits and trained a SEResNext101 to 70+% average precision. Then I used this pretrained model as a warm start to continue training on the multilabel classification problem. This pretrained model is not providing much boost comparing to training the multilabel classfication problem from scratch. Perhaps the bounding box labels are too limited (500+ classes), the bboxes have varying resolution (a few pixels to a few handred pixels) and each bbox only belongs to one label, all of which make it diffcult to generalize to a much more challenging dataset - 7000+ classes multilabel classification.

- My main failure of this competition is to use SEResNext101 network. I thought the SE architechture can give me more global information due to its Squeeze & Excitation operations, but it was not. This is also observed by another Kaggler [here](https://www.kaggle.com/c/inclusive-images-challenge/discussion/71525). Unfortunately I didn't have enough computing power / cloud credits to try another network. 

Success trials:

- Since the F2 score is defined for the average over all samples, we can write a soft F2 loss function and optimize it directly. I took the example of soft f2 loss from this [post](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/36809#390022). This is reminded by another Kaggler [Ren](https://www.kaggle.com/ryanzhang). 

- When optimization stagnates, changing optimizer can be effective.

- Changing the pooling layer to adaptive pooling according to fastai to enable multiscale training is very effective. This is also mentioned by another (post)[]. 

- Dataparallelism is effective and simple in pytorch.

- Dataloader can be tricky. I was stuck at some dead threading for quite some time until I updated to a nightly version of pytorch. It turns out that the annoying threading problem in dataloader is solved once and for all [here](https://github.com/pytorch/pytorch/pull/11985). Kudos to [SsnL](https://github.com/SsnL)!

## Lessons Learned

- Always start from the vanilla resnet50 for computer vision problems. For this competition, running resnet50 for tens of epoch with conventional techniques will simply give ~top 30 results.

- NASnet seems to be a good choice based on this [solution](https://www.kaggle.com/c/inclusive-images-challenge/discussion/71525) and he also used only ~1/8 data. Will test NASnet later.

- [Albumentation](https://github.com/albu/albumentations) is a good image augmentation tool in addition to [imgaug](https://github.com/aleju/imgaug) and [Augmentor](https://github.com/mdbloice/Augmentor).

- Testing Time Augmentation (TTA) can make validation error more robust.

- Fastai's 18 minutes to train imagenet is amazing, [blog](https://www.fast.ai/2018/08/10/fastai-diu-imagenet/), [repo](https://github.com/diux-dev/imagenet18). They seem to use multiple threads to load the same batch instead of one threads per batch as in the default dataloader. This is to be investigated.

## Complaints

- This competition is a bit less fair than I expected. The host claims that the data in Stage 1 and Stage 2 are very different in terms of distribution and they expect solutions that address this problem. However, people, who take advantage of exploiting the label distribution at Stage 1, get very high scores in the final standing. I doubt that the 20+ people in the final leaderboard who has the same score just submitted some top N popular labels. This clearly defeats the purpose of the host and the host fails to make this a fair game to all Kagglers. 

- Although the huge dataset also makes it less accessible to more people, it is very kind of Google to provide $1000 cloud credits.
