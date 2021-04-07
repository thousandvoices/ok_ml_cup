## Overview
This repository contains a winning solution to [OK ML Cup](https://cups.mail.ru/ru/tasks/1048), a competition devoted to detection of toxic content in comments from russian social network. The major challenge was maintaining good accuracy and throughput simultaneously in extremely resource-constrained environment (3 slow CPU cores and no GPU) provided by competition sponsors. The dataset was later made public and is now available at <https://www.kaggle.com/alexandersemiletov/toxic-russian-comments>.

The surprising result is that pruning, quantization and knowledge distillation can make transformer models order of magnitude smaller and faster without a significant loss of prediction quality.

Whenever model quality is concerned, we report mean average precision on validation set.

## Building accurate models
Pretrained language models have revolutionized the field of natural language processing, achieving state-of-the-art results in almost every task they were applied to. While for russian texts there aren't as many options as for english, Converstaional RuBERT from DeepPavlov and XLM-Roberta show excellent results, beating simpler methods, such as logistic regression with tf-idf features and supervised fasttext, by a margin. Furthermore, averaging their predictions boosts the performance significantly.

| Model | Mean Average Precision | Throughput, sentences per second |
| --------------------- |:----------------------:|:-:|
| Logistic regression | 0.9434 | 2100 |
| Conversational RuBERT | 0.9594 | 32 |
| XLM-Roberta-large | 0.9638 | 8 |
| Averaged predictions | 0.9682 | 7 |

## Making the models faster...
However, such ensembles are known to be slow and impractical. That's why we use knowledge distillation to extract and transfer information into a much smaller student. Several approaches to students construction were proposed in recent years. <https://arxiv.org/pdf/1908.08962.pdf> reports that smaller models trained on language modeling task show best results, but expensive pretraining step makes it infeasible for a single-GPU setup. Instead, we decided to use an approach proposed in <https://arxiv.org/pdf/2004.03844.pdf> and initialize our student with bottom layers from existing large model, namely conversational rubert. This method works reasonably well without relying on additional computational resources.

| Model | Mean Average Precision | Throughput, sentences per second |
| --------------------- |:----------------------:|:-:|
| 4 bottom layers of RuBERT trained on original labels | 0.9548 | 87 |
| 4 bottom layers of RuBERT trained with knowledge distillation | 0.9634 | 87 |

We used mean squared error loss for knowledge distillation. Training dataset was augmented with randomly truncated and concatenated sentences, which gave us a small but consistent performance improvement.

Model quantization is the key for further preformance gains as it both reduces memory bandwidth requirements and enables much more efficient SIMD registers usage. Several frameworks provide this functionality out of the box. We opted for ONNX runtime as an engine for inference mainly because of its high-quality quantization implementation (but excellent performance and small binaries were important as well). As you can see, models become 4 times faster while quality loss remains negligible.

| Framework | Mean Average Precision | Throughput, sentences per second |
| ----- |:----:|:----:|
| Pytorch, full precision | 0.9634 | 87 |
| Pytorch, dynamic quantization | 0.9635 | 156 |
| ONNX runtime, 8-bit quantization| 0.9632 | 313 |

## ... and smaller
Another interesting factor is model size. Pruning, quantization and gzip compression all show strong improvements. Moreover, the gains are complimentary — when applied simultaneously, they reduce model size by a factor of 12.

| Model | Size, megabytes |
| ----- |:---------------:|
| Conversational RuBERT | 712 |
| 4-layer student | 484 |
| 4-layer student + quantization| 121 |
| 4-layer student + quantization + gzip compression | 62 |

## One more trick
At this point my Core i7-7700 CPU was able to process more than 300 sentences a second with the quantized model. But it still wasn't fast enough to be accepted for the competition.

We noticed that most of the provided examples are rather easy and can be reliably classified with simpler methods. So we decided to predict all of the examples with logistic regression and used BERT for final prediction only if this model was not confident enough (namely, the predictions fell inside `(threshold, 1 - threshold)` range; the threshold was chosen to be 0.04 based on validation set performance). It turned out that we could pass as few as 8% of all data to the slow language model based classifier without losing prediction quality, achieving another order of magnitude speed-up.

## Pretrained models
We provide the resulting models for russian toxic comments detection at the releases page.

To use the models, you'll need to install the library
```
python -m pip install --user git+https://github.com/thousandvoices/ok_ml_cup.git#egg=ok_ml_cup[bert]
```

Invoking them from python is simple now. They will be downloaded automatically and cached at ```~/.toxic_text_classifier```.
```python
from toxic_text_classifier.inference.classifier import Classifier

classifier = Classifier.load('https://github.com/thousandvoices/ok_ml_cup/releases/download/v0.0.1/rubert_conversational_4_layers.zip')
print(classifier.predict(['ну и тупой же ты', 'я тебя люблю']))
```

Manual downloads are supported as well:
```bash
wget https://github.com/thousandvoices/ok_ml_cup/releases/download/v0.0.1/rubert_conversational_4_layers.zip
unzip rubert_conversational_4_layers.zip
```
```python
from toxic_text_classifier.inference import Classifier

classifier = Classifier.load('rubert_conversational_4_layers')
```

You can visit the online demo (in russian) at <https://stark-badlands-29727.herokuapp.com> to see the models in action. Please note that the first page load can take a couple of seconds due to heroku free instances limitations.

## License
The code is released under MIT license.

The models are distributed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) to comply with the training dataset license.
