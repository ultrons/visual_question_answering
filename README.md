This is a neural network architecture for visual question answering roughly based on the paper "Dynamic Memory Networks for Visual and Textual Question Answering" by Xiong et al. (ICML2016). The input includes an image and a question about the image, and the output is the answer to the question. It first uses a convolutional neural network to extract a set of features of the image, which are called "contexts" or "facts". These facts are fused by a bidirectional GRU. It also uses a GRU recurrent neural network to encode the question. Then the encodings of image and question are fed into an episodic memory module, which consists of the attention mechanism (responsible for producing a contextual vector) and the memory update mechanism (responsible for generating the episode memory based upon the contextual vector and previous episode memory). Both the soft and GRU-based attention mechanisms are implemented. Finally, the model generates a one-word answer based on the last memory and the encoding of question. 

This project is implemented in Tensorflow, and allows end-to-end training of both CNN and RNN parts. To use it, you will need the Tensorflow version of VGG16 or ResNet(50, 101, 152) model, which can be obtained by using Caffe-to-Tensorflow. 

Examples
----------
![img](examples/COCO_val2014_000000393282_3932820_result.jpg)
![img](examples/COCO_val2014_000000052527_525272_result.jpg)
![img](examples/COCO_val2014_000000131207_1312070_result.jpg)
![img](examples/COCO_val2014_000000235984_2359841_result.jpg)
![img](examples/COCO_val2014_000000078820_788200_result.jpg)
![img](examples/COCO_val2014_000000367029_3670291_result.jpg)
![img](examples/COCO_val2014_000000052470_524702_result.jpg)
![img](examples/COCO_val2014_000000576827_5768271_result.jpg)
![img](examples/COCO_val2014_000000445682_4456821_result.jpg)

References
----------

* [Dynamic Memory Networks for Visual and Textual Question Answering](https://arxiv.org/abs/1603.01417) Caiming Xiong, Stephen Merity, Richard Socher. ICML 2016.
* [Visual Question Answering (VQA) dataset](http://visualqa.org/)
* [Implementing Dynamic memory networks by YerevaNN](https://yerevann.github.io/2016/02/05/implementing-dynamic-memory-networks/)
* [Dynamic memory networks in Theano](https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano)
* [Dynamic Memory Networks in Tensorflow](https://github.com/therne/dmn-tensorflow)
* [Caffe to Tensorflow](https://github.com/ethereon/caffe-tensorflow)


Temporary Description of the repository:



.
├── base_model.py
├── dataset.py
├── episodic_memory.py
├── examples
│   ├── COCO_val2014_000000052470_524702_result.jpg
│   ├── COCO_val2014_000000052527_525272_result.jpg
│   ├── COCO_val2014_000000078820_788200_result.jpg
│   ├── COCO_val2014_000000131207_1312070_result.jpg
│   ├── COCO_val2014_000000235984_2359841_result.jpg
│   ├── COCO_val2014_000000367029_3670291_result.jpg
│   ├── COCO_val2014_000000393282_3932820_result.jpg
│   ├── COCO_val2014_000000445682_4456821_result.jpg
│   └── COCO_val2014_000000576827_5768271_result.jpg
├── LICENSE.md
├── main.py
├── model.py
├── models
│   └── vgg16
│       ├── -10999.data-00000-of-00001
│       ├── -10999.index
│       ├── -10999.meta
│       ├── -1999.data-00000-of-00001
│       ├── -1999.index
│       ├── -1999.meta
│       ├── -2999.data-00000-of-00001
│       ├── -2999.index
│       ├── -2999.meta
│       ├── -3999.data-00000-of-00001
│       ├── -3999.index
│       ├── -3999.meta
│       ├── -4999.data-00000-of-00001
│       ├── -4999.index
│       ├── -4999.meta
│       ├── -5999.data-00000-of-00001
│       ├── -5999.index
│       ├── -5999.meta
│       ├── -6999.data-00000-of-00001
│       ├── -6999.index
│       ├── -6999.meta
│       ├── -7999.data-00000-of-00001
│       ├── -7999.index
│       ├── -7999.meta
│       ├── -8999.data-00000-of-00001
│       ├── -8999.index
│       ├── -8999.meta
│       ├── -9999.data-00000-of-00001
│       ├── -9999.index
│       ├── -9999.meta
│       ├── -999.data-00000-of-00001
│       ├── -999.index
│       ├── -999.meta
│       └── checkpoint
├── __pycache__
│   ├── base_model.cpython-35.pyc
│   ├── dataset.cpython-35.pyc
│   ├── episodic_memory.cpython-35.pyc
│   └── model.cpython-35.pyc
├── README.md
├── requirements.txt
├── run.sh
├── train
│   ├── anns.csv
│   ├── images -> /home/tsaikevin/VQA/dataset/Images/train2014
│   ├── mscoco_train2014_annotations.json -> /home/tsaikevin/VQA/dataset/Annotations/v2_mscoco_train2014_annotations.json
│   ├── mscoco_val2014_annotations.json -> /home/tsaikevin/VQA/dataset/Annotations/v2_mscoco_val2014_annotations.json
│   ├── OpenEnded_mscoco_test2015_questions.json -> /home/tsaikevin/VQA/dataset/Questions/v2_OpenEnded_mscoco_test2015_questions.json
│   ├── OpenEnded_mscoco_test-dev2015_questions.json -> /home/tsaikevin/VQA/dataset/Questions/v2_OpenEnded_mscoco_test-dev2015_questions.json
│   ├── OpenEnded_mscoco_train2014_questions.json -> /home/tsaikevin/VQA/dataset/Questions/v2_OpenEnded_mscoco_train2014_questions.json
│   ├── OpenEnded_mscoco_val2014_questions.json -> /home/tsaikevin/VQA/dataset/Questions/v2_OpenEnded_mscoco_val2014_questions.json
│   └── rename.sh
├── utils
│   ├── ilsvrc_2012_mean.npy
│   ├── __init__.py
│   ├── nn.py
│   ├── __pycache__
│   │   ├── __init__.cpython-35.pyc
│   │   ├── nn.cpython-35.pyc
│   │   └── words.cpython-35.pyc
│   ├── vqa
│   │   ├── __init__.py
│   │   ├── __init__.pyc
│   │   ├── license.txt
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-35.pyc
│   │   │   ├── vqa.cpython-35.pyc
│   │   │   └── vqaEval.cpython-35.pyc
│   │   ├── readme.md
│   │   ├── README.md
│   │   ├── vqaEval.py
│   │   ├── vqaEval.pyc
│   │   ├── vqa.py
│   │   └── vqa.pyc
│   └── words.py
└── words
    └── word_table.pickle



Note:

Following dirtories have to be created inorder for the run to proceed.
./train  (create appropriate links of the images annotations etc, filename are important)
likewise for val and test.
./models (This is where it is going to savew models/checkpoint etc.)






