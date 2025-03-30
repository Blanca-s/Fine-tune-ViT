# Fine-tune-ViT
This is an attempt on fine-tuning a simple LLM: Vision Transformer.\
Because this repository is a guidance written by a beginner and for beginner, so, the introduction will be easy and detailed, not like those professional ones. 
# Enviroment settings
In the process of LLMs related projects, the package "transformers" is a must. So, it is very basic to read its corresponding repository first. Please see: [transformers](https://github.com/huggingface/transformers).\
In its introduction, we could see that it requires enviornment powered by python>=3.9 and other basic packages like Pytorch 2.0+. 
\
We use the following commands to create the environment of this project:
```
conda create -n python-transformers python=3.10\
conda activate python-transformers
```
Please find a Pytorch version that satisifying your device. Details related to Pytorch will not be described here. It's too basic. But I do have a suggest for you: use `pip` instead of `conda` while installing this package. \
After installing package "Pytorch" and verifying it is usable, we could install `transformers`. Use command:
```
pip install transformers
```
Because this project is about Vision Transformer, which is used to process images. It is better for us to install another package:
```
pip install pillow
```
# A simple verification
After all packages are set, you could run the script `TheuseofViT.py` to see if everingthing is OK. \
If you directly run this script, you will get a two dimensional tensor whose first dimension has only one element and second dimension has 1000 elements, which is obviously inconsistent with our task: trying to deal with CIFAR-10, where only ten classes exist. What's more, if you use an evaluation script to evaluate the model accuracy on CIFAR-10, you will find the result is no more accurate than direct guessing. Here, we provide a evaluation script `Evaluation-ViT-on-CIFAR-10.py`. \
Then, if you think deeper, you will come up with an idea that training a neural netwrok as a classifier for ViT model, which could map the 1000 classes to the 10 classes in CIFAR-10. \
Knowing all that, you may try to convince yourself building a neural network following the same old story like instantiating a `nn.Module` class, difining linear layers, activatiion layers, balabalabala $\cdots$. However, in nowadays, developing is much more simpler. In the model ViT provided by google, it contains a classifer inside itself and all you need to do is to assigning the parameter `num_labels` when instantiating the model. For example:
```
model = ViTForImageClassification.from_pretrained(
    "/224015062/ViT",
    num_labels=10
)
```
Then, the model will randomly initiate a classifer itself. \
And please note my word: <font color=red>"randomly"</font>. Yes, when assigning this parameter, doesn't mean that the model will get a classifier that could be directly used. Instead, you need to train it. And that, is called "**fine-tune**".\
 So, as a beginner, even you are not a researcher in the fields related to LLMs, you've been hearing the concept of "fine-tune". It's like, everyone is talking about it, making you so confused. And now, you have experienced the construction of the concept "fine-tune" on your own. I believe you already have a better understanding of this concept than those who simply talk from all the time. After all, talk is cheap. \
 Also, we have provided a script `fine-tune_on_CIFAR-10.py` that helps you to fine-tune your own ViT model on CiFAR-10. In which, we simply uses one batch of CIFAR-10 to train the classifier. And if you test the model you fine-tune, you will find the accuracy is very high, about 96%. It is a very intuitionistic feeling of the power of LLMs. 
