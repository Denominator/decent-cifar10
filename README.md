# decent-cifar10
This is "Decent Cifar10" model for TensorFlow.

It is heavily based on this extremely useful Torch model
http://torch.ch/blog/2015/07/30/cifar.html
done by http://imagine.enpc.fr/~zagoruys.

This TensorFlow model was used as an entry point:
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/models/image/cifar10

Training:
1. git clone https://github.com/Denominator/decent-cifar10.git
2. cd tf_decent_cifar10
3. python cifar10_train.py

Evaluation/Testing:
In the same directory: python cifar10_eval.py