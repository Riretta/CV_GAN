# CV_GAN
Image Generation with Capsule Vector-VAE
We propose <b>Capsule Vector - VAE(CV-VAE)</b>, a new model based on VQ-VAE architecture where the <em>discrete bottleneck</em> represented by the quantization code-book
is replaced with a capsules layer. 
We demonstrate that the capsules can be successfully applied for the clusterization procedure reintroducing the differentiability of the bottleneck in the model. 
![A test image](CVGAN.jpg)<br>
The capsule layer clusters the encoder outputs considering the agreement among capsules. 
<br>
The CV-VAE is trained within Generative Adversarial Paradigm (GAN), CVGAN in short. Our model is shown to perform on par with the original VQGAN, VAE in GAN. 
CVGAN obtains images with higher quality after few epochs of training. The interpretability of the training process for the latent representation is significantly increased maintaining the structured bottleneck idea. <br>
![A test image](CV.jpg)<br>
This has practical benefits, for instance, in unsupervised representation learning, where a large number of capsules may lead to the disentanglement of latent representations
![A test image](Results.jpg)<br>
We present results on ImageNet, COCOStuff, and FFHQ datasets, and we compared the obtained images with results with VQGAN. 

