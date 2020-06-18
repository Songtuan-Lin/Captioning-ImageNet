---


---

<h2 id="captioning-imagenet">Captioning ImageNet</h2>
<p>This project captions the images within ImageNet dataset in a semi-autonomous way. Specifically, we used state-of-the-art caption generator plus constrained beam search algorithm to accomplish this task.</p>
<h2 id="motivation">Motivation</h2>
<p>This project is motivated by the situation that the avaliable datasets used to train image captioning model are limited to Microsoft COCO and Flickr. Therefore, we want to caption the images in ImageNet and extend the avaliable datasets.</p>
<h2 id="framework-used">Framework Used</h2>
<p>The caption generator in this project is built upon Pythia, which is developed by facebook research group and provides a pre-trained BUTD caption generator.  Moreover, since Pythia rely on Pytorch, this project also requires Pytorch installed.</p>
<h2 id="features">Features</h2>
<p>The basic feature of this project is that it can generate caption for images in ImageNet by using the method described in the project report. However, the most exciting part about this project is that it can accept almost any regular expression which specify the format of caption and  implements this regular expression as constrained beam search.  More precisely, this project can be used to caption arbitary image with it’s generated caption follow some constrains specified by user-defined regular expression.</p>
<h2 id="installation">Installation</h2>
<p>In order to use this project, Pythia should first be installed:</p>
<pre class=" language-console"><code class="prism  language-console">git clone https://github.com/Songtuan-Lin/pythia.git
cd pythia/
git reset --hard 33225b89023472f9307b4e665e6429dbcbe01d77
sed -i '/torch/d' requirements.txt
pip install -e .
</code></pre>
<pre class=" language-console"><code class="prism  language-console">git clone https://gitlab.com/meetshah1995/vqa-maskrcnn-benchmark.git
cd vqa-maskrcnn-benchmark/
python setup.py build develop
</code></pre>
<p>If the installation failed, check whether all dependencies are installed:</p>
<pre class=" language-console"><code class="prism  language-console">pip install ninja yacs cython matplotlib demjson
</code></pre>
<p>Then, clone this repo and make a directory called <em>model_data</em>. This <em>model_data</em> directory is used to hold pre-trained model data:</p>
<pre class=" language-console"><code class="prism  language-console">git clone https://gitlab.cecs.anu.edu.au/u6162630/Captioning-ImageNet-Pythia.git
cd Captioning-ImageNet-Pythia/
mkdir model_data/
</code></pre>
<p>Finally, download pre-trained model data:</p>
<pre class=" language-console"><code class="prism  language-console">wget -O model_data/vocabulary_captioning_thresh5.txt https://dl.fbaipublicfiles.com/pythia/data/vocabulary_captioning_thresh5.txt
wget -O /model_data/detectron_model.pth https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.pth
wget -O model_data/butd.pth https://dl.fbaipublicfiles.com/pythia/pretrained_models/coco_captions/butd.pth
wget -O model_data/butd.yaml https://dl.fbaipublicfiles.com/pythia/pretrained_models/coco_captions/butd.yml
wget -O model_data/detectron_model.yaml https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.yaml
wget model_data/detectron_weights.tar.gz https://dl.fbaipublicfiles.com/pythia/data/detectron_weights.tar.gz
tar xf model_data/detectron_weights.tar.gz
</code></pre>
<p>Now, we are ready to go!</p>
<h2 id="usage">Usage</h2>
<p>To caption the images in ImageNet, simply execute <em>caption_imagenet.py</em> file by giving it three command line arguments:  the root directory of ImageNet dataset, the target directory to hold the caption results and the upper levels of ImageNet tag to trace(mentioned in project report)</p>
<pre class=" language-console"><code class="prism  language-console">python caption_imagenet.py --root_dir root directory of ImageNet --save_dir directory to save results --up_level
</code></pre>
<p>To support input regular expression, we also provide following classes:</p>
<ol>
<li>utils.finite_automata.FiniteAutomata: Construct finite automata by giving an regular expression as input.</li>
<li>utils.table_tensor.TableTensor: Transfer transition tables of a finite automata to Pytorch tensor.</li>
<li>dataset.customized_dataset.CustomizedDataset: Load arbitrary dataset and transition tables which are represented as Pytorch tensor.</li>
</ol>
<p>Additionally, regular expression should be consist of following symbols:</p>
<ol>
<li>.: Match any single character.</li>
<li>?: Match zero or more occurrences of the preceding element.</li>
<li>( : Worked as delimiter, do not match any symbol.</li>
<li>): the same as (</li>
<li>a-zA-Z: Alphabet</li>
<li>space: used to seperate token, does not match any symbol.</li>
</ol>
<p>Particularly, wildcard matching can be replaced as: (.?). Moreover, it is strongly suggest that using ‘(’ and ‘)’ to seperate each component in the regular expression. For example, if we want to input regular expression ‘dog|cat’, we strongly recommand rewrite it as ‘(dog|cat)’.</p>
<p>The example which demonstrate how to construct and use regular expression will be presented in Demo.</p>
<h2 id="demo">Demo</h2>
<p>The following code snippet demonstrate how to construct finite automata and visualize it by giving a regular expression:</p>
<pre class=" language-console"><code class="prism  language-console">from utils.finite_automata import FiniteAutomata
from utils.table_tensor import TableTensor

reg = '.?(animal|bird).?'
nfa = FiniteAutomata(reg)
nfa.visualize()
</code></pre>
<p>The complete demo, which shows how to use our code to caption ImageNet and how to caption image with arbitrary regular expression, can be find here:</p>
<p><a href="https://colab.research.google.com/drive/1YSxnFoBQ-2EQVVsWJIOj0w_5oFqv6Xaa?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a></p>
<h2 id="api-reference">API Reference</h2>
<p>class utils.finite_automata.FiniteAutomata (reg): This class take an input regular expression and produce the corresponding NFA. The main class methods include:</p>
<ol>
<li>transitions(): This method returns the transition table which is corresponding to input regular expression.</li>
<li>visualize(): This method visualize the finite automata.</li>
</ol>
<p>class utila.table_tensor.TableTensor(vocab, table): This class takes two arguments: a pre-defined vocabulary and a transition table produced by class FiniteAutomata. The main class methods include:</p>
<ol>
<li>to_tensors(): Convert the transition table to Pytorch tensor.</li>
</ol>
<p>class dataset.customized_dataset.CustomizedDataset(root_dir, transitions): This class takes a file directory and a transition table which produced by TableTensor as arguments and construct a Pytorch dataset.</p>
<h2 id="implementation-note">Implementation Note</h2>
<p>The core of our implementation is how we represent the transition table of a finite automata as Pytorch tensor. In our code, we represent the transition table as a list of tensors. This list contains k tensors, where k equals to the number of states in finite automata. The ith tensor in the list indicates which tokens in the vocabulary can trigger the statee transition from another states to state i. More precisely, if we denote the ith tensor in the list as Ti, then, Ti has size (num_states, vocab_size) and the transition table is interpreted as:</p>
<ol>
<li>If Ti[j, k] = 0, then the kth token in the vocabulary can trigger the state transition from state j to state i.</li>
<li>If Ti[j, k] = 1, otherwise.</li>
</ol>
<p>By representing transition table as list of tensor, we can then implement constrained beam search. This part of code has been well commented and hence will not be explained here.</p>
<h2 id="script-to-download-captioned-data">Script to Download Captioned Data</h2>
<p>Since our captioned images are stored in AWS S3 storage, which can only be accessed under certain permission, we provide a script to download captioned data:<br>
<a href="https://colab.research.google.com/drive/1Fhset-K0NCUdcl4tZSAHsyo4_s07qy6I?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a></p>

