deep-learning
=============
Because leading computer scientists working on neural networks have concluded that
<figure style="align: left; text-align:center;">
    <img src="http://i3.kym-cdn.com/photos/images/original/000/531/557/a88.jpg">
    <figcaption>(<a href="http://knowyourmeme.com/memes/we-need-to-go-deeper">What?</a>),</figcaption>
</figure>

this project contains python code that attempts to go deeper than any physicist has gone before by applying deep learning to high-energy physics in novel ways.


Installing the stuff
--------------------
1. Get the stuff we didn't write
<ul>
<li><a href="http://deeplearning.net/software/theano/">Theano</a></li>
<li><a href="http://deeplearning.net/software/pylearn2/>Pylearn2</a>"</li>
</ul>
2. Get the stuff we *did* write
<ul>
<li><code>git clone https://github.com/klannon/deep-learning.git</code></li>
</ul>
3. Set the `PYTHONPATH` environment variable
<ul>
<li><code>cd deep-learning</code></li>
<li><code>setenv PYTHONPATH $PWD</code></li>
</ul>

If you're planning on using this a lot, consider adding the PYTHONPATH environment variable to your .cshrc or .profile so you don't have to keep setting it