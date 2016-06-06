deep-learning
=============

deep-learning contains code to train and evaluate deep neural networks on data from the CMS Experiment at CERN's LHC.  It also contains some scripts to graph the performance of networks.

Installing:
-----------

<ol>
    <li>Get the stuff we wrote
        <ul>
            <li><code>git clone https://github.com/klannon/deep-learning.git deep_learning</code></li>
            <li>It is VERY IMPORTANT that you clone the repository into the folder "deep_learning" (which will happen by default) instead of some other directory of your choosing.  The imports depend on the top-level module being "deep_learning."</li>
        </ul>
    </li>
    <li>Create your virtual environment
    <p>To keep this repository's dependencies separate from the rest of the python packages you have installed on your system, we recommend setting up a <a href="https://virtualenv.pypa.io/en/latest/">virtual envirtonment</a> to contain the software you install in step 3.</p>
        <ol>
            <li><a href="https://virtualenv.pypa.io/en/latest/installation.html">Install virtualenv</a></li>
            <li>Because some of the software in step 3 depends on <code><a href="https://www.scipy.org/">scipy</a></code> and <code><a href="http://www.numpy.org/">numpy</a></code>, each of which is tightly coupled with your system BLAS libraries, set up your virtual environment like this: <code>virtualenv --system-site-packages env</code> so virtualenv will know about your system installations of numpy and scipy.</li>
            <li>Since your virtual environment wouldn't be very useful if you didn't activate it, <code>source env/bin/activate</code> if you use BASH or <code>source env/bin/activate.csh</code> if you use CSH</li>
        </ol>
        
    </li>
    <li>Get the stuff we <em>didn't</em> write
        <ul>
            <li><a href="http://keras.io/">Keras</a>: <code>pip install keras</code> Deep learning library used for designing/executing experiments</li>
            <li><a href="https://developers.google.com/protocol-buffers/">Google Protocol Buffers</a>: <code>pip install protobuf==3.0.0b2</code> Data serialization library used to save the results of Keras experiments in an intelligible format</li>
            <li><a href="https://plot.ly/">Plotly</a>: <code>pip install plotly</code> Graphing library used by the visualization dashboard to plot the results of experiments</li>
            <li><a href="http://flask.pocoo.org/">Flask</a>: <code>pip install flask</code> Web application framework used to write the visualization dashboard</li>
            <li><a href="http://scikit-learn.org/stable/index.html">Scikit-learn</a>: <code>pip install scikit-learn</code> Library we use to normalize batches of data</li>
        </ul>
        Note: If you want to visualize the results of the networks you train, you can run the script "run_dashboard.py" in deep_learning/dashboard (<code>python run_dashboard.py</code>) to launch a web application that will let you visualize the results of all the networks you've trained!  More features coming soon!
    </li>
    <li>Set the <code>PYTHONPATH</code> environment variable to the <em>PARENT</em> directory of the directory that you cloned this repository into (whatever directory containes the folder "deep_learning" by default) so python can resolve our imports correctly.
        <ul>
            <li>CSH: <code>setenv PYTHONPATH /path/to/parent/directory</code></li>
            <li>BASH: <code>export PYTHONPATH="/path/to/parent/directory"</code></li>
        </ul>
    </li>
</ol>

If you're planning on using this code frequently, consider adding the PYTHONPATH environment variable to your .cshrc, .bashrc, .bash_profile, or .profile (depending on your shell of choice) so you don't have to keep setting it
