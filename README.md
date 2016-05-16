deep-learning
=============

deep-learning contains code to train and evaluate deep neural networks on data from the CMS Experiment at CERN's LHC.  It also contains some scripts to graph the performance of networks.

Installing:
-----------
<ol>
    <li>Get the stuff we didn't write
        <ul>
            <li><a href="http://keras.io/">Keras</a>: <code>pip install keras</code></li>
            <li><a href="https://developers.google.com/protocol-buffers/">Google Protocol Buffers</a>: <code>pip install protobuf==3.0.0b2</code></li>
        </ul>
        Note: If you want to visualize the results of the networks you train, you can run the script "run_dashboard.py" in /dashboard (<code>python run_dashboard.py</code>)
    </li>
    <li>Get the stuff we <em>did</em> write
        <ul>
            <li><code>git clone https://github.com/klannon/deep-learning.git</code></li>
        </ul>
    </li>
    <li>Set the <code>PYTHONPATH</code> environment variable to the directory you cloned this repository into ("deep-learning" by default) so python can resolve our imports correctly.
        <ul>
            <li><code>cd deep-learning</code></li>
            <li><code>setenv PYTHONPATH $PWD</code></li>
        </ul>
    </li>
</ol>

If you're planning on using this code frequently, consider adding the PYTHONPATH environment variable to your .cshrc or .profile so you don't have to keep setting it
