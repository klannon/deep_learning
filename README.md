deep-learning
=============

deep-learning contains code to train and evaluate deep neural networks on data from the CMS Experiment at CERN's LHC.  It also contains some scripts to graph the performance of networks.

Installing:
-----------
<ol>
    <li>Get the stuff we didn't write
        <ul>
            <li><a href="http://keras.io/">Keras</a>: <code>pip install keras</code> Deep learning library used for designing/executing experiments</li>
            <li><a href="https://developers.google.com/protocol-buffers/">Google Protocol Buffers</a>: <code>pip install protobuf==3.0.0b2</code> Data serialization library used to save the results of Keras experiments in an intelligible format</li>
            <li><a href="https://plot.ly/">Plotly</a>: <code>pip install plotly</code> Graphing library used by the visualization dashboard to plot the results of experiments</li>
            <li><a href="http://flask.pocoo.org/">Flask</a>: <code>pip install flask</code> Web application framework used to write the visualization dashboard</li>
        </ul>
        Note: If you want to visualize the results of the networks you train, you can run the script "run_dashboard.py" in /dashboard (<code>python run_dashboard.py</code>)
    </li>
    <li>Get the stuff we <em>did</em> write
        <ul>
            <li><code>git clone https://github.com/klannon/deep-learning.git deep_learning</code></li>
            <li>It is VERY IMPORTANT that you clone the repository into the folder "deep_learning" instead of "deep-learning" because python can not import modules from directories that contain a "-" without some headaches</li>
        </ul>
    </li>
    <li>Set the <code>PYTHONPATH</code> environment variable to the PARENT directory of the directory that you cloned this repository into (whatever directory containes the folder "deep_learning" by default) so python can resolve our imports correctly.
        <ul>
            <li>CSH: <code>setenv PYTHONPATH /path/to/parent/directory</code></li>
            <li>BASH: <code>export PYTHONPATH="/path/to/parent/directory"</code></li>
        </ul>
    </li>
</ol>

If you're planning on using this code frequently, consider adding the PYTHONPATH environment variable to your .cshrc, .bashrc, .bash_profile, or .profile (depending on your shell of choice) so you don't have to keep setting it
