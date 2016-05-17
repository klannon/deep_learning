from flask import Flask, render_template, url_for, abort
app = Flask(__name__)

import utils
import graph


######################
## Helper Functions ##
######################

def dataset_page(dataset_name):
    files = utils.get_files_from_dataset(dataset_name)
    return render_template("dataset.html", dataset_name = dataset_name, file_names = files)

def file_page(dataset_name, file_name):
    graph_test = graph.make_test_plot()
    file_graph_html = graph.make_plots_from_single_file(dataset_name,
                                                        file_name)
    return render_template("file.html", file_name = file_name,
                           dataset_name = dataset_name, graph_html = file_graph_html)


#######################
## Routing Functions ##
#######################

@app.errorhandler(404)
def page_not_found(error):
    return (('<html><head><title>404 Error</title></head><body>ERROR 404: '
             + 'This page does not exist</body></html>'), 404)

@app.route('/datasets/<path:name>')
def dataset_and_file_handler(name):
    """ Note the use of the :path.  This is because I want this function to be
    called in 2 situations: (1) if the user supplies just a dataset and wants
    to see a list of the files in the dataset, or (2) if the user supplies a
    and a file and wants to see the information we have on that file.
    If the url provided doesn't satisfy either of those rules, throw a 404 """
    components = name.split("/")
    # situation (1)
    if len(components) == 1:
        return dataset_page(name) # (dataset)
    
    # situation (2)
    elif len(components) == 2:
        return file_page(components[0], components[1]) # (dataset, file)
        
    else:
        return abort(404)



@app.route('/')
def home_page():
    datasets = utils.get_datasets()
    return render_template("home.html", dataset_names=datasets)

if __name__ == '__main__':
    app.run(debug=True)
    
