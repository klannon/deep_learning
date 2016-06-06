import os

from deep_learning.protobuf import load_experiment
from deep_learning.protobuf import Experiment
import deep_learning.utils.dataset as ds

import plotly
from plotly import tools
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF

# f = open("train_y_misclass_pylearn_512.log", "r")

# x = []
# y = []
# epoch_number = 1
# for line in f:
#     x.append(float(line))
#     y.append(epoch_number)
#     epoch_number += 1

# f.close()

# text = plotly.offline.plot({
#     "data": [
#             Scatter(x=y, y=x)
#         ],
#     "layout": Layout(
#             title=""
#         )
#     }, output_type="div", show_link=False)

# # print text


def make_plots_from_single_file(dataset_name, file_name):
    data_path = ds.get_data_dir_path()
    file_path = os.path.join(data_path, dataset_name,
                             ("%s.exp" % file_name))
    exp = load_experiment(file_path)

    accuracies = map(lambda a: a.train_accuracy, exp.results[:])
    total_times = [0]
    times = map(lambda a: a.num_seconds, exp.results[:])
    for i in range(len(times)):
        total_time = total_times[len(total_times)-1] + times[i]
        total_times.append(total_time)

    trace1 = go.Scatter(x=range(len(accuracies)), y=accuracies)

    trace2 = go.Scatter(x=total_times, y=accuracies)

    titles = ('Train Accuracy vs Epochs','Train Accuracy vs Time')
    fig = tools.make_subplots(rows=2,cols=1,subplot_titles=titles)

    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 2, 1)
    # All of the axes properties here: https://plot.ly/python/reference/#XAxis
    fig['layout']['xaxis1'].update(title='Epochs')
    fig['layout']['xaxis2'].update(title='Time (seconds)')

    # All of the axes properties here: https://plot.ly/python/reference/#YAxis
    fig['layout']['yaxis1'].update(title='Train Accuracy')
    fig['layout']['yaxis2'].update(title='Train Accuracy')

    fig['layout'].update(height=900)

    text = plotly.offline.plot(fig, output_type="div", show_link=False)
    return text


def make_test_plot():
    # Add table data
    table_data = [['Team', 'Wins', 'Losses', 'Ties'],
                  ['Montreal<br>Canadiens', 18, 4, 0],
                  ['Dallas Stars', 18, 5, 0],
                  ['NY Rangers', 16, 5, 0],
                  ['Boston<br>Bruins', 13, 8, 0],
                  ['Chicago<br>Blackhawks', 13, 8, 0],
                  ['Ottawa<br>Senators', 12, 5, 0]]
    # Initialize a figure with FF.create_table(table_data)
    figure = FF.create_table(table_data, height_constant=60)
    
    # Add graph data
    teams = ['Montreal Canadiens', 'Dallas Stars', 'NY Rangers',
             'Boston Bruins', 'Chicago Blackhawks', 'Ottawa Senators']
    GFPG = [3.54, 3.48, 3.0, 3.27, 2.83, 3.18]
    GAPG = [2.17, 2.57, 2.0, 2.91, 2.57, 2.77]
    # Make traces for graph
    trace1 = go.Bar(x=teams, y=GFPG, xaxis='x2', yaxis='y2',
                    marker=dict(color='#0099ff'),
                    name='Goals For<br>Per Game')
    trace2 = go.Bar(x=teams, y=GAPG, xaxis='x2', yaxis='y2',
                    marker=dict(color='#404040'),
                    name='Goals Against<br>Per Game')
    
    # Add trace data to figure
    figure['data'].extend(go.Data([trace1, trace2]))
    
    # Edit layout for subplots
    figure.layout.yaxis.update({'domain': [0, .45]})
    figure.layout.yaxis2.update({'domain': [.6, 1]})
    # The graph's yaxis2 MUST BE anchored to the graph's xaxis2 and vice versa
    figure.layout.yaxis2.update({'anchor': 'x2'})
    figure.layout.xaxis2.update({'anchor': 'y2'})
    figure.layout.yaxis2.update({'title': 'Goals'})
    # Update the margins to add a title and see graph x-labels. 
    figure.layout.margin.update({'t':75, 'l':50})
    figure.layout.update({'title': '2016 Hockey Stats'})
    # Update the height because adding a graph vertically will interact with
    # the plot height calculated for the table
    figure.layout.update({'height':800})
    
    # Plot!
    return plotly.offline.plot(figure, output_type='div')
