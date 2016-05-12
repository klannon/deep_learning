import plotly
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF

# import plotly
# from plotly.graph_objs import Scatter, Layout

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