from flask import Flask
app = Flask(__name__)

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


@app.route('/')
def home_page():
    return render_template("home.html")

if __name__ == '__main__':
    app.run(host='127.1.1.23')
