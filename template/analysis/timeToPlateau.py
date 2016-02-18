from __future__ import print_function

def extractDatapoints(filename):

    datapoints = []
    timer = []
    with open(filename) as f:
        epochs = -1
        for line in f:  # Copied from get_channels.py (mostly)
            acc = line.find("test_y_misclass") >= 0
            sec = line.find("total_seconds") >= 0
            channel_str = None
            if acc or sec:
                try:
                    channel_str = line.split(":")[1]
                except:
                    continue
            if channel_str is not None:
                channel_str = channel_str.strip()
                if acc:
                    accuracy = 1 - float(channel_str)
                    epochs += 1
                    datapoints.append((epochs, accuracy))
                elif sec:
                    timer.append(float(channel_str))
    timer.append(timer[-1])
    return timer, datapoints

def convertTime(times):

    rval = []

    for t in times:
        seconds = t%60
        times = ((t//60%60, t//60),  # Minutes
                 (t//60//60%24, t//60//60),  # Hours
                 (t//60//60//24%7, t//60//60//24),  # Days
                 (t//60//60//24//7%52, t//60//60//24//7),  # Weeks
                 (t//60//60//24//7//52, t//60//60//24//7//52))  # Years
        rval.append(':'.join([str(int(val)) for val, test in reversed(times) if test > 0]+[str(seconds)]))
    return tuple(rval)

def findPlateau(datapoints, avg_width=10, decay_percent=0.0002):

    plateaus = []
    _last_point = 0
    _last_avg = 0
    _found = False
    for n in xrange(len(datapoints)//avg_width):
        first_point = n*avg_width
        avg = 0
        for i in xrange(avg_width):
            ix, acc = datapoints[i+n*avg_width]
            avg += acc
        avg = avg/avg_width
        slope = abs((avg-_last_avg)/avg_width)
        if slope <= decay_percent:
            plateaus.append((_last_point, _last_avg))
            _found = True
        elif _found and slope > decay_percent:
            _found = False
            plateaus = []
        _last_point = first_point
        _last_avg = avg

    return tuple(plateaus) if plateaus else None

def totalTime(plateau, timer):

    plateau = (plateau,) if type(plateau[0]) is int else plateau

    return tuple([sum(timer[:p[0]]) for p in tuple(plateau)])

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Find the time to the final plateau for each .log file.")
    parser.add_argument("-a", "--all", help="Show all plateaus for each file", action="store_true", default=False)
    parser.add_argument("-w", "--width", help="The number of datapoints to average across", default=10, type=int)
    parser.add_argument("-d", "--decay", help="The slope between points that determines a plateau", default=0.0002, type=float)
    parser.add_argument("-s", "--seconds", help="Show the time to plateau in seconds", action="store_true", default=False)
    parser.add_argument("f", help="List of files to analyze", nargs="+")
    args = parser.parse_args()

    for f in args.f:
        t, d = extractDatapoints(f)
        p = findPlateau(d, avg_width=args.width, decay_percent=args.decay)
        try:
            p = p if args.all else p[0]
            s = totalTime(p, t) if args.seconds else convertTime(totalTime(p, t))
            s = s if args.all else s[0]
        except TypeError as e:
            print(e)
            p = None
            s = None
        print(p, s)