from __future__ import print_function
import cProfile as profile
import pstats2
import os
import line_profiler
from template import run_3v
from pylearn2.monitor import Monitor

__author__ = 'Matthew Drnevich'

if __name__ == '__main__':

    hostname = os.getenv("HOST", os.getpid())
    results_dir = "{1}{0}results{0}".format(os.sep, os.getcwd())
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    idpath = "{}{}_run_3v_main".format(results_dir, hostname)

    if False:
        stats_file = idpath+'_pstats.stat'

        profile.run('run_3v.run(15)', filename=stats_file)

        s1 = pstats2.Stats(stats_file, stream=open(idpath+"_sorted_cumpercall.txt", 'w'))
        s2 = pstats2.Stats(stats_file, stream=open(idpath+"_sorted_tottime.txt", 'w'))
        s3 = pstats2.Stats(stats_file, stream=open(idpath+"_sorted_percall.txt", 'w'))
        s4 = pstats2.Stats(stats_file, stream=open(idpath+"_sorted_callees.txt", 'w'))
        s5 = pstats2.Stats(stats_file, stream=open(idpath+"_sorted_callers.txt", 'w'))

        s1.sort_stats("cumpercall")
        s2.sort_stats("tottime")
        s3.sort_stats("percall")
        s4.sort_stats("cumpercall")
        s5.sort_stats("cumpercall")

        s1.print_stats()
        s2.print_stats()
        s3.print_stats()
        s4.print_callees()
        s5.print_callers()

    if True:
        stats_file = idpath+'_line_stats.stat'

        profile = line_profiler.LineProfiler(Monitor.__call__)
        profile.run('run_3v.run(15)')

        profile.print_stats(open(idpath+"_line_stats.txt", 'w'))