import cProfile
import main
import pstats
from pstats import SortKey
print("TEST")
#cProfile.run('main.main()', 'restats')
p = pstats.Stats('restats')
p.strip_dirs().sort_stats(2).print_stats(.25)
