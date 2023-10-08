import numpy as np
from typing import List, Tuple
from lle import World
from problem import SimpleSearchProblem, GemSearchProblem, CornerSearchProblem
from search import bfs, dfs, astar, Solution
from utils import print_items
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors

# execute the 3 search algorithms on the level 3
# , and compare the size of the paths found for the three search algorithms on the level 3
# , and compare the number of nodes extended during the search for BFS, DSF and A∗ when searching
# use a annoted graph to show the size of the paths found for the three search algorithms on the level 3.

world = World.from_file("level3")
world.reset()


problem = SimpleSearchProblem(world)
solution = dfs(problem)
dfs_path_size = len(solution.actions)
dfs_nodes_expanded = problem.nodes_expanded

problem = SimpleSearchProblem(world)
solution = bfs(problem)
bfs_path_size = len(solution.actions)
bfs_nodes_expanded = problem.nodes_expanded

problem = SimpleSearchProblem(world)
solution = astar(problem)
astar_path_size = len(solution.actions)
astar_nodes_expanded = problem.nodes_expanded

print("dfs_path_size= ", dfs_path_size)
print("bfs_path_size= ", bfs_path_size)
print("astar_path_size= ", astar_path_size)

print("dfs_nodes_expanded= ", dfs_nodes_expanded)
print("bfs_nodes_expanded= ", bfs_nodes_expanded)
print("astar_nodes_expanded= ", astar_nodes_expanded)