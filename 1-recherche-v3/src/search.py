from dataclasses import dataclass
from typing import Optional
from lle import Action, World, WorldState

from problem import GemSearchProblem, SearchProblem, SimpleSearchProblem
from priority_queue import PriorityQueue, PriorityQueueOptimized
import sys
import auto_indent

sys.stdout = auto_indent.AutoIndent(sys.stdout)

@dataclass
class Solution:
    actions: list[tuple[Action]]

    @property
    def n_steps(self) -> int:
        return len(self.actions)

    ...


def serialize(world_state: WorldState) -> tuple:
    return (tuple(world_state.agents_positions), tuple(world_state.gems_collected))

def was(world_state: WorldState, visited: set) -> bool:
    return serialize(world_state) in visited

# function to print visited set or stack items in terminal
def print_items(items, transform=None) -> None:
    """Prints items in terminal
    Args:
        items: items to print
    T is a generic type variable
    possible types for T:
    set, list, tuple, dict, etc."""
    # if items is a set
    if isinstance(items, set):
        print("set: ")
        print("visited: ")
    # if items is a stack
    elif isinstance(items, list):
        print("stack: ")
    # if transform == "hash":
    for item in items:
        print(item)
    print("")

def is_empty(data_structure) -> bool:
    """Returns True if data_structure is empty, False otherwise"""
    if isinstance(data_structure, list):
        return len(data_structure) == 0
    elif isinstance(data_structure, set):
        return len(data_structure) == 0
    elif isinstance(data_structure, PriorityQueueOptimized) or isinstance(data_structure, PriorityQueue):
        return data_structure.is_empty()

def tree_search(problem: SearchProblem, mode: str) -> Optional[Solution]:
    """Tree search algorithm.
    Args:
        problem: the problem to solve.
        mode: the search mode to use:
            - "dfs": Depth-First Search
            - "bfs": Breadth-First Search
            - "astar": A* Search

    Returns:
        A solution to the problem, or None if no solution exists.
    """
    # set problem's initial state
    initial_state = problem.initial_state
    # check if initial state is goal state
    current_state_is_goal_state = problem.is_goal_state(initial_state)

    
    if mode == "astar":
        # data_structure = PriorityQueue() 
        data_structure = PriorityQueueOptimized()  
        data_structure.push((initial_state, []), 0)  # Initial state with priority 0
    else:
        data_structure = [(problem.initial_state, [])]  #  to keep track of states
    visited = set()  # Set to keep track of visited states

    while not is_empty(data_structure):
        # print terminal line spacer empty line
        print("")
        # print terminal line separator
        print("--------------------------------------------------")
        # print_items(data_structure)

        # Pop the top state from the data_structure
        if mode == "bfs":
            current_state, actions = data_structure.pop(0)
        else:
            current_state, actions = data_structure.pop()

        print("current_state: ", current_state)
        # print("actions: ", actions)
        current_state_hashable = serialize(current_state)
        # print_items(visited)

        # compare hash of current_state to hash of visited states in terminal
        if was(current_state, visited):
            # state has already been visited
            # Skip it
            continue
        visited.add(current_state_hashable)

        # Check if the current state is the goal state
        if problem.is_goal_state(current_state):
            print("Solution found!")
            print("actions: ", actions)
            return Solution(actions)

        # Add successors to data_structure
        successors = problem.get_successors(current_state)
        print("successors: ")
        for successor, action, cost in successors:  # assuming get_successors returns (state, action) tuples
            print(successor)
            # Skip this successor if it has already been visited
            if was(successor, visited):
                # print("successor was visited")
                continue

            # print("actions: ", actions)
            # print("action: ", action)
            new_actions = actions + [action]
            # print("new_actions: ", new_actions)

            if mode == "astar":
                heuristic = problem.heuristic(successor)
                total_cost = cost + heuristic
                data_structure.push((successor, new_actions), total_cost)
            else:
                data_structure.append((successor, new_actions))
            # print_items(queue)
    # No solution found
    return None

def dfs(problem: SearchProblem) -> Optional[Solution]:
    """Depth-First Search"""
    return tree_search(problem, "dfs")

def bfs(problem: SearchProblem) -> Optional[Solution]:
    """Breadth-First Search"""
    return tree_search(problem, "bfs")

def astar(problem: SearchProblem) -> Optional[Solution]:
    """A* Search"""
    return tree_search(problem, "astar")

if __name__ == "__main__":
    # world = World.from_file("cartes/1_agent/vide")
    # world = World.from_file("level3")

    # problem = SimpleSearchProblem(world)
    # solution = dfs(problem)
    # print("solution: ", solution)

    # world = World.from_file("cartes/gems_simplest")
    world = World.from_file("cartes/2_agents/zigzag_gems")
    # world = World.from_file("cartes/gems")
    problem = GemSearchProblem(world)
    solution = astar(problem)
    print("solution: ", solution)
    # check_world_done(problem, solution)
    # if world.n_gems != world.gems_collected:
    #     raise AssertionError("Your is_goal_state method is likely erroneous beacuse some gems have not been collected")



    
