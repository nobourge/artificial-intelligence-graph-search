from dataclasses import dataclass
from typing import Optional
from lle import Action, World, WorldState

from problem import SearchProblem, SimpleSearchProblem

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

def dfs(problem: SearchProblem) -> Optional[Solution]:
    """Depth-First Search"""
    # set problem's initial state
    initial_state = problem.initial_state
    # check if initial state is goal state
    current_state_is_goal_state = problem.is_goal_state(initial_state)

    # apply dfs
    stack = [(problem.initial_state, [])]  # Stack to keep track of states
    visited = set()  # Set to keep track of visited states

    while stack:
        # print terminal line spacer empty line
        print("")
        # print terminal line separator
        print("--------------------------------------------------")

        #print stack
        # print_items(stack)

        # Pop the top state from the stack
        current_state, actions = stack.pop()
        print("current_state: ", current_state)
        # print("actions: ", actions)
        current_state_hashable = serialize(current_state)
        # print_items(visited)

        # Skip this state if it has already been visited
        # if hash(current_state) in visited:

        # compare hash of current_state to hash of visited states in terminal
        if was(current_state, visited):
            print("current_state was visited")
            continue
        visited.add(current_state_hashable)

        # Check if the current state is the goal state
        if problem.is_goal_state(current_state):
            print("Solution found!")
            print("actions: ", actions)

            return Solution(actions)

        # Add successors to stack
        successors = problem.get_successors(current_state)
        print("successors: ")
        for successor, action, heuristic in successors:  # assuming get_successors returns (state, action) tuples
            print(successor)
            # Skip this successor if it has already been visited
            if was(successor, visited):
                # print("successor was visited")
                continue

            # print("actions: ", actions)
            # print("action: ", action)
            new_actions = actions + [action]
            # print("new_actions: ", new_actions)
            stack.append((successor, new_actions))
            # print_items(stack)
    # No solution found
    return None
        



def bfs(problem: SearchProblem) -> Optional[Solution]:
    ...


def astar(problem: SearchProblem) -> Optional[Solution]:
    print("astar")
    print("problem: ", problem)
    print("problem.world: ", problem.world)

    print("problem.world.get_state(): ", problem.world.get_state())

    # apply astar



#main


if __name__ == "__main__":
    # world = World.from_file("cartes/1_agent/vide")
    world = World.from_file("level3")

    problem = SimpleSearchProblem(world)
    solution = dfs(problem)
    print("solution: ", solution)


    
