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

class HashableAction:
    def __init__(self, action):
        self.action = action

    def __hash__(self):
        # return hash((self.action.some_attribute, self.action.some_other_attribute))
        return hash((self.action))

    def __eq__(self, other):
        if not isinstance(other, HashableAction):
            return False
        # return self.action.some_attribute == other.action.some_attribute and \
            #    self.action.some_other_attribute == other.action.some_other_attribute
        return self.action == other.action

# visited = {}
# # To add an action:
# visited[str(action)] = True
# # To check for an action:
# if str(action) in visited:
#     pass

def serialize_state(world_state: WorldState) -> tuple:
    return (tuple(world_state.agents_positions), tuple(world_state.gems_collected))

def was(world_state: WorldState, visited: set) -> bool:
    return serialize_state(world_state) in visited

# def print_items(items: T) -> None:
#     """Prints items in terminal
#     Args:
#         items: items to print
#     T is a generic type variable
#     possible types for T:
#     set, list, tuple, dict, etc."""
#     print("items: ")
#     for item in items:
#         print(item)

# function to print visited set or stack items in terminal
def print_items(items) -> None:
    """Prints items in terminal
    Args:
        items: items to print
    T is a generic type variable
    possible types for T:
    set, list, tuple, dict, etc."""
    print("items: ")
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
        print_items(stack)

        # Pop the top state from the stack
        current_state, actions = stack.pop()
        print("current_state: ", current_state)
        print("actions: ", actions)

        # current_state_hashable = problem.serialize_state(current_state)
        current_state_hashable = serialize_state(current_state)
        print("current_state_hashable: ", current_state_hashable)
        print("current_state_hashable hash: ")
        print(hash(current_state_hashable))

        # print the set of visited states in their hashed form
        print("visited_state hashes: ")

        for visited_state in visited:
            # print("visited_state: ", visited_state)
            print(hash(visited_state))
        # print the set of visited states
        print_items(visited)

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
                print("successor was visited")
                continue

            print("actions: ", actions)
            print("action: ", action)
            new_actions = actions + [action]
            print("new_actions: ", new_actions)
            stack.append((successor, new_actions))
            print_items(stack)

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
    world = World.from_file("cartes/1_agent/vide")
    problem = SimpleSearchProblem(world)
    solution = dfs(problem)
    print("solution: ", solution)


    
