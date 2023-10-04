from dataclasses import dataclass
from typing import Optional
from lle import Action, WorldState

from problem import SearchProblem


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



def dfs(problem: SearchProblem) -> Optional[Solution]:
    """Depth-First Search"""
    # set problem's initial state
    initial_state = problem.initial_state
    # check if initial state is goal state
    current_state_is_goal_state = problem.is_goal_state(initial_state)

    # apply dfs
    stack = [(problem.initial_state, [])]  # Initialize the stack with the initial state and an empty action list
    visited = set()  # Set to keep track of visited states

    while stack:
        # Pop the top state from the stack
        current_state, actions = stack.pop()

        # Skip this state if it has already been visited
        # if hash(current_state) in visited:
        if current_state in visited:
            continue
        visited.add(current_state)

        # Check if the current state is the goal state
        if problem.is_goal_state(current_state):
            return Solution(actions)

        # Add successors to stack
        successors = problem.get_successors(current_state)
        for successor, action, heuristic in successors:  # assuming get_successors returns (state, action) tuples
            new_actions = actions + [action]
            stack.append((successor, new_actions))

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
    astar("problem")
    
