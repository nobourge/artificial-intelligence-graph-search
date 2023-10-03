from dataclasses import dataclass
from typing import Optional
from lle import Action

from problem import SearchProblem


@dataclass
class Solution:
    actions: list[tuple[Action]]

    @property
    def n_steps(self) -> int:
        return len(self.actions)

    ...


def dfs(problem: SearchProblem) -> Optional[Solution]:
    """Depth-First Search"""
    # set problem's initial state
    initial_state = problem.initial_state
    # check if initial state is goal state
    current_state_is_goal_state = problem.is_goal_state(initial_state)

    # apply dfs
    stack = [problem.initial_state]  # Initialize the stack with the initial state
    visited = set()  # Set to keep track of visited states

    while stack:
        current_state = stack.pop()
        
        # Skip this state if it has already been visited
        if current_state in visited:
            continue
        visited.add(current_state)
        
        # Check if the current state is the goal state
        if problem.is_goal_state(current_state):
            return Solution(current_state)
        
        # Add successors to stack
        successors = problem.get_successors(current_state)
        stack.extend(successors)
    
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
    
