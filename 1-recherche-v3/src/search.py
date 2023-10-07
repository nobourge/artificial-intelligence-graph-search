from dataclasses import dataclass
from typing import Optional
from lle import Action, World, WorldState

from problem import CornerSearchProblem, GemSearchProblem, SearchProblem, SimpleSearchProblem, serialize, was
from priority_queue import PriorityQueue, PriorityQueueOptimized
import sys
import auto_indent
from utils import print_items

sys.stdout = auto_indent.AutoIndent(sys.stdout)

@dataclass
class Solution:
    actions: list[tuple[Action]]

    @property
    def n_steps(self) -> int:
        return len(self.actions)

    ...

def is_empty(data_structure) -> bool:
    """Returns True if data_structure is empty, False otherwise"""
    if isinstance(data_structure, list):
        return len(data_structure) == 0
    elif isinstance(data_structure, set):
        return len(data_structure) == 0
    elif isinstance(data_structure, PriorityQueueOptimized) or isinstance(data_structure, PriorityQueue):
        return data_structure.is_empty()

def check_goal_state(problem: SearchProblem
                     , current_state: WorldState
                     , actions: list[tuple[Action]]
                     , objectives_reached = None
                     ) -> bool:
    # Check if the current state is the goal state
    if isinstance(problem, CornerSearchProblem) or isinstance(problem, GemSearchProblem):
        current_state_is_goal_state = problem.is_goal_state(current_state, objectives_reached)
    else:
        current_state_is_goal_state = problem.is_goal_state(current_state)
        
    if current_state_is_goal_state:
        print("Solution found!")
        print("actions: ", actions)
        print( "n_steps: ", len(actions))
        return Solution(actions)
    
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
    actions = []
    cost = 0
    if mode == "astar":
        # data_structure = PriorityQueue() 
        data_structure = PriorityQueueOptimized()  
        if isinstance(problem, CornerSearchProblem) or isinstance(problem, GemSearchProblem):
            objectives_reached = []
            # heuristic = problem.heuristic(initial_state)
            data_structure.push((initial_state
                                 , actions
                                 , objectives_reached)
                                , cost)
        else:
            data_structure.push((initial_state
                                 , actions
                                 )
                                , cost)
    else:
        data_structure = [(initial_state
                           , actions)]  #  to keep track of states
    visited = set()  # Set to keep track of visited states

    while not is_empty(data_structure):
        # print terminal line spacer empty line
        print("")
        # print terminal line separator
        print("--------------------------------------------------")
        # print_items(data_structure)
        # print_items("data_structure: ", data_structure)
        # print("data_structure: ", data_structure)

        # Pop the top state from the data_structure
        if mode == "bfs":
            current_state, actions = data_structure.pop(0)
        else:
            if isinstance(problem, CornerSearchProblem) or isinstance(problem, GemSearchProblem):
                current_state, actions, objectives_reached = data_structure.pop()
            else:
                current_state, actions = data_structure.pop()

        # Check if the current state is in the visited set
        if was(current_state, visited):
            continue
        solution = None
        if isinstance(problem, CornerSearchProblem):
            solution = check_goal_state(problem
                            , current_state
                            , actions
                            , objectives_reached)
        else:
            solution = check_goal_state(problem
                            , current_state
                            , actions
                            , None)
        if solution:
            return solution

        print("current_state: ", current_state)
        # print("actions: ", actions)
        current_state_hashable = serialize(current_state)
        visited.add(current_state_hashable)
        # print_items("visited: ", visited)

        # Add successors to data_structure
        if isinstance(problem, CornerSearchProblem) or isinstance(problem, GemSearchProblem):
            successors, objectives_reached = problem.get_successors(current_state
                                                ,visited
                                                ,objectives_reached)
        else:
            successors = problem.get_successors(current_state
                                            ,visited)
        print("")
        print("successors: ")
        # print_items("successors:", successors)
        for successor, action, cost in successors:  # assuming get_successors returns (state, action) tuples
            print(successor)
            # print("actions: ", actions)
            # print("action: ", action)
            new_actions = actions + [action]
            # print("new_actions: ", new_actions)
            if mode == "astar":
                successor_cost = problem.heuristic(successor)
                total_cost = cost + successor_cost
                if isinstance(problem, CornerSearchProblem) or isinstance(problem, GemSearchProblem):
                    # corners_reached = problem.corners_reached(successor
                    #                                           , corners_reached)
                    data_structure.push((successor
                                         , new_actions
                                         , objectives_reached)
                                        , total_cost)
                else:
                    data_structure.push((successor
                                         , new_actions)
                                        , total_cost)
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

    # world = World.from_file("cartes/1_agent/simplest")
    # world = World.from_file("cartes/1_agent/impossible_simplest")
    world = World.from_file("cartes/2_agents/impossible")
    # world = World.from_file("cartes/1_agent/zigzag")
    # world = World.from_file("cartes/2_agents/zigzag")
    # world = World.from_file("cartes/2_agents/zigzag_simpler")

    # world = World.from_file("level3")
    world.reset()

    problem = SimpleSearchProblem(world)
    # solution = dfs(problem)
    solution = astar(problem)
    print("solution: ", solution)

    # world = World.from_file("cartes/gems_simplest")
    # world = World.from_file("cartes/2_agents/zigzag")
    # world = World.from_file("cartes/2_agents/zigzag_gems")
    # world = World.from_file("cartes/gems")
    # problem = GemSearchProblem(world)
    # solution = astar(problem)
    # print("solution: ", solution)
    # check_world_done(problem, solution)
    # if world.n_gems != world.gems_collected:
    #     raise AssertionError("Your is_goal_state method is likely erroneous beacuse some gems have not been collected")



    
