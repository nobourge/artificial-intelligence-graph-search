from abc import ABC, abstractmethod
import copy
from itertools import product
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import Tuple, Iterable, Generic, TypeVar
from lle import Position, World, Action, WorldState


T = TypeVar("T")


class SearchProblem(ABC, Generic[T]):
    """
    A Search Problem is a problem that can be solved by a search algorithm.

    The generic parameter T is the type of the problem state, which must inherit from WorldState.
    """

    def __init__(self, world: World):
        self.world = world
        world.reset()
        self.initial_state = world.get_state()
        self.nodes_expanded = 0

    @abstractmethod
    def is_goal_state(self, problem_state: T) -> bool:
        """Whether the given state is the goal state"""
        

    @abstractmethod
    def get_successors(self, state: T) -> Iterable[Tuple[T, Tuple[Action, ...], float]]:
        """
        Yield all possible states that can be reached from the given world state.
        Returns
            - the new problem state
            - the joint action that was taken to reach it
            - the cost of taking the action
        """

    def heuristic(self, problem_state: T) -> float:
        return 0.0
    
    def print_state(self, problem_state: T):
        """Print the state of the world."""
        map_str = self.world.world_string
        print(map_str)


class SimpleSearchProblem(SearchProblem[WorldState]):

    def each_agent_on_different_pos(self, state: WorldState) -> bool:
        """Whether each agent is on a different position."""
        print("each_on_different_pos()")
        print("state", state)
        print("state.agents_positions", state.agents_positions)

        # Create a set of the agents' positions
        agent_positions = set(state.agents_positions)  
        # Check if the number of agents on exits is equal to the total number of agents
        # and if each agent is on a different exit
        return len(agent_positions) == len(state.agents_positions)
    
    def each_agent_on_different_exit_pos(self, state: WorldState) -> bool:
        """Whether each agent is on a different exit position."""
        print("each_agent_on_different_exit_pos()")
        print("state", state)
        print("state.agents_positions", state.agents_positions)
        print("self.world.exit_pos", self.world.exit_pos)

        agent_positions = set(state.agents_positions)  
        exit_positions = set(self.world.exit_pos)  
        
        # Intersect the sets to find agents that are on exit positions
        agents_on_exits = agent_positions.intersection(exit_positions)
        
        # Check if the number of agents on exits is equal to the total number of agents
        # and if each agent is on a different exit
        return len(agents_on_exits) == len(agent_positions) # and len(agents_on_exits) == len(exit_positions)



    def is_goal_state(self, state: WorldState) -> bool:
        """Whether the given 
        SimpleStateProblem state is the 
        SimpleSearchProblem goal state

        Hint: you can use `self.world.done()` to check if the world is done.
        """
        # is_done means the game is over, i.e. agents can no longer perform joint_actions. 
        #   This happens when an agent is dead or all agents are on fini tiles.
        # if world is done & agents are alive, then it is the goal state
        # is_done = self.world.done()
        # if is_done and self.world.agents_alive:
        #     return True
        # else:
        #     return False

        # return self.world.done() and self.world.agents_alive

        # true if all agents are on exit tiles
        return self.each_agent_on_different_exit_pos(state)

    def is_valid_joint_action(self, state: WorldState, joint_action: Tuple[Action, ...]) -> bool:
        """Whether the given joint action is valid.
        an action is valid if it is available for an agent 
        and if it does not lead the agent to be on the same position as another agent"""
        print("is_valid_joint_action()")
        print("state", state)
        print("joint_action", joint_action)
        print("state.agents_positions", state.agents_positions)
        #todo: check if the joint action is valid


        # # calculate agent positions after applying the joint action
        # agent_positions_after_joint_action = []
        # for i, agent_pos in enumerate(state.agents_positions):

        # # if 
        return False
    
    def is_valid_state(self, state: WorldState) -> bool:
        """Whether the given state is valid."""
        print("is_valid_state()")
        print("state", state)
        print("state.agents_positions", state.agents_positions)

        # Create a set of the agents' positions
        agent_positions_set = set(state.agents_positions)  
        # Check if the length of the set is equal to the total number of agents
        return len(agent_positions_set) == len(state.agents_positions)

    
    def get_valid_joint_actions(self, state: WorldState) -> Iterable[Tuple[Action, ...]]:
        """Yield all possible joint actions that can be taken from the given state.
        Hint: you can use `self.world.available_actions()` to get the available actions for each agent.
        """
        print("available_actions", self.world.available_actions())
        # For each possible joint actions set (i.e. cartesian product of the agents' actions)
        for joint_actions in product(*self.world.available_actions()):
            print("joint_actions", joint_actions)
            # Create a copy of the world state to avoid modifying the original state
            World_copy = copy.deepcopy(self.world)
            # Apply the joint_actions to the new world 
            World_copy.step(list(joint_actions))
            new_state = World_copy.get_state()
            # Check if the new state is valid
            if self.is_valid_state(new_state):
                # If so, yield the joint_actions
                yield joint_actions

    def get_successors(self, state: WorldState) -> Iterable[Tuple[WorldState, Tuple[Action, ...], float]]:
        # - N'oubliez pas de jeter un oeil aux méthodes de a classe World (set_state, done, step, available_actions, ...)
        # - Vous aurez aussi peut-être besoin de `from itertools import product`
        """Yield all possible states that can be reached from the given world state."""
        print("get_successors()")
        self.nodes_expanded += 1

        # valid_joint_actions = self.get_valid_joint_actions(state)
        # print("valid_joint_actions", valid_joint_actions)
        
        # For each possible joint actions set (i.e. cartesian product of the agents' actions)
        for joint_actions in product(*self.world.available_actions()):
        # for joint_actions in product(*valid_joint_actions):
            print("joint_actions", joint_actions)
            # Create a copy of the world state to avoid modifying the original state
            # new_state = self.world.copy()
            new_world = copy.deepcopy(self.world)
            print("new_world", new_world)
            # Apply the joint_actions to the new world 
            self.world.step(list(joint_actions))
            new_state = self.world.get_state()

            if self.is_valid_state(new_state):

                # Compute the cost of the new state
                cost = self.heuristic(new_state)
                # Yield the new state, the joint_actions taken, and the cost
                yield new_state, joint_actions, cost

    def manhattan_distance(self, pos1: Position, pos2: Position) -> float:
        """The Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    # def agent_closest_exit(self, agent_pos: Position) -> float:
    def min_distance_pairing(self
                             , list_1
                             , list_2):
        # Create a cost matrix
        cost_matrix = np.zeros((len(list_1), len(list_2)))
        for i, point1 in enumerate(list_1):
            for j, point2 in enumerate(list_2):
                cost_matrix[i, j] = ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
        
        # Hungarian algorithm: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
        # from cost_matrix, it does the pairing by minimizing the total distance
        # Use the Hungarian algorithm to find the optimal pairing
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Extract the paired points, their distances, and the minimum total distance
        paired_points = []
        distances = []
        min_total_distance = 0
        for i, j in zip(row_ind, col_ind):
            paired_points.append((list_1[i], list_2[j]))
            distances.append(cost_matrix[i, j])
            min_total_distance += cost_matrix[i, j]
        
        return paired_points, distances, min_total_distance
   
    def heuristic(self, state: WorldState) -> float:
        """Manhattan distance for each agent to the closest exit"""
        agent_positions = self.world.agents_positions
        print("agent_positions", agent_positions)
        exit_positions = self.world.exit_pos
        print("exit_positions", exit_positions)
        # for each agent, compute its closest exit, if exit 
        min_distance_pairing_result = self.min_distance_pairing(agent_positions, exit_positions)
        return min_distance_pairing_result[2]

class CornerProblemState:
    ...


class CornerSearchProblem(SearchProblem[CornerProblemState]):
    def __init__(self, world: World):
        super().__init__(world)
        self.corners = [(0, 0), (0, world.width - 1), (world.height - 1, 0), (world.height - 1, world.width - 1)]
        self.initial_state = ...

    def is_goal_state(self, state: CornerProblemState) -> bool:
        raise NotImplementedError()

    def heuristic(self, problem_state: CornerProblemState) -> float:
        raise NotImplementedError()

    def get_successors(self, state: CornerProblemState) -> Iterable[Tuple[CornerProblemState, Action, float]]:
        self.nodes_expanded += 1
        raise NotImplementedError()


class GemProblemState:
    ...


class GemSearchProblem(SearchProblem[GemProblemState]):
    def __init__(self, world: World):
        super().__init__(world)
        self.initial_state = ...

    def is_goal_state(self, state: GemProblemState) -> bool:
        raise NotImplementedError()

    def heuristic(self, state: GemProblemState) -> float:
        """The number of uncollected gems"""
        raise NotImplementedError()

    def get_successors(self, state: GemProblemState) -> Iterable[Tuple[GemProblemState, Action, float]]:
        self.nodes_expanded += 1
        raise NotImplementedError()
