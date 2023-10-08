from abc import ABC, abstractmethod
import copy
from itertools import product
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import Tuple, Iterable, Generic, TypeVar
from lle import Position, World, Action, WorldState

from travel_sales_man import balanced_multi_salesmen_greedy_tsp
from utils import print_items


# T = TypeVar("T")
T = TypeVar('T', bound=WorldState)  # Declare the generic type variable with a default bound



def serialize(world_state: WorldState
              ,objectives_reached: list[Position] = None
              ) -> tuple:
    """Serialize the given world state.
    Args:
        world_state: the world state to serialize.
    Returns:
        A tuple that represents the given world state.
    """
    if objectives_reached:
        return (tuple(world_state.agents_positions), tuple(world_state.gems_collected), tuple(objectives_reached))
    else:
        return (tuple(world_state.agents_positions), tuple(world_state.gems_collected))

def was(state: WorldState
        , objectives_reached: list[Position]
        , visited: set) -> bool:
    # print("was()")
    # print("state", state)
    return serialize(state, objectives_reached) in visited

def min_distance_pairing(list_1
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
        print("paired_points", paired_points)
        print("distances", distances)
        
        return paired_points, distances, min_total_distance

def min_distance_road(positions: list[Position]) -> float:
    """The minimum distance between two positions in a list of positions"""
    min_distance = np.inf
    for i, pos1 in enumerate(positions):
        for j, pos2 in enumerate(positions):
            if i != j:
                distance = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
    return min_distance
class SearchProblem(ABC, Generic[T]):
    """
    A Search Problem is a problem that can be solved by a search algorithm.

    The generic parameter T is the type of the problem state, 
    which must inherit from WorldState.
    """

    def __init__(self, world: World):
        self.world = world
        world.reset()
        self.initial_state = world.get_state()
        self.objectives = []

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
class SimpleSearchProblem(SearchProblem[T], Generic[T]):  # Use Generic[T] to make the class generic

    def no_duplicate_in(self, agents_positions: list[Position]) -> bool:
        """Whether each agent is on a different position."""
        agents_positions_set = set(agents_positions)  
        # Check if the number of agents on exits is equal to the total number of agents
        # and if each agent is on a different exit
        result = len(agents_positions) == len(agents_positions_set)
        return result
    
    def agents_each_on_different_exit_pos(self, state: WorldState) -> bool:
        """Whether each agent is on a different exit position."""
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
        SimpleSearchProblem goal state.
        True if all agents are on exit tiles
        """
        return self.agents_each_on_different_exit_pos(state)
    
    def agent_position_after_action(self, agent_pos: Position, action: Action) -> Position:
        """The position of an agent after applying the given action."""
        # print("agent_position_after_action()")
        # print("agent_pos", agent_pos)
        # print("action", action)
        agent_pos_after_action = None
        # Apply the action to the agent's position
        if action == Action.NORTH:
            agent_pos_after_action = (agent_pos[0] - 1, agent_pos[1])
        elif action == Action.SOUTH:
            agent_pos_after_action = (agent_pos[0] + 1, agent_pos[1])
        elif action == Action.WEST:
            agent_pos_after_action = (agent_pos[0], agent_pos[1] - 1)
        elif action == Action.EAST:
            agent_pos_after_action = (agent_pos[0], agent_pos[1] + 1)
        elif action == Action.STAY:
            agent_pos_after_action = (agent_pos[0], agent_pos[1])
        else:
            raise ValueError("Invalid action")
        return agent_pos_after_action

    def are_valid_joint_actions(self, state: WorldState, joint_actions: Tuple[Action, ...]) -> bool:
        """Whether the given joint actions are valid.
        an action is valid if it is available for an agent 
        and if it does not lead the agent to be on the same position as another agent"""
        # print("are_valid_joint_actions()")
        # print("state", state)
        # print("joint_actions", joint_actions)
        # print("state.agents_positions", state.agents_positions)
        # # calculate agent positions after applying the joint action
        agents_positions_after_joint_actions = []
        for i, agent_pos in enumerate(state.agents_positions):
            agent_pos_after_action = self.agent_position_after_action(agent_pos, joint_actions[i])
            agents_positions_after_joint_actions.append(agent_pos_after_action)
        return self.no_duplicate_in(agents_positions_after_joint_actions)

    
    def get_valid_joint_actions(self
                                , state: WorldState
                                , available_actions: Tuple[Tuple[Action, ...], ...]) -> Iterable[Tuple[Action, ...]]:
        """Yield all possible joint actions that can be taken from the given state.
        Hint: you can use `self.world.available_actions()` to get the available actions for each agent.
        """
        # print("available_actions", available_actions)
        # cartesian product of the agents' actions
        for joint_actions in product(*available_actions):
            # print("joint_actions", joint_actions)
           
            if self.are_valid_joint_actions(state, joint_actions):
                yield joint_actions
    
    def get_successor_state(self
                            , state: WorldState
                            , joint_actions: Tuple[Action, ...]) -> WorldState:
        """The successor state of the given state after applying the given joint actions."""
        self.world.set_state(state)
        self.world.step(list(joint_actions))
        successor_state = self.world.get_state()
        # print("successor_state", successor_state)
        return successor_state

    def get_successors(self
                       , state: WorldState
                       , visited: set = None
                       , objectives_reached_before_successor: list[Position] = None
                       ):
        # - N'oubliez pas de jeter un oeil aux méthodes de la classe World (set_state, done, step, available_actions, ...)
        # - Vous aurez aussi peut-être besoin de `from itertools import product`
        """Yield all possible states that can be reached from the given world state."""
        # print("get_successors()")
        # print("state", state)
        if visited is None: # for tests
            visited = set()
        # print_items("visited", visited)
        # print("objectives_reached_before_successor", objectives_reached_before_successor)
        self.nodes_expanded += 1
        real_state = self.world.get_state()
        # simulation = copy.deepcopy(self.world)
        self.world.set_state(state)
        # For each possible joint actions set (i.e. cartesian product of the agents' actions)
        available_actions = self.world.available_actions()
        # print("available_actions", available_actions)
        valid_joint_actions = self.get_valid_joint_actions(state, available_actions)
        # print_items("valid_joint_actions", valid_joint_actions)
        # print("successors: ")

        i = 0
        for joint_actions in valid_joint_actions:
            i += 1
            objectives_reached_by_successor = objectives_reached_before_successor
            # print("successor", i)
            # print(" joint_actions", joint_actions)
            # print("objectives_reached_before_successor", objectives_reached_before_successor)
            # print("objectives_reached_by_successor", objectives_reached_by_successor)
            # simulation_copy = copy.deepcopy(simulation)
            try:
                successor_state = self.get_successor_state(state, joint_actions)
            except ValueError:
                # print("ValueError: World is done, cannot step anymore")
                continue
            if isinstance(self, CornerSearchProblem):
                objectives_reached_by_successor = self.update_corners_reached(copy.deepcopy(objectives_reached_before_successor)
                                                              , joint_actions
                                                              , successor_state.agents_positions
                                                              )
            elif isinstance(self, GemSearchProblem):
                objectives_reached_by_successor = self.update_gems_collected(copy.deepcopy(objectives_reached_before_successor)
                                                              , joint_actions
                                                              , successor_state.agents_positions
                                                              )
            if was(successor_state
                    , objectives_reached_by_successor
                   , visited):
                continue
            if isinstance(self, CornerSearchProblem):
                # Compute the cost of the new state
                cost = self.heuristic(successor_state, objectives_reached_by_successor)
                yield successor_state, joint_actions, cost, objectives_reached_by_successor
            elif isinstance(self, GemSearchProblem):
                # Compute the cost of the new state
                cost = self.heuristic(successor_state, objectives_reached_by_successor)
                yield successor_state, joint_actions, cost, objectives_reached_by_successor
            elif not isinstance(self, CornerSearchProblem) and not isinstance(self, GemSearchProblem):
                # Compute the cost of the new state
                cost = self.heuristic(successor_state)
                yield successor_state, joint_actions, cost #todo must not change for test
        self.world.set_state(real_state)

    def manhattan_distance(self, pos1: Position, pos2: Position) -> float:
        """The Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def average_manhattan_distance_from_agents_to_exits(self, state: WorldState) -> float:
        """The average Manhattan distance from each agent to each exit
        divided by the number of agents"""
        # Create a list of the agents' positions
        agents_positions = state.agents_positions
        # Create a list of the exits' positions
        exit_positions = self.world.exit_pos
        # For each agent, compute its Manhattan distance to each exit
        total_distance = 0
        for agent_pos in agents_positions:
            for exit_pos in exit_positions:
                total_distance += self.manhattan_distance(agent_pos, exit_pos)
        # Divide the total distance by the number of agents
        average_distance = total_distance / len(agents_positions)
        return average_distance
    
    def heuristic(self
                  , state: WorldState
                  , last_actions: Tuple[Action, ...] = None
                  ) -> float:
        """Manhattan distance for each agent to the closest exit"""
        agent_positions = self.world.agents_positions
        exit_positions = self.world.exit_pos
        total_distance = self.average_manhattan_distance_from_agents_to_exits(state)
        # problem:  if agent has to get away from exit to get around a wall to reach the exit, a star chooses for him to stay in place
        # tie breaking with the last actions 
        # a last action STAY agent (if he had other options, but we don't take that into account here for simplicity) could have moved
        # so he actually lost 1 turn
        # for each action, if it was STAY
        # and if the agent is not on an exit
        # , add 1 to the total distance
        if last_actions:
            for i, action in enumerate(last_actions):
                if action == Action.STAY and agent_positions[i] not in exit_positions:
                    total_distance += 1
        return total_distance

class CornerProblemState:
    def __init__(self, world_state: WorldState):
        self.agents_positions = world_state.agents_positions
        self.gems_collected = world_state.gems_collected
        self.world_state = world_state


# class CornerSearchProblem(SearchProblem[CornerProblemState]):
class CornerSearchProblem(SimpleSearchProblem[WorldState]):
    """Problème qui consiste à passer par les quatre coins du World 
    puis d’atteindre une sortie."""
    def __init__(self, world: World):
        super().__init__(world)
        self.corners = [(0, 0), (0, world.width - 1), (world.height - 1, 0), (world.height - 1, world.width - 1)]
        # self.corners_reached = []
        self.initial_state = world.get_state()
        # self.initial_state = CornerProblemState(world.get_state())
        self.corners_to_exits_minimum_distance_pairing = min_distance_pairing(self.corners, self.world.exit_pos)  
        self.agents_to_corners_minimum_distance_pairing = min_distance_pairing(self.world.agents_positions, self.corners)

    def corners_to_exits_manhattan_distances(self) -> list[float]:
        corners_to_exits_manhattan_distances = []
        exit_positions = self.world.exit_pos
        min_distance_pairing_result, distances, min_total_distance = min_distance_pairing(self.corners, exit_positions)
        return distances
    
    def update_corner_reached(self
                               , corners_reached: list[Position]
                               , agent_position: Position) -> list[Position]:
        """Update the list of corners reached"""
        corners_reached.append(agent_position)
        return corners_reached
    
    def update_corners_reached(self
                                 , corners_reached: list[Position]
                                    , joint_actions: Tuple[Action, ...]
                                    , agent_positions: list[Position]) -> list[Position]:
        """Update the list of corners reached"""

        for action in joint_actions:
            if action != Action.STAY:
                agent_position = agent_positions[joint_actions.index(action)]
                if agent_position not in corners_reached and agent_position in self.corners:
                    corners_reached.append(agent_position)

        return corners_reached

    def all_corners_reached(self
                            , state
                            , corners_reached: list[Position]) -> bool:
        """Whether all corners are reached"""
        # print("all_corners_reached()")
        # print("state", state)

        return len(corners_reached) == len(self.corners)

    def is_goal_state(self
                      , state: WorldState
                      , corners_reached: list[Position]) -> bool:
        """Whether the given state is the goal state
        """
        # # if a new position is reached, check if it is a corner
        # # if it is, add it to the list of corners reached
        # agents_positions = state.agents_positions
        # for agent_pos in agents_positions:
        #     if agent_pos not in corners_reached and agent_pos in self.corners:
        #         corners_reached.append(agent_pos)

        # if all corners are reached, check if it is the goal state

        return self.all_corners_reached(state, corners_reached) and SimpleSearchProblem.is_goal_state(self, state)

    # def heuristic(self, problem_state: CornerProblemState) -> float:
    def heuristic(self
                  , state: WorldState
                  , corners_reached: list[Position]
                  ) -> float:
        """"""
        # print("heuristic()")
        # print("state", state)
        # print("corners_reached", corners_reached)
        # print("state.agents_positions", state.agents_positions)
        # print("self.world.exit_pos", self.world.exit_pos)

        # Create a list of the agents' positions
        agents_positions = state.agents_positions
        # print("agents_positions", agents_positions)

        # Create a list of the corners to reach
        corners_to_reach = [corner for corner in self.corners if corner not in corners_reached]
        # print("corners_to_reach", corners_to_reach)

        cost = balanced_multi_salesmen_greedy_tsp(corners_to_reach
                                                  , len(agents_positions)
                                                  , agents_positions
                                                  , self.world.exit_pos
                                                  )[2]
        return cost

class GemProblemState:
    """The state of the GemSearchProblem"""
    def __init__(self, world_state: WorldState):
        self.agents_positions = world_state.agents_positions
        self.gems_collected = world_state.gems_collected
        self.world_state = world_state

class GemSearchProblem(SimpleSearchProblem[WorldState]):
    """Modéliez le problème qui consiste à collecter toutes les gemmes de l’environnement 
    puis à rejoindre les cases de sortie"""
    def __init__(self, world: World):
        super().__init__(world)
        self.initial_state = world.get_state()

    def update_gems_collected(self
                                , gems_collected: list[Position]
                                , joint_actions: Tuple[Action, ...]
                                , agent_positions: list[Position]) -> list[Position]:
        """Update the list of gems collected"""
        for action in joint_actions:
            if action != Action.STAY:
                agent_position = agent_positions[joint_actions.index(action)]
                if agent_position not in gems_collected and agent_position in [pos for pos, gem in self.world.gems]:
                    gems_collected.append(agent_position)

        return gems_collected

    def all_gems_collected(self, state):
        return sum(state.gems_collected) == self.world.n_gems

    def is_goal_state(self
                      , state
                      ) -> bool:
        return self.all_gems_collected(state) and super().is_goal_state(state)

    def heuristic(self
                  , state: WorldState
                  , gems_collected
                  ) -> float:
        """The distance of each agent to each uncollected gem and to the closest exit
        when all gems are collected, the distance of each agent to the closest exit"""
        # print("heuristic()")
        # print("state", state)
        # print("state.agents_positions", state.agents_positions)
        # print("state.gems_collected", state.gems_collected)
        # print("self.world.exit_pos", self.world.exit_pos)
        # print("self.world.n_gems", self.world.n_gems)

        gems_to_collect = [gem[0] for gem in self.world.gems if not gem[0] in gems_collected]
        # print("gems_to_collect", gems_to_collect)

        cost = balanced_multi_salesmen_greedy_tsp(gems_to_collect
                                                  , len(state.agents_positions)
                                                  , state.agents_positions
                                                  , self.world.exit_pos
                                                  )[2]

        return cost

    