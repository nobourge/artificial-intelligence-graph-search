from abc import ABC, abstractmethod
import copy
from itertools import product
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import Tuple, Iterable, Generic, TypeVar
from lle import Position, World, Action, WorldState


# T = TypeVar("T")
T = TypeVar('T', bound=WorldState)  # Declare the generic type variable with a default bound

def serialize(world_state: WorldState) -> tuple:
    return (tuple(world_state.agents_positions), tuple(world_state.gems_collected))

def was(world_state: WorldState, visited: set) -> bool:
    return serialize(world_state) in visited

# function to print visited set or stack items in terminal
def print_items(title, items, transform=None) -> None:
    """Prints items in terminal
    Args:
        items: items to print
    T is a generic type variable
    possible types for T:
    set, list, tuple, dict, etc."""
    print(title)
    for item in items:
        print(item)
    print("")

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

# class SimpleSearchProblem(SearchProblem[WorldState]):
# class SimpleSearchProblem(SearchProblem[T] = WorldState, Generic[T] = WorldState):  # Use Generic[T] to make the class generic
class SimpleSearchProblem(SearchProblem[T], Generic[T]):  # Use Generic[T] to make the class generic

    def no_duplicate_in(self, agents_positions: list[Position]) -> bool:
        """Whether each agent is on a different position."""
        # print("each_on_different_pos()")

        # Create a set of the agents' positions
        agents_positions_set = set(agents_positions)  
        # Check if the number of agents on exits is equal to the total number of agents
        # and if each agent is on a different exit
        result = len(agents_positions) == len(agents_positions_set)
        # print("result", result)
        # print("agents")
        return result
    
    def agents_each_on_different_exit_pos(self, state: WorldState) -> bool:
        """Whether each agent is on a different exit position."""
        # print("each_agent_on_different_exit_pos()")
        # print("state", state)
        # print("state.agents_positions", state.agents_positions)
        # print("self.world.exit_pos", self.world.exit_pos)

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
        

        # true if all agents are on exit tiles
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
        print("agents_positions_after_joint_actions", agents_positions_after_joint_actions)

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
                # If so, yield the joint_actions
                yield joint_actions
    
    def get_successor_state(self
                            , state: WorldState
                            , joint_actions: Tuple[Action, ...]) -> WorldState:
        """The successor state of the given state after applying the given joint actions."""
        self.world.set_state(state)
        print("joint_actions", joint_actions)
        # Apply the joint_actions to the new world 
        print("world.step()")
        self.world.step(list(joint_actions))
        successor_state = self.world.get_state()
        print("successor_state", successor_state)
        return successor_state

    def get_successors(self
                       , state: WorldState
                       , visited: set = None
                       , corners_reached: list[Position] = None
                       ) -> Iterable[Tuple[WorldState, Tuple[Action, ...]
                                           , float
                                           , list[Position]]]:
        # - N'oubliez pas de jeter un oeil aux méthodes de la classe World (set_state, done, step, available_actions, ...)
        # - Vous aurez aussi peut-être besoin de `from itertools import product`
        """Yield all possible states that can be reached from the given world state."""
        # print("get_successors()")
        self.nodes_expanded += 1
        real_state = self.world.get_state()
        # simulation = copy.deepcopy(self.world)
        self.world.set_state(state)
        # For each possible joint actions set (i.e. cartesian product of the agents' actions)
        available_actions = self.world.available_actions()
        print("available_actions", available_actions)
        valid_joint_actions = self.get_valid_joint_actions(state, available_actions)
        # print_items("valid_joint_actions", valid_joint_actions)
        for joint_actions in valid_joint_actions:
            # simulation_copy = copy.deepcopy(simulation)
            try:
                successor_state = self.get_successor_state(state, joint_actions)
            except ValueError:
                print("ValueError: World is done, cannot step anymore")
                continue
            if was(successor_state, visited):
                continue
            # Compute the cost of the new state
            cost = self.heuristic(successor_state)
            # Yield the new state, the joint_actions taken, and the cost
            yield successor_state, joint_actions, cost
            print("hello")
        print("bye")
        self.world.set_state(real_state)
        print("self.world.get_state()", self.world.get_state())


    def manhattan_distance(self, pos1: Position, pos2: Position) -> float:
        """The Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    # def agent_closest_exit(self, agent_pos: Position) -> float:
    
   
    def heuristic(self, state: WorldState) -> float:
        """Manhattan distance for each agent to the closest exit"""
        agent_positions = self.world.agents_positions
        print("agent_positions", agent_positions)
        exit_positions = self.world.exit_pos
        # print("exit_positions", exit_positions)
        # for each agent, compute its closest exit, if exit 
        min_distance_pairing_result = min_distance_pairing(agent_positions, exit_positions)
        return min_distance_pairing_result[2]

class CornerProblemState:
    def __init__(self, world_state: WorldState):
        self.agents_positions = world_state.agents_positions
        self.gems_collected = world_state.gems_collected
        self.world_state = world_state


class CornerSearchProblem(SearchProblem[CornerProblemState]):
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
    
    def min_distance_road_between_corners(self) -> float:
        """The minimum distance road to reach all corners"""
        min_distance = np.inf

        return min_distance

    def corners_to_exits_manhattan_distances(self) -> list[float]:
        corners_to_exits_manhattan_distances = []
        exit_positions = self.world.exit_pos
        min_distance_pairing_result, distances, min_total_distance = min_distance_pairing(self.corners, exit_positions)
        # for corner in self.corners:
        #     corner_to_exits_manhattan_distances.
        print("distances", distances)
        return distances
    
    def update_corners_reached(self
                               , corners_reached: list[Position]
                               , agent_position: Position) -> list[Position]:
        """Update the list of corners reached"""
        corners_reached.append(agent_position)
        return corners_reached
    
    # def check_corners_reached(self
    #                             , corners_reached: list[Position]
    #                             , state: WorldState) -> bool:
        

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
        # if a new position is reached, check if it is a corner
        # if it is, add it to the list of corners reached
        agents_positions = state.agents_positions
        for agent_pos in agents_positions:
            if agent_pos not in corners_reached and agent_pos in self.corners:
                corners_reached.append(agent_pos)

        # if all corners are reached, check if it is the goal state

        return self.all_corners_reached(state, corners_reached) and SimpleSearchProblem.is_goal_state(self, state)

    # def heuristic(self, problem_state: CornerProblemState) -> float:
    def heuristic(self, state: WorldState) -> float:
        """minimum distance pairing between agents and corners
        + minimum distance pairing between corners and exits
        
        The distance of each agent to its corner road closest corner and to the closest exit"""
        # print("heuristic()")
        # print("state", state)
        # print("state.agents_positions", state.agents_positions)
        # print("state.gems_collected", state.gems_collected)
        # print("self.world.exit_pos", self.world.exit_pos)
        # print("self.world.n_gems", self.world.n_gems)

        cost = 0.0

        # Create a list of the agents' positions
        agents_positions = state.agents_positions
        # print("agents_positions", agents_positions)

        # if len(agents_positions) == 0:
        #     return cost
        # elif len(agents_positions) == 1:


        # minimum distance pairing between agents and corners
        agents_to_corners_min_distance_pairing_result = min_distance_pairing(agents_positions, self.corners)
        # print("agents_to_corners_min_distance_pairing_result", agents_to_corners_min_distance_pairing_result)
        # add the minimum total distance to the heuristic
        min_total_distance = agents_to_corners_min_distance_pairing_result[2]
        corners_cost = min_total_distance
        cost += corners_cost

        # Create a list of the exit positions
        exit_positions = self.world.exit_pos
        # print("exit_positions", exit_positions)
        # minimum distance pairing between agents and exits
        corners_to_exits_min_distance_pairing_result = min_distance_pairing(self.corners, exit_positions)
        # print("corners_to_exits_min_distance_pairing_result", corners_to_exits_min_distance_pairing_result)
        # add the minimum total distance to the heuristic
        min_total_distance = corners_to_exits_min_distance_pairing_result[2]
        exit_cost = min_total_distance
        cost += exit_cost

        return cost

    def get_successors(self, state: CornerProblemState) -> Iterable[Tuple[CornerProblemState, Action, float]]:
        self.nodes_expanded += 1
        # use SimpleSearchProblem.get_successors()
        for successor in SimpleSearchProblem.get_successors(self, state):
            yield successor

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

    def all_gems_collected(self, state):
        return sum(state.gems_collected) == self.world.n_gems

    def is_goal_state(self, state):
        return self.all_gems_collected(state) and super().is_goal_state(state)

    def heuristic(self, state: WorldState) -> float:
        """The distance of each agent to each uncollected gem and to the closest exit
        when all gems are collected, the distance of each agent to the closest exit"""
        print("heuristic()")
        print("state", state)
        print("state.agents_positions", state.agents_positions)
        print("state.gems_collected", state.gems_collected)
        print("self.world.exit_pos", self.world.exit_pos)
        print("self.world.n_gems", self.world.n_gems)

        cost = 0.0

        # Create a list of the agents' positions
        agents_positions = state.agents_positions
        print("agents_positions", agents_positions)

        if not self.all_gems_collected(state):
            # Create a list of the uncollected gems
            uncollected_gems_positions = []
            for i, gem_collected in enumerate(state.gems_collected):
                if gem_collected == 0:
                    uncollected_gems_positions.append(self.world.gems[i][0])
            print("uncollected_gems_positions", uncollected_gems_positions)

            # minimum distance pairing between agents and uncollected gems
            agents_to_gems_min_distance_pairing_result = min_distance_pairing(agents_positions, uncollected_gems_positions)
            print("agents_to_gems_min_distance_pairing_result", agents_to_gems_min_distance_pairing_result)
            # add the minimum total distance to the heuristic
            min_total_distance = agents_to_gems_min_distance_pairing_result[2]
            # exit < gem collection
            uncollected_gems_cost = min_total_distance*2
            cost += uncollected_gems_cost
    

        # Create a list of the exit positions
        exit_positions = self.world.exit_pos
        # print("exit_positions", exit_positions)
        # minimum distance pairing between agents and exits
        agents_to_exits_min_distance_pairing_result = min_distance_pairing(agents_positions, exit_positions)
        print("agents_to_exits_min_distance_pairing_result", agents_to_exits_min_distance_pairing_result)
        # add the minimum total distance to the heuristic
        min_total_distance = agents_to_exits_min_distance_pairing_result[2]
        exit_cost = min_total_distance
        cost += exit_cost

        return cost

    