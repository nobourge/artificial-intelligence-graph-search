import sys
sys.path.append("D:\\bourg\Documents\GitHub\\artificial-intelligence-graph-search\\1-recherche-v3\src\search.py")  # Replace "/path/to/folder" with the actual path to the directory containing search.py
for p in sys.path:
    print(p)
from lle import World, Action
from search import astar
from problem import SimpleSearchProblem

from .utils import check_world_done


def test_1_agent_empty():
    world = World.from_file("cartes/1_agent/vide")
    problem = SimpleSearchProblem(world)
    solution = astar(problem)
    assert solution.n_steps == 8
    assert solution.actions.count((Action.EAST,)) == 6
    assert solution.actions.count((Action.SOUTH,)) == 2
    check_world_done(problem, solution)


def test_1_agent_zigzag():
    world = World.from_file("cartes/1_agent/zigzag")
    problem = SimpleSearchProblem(world)
    solution = astar(problem)
    assert solution.n_steps == 19
    assert solution.actions == [
        (Action.NORTH,),
        (Action.NORTH,),
        (Action.EAST,),
        (Action.EAST,),
        (Action.SOUTH,),
        (Action.SOUTH,),
        (Action.EAST,),
        (Action.EAST,),
        (Action.NORTH,),
        (Action.NORTH,),
        (Action.EAST,),
        (Action.EAST,),
        (Action.SOUTH,),
        (Action.SOUTH,),
        (Action.EAST,),
        (Action.EAST,),
        (Action.NORTH,),
        (Action.NORTH,),
        (Action.EAST,),
    ]
    check_world_done(problem, solution)


def test_1_agent_impossible():
    world = World.from_file("cartes/1_agent/impossible")
    problem = SimpleSearchProblem(world)
    assert astar(problem) is None
    assert problem.nodes_expanded > 0


def test_2_agents_empty():
    world = World.from_file("cartes/2_agents/vide")
    problem = SimpleSearchProblem(world)
    solution = astar(problem)
    assert solution.n_steps == 8
    check_world_done(problem, solution)


def test_2_agents_zigzag():
    world = World.from_file("cartes/2_agents/zigzag")
    world.reset()
    problem = SimpleSearchProblem(world)
    solution = astar(problem)
    assert solution.n_steps == 12
    check_world_done(problem, solution)


def test_2_agents_impossible():
    world = World.from_file("cartes/2_agents/impossible")
    problem = SimpleSearchProblem(world)
    assert astar(problem) is None
    assert problem.nodes_expanded > 0


def test_level3():
    world = World.from_file("level3")
    problem = SimpleSearchProblem(world)
    solution = astar(problem)
    check_world_done(problem, solution)

#main
if __name__ == "__main__":
    print(sys.path)

    test_1_agent_empty()
    test_1_agent_zigzag()
    test_1_agent_impossible()
    test_2_agents_empty()
    test_2_agents_zigzag()
    test_2_agents_impossible()
    test_level3()
    print("ok") 

    