from search import Solution
from problem import SimpleSearchProblem

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

def check_world_done(problem: SimpleSearchProblem, solution: Solution):
    world = problem.world
    world.reset()
    for action in solution.actions:
        world.step(action)
    assert world.done
