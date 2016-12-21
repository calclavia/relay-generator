# Given a world, produce all possible solutions.
from .util import PriorityQueue

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def search(problem, heuristic=nullHeuristic):
    """
    A* search algorithm.
    Search the node that has the lowest combined cost and heuristic first.
    """

    # Data structure used for search
    search = PriorityQueue()
    # A dictionary that maps a state from the previous state
    come_from = {}
    # A dictionary that maps a state and the action that led to it
    action_map = {}
    # A dictionary that maps a state to its cost
    g_score = {}
    # A set containing visited states
    visited = set()

    # Initialize starting g-score to 0
    g_score[problem.get_start()] = 0
    search.push(problem.get_start(), 0)

    # Keep searching while we have more positions
    while not search.isEmpty():
        current = search.pop()
        visited.add(current)

        if problem.is_goal(current):
            # Compute path of nodes by backtracking
            path = backTrack(current, come_from) + [current]
            # Return a list of actions to reach a goal
            return [action_map[n] for n in path[1:]]
        else:
            g_current = g_score[current]
            # Iterate all successors
            for succ, action, cost in problem.get_next(current):
                if succ not in visited:
                    # Compute current g-score by adding it with the cost to take the path
                    new_g = g_current + cost

                    if succ not in g_score or new_g < g_score[succ]:
                        # We either found a better path, or found this path the first time
                        # Set this position's prev pos, the action it came from and the g_score
                        come_from[succ] = current
                        action_map[succ] = action
                        g_score[succ] = new_g

                    # f(n) = g(n) + h(n)
                    newfScore = new_g + heuristic(succ, problem)
                    search.update(succ, newfScore)
    # No solution
    return None

def backTrack(target, come_from):
    """
    Takes a target and returns a list based on a dictionary "come_from"
    that back tracks all paths visited to the starting point.
    """
    if target not in come_from:
        return []
    prev = come_from[target]
    return backTrack(prev, come_from) + [prev]
