import heapq
from enum import Enum

num_directions = 4

class Direction(Enum):
    right = (0, (1, 0))
    left = (1, (-1, 0))
    up = (2, (0, 1))
    down = (3, (0, -1))

DirectionMap = {
    0: Direction.right,
    1: Direction.left,
    2: Direction.up,
    3: Direction.down,
}

# Movement directions.
num_move_dirs = 3

class MoveDirection(Enum):
    left = 0
    forward = 1
    right = 2

# Types of blocks available
num_block_type = 4

class BlockType(Enum):
    empty = 0
    solid = 1
    start = 2
    end = 3

class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)
