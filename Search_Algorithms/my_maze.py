# This code is an exercise concerning Search Algorithms
# I am going to code a AI Search algorithm capable of taking a maze as input and solve it
# The maze is written using a txt file (., , *, A, B) and Depth-first / Breadth-first Search

# Node definition: the node is a data structure containing a current state, a parent state and
# a series of possble actions
class Node():
    def __init__(self, state, parent, action):
        self.state = state #This is actually conceived as a (i, j) tuple defined in the Maze Class
        self.parent = parent
        self.action = action # Lists of instructions defined in the Maze class

# Frontier represents the mechanism that manage the nodes and contains the logic behind the way
# we select, remove and add nodes to an ensable of elements for their examination
# Depth-first Search and Breadth-first Search Algorithms are two possibilities, I examin the DFS 
# which utilize a stack logic based on last-in first-out
class FrontierLogic():
    def __init__(self):
        self.frontier = [] # inizialisation of the frontier
    
    # Check for empty frontier: return = True means empty / return = False means full !!!
    def empty(self):
        return len(self.frontier) == 0
    
    # Adding a node to the frontier for their examination until the goal is reached
    def add(self, node):
        self.frontier.append(node)

    # Check for nodes present inside the frontier: return a list of True and False if a 
    # specific state is present inside the frontier
    def contains_state(self, state):
        return any(node.state == state for node in self.frontier) # elements of frontier will be Node classes with 'state' characteristic
    
    # Removing Logic: how we remove a node from the frontier to examine it and compare with the goal?
    # how we proceed on in filling the Frontier? DFS or BFS?
    # the return of remove() is the element under examination for each step, so the last element of the frontier itself following the DFS logic
class DepthFirstSearch(FrontierLogic):
    def remove(self):
        if self.empty():
            raise Exception("Empty Frontier!")
        else:
            last_node = self.frontier[-1] # last element is the node under examination
            self.frontier = self.frontier[:-1] # new frontier is the previous minus the last element which is under investigation
            return last_node
class BreadthFirstSearch(FrontierLogic):
    def remove(self):
        if self.empty():
            raise Exception("Empty Frontier!")
        else:
            last_node = self.frontier[0] # last element is the node under examination
            self.frontier = self.frontier[1:] # new frontier is the previous minus the last element which is under investigation
            return last_node
        
#-----------------------------------------------------------------------------------------------
####################################################
####################################################
####                                            ####
####     MMM   MMM  AAAAAAA  ZZZZZZ    EEEEE    ####
####     MM  M  MM  AA   AA      Z     EEE      ####
####     MM  M  MM  AA   AA     Z      EE       ####
####     MM     MM  AA   AA   ZZZZZZ   EEEEE    ####               
####                                            ####
####################################################
####################################################

class Maze():

    def __init__(self, filename): 
    # in the init I insert: opening the file, check the validity, check the dimensions
    # and tracking the walls
        
        # Read the file
        with open(filename) as link:
            content = link.read()
            content_rows = content.splitlines() # we have a list of strings, representing each single line
        
        # Validity Check: count how many starting and ending points are present!
        if content.count('A') != 1:
            raise Exception("More Starting points detected!!!")
        if content.count('B') != 1:
            raise Exception("More Goal points detected!!!")
        
        # Store the maze dimensions
        self.height = len(content_rows) # number of rows represents the height of the Maze
        self.width = len(content_rows[0]) #  supposing all lines are equally long, their lenght is the width of the Maze

        # Walls, Path, A and B points: *
        # States are (i, j) data!!!! so they are coordinates
        self.walls = [] # actually walls list represents the list of lists containg True of False where a wall is present
        for i in range(self.height):
            row = [] #we have to fill the row list, for every row (i) of True and False

            for j in range(self.width):
                try:
                    if content_rows[i][j] == 'A':
                        self.start_state = (i, j) # we initialize the starting position as a tuple
                        row.append(False)
                    elif content_rows[i][j] == 'B':
                        self.goal_state = (i, j)  # we initialize the final goal
                        row.append(False)
                    elif content_rows[i][j] == ' ':
                        self.path = (i, j)  # we sign the path
                        row.append(False)
                    else:
                        row.append(True)
                except IndexError:
                    row.append(False)
            
            self.walls.append(row) # at the end of each row we append the list to the wall matrix
        
        # Solution of the Maze at the beginning is obviously Empty
        # Solution will be a set of (actions, states)
        self.solution_maze = None
    
    # Printing Function: Printing the maze in the terminal directly!!!
    def print_solution(self):
        solution = self.solution_maze[1] if self.solution_maze is not None else None
        
        for i in range(self.height):
            for j in range(self.width):

                if (i, j) == self.start_state:
                    print('A', end="")
                elif (i, j) == self.goal_state:
                    print('B', end="")
                elif self.walls[i][j]:
                    print('=', end="")
                elif solution is not None and (i, j) in solution: # if in the final solution there are those coordinates
                    print('*', end="")
                else:
                    print(' ', end="") #represents the investigated or not investigated cells that do not constitute the solution of the Maze
            print()
        print()

    # Actions: possible moves that we have to describe somehow
    def Actions(self, state):
        # actions are possible movements that AI can do if it is in a specific position (i,j)
        # returns a list of (action, (i',j'))

        # I give as input (in the solver) a 'state', and set possible general moves that must be
        # validated as possible: imagine being on the extreme left of the maze or against a wall,
        # the move 'left' is not avalaible as 'possible move'
        i, j = state
        all_possible_moves = [
            ("up", (i-1, j)),
            ("down", (i+1, j)),
            ("right", (i, j+1)),
            ("left", (i, j-1))
        ]

        # real possible moves must be identified for each single state
        # possible moves must be that they do not belong to the wall (where the wall is False)
        real_possible_moves = [] 
        for action, (n,p) in all_possible_moves:
            if (0 <= n < self.height) and (0 <= p < self.width) and (self.walls[n][p] == False):
                real_possible_moves.append((action, (n,p)))
        return real_possible_moves


    # MAZE SOLVER FUNCTION
    def maze_solver(self):

        #Initialize the starting point and the Frontier
        # the state of the node is the start_state obtained from the txt file analysis in (i,j) form
        start_point = Node(state=self.start_state, parent=None, action=None) #starting node
        frontier = DepthFirstSearch() #Frontier logic: add, remove, check, empty frontier
        #frontier = BreadthFirstSearch()
        self.explored = set() #Empty explored set
        frontier.add(start_point) #adding the first node: the starting point
        self.number_exploration_steps = 0
        self.cost = 0

        # Loop until solution!!!:
        while True:
            
            # Empty Frontier check
            if frontier.empty():
                raise Exception("Empty Frontier!!! No solution to the problem!!!")
            
            # Frontier should be always full if A is present:
            # Remove the first state to check it:
            node_removed = frontier.remove() # return the last node in the frontier
            self.number_exploration_steps += 1
            
            # Goal Test: if the node_removed is the goal we have to return the whole solution
            if node_removed.state == self.goal_state:
                actions = []
                cells = []
                # Recursive cycle: we move backwards untill the starting point is reached
                while node_removed.parent is not None: # until we arrive at the starting point that have None
                    actions.append(node_removed.action)
                    cells.append(node_removed.state)
                    self.cost += 1
                    node_removed = node_removed.parent
                
                #Revering the lists to obtain the right direction of events
                actions.reverse()
                cells.reverse()
                self.solution_maze = (actions, cells)
                return

            # Explored node:
            self.explored.add(node_removed.state)

            # Adding process to the frontier to explore the Maze
            # We ask for each action and state possible in a given state removed to be analized
            # if that state is not in the frontier, and if the already explored list
            for action, state in self.Actions(node_removed.state):
                if (frontier.contains_state(state)==False) and (state not in self.explored):
                    next_node = Node(state=state, parent=node_removed, action=action)
                    frontier.add(next_node)
        
maze_filename = "maze1"
m = Maze(f"0. Search Algorithm\{maze_filename}.txt")

print("Initial Maze:\n")
m.print_solution()

print("Solved Maze:\n")
m.maze_solver()
m.print_solution()
print(f"Number of steps: {m.number_exploration_steps}")

#---------------------------------------------------------------

# GRAPHICAL REPRESENTATION
from PIL import Image, ImageDraw, ImageShow
import matplotlib.pyplot as plt
size = 50
border = 20
# Create a black canva using txt dimensions to draw squares
img = Image.new("RGBA", (m.width*size, m.height*size), "purple")
draw = ImageDraw.Draw(img)

# Filling the coordinates
solution = m.solution_maze[1] if m.solution_maze is not None else None
for i in range(m.height):
    for j in range(m.width):
        if (i, j) == m.start_state:
            fill = (0, 255, 0) # starting point is green
        elif (i, j) == m.goal_state:
            fill = (255, 0, 0)
        elif m.walls[i][j] == True:
            fill = (10, 10, 10)
        elif (i, j) in solution and solution is not None:
            fill = (220, 230, 110)
        elif (i, j) in m.explored:
            fill = (255, 255, 255)
        else:
            fill = (100, 100, 100)

        # we draw in the canva, a series of rectangles of fill colors in position given for each iteration
        draw.rectangle([(j*size, i*size), ((j+1)*size, (i+1)*size)], fill=fill)


# Plotting image using Matplotlib and not ImageShow!!!
plt.imshow(img)
plt.axis('off')
plt.title(maze_filename.upper() + " - " + f"cost: {m.cost} / explored: {m.number_exploration_steps}")
plt.show()