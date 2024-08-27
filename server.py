from flask import Flask, jsonify
import agentpy as ap
import numpy as np
import heapq

app = Flask(__name__)

class HarvestModel(ap.Model):
    def a_star(self, grid, start, goal, invalid_values):
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        open_list = []
        heapq.heappush(open_list, (0, start))

        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        came_from = {}

        while open_list:
            _, current = heapq.heappop(open_list)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            for direction in directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])

                if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and grid[neighbor[0]][neighbor[1]] not in invalid_values:
                    tentative_g_score = g_score[current] + 1

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_list, (f_score[neighbor], neighbor))
        return None

    def setup(self):
        self.truck_agents = [Harvester(self, ((self.p['size'][0] - 2), 1)), Collecter(self, ((self.p['size'][0] - 2), 0))]
        
        # Initialize grid matrix
        self.aux_grid_matrix = np.zeros(self.p.size, dtype=int)

        self.path = []  # Store the A* path here
        self.path_index = 0  # Track current position in the A* path
        self.collecting = False  # Track whether the collector is moving along the A* path
        self.crops_collected = set()  # Track which crops have been collected
        self.total_crops = 0  # Track the total number of crops

        self.harvester_path = []  # Store harvester's path
        self.collector_path = []  # Store collector's path

        self.custom_t = 0

        # List of crop coordinates
        self.lista_coordenadas = []

        for i in range(1, (self.p['size'][0] - 1), 1):
            if i % 2 != 0:
                for j in range((self.p['size'][0] - 2), 0, -1):
                    self.lista_coordenadas.append((j, i))
            else:
                for j in range(1, (self.p['size'][0] - 1), 1):
                    self.lista_coordenadas.append((j, i))

        # List of obstacle coordinates
        self.lista_obstaculos = []

        # Coordinates to exclude
        exclude_coords = {((self.p['size'][0] - 2), 1), ((self.p['size'][0] - 2), 0)}  # Exclude the starting positions

        # Create obstacles at random coordinates
        while len(self.lista_obstaculos) < self.p['obstacles']:
            x = np.random.randint(0, self.p['size'][0] - 1)
            y = np.random.randint(0, self.p['size'][1] - 1)
            coord = (x, y)
            if coord not in exclude_coords and coord not in self.lista_obstaculos:
                self.lista_obstaculos.append(coord)

        self.truck_agents[1].capacity = self.p['capacity']  # Initial capacity

        # Remove obstacles from the list of coordinates
        for i in range(len(self.lista_obstaculos)):
            if self.lista_obstaculos[i] in self.lista_coordenadas:
                self.lista_coordenadas.remove(self.lista_obstaculos[i])

        # Count the total number of crops
        self.total_crops = len(self.lista_coordenadas)

        # Mark the grid with the obstacles
        for i in range(len(self.lista_obstaculos)):
            self.aux_grid_matrix[self.lista_obstaculos[i]] = -1

        # Mark the grid with the crops
        for i in range(len(self.lista_coordenadas)):
            self.aux_grid_matrix[self.lista_coordenadas[i]] = 1

        # Mark the grid with the initial positions of the harvester and collector
        self.aux_grid_matrix[self.truck_agents[0].index] = 5
        self.aux_grid_matrix[self.truck_agents[1].index] = 10

        # Save the initial state of the grid
        self.initial_grid = self.aux_grid_matrix.copy()

    def step(self):
        if self.collecting:
            # Move the collector along the A* path one step at a time
            if self.path_index < len(self.path):
                step = self.path[self.path_index]

                # Record the path for the collector
                self.collector_path.append(step)

                if self.aux_grid_matrix[step] == 1:
                    self.truck_agents[1].next_index_was1 = True
                else:
                    self.truck_agents[1].next_index_was1 = False

                self.truck_agents[1].last_index = self.truck_agents[1].index
                self.truck_agents[1].index = step

                # Update the grid
                if self.truck_agents[1].last_index != self.truck_agents[1].index:
                    if self.truck_agents[1].next_index_was1:
                        self.aux_grid_matrix[self.truck_agents[1].last_index] = 1
                    else:
                        self.aux_grid_matrix[self.truck_agents[1].last_index] = 0

                self.aux_grid_matrix[self.truck_agents[1].index] = 10

                self.path_index += 1  # Move to the next step in the path

            if self.path_index == len(self.path):
                # Finished moving along the A* path
                self.truck_agents[1].capacity = self.p['capacity']
                self.collecting = False

        else:
            if self.truck_agents[1].capacity == 0 and self.custom_t > 0:
                start = self.truck_agents[1].index
                goal = (0, self.p['size'][0] - 1)
                path_to_goal = self.a_star(self.aux_grid_matrix, start, goal, {5, -1}) or []
                path_back = self.a_star(self.aux_grid_matrix, goal, start, {5, -1}) or []

                # Combine the path to the goal and back to the start
                self.path = path_to_goal + path_back

                if len(self.path) > 1:
                    self.path = self.path[1:]

                self.path_index = 0
                self.collecting = True
            else:
                # Move both agents together

                path_to_tractor = self.a_star(self.aux_grid_matrix, self.truck_agents[1].index, self.truck_agents[0].index, {-1})

                while self.custom_t < len(self.lista_coordenadas) and self.lista_coordenadas[self.custom_t] in self.crops_collected:
                    self.custom_t += 1

                path_to_crop = self.a_star(self.aux_grid_matrix, self.truck_agents[0].index, self.lista_coordenadas[self.custom_t], {-1, 10})

                self.truck_agents[0].last_index = self.truck_agents[0].index
                self.truck_agents[1].last_index = self.truck_agents[1].index

                if path_to_crop:
                    if len(path_to_crop) > 1:
                        self.truck_agents[0].index = path_to_crop[1]
                        self.harvester_path.append(self.truck_agents[0].index)

                if self.truck_agents[0].index in self.lista_coordenadas:
                    self.crops_collected.add(self.truck_agents[0].index)

                if path_to_tractor:
                    if len(path_to_tractor) > 1 and path_to_tractor[1] != self.truck_agents[0].index:
                        self.truck_agents[1].index = path_to_tractor[1]
                        self.collector_path.append(self.truck_agents[1].index)

                self.aux_grid_matrix[self.truck_agents[0].last_index] = 0
                self.aux_grid_matrix[self.truck_agents[1].last_index] = 0

                self.aux_grid_matrix[self.truck_agents[0].index] = 5
                self.aux_grid_matrix[self.truck_agents[1].index] = 10

                self.truck_agents[1].capacity -= 1

    def update(self):
        if self.total_crops == len(self.crops_collected):
            self.stop()

    def get_simulation_state(self):
        initial_grid = self.initial_grid.tolist()
        harvester_path = self.harvester_path
        collector_path = self.collector_path
        state = {
            "initial_grid": initial_grid,
            "harvester_path": harvester_path,
            "collector_path": collector_path,
            "size" : self.p['size']
        }
        return state

class Harvester(ap.Agent):
    def setup(self, index):
        self.index = index  # Current position
        self.last_index = index  # Last position

class Collecter(ap.Agent):
    def setup(self, index):
        self.index = index  # Current position
        self.last_index = index  # Last position
        self.capacity = 0  # Collector's capacity
        self.next_index_was1 = False  # Track if next step is over a crop

@app.route('/simulate', methods=['GET'])
def simulate():
    parameters = {
        'steps': 500,
        'size': (15, 15),  # Grid size
        'capacity': 20,  # Collector capacity
        'obstacles': 15  # Number of obstacles
    }
    model = HarvestModel(parameters)
    model.run()  # Run the model

    # Get the current state after the simulation
    simulation_state = model.get_simulation_state()

    response = {
        "status": "success",
        "message": "Simulation data",
        "data": simulation_state
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
