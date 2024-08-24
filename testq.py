import agentpy as ap
import IPython
import numpy as np
import heapq
import matplotlib.pyplot as plt


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
                if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and grid[neighbor[0]][
                    neighbor[1]] not in invalid_values:
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_list, (f_score[neighbor], neighbor))
        return None

    def setup(self):
        self.collector = Collecter(self, ((self.p['size'][0] - 2), 0))
        self.harvesters = [
            Harvester(self, ((self.p['size'][0] - 2), i + 1)) for i in range(self.p['tractors'])
        ]

        self.aux_grid_matrix = np.zeros(self.p.size, dtype=int)
        self.path = []
        self.path_index = 0
        self.collecting = False
        self.crops_collected = set()
        self.total_crops = 0
        self.custom_t = 0

        self.lista_coordenadas = []
        for i in range(1, (self.p['size'][0] - 1), 1):
            if i % 2 != 0:
                for j in range((self.p['size'][0] - 2), 0, -1):
                    self.lista_coordenadas.append((j, i))
            else:
                for j in range(1, (self.p['size'][0] - 1), 1):
                    self.lista_coordenadas.append((j, i))

        self.lista_obstaculos = []
        exclude_coords = {
            ((self.p['size'][0] - 2), i) for i in range(self.p['tractors'] + 1)
        }
        while len(self.lista_obstaculos) < self.p['obstacles']:
            x = np.random.randint(0, self.p['size'][0] - 1)
            y = np.random.randint(0, self.p['size'][1] - 1)
            coord = (x, y)
            if coord not in exclude_coords and coord not in self.lista_obstaculos:
                self.lista_obstaculos.append(coord)

        self.collector.capacity = self.p['capacity']

        for i in range(len(self.lista_obstaculos)):
            if self.lista_obstaculos[i] in self.lista_coordenadas:
                self.lista_coordenadas.remove(self.lista_obstaculos[i])

        self.total_crops = len(self.lista_coordenadas)

        for i in range(len(self.lista_obstaculos)):
            self.aux_grid_matrix[self.lista_obstaculos[i]] = -1

        for i in range(len(self.lista_coordenadas)):
            self.aux_grid_matrix[self.lista_coordenadas[i]] = 1

    def step(self):
        if self.collecting:
            if self.path_index < len(self.path):
                step = self.path[self.path_index]

                if self.aux_grid_matrix[step] == 1:
                    self.collector.next_index_was1 = True
                else:
                    self.collector.next_index_was1 = False

                self.collector.last_index = self.collector.index
                self.collector.index = step

                if self.collector.last_index != self.collector.index:
                    if self.collector.next_index_was1:
                        self.aux_grid_matrix[self.collector.last_index] = 1
                    else:
                        self.aux_grid_matrix[self.collector.last_index] = 0

                self.aux_grid_matrix[self.collector.index] = 10
                self.path_index += 1

            if self.path_index == len(self.path):
                self.collector.capacity = self.p['capacity']
                self.collecting = False

        else:
            if self.collector.capacity == 0 and self.custom_t > 0:
                start = self.collector.index
                goal = (0, self.p['size'][0] - 1)
                path_to_goal = self.a_star(self.aux_grid_matrix, start, goal, {5, -1}) or []
                path_back = self.a_star(self.aux_grid_matrix, goal, start, {5, -1}) or []

                self.path = path_to_goal + path_back
                if len(self.path) > 1:
                    self.path = self.path[1:]

                self.path_index = 0
                self.collecting = True
            else:
                occupied_positions = {harvester.index for harvester in self.harvesters}
                occupied_positions.add(self.collector.index)

                for harvester in self.harvesters:
                    harvester.move()

                self.collector.capacity -= 1

    def update(self):
        if self.total_crops == len(self.crops_collected):
            self.stop()


class Harvester(ap.Agent):
    def setup(self, index):
        self.index = index
        self.last_index = index
        self.q_table = np.zeros((self.model.p['size'][0], self.model.p['size'][1], 4))  # 4 possible actions
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

    def choose_action(self, state):
        if np.random.rand() < self.model.p['epsilon']:
            return np.random.choice(range(len(self.actions)))  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update_q_value(self, state, action, reward, new_state):
        best_future_q = np.max(self.q_table[new_state])
        self.q_table[state][action] = self.q_table[state][action] + self.model.p['alpha'] * (
                reward + self.model.p['gamma'] * best_future_q - self.q_table[state][action])

    def move(self):
        state = self.index
        action = self.choose_action(state)
        next_position = (self.index[0] + self.actions[action][0], self.index[1] + self.actions[action][1])

        if self.is_valid(next_position):
            reward = self.get_reward(next_position)
            self.update_q_value(state, action, reward, next_position)
            self.index = next_position

    def is_valid(self, position):
        x, y = position
        return 0 <= x < self.model.p['size'][0] and 0 <= y < self.model.p['size'][1] and self.model.aux_grid_matrix[x, y] != -1

    def get_reward(self, position):
        if position in self.model.lista_coordenadas:
            self.model.crops_collected.add(position)
            return 10  # Reward for harvesting a crop
        elif self.model.aux_grid_matrix[position[0], position[1]] == -1:
            return -10  # Penalty for hitting an obstacle
        return -1  # Small penalty for each step


class Collecter(ap.Agent):
    def setup(self, index):
        self.index = index
        self.last_index = index
        self.capacity = 0
        self.next_index_was1 = False


parameters = {
    'steps': 397,
    'size': (15, 15),
    'capacity': 20,
    'obstacles': 15,
    'tractors': 3,  # Number of tractors (harvesters)
    'alpha': 0.1,  # Learning rate
    'gamma': 0.9,  # Discount factor
    'epsilon': 0.2  # Exploration rate
}


def my_plot(model, ax):
    ax.set_title(f'Step: {model.t}')
    ax.grid(True)
    ax.clear()
    ax.imshow(model.aux_grid_matrix, cmap='ocean', interpolation='nearest')


fig, ax = plt.subplots()
model = HarvestModel(parameters)
animation = ap.animate(model, fig, ax, my_plot)
IPython.display.HTML(animation.to_jshtml(fps=3))
