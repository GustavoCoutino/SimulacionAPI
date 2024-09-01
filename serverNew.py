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

                if (
                    0 <= neighbor[0] < len(grid)
                    and 0 <= neighbor[1] < len(grid[0])
                    and grid[neighbor[0]][neighbor[1]] not in invalid_values
                ):
                    tentative_g_score = g_score[current] + 1

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(
                            neighbor, goal
                        )
                        heapq.heappush(open_list, (f_score[neighbor], neighbor))
        return None

    def followRefillPath(self):
        # MOVER EL COLLECTER Y SEGUIR LA RUTA DE A*

        # Move collector along A* path one step at a time
        if self.path_index < len(self.path):
            step = self.path[self.path_index]

            # # Check if the collector will be on a crop to paint it as a crop when the collecter passes over it
            # if self.aux_grid_matrix[step] == 1:
            #     self.collecter.next_index_was1 = True
            # else:
            #     self.collecter.next_index_was1 = False

            self.collecter.last_index = self.collecter.index
            self.collecter.index = step

            # # Update the grid
            # if self.collecter.last_index != self.collecter.index:
            #   # si en el paso anterior se detectó un cultivo en el siguiente step se marca como cultivo, si no se marca como 0
            #   if self.collecter.next_index_was1:
            #     self.aux_grid_matrix[self.collecter.last_index] = 1
            #   else:
            self.aux_grid_matrix[self.collecter.last_index] = 0

            # Set the new position
            self.aux_grid_matrix[self.collecter.index] = 10

            self.path_index += 1  # Move to the next step in the path

        if self.path_index == len(self.path):
            # Finished moving along the A* path
            self.collecter.capacity = self.p["capacity"]

            self.refilling = False

            # find closest harvester to the collector
            distances = []
            for harvester in self.harvesters:
                distances.append(
                    abs(self.collecter.index[0] - harvester.index[0])
                    + abs(self.collecter.index[1] - harvester.index[1])
                )

            self.target_harvester = distances.index(min(distances))

    def move_harvester(self):
        self.collecter.last_index = self.collecter.index

        for index, harvester in enumerate(self.harvesters):
            if harvester.cropStorage > 0:
                # obtener un arreglo de distancias entre el tractor y los cultivos
                distances = []
                for crop in self.lista_coordenadas:
                    distances.append(
                        abs(harvester.index[0] - crop[0])
                        + abs(harvester.index[1] - crop[1])
                    )

                # se consigue el índice del cultivo más cercano
                if distances:
                    closest_crop_index = distances.index(min(distances))
                    path_to_crop = self.a_star(
                        self.aux_grid_matrix,
                        harvester.index,
                        self.lista_coordenadas[closest_crop_index],
                        {-1, 10, 5},
                    )

                if closest_crop_index == self.collecter.index:
                    self.collecter.is_on_crop = True

                # consigue ruta para llegar al siguiente cultivo

                # se actuliza la posición pasada para que equivalga a la posición actual
                harvester.last_index = harvester.index

                # actualizar posición del tractor
                if path_to_crop:
                    if len(path_to_crop) > 1:
                        # se reduce el almacenamiento del tractor
                        harvester.index = path_to_crop[1]

                        if self.aux_grid_matrix[path_to_crop[1]] == 1:
                            harvester.cropStorage -= 1
                            # se actualiza la posición del tractor

                # si el tractor llega a una posición de un cultivo se agrega a la lista de cultivos recolectados
                if harvester.index in self.lista_coordenadas:
                    self.crops_collected.add(harvester.index)
                    self.lista_coordenadas.remove(harvester.index)
                # Clear the last positions

                if harvester.last_index != harvester.index:
                    self.aux_grid_matrix[harvester.last_index] = 0

            elif self.collecter.collecting == False:
                # obtener distancia entre el colector y el tractor sin capacidad para encontrar el más cercano e ir a ese
                new_distance = abs(self.collecter.index[0] - harvester.index[0]) + abs(
                    self.collecter.index[1] - harvester.index[1]
                )
                old_distance = abs(
                    self.collecter.index[0]
                    - self.harvesters[self.target_harvester].index[0]
                ) + abs(
                    self.collecter.index[1]
                    - self.harvesters[self.target_harvester].index[1]
                )

                if old_distance < new_distance:
                    self.target_harvester = index

                self.collecter.collecting = True

            # Set the new positions
            self.aux_grid_matrix[harvester.index] = 5

    def setup(self):
        self.harvesters = []
        exclude_coords = []

        for i in range(self.p["tractors"]):
            # self.truck_agents = [Harvester(self, ((self.p['size'][0] - 2), 1)), Collecter(self, ((self.p['size'][0] - 2), 0))]
            self.harvesters.append(
                Harvester(self, ((self.p["size"][0] - 2), 1 + i * 2))
            )
            exclude_coords.append(((self.p["size"][0] - 2), 1 + i * 2))

        self.collecter = Collecter(self, ((self.p["size"][0] - 2), 0))

        # MATRIX
        # 1 = crops
        # -1 = obstacles
        # 5 = tractor
        # 10 = collector
        # 0 = empty space
        self.aux_grid_matrix = np.zeros(self.p.size, dtype=int)

        self.path = []  # Store the A* path here
        self.path_index = 0  # Track current position in the A* path
        self.refilling = False  # Track whether the collector is moving along the A* path to the gas station
        self.crops_collected = set()  # Track which crops have been collected
        self.total_crops = 0  # Track the total number of crops

        self.target_harvester = 0

        # lista de coordenadas en donde hay cultivos
        self.lista_coordenadas = []

        for i in range(1, (self.p["size"][0] - 1), 1):
            if i % 2 != 0:
                for j in range((self.p["size"][0] - 2), 0, -1):
                    self.lista_coordenadas.append((j, i))
            else:
                for j in range(1, (self.p["size"][0] - 1), 1):
                    self.lista_coordenadas.append((j, i))

        # lista de coordenadas para generar obstáculos
        self.lista_obstaculos = []

        # Create obstacles at random coordinates
        while len(self.lista_obstaculos) < self.p["obstacles"]:
            x = np.random.randint(0, self.p["size"][0] - 1)
            y = np.random.randint(0, self.p["size"][1] - 1)
            coord = (x, y)
            if coord not in exclude_coords and coord not in self.lista_obstaculos:
                self.lista_obstaculos.append(coord)

        self.collecter.capacity = self.p["capacity"]  # capacidad inicial

        # remove the obstacles from the list of coordinates
        for i in range(len(self.lista_obstaculos)):
            if self.lista_obstaculos[i] in self.lista_coordenadas:
                self.lista_coordenadas.remove(self.lista_obstaculos[i])

        # count the total number of crops
        self.total_crops = len(self.lista_coordenadas)

        # Mark the grid with the obstacles
        for i in range(len(self.lista_obstaculos)):
            self.aux_grid_matrix[self.lista_obstaculos[i]] = -1

        # Mark the grid with the crops
        for i in range(len(self.lista_coordenadas)):
            self.aux_grid_matrix[self.lista_coordenadas[i]] = 1

    def step(self):
        if self.refilling:
            self.followRefillPath()
            self.move_harvester()

        else:
            # REVISAR SI SE ACABO LA CAPACIDAD PARA CONSEGUIR LA RUTA DE IDA Y VUELTA AL ALMACÉN
            if self.collecter.capacity == 0:
                start = self.collecter.index

                # Find the closest gas station, they are always at each corner of the grid
                gas_stations = [
                    (0, 0),
                    (0, self.p["size"][0] - 1),
                    (self.p["size"][0] - 1, 0),
                    (self.p["size"][0] - 1, self.p["size"][0] - 1),
                ]

                gas_station_distances = []
                # Find the closest gas station
                for station in gas_stations:
                    gas_station_distances.append(
                        abs(start[0] - station[0]) + abs(start[1] - station[1])
                    )

                closest_gas_station = gas_stations[
                    gas_station_distances.index(min(gas_station_distances))
                ]

                goal = closest_gas_station
                path_to_goal = self.a_star(
                    self.aux_grid_matrix, start, goal, {5, -1, 1}
                )
                # path_back = self.a_star(self.aux_grid_matrix, goal, start, {5, -1, 1})

                # Combine the path to the goal and back to the start
                self.path = path_to_goal  # + path_back

                # Remove the first coordinate from the path (usually the starting point)
                if self.path:
                    if len(self.path) > 1:
                        self.path = self.path[1:]

                self.path_index = 0  # Reset path index
                self.refilling = True  # Start moving the collector along the path
            else:
                # AMBOS CAMINAN JUNTOS
                # se consigue el camino del colector al tractor para que estén lo más cerca posible

                self.move_harvester()
                # lista de distancias, si hay una que sea igual a 1, el recolector va a tener que negociar con los agentes y no moverse
                list_distances = []

                # calcular distancia entre cada tractor y el colector
                for harvester in self.harvesters:
                    distance = abs(self.collecter.index[0] - harvester.index[0]) + abs(
                        self.collecter.index[1] - harvester.index[1]
                    )
                    list_distances.append(distance)

                if 1 in list_distances:

                    # # aquí se revisa que el camino no sea nulo y se actualiza la posición del colector para que sea el segundo paso del camino
                    # if len(path_to_tractor) > 1 and path_to_tractor[1] != self.harvesters[self.target_harvester].index:
                    #   # se actualiza la posición del colector
                    #   self.collecter.index = path_to_tractor[1]

                    # obtener lista de indices de los tractores que estan en capacidad 0
                    list_harvesters = []
                    for index, harvester in enumerate(self.harvesters):
                        distance = abs(
                            self.collecter.index[0] - harvester.index[0]
                        ) + abs(self.collecter.index[1] - harvester.index[1])

                        if harvester.cropStorage == 0 and distance == 1:
                            list_harvesters.append(index)

                    if self.collecter.collecting:
                        self.collecter.collecting = False
                        for index in list_harvesters:
                            self.harvesters[index].cropStorage = np.random.randint(
                                10, 26
                            )
                else:
                    path_to_tractor = self.a_star(
                        self.aux_grid_matrix,
                        self.collecter.index,
                        self.harvesters[self.target_harvester].index,
                        {-1, 1},
                    )

                    if len(path_to_tractor) > 1:
                        self.collecter.index = path_to_tractor[1]
                        self.aux_grid_matrix[self.collecter.last_index] = 0
                        self.aux_grid_matrix[self.collecter.index] = 10

                    if self.collecter.index != self.collecter.last_index:
                        self.collecter.capacity -= 1

        # print(self.aux_grid_matrix)

    # el modelo termina cuando el tractor ha pasado por todos los cultivos
    def update(self):
        if self.total_crops == len(self.crops_collected):
            self.stop()


class Harvester(ap.Agent):
    def setup(self, index):
        # posición del harvester en formato de tupla
        self.index = index  # posición actual
        self.last_index = index  # posición del paso anterior, se usa para marcar con 0 el paso anterior

        self.cropStorage = np.random.randint(
            10, 26
        )  # capacidad del tractor, ira reduciendo -1 cada vez que recolecta un cultivo


class Collecter(ap.Agent):
    def setup(self, index):
        # del colelcter en formate de tupla
        self.index = index  # posición actual
        self.last_index = index  # posición del paso anterior, se usa para marcar con 0 el paso anterior

        self.capacity = 0  # capacidad del colector, ira reduciendo -1 cada vez que camina junto al tractor

        self.next_index_was1 = False  # se usa para saber si el siguiente paso es un cultivo y poder marcarlo como cultivo después de pasar por él

        self.collecting = False
        self.is_on_crop = False


parameters = {
    "steps": 500,  # cantidad de pasos
    "size": (20, 20),  # tamaño de la matriz
    "capacity": 50,  # capacidad del colector
    "obstacles": 15,  # cantidad de obstáculos
    "tractors": 4,
}


def my_plot(model, ax):
    ax.clear()

    ax.set_title(f"Step: {model.t}")
    ax.imshow(model.aux_grid_matrix, cmap="ocean", interpolation="nearest")


# Initialize the model and figure
fig, ax = plt.subplots()
model = HarvestModel(parameters)
# results = model.run()

# Run the animation
animation = ap.animate(model, fig, ax, my_plot)
IPython.display.HTML(animation.to_jshtml(fps=12))
