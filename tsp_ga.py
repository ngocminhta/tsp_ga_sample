from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from typing import List
import random
import numpy
import math

# Initialize City

class CityInit:
    def __init__(self, x, y):
        self.x = x
        self.y = y
     
def read_cities():
    CITY_COORDINATES = []
    with open('test_data/eil51.tsp', 'r') as handle:
        lines = handle.readlines()
        for line in lines:
            coor = []
            _, x, y = map(float, line.split())
            coor.append(x)
            coor.append(y)
            CITY_COORDINATES.append(coor)
        return CITY_COORDINATES

# Initialize parameter

CITY_COORDINATES = read_cities()
TOTAL_NST = len(CITY_COORDINATES) - 1
POPULATION_SIZE = 100
MAX_GENERATION = 2000
MUTATION_RATE = 0.01


class City():
    def __init__(self):
        self.nst = []
        self.cost = 0

    def __str__(self):
        return "NST: {0} Cost: {1}\n".format(self.nst, self.cost) 
    
    def __repr__(self):
        return str(self)

def create_city() -> City:
    city = City()
    city.nst = random.sample(range(1, TOTAL_NST + 1), TOTAL_NST)
    city.cost = eval_nst(city.nst)
    return city

def distance(a, b) -> float:
    dis = math.hypot((a[0] - b[0]), (a[1] - b[1]))
    return dis

def get_fittest_city(cities: List[City]) -> City:
    city_cost = [city.cost for city in cities]
    return cities[city_cost.index(min(city_cost))]

def eval_nst(nst: List[int]) -> float:
    # Add 0 to beginning and ending of nst
    arr = [0] * (len(nst) + 2)
    arr[1:-1] = nst
    cost = 0
    for i in range(len(arr) - 1):
        p1 = CITY_COORDINATES[arr[i]]
        p2 = CITY_COORDINATES[arr[i + 1]]
        cost += distance(p1, p2)
    return numpy.round(cost, 2)

def tournament_selection(population:List[City], k:int) -> List[City]:
    selected_cities = random.sample(population, k)
    selected_parent = get_fittest_city(selected_cities)
    return selected_parent

def order_crossover(parents: List[City]) -> City:
    child_chro = [-1] * TOTAL_NST

    subset_length = random.randrange(2, 5)
    crossover_point = random.randrange(0, TOTAL_NST - subset_length)

    child_chro[crossover_point:crossover_point+subset_length] = parents[0].nst[crossover_point:crossover_point+subset_length]

    j, k = crossover_point + subset_length, crossover_point + subset_length
    while -1 in child_chro:
        if parents[1].nst[k] not in child_chro:
            child_chro[j] = parents[1].nst[k]
            j = j+1 if (j != TOTAL_NST-1) else 0
        
        k = k+1 if (k != TOTAL_NST-1) else 0

    child = City()
    child.nst = child_chro
    child.cost = eval_nst(child.nst)
    return child

def scramble_mutation(city: City) -> City:
    subset_length = random.randint(2, 6)
    start_point = random.randint(0, TOTAL_NST - subset_length)
    subset_index = [start_point, start_point + subset_length]

    subset = city.nst[subset_index[0]:subset_index[1]]
    random.shuffle(subset)

    city.nst[subset_index[0]:subset_index[1]] = subset
    city.cost = eval_nst(city.nst)
    return city

def reproduction(population: List[City]) -> City:
    parents = [tournament_selection(population, 20), random.choice(population)] 

    child = order_crossover(parents)
    
    if random.random() < MUTATION_RATE:
        scramble_mutation(child)

    return child

def visualize(all_fittest: List[City], all_pop_size: List[int]):
    fig = plt.figure(tight_layout=True, figsize=(10, 10))
    gs = gridspec.GridSpec(2, 1)

    # Route
    nst = [0] * (len(all_fittest[-1].nst) + 2)
    nst[1:-1] = all_fittest[-1].nst
    coordinates = [CITY_COORDINATES[i] for i in nst]
    x, y = zip(*coordinates)
    
    ax = fig.add_subplot(gs[0, :])
    ax.plot(x, y, color="green")
    ax.scatter(x, y, color="red")

    for i, xy in enumerate(coordinates[:-1]):
        ax.annotate(i, xy, xytext=(-20, -4), textcoords="offset points", color="tab:blue")
    
    ax.set_title("Route")

    # Optimization Graph
    ax = fig.add_subplot(gs[1, :])
    all_cost = [city.cost for city in all_fittest]
    ax.plot(all_cost, color="midnightblue")
    
    at = AnchoredText(
        "Minimum Cost: {0}".format(all_fittest[-1].cost), prop=dict(size=10), 
        frameon=True, loc='upper right')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    
    ax.set_title("Optimization Graph")
    ax.set_ylabel("Cost")
    ax.set_xlabel("Generation")
    
    fig.align_labels()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    generation = 0

    population = [create_city() for x in range (POPULATION_SIZE)]

    all_fittest = []
    all_pop_size = []
    
    while generation != MAX_GENERATION:
        generation += 1
        print("Generation: {0} -- Best Cost: {1}"
            .format(generation, get_fittest_city(population).cost))

        childs = []
        for x in range(POPULATION_SIZE):
            child = reproduction(population)
            childs.append(child)
        population.extend(childs)

        # Renew population
        sorted_list = sorted(population, key = lambda x : x.cost, reverse= True)
        population = sorted_list[-POPULATION_SIZE:]

        all_fittest.append(get_fittest_city(population))
        all_pop_size.append(len(population))

    visualize(all_fittest, all_pop_size)