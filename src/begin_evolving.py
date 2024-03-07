from ea import EA
import sys 
import random

seed = int(sys.argv[1])

output = 3
popsize = 10
grain_diameter = 10 
num_rows = 5 
cols_structure = [5,4,5,4,5]


ea = EA(seed=seed, popsize=popsize, generations=100, output=output)
ea.create_materials(grain_diameter, num_rows, cols_structure)
ea.run_generation_one()
ea.hillclimber()
# ea.visualize_best()