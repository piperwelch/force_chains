import numpy as np 
import random 
import os 
import constants as c 
import os 
from datetime import datetime
import copy 
import time
import pickle 
import random 
from glob import glob 
import subprocess 

class Grain:
    def __init__(self, x, y, grain_type, grain_id, is_target=False):
        self.x = x
        self.y = y
        self.type = grain_type
        self.is_target = is_target
        self.grain_id = grain_id


    def mutate(self, mutation_rate=0.08):
        # Mutate the density of the grain with a certain probability
        if random.random() < mutation_rate:
            self.type = 1 if self.type == 2 else 2



class Material:
    def __init__(self, id, output,  grain_diameter, num_rows, cols_structure, number_grains = 23, seed = 0):
        self.id = id
        self.seed = seed
        self.number_grains = number_grains 
        self.output = output

        self.make_grains(grain_diameter, num_rows, cols_structure)
        

    def read_sim(self):

        grain_ids = {self.output}
        forces = {grain_id: [] for grain_id in grain_ids}

        with open(f"../dumps/dump.force_chain_id{self.id}_seed{self.seed}", 'r') as file:
            data = file.read()

            sections = data.strip().split("ITEM: ATOMS id fy\n")[1:]

            for section in sections:
                lines = section.strip().split('\n')

                for line in lines:
                    parts = line.strip().split()

                    if len(parts) == 2:
                        try:
                            grain_id = int(parts[0])
                            if grain_id in grain_ids:
                                force_y = float(parts[1])
                                forces[grain_id].append(force_y)
                        except:
                            continue

        return forces
    

    def make_grains(self, grain_diameter, num_rows, cols_structure): 
        grain_count = 1
        offset = (grain_diameter**2 - (grain_diameter/2)**2)**0.5
 
        grains = []

        for row in range(num_rows):
            num_grains_in_row = cols_structure[row]
            y = row * offset + grain_diameter/2
            for j in range(num_grains_in_row):
                if row % 2:
                    x = j * grain_diameter + grain_diameter
                else: 
                    x = j * grain_diameter + grain_diameter/2

                grain_type = np.random.choice([1,2])
                is_target = False

                if grain_count == self.output: 
                    is_target = True

                grain = Grain(grain_id=grain_count, grain_type=grain_type, x=x, y=y, is_target=is_target)
                grain_count += 1
                grains.append(grain)
        self.grains = grains


    def write_data(self):
        data_file = open(f"../data_ins/data_id{self.id}_seed{self.seed}", "w")

        data_file.write(f"\n{self.number_grains} atoms\n")
        data_file.write("2 atom types\n\n")

        data_file.write(f"{0} {50} xlo xhi\n")
        data_file.write(f"{0} {50} ylo yhi\n")
        data_file.write(f"{-5} {5} zlo zhi\n")
        data_file.write("0 0 0 xy xz yz\n\n")

        data_file.write("Atoms # sphere\n\n")   
        grain_count = 1

        for grain in self.grains:
            #                     id        type       diameter  density         x        y      z
            data_file.write(f"{grain_count} {grain.type}  10 {1} {grain.x} {grain.y} 0\n")
            grain_count+=1 

        data_file.close()


    def get_fitness(self, fy):
        self.fitness1 = min(fy)
        self.fitness2 = min(fy)
        print(self.id, self.fitness1)


    def mutate(self):
        ''' mutates all grains in a material 
        '''
        for grain in self.grains:
            grain.mutate()


    def replay_material(self, replay=False, poly=False):

        self.write_data()
        current_datetime = datetime.now()
        if replay:
            data_file = f"../results/data_id{self.id}_seed{self.seed}"

        else:
            data_file = f"../data_ins/data_id{self.id}_seed{self.seed}"

        os.system(f"sbatch single_run.sh {self.id} {self.seed} {self.output} True {data_file} {current_datetime}")


    def visualize_poly(self):

        data_file = f"../data_ins/data_id{self.id}_seed{self.seed}"

        os.system(f"sbatch check_indp.sh {data_file} {self.output}")


class EA:
    def __init__(self, seed, popsize, generations, output):
        self.seed = seed 
        self.popsize = popsize 
        self.num_generations = generations 
        self.population = []
        self.next_avaliable_id = 0 
        self.output = output 
        self.generation = 0 
        os.makedirs(f"../dumps/", exist_ok=True)
        np.random.seed(seed)
        random.seed(seed)
        self.fitness_data = np.zeros(shape=(self.num_generations+1, self.popsize, 2))
    
    
    def create_materials(self, grain_diameter, num_rows, cols_structure):
        for _ in range(self.popsize*2):
            self.population.append(Material(id=self.next_avaliable_id,  grain_diameter=grain_diameter, num_rows=num_rows, cols_structure=cols_structure, number_grains=23,output=self.output, seed=self.seed))
            self.next_avaliable_id+=1 


    def run_batch(self, organisms): 
        data_files = ""
        for organism in organisms: 
            data_file = f"../data_ins/data_id{organism.id}_seed{self.seed} "
            data_files += data_file
        n_jobs = str(subprocess.check_output(['squeue', '-u', 'pwelch1']))

        n_jobs = n_jobs.split('\\n')

        while len(n_jobs) > 980:

            time.sleep(4)
            n_jobs = str(subprocess.check_output(['squeue', '-u', 'pwelch1']))
            n_jobs = n_jobs.split('\\n')

        current_datetime = datetime.now()
        os.system(f"sbatch batch10_run.sh {data_files} False {self.output} {current_datetime}")


    def run_generation_one(self):

        for material in self.population:
            material.write_data()

        for i in range(0, len(self.population), 10):
            run_batch = self.population[i:i+10]  # Get the segment of length 10
            self.run_batch(run_batch)  # Pass the segment to another functi
  
        self.wait_for_sims_to_finish()

        for material in self.population:
            y_forces = material.read_sim()
            material.get_fitness(y_forces[self.output])
        # quit()
        self.generation+=1

        os.system(f"rm -rf ../data_ins/*seed{self.seed}")
        os.system(f"rm -rf ../dumps/*seed{self.seed}*")
        os.system(f"rm outputs/*")


    def wait_for_sims_to_finish(self):

        all_sims_started = False
        
        n_sims = len(glob(f"../data_ins/*seed{self.seed}*"))

        while not all_sims_started:
            total_finished_sims = len(glob(f'../dumps/*seed{self.seed}*')) 
            if total_finished_sims == n_sims:
                all_sims_started = True
            else: 
                time.sleep(10) # check in increments of 1 seconds
                print("lines 240", self.seed, len(glob(f'../dumps/*seed{self.seed}*')), n_sims, flush=True)
        finished_sims = []
        while len(finished_sims) != n_sims:
            for out_file in glob(f'../dumps/*seed{self.seed}*'):
                
                f = open(out_file, 'r')
                lines = f.readlines()

                sim_name = out_file.split(f'/dumps/')[1]
                if sim_name not in finished_sims:

                    if len(lines) > 6000:
                        finished_sims.append(sim_name)

            time.sleep(1) # check in increments of 1 seconds   
            # print("lines 253", n_sims, len(finished_sims))  


    def open_csv(self, file_name, open_type):
        f = open(file_name, open_type)
        
        return f
    
    
    def save_checkpoint(self, j):

        filename = '../checkpoints/run{}_{}gens.p'.format(self.seed, j)

        rng_state = random.getstate()
        np_rng_state = np.random.get_state()

        with open(filename, 'wb') as f:
            pickle.dump([self, rng_state, np_rng_state], f)


    def survivor_selection(self):

        while len(self.population) > self.popsize:

            # Choose two different individuals from the population
            ind1 = np.random.randint(len(self.population))
            ind2 = np.random.randint(len(self.population))
            while ind1 == ind2:
                ind2 = np.random.randint(len(self.population))

            if self.dominates(ind1, ind2):  # ind1 dominates
                # remove ind2 from population and shift following individuals up in list
                for i in range(ind2, len(self.population)-1):
                    self.population[i] = self.population[i+1]
                self.population.pop() # remove last element from list (because it was shifted up)

            elif self.dominates(ind2, ind1):  # ind2 dominates

                # remove ind1 from population and shift following individuals up in list
                for i in range(ind1, len(self.population)-1):
                    self.population[i] = self.population[i+1]
                self.population.pop() # remove last element from list (because it was shifted up)

        assert len(self.population) == self.popsize


    def dominates(self, ind1, ind2):

        if self.population[ind1].fitness1 <= self.population[ind2].fitness1 and self.population[ind1].fitness2 <= self.population[ind2].fitness2:
            return True
        else:
            return False


    def hillclimber(self):

        for j in range(self.generation, self.num_generations):
            self.survivor_selection()

            for i, mat in enumerate(self.population):   
                self.fitness_data[self.generation,i,0] = mat.fitness1
                self.fitness_data[self.generation,i,1] = mat.fitness2

            if j % 2 == 0: 
                self.save_checkpoint(j)
            
            new_orgs = []
            num_children_per_parent = 2 

            #tournament winners breed 
            for parent in self.population:
                for m in range(num_children_per_parent): #for the num children per parent
                    child = copy.deepcopy(parent)
                    child.id = self.next_avaliable_id
                    self.next_avaliable_id+=1
                    child.mutate()
                    child.write_data()
                    new_orgs.append(child)

            for i in range(0, len(new_orgs), 10):
                run_batch = new_orgs[i:i+10]  # Get the segment of length 10
                self.run_batch(run_batch)  # Pass the segment to another functi

            self.wait_for_sims_to_finish()

            for material in new_orgs:
                y_forces = material.read_sim()
                material.get_fitness(y_forces[self.output])

            self.population.extend(new_orgs)

            self.generation = j

            os.system(f"rm -rf ../data_ins/*seed{self.seed}")
            os.system(f"rm -rf ../dumps/*seed{self.seed}*")
            os.system(f"rm outputs/*")


    def visualize_best(self):
        top = max(self.population, key=lambda x: x.fitness)
        top.replay_material()
