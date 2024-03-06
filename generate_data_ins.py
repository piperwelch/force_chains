import math 


def create_offset_lattice_coords(num_grains, grain_diameter, num_rows, cols_structure):

    box_width = max(cols_structure) * grain_diameter
    box_height = num_rows * grain_diameter
    offset = (grain_diameter**2 - (grain_diameter/2)**2)**0.5

    with open("data_file", "w") as f:
        f.write(f"\n{num_grains} atoms\n")
        f.write(f"{2} atom types\n\n")
        f.write(f"{0} {box_width} xlo xhi\n")
        f.write(f"{0} {box_height} ylo yhi\n")
        f.write(f"{-5} {5} zlo zhi\n")
        f.write(f"0 0 0 xy xz yz\n\n")
        f.write(f"Atoms # sphere\n\n")

        grain_density = 1
        grain_id = 1 
        grain_type = 1
        for row in range(num_rows):
            num_grains_in_row = cols_structure[row]
            y = row * offset + grain_diameter/2
            for j in range(num_grains_in_row):
                if row%2:
                    x = j * grain_diameter + grain_diameter
                else: 
                    x = j * grain_diameter + grain_diameter/2

                f.write(f"{grain_id} {grain_type} {grain_diameter} {grain_density} {x} {y} {0}\n")

                grain_id+=1
                grain_type = 1 + (grain_type % 2)
 

num_grains = 23
grain_diameter = 10
num_rows = 5
cols_structure = [5, 4, 5, 4, 5]
create_offset_lattice_coords(num_grains, grain_diameter, num_rows, cols_structure)
