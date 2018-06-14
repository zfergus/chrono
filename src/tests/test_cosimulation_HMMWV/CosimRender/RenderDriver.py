# usage: python TerrainDATtoVTK.py <tire number> <timestep> <ancf_flag> <list of files for this frame...>
# usage: python TerrainDATtoVTK.py <timestep> <list of files for this frame...>

# usage: python RenderDriver.py <num_tires> <ancf_flag> <list of tire directories...>
#       <num_terrain_ranks> <terrain directory> <end_timestep> <settle_flag>

import sys
import TerrainDATtoVTK as TerrainRender
import TireDATtoVTK as TireRender

if len(sys.argv) < 7:
	print('''usage: python RenderDriver.py <num_tires> <ancf_flag> <list of tire directories...>
	       <num_terrain_ranks> <terrain directory> <start_timestep> <end_timestep> <settling_flag> <num_settling_frames>''')
	exit(1)

arg = 1
num_tires = int(sys.argv[arg])
arg = arg + 1
use_ancf = bool(int(sys.argv[arg]))
arg = arg + 1

tire_dirs = []
for i in range(arg,arg+num_tires):
	tire_dirs.append(sys.argv[i])

arg = arg + num_tires
num_terrain_ranks = int(sys.argv[arg])
arg = arg + 1
terrain_dir = sys.argv[arg]
if terrain_dir[-1] != '/':
	terrain_dir = terrain_dir + '/'

arg = arg + 1

for i in range(len(tire_dirs)):
	if tire_dirs[i][-1] != '/':
		tire_dirs[i] = tire_dirs[i] + '/'

start_timestep = int(sys.argv[arg])
arg = arg +1

end_timestep = int(sys.argv[arg])
arg = arg + 1

settle = bool(int(sys.argv[arg]))
arg = arg + 1
if settle:
	num_settle_steps = int(sys.argv[arg])
	arg = arg + 1


# Debug args
print('num_tires ' + str(num_tires))
print('use_ancf ' + str(use_ancf))
print('tire_dirs ' + str(tire_dirs))
print('num_terrain_ranks ' + str(num_terrain_ranks))
print('terrain_dir ' + str(terrain_dir))
print('end_timestep ' + str(end_timestep))
print('settle ' + str(settle))
print('num_settle_steps ' + str(num_settle_steps))
print('start_timestep ' + str(start_timestep))


# Render settling
if settle:
	print('Settling')
	for step in range(1,num_settle_steps+1):
		step = str(step).zfill(4)
		print('step ' + str(step))
		settling_files = []
		for rank in range(num_terrain_ranks):
			rank = str(rank).zfill(3)
			filename = terrain_dir + 'results_' + str(rank) + '/settling_' + str (step) + '.dat'
			settling_files.append(filename)

		TerrainRender.Parse(step,settling_files,1)


# Render Main Simulation
# For each frame
print('Main')
for step in range(start_timestep, end_timestep):
	step = str(step).zfill(4)
	print('step ' + str(step))

	# Terrain
	terrain_files = []
	for rank in range(num_terrain_ranks):
		rank = str(rank).zfill(3)
		# print('rank ' + rank)
		filename = terrain_dir + 'results_' + str(rank) + '/data_' + str(step) + '.dat'
		# print(filename)
		terrain_files.append(filename)

	TerrainRender.Parse(step,terrain_files,0)

	# Tires
	for tire in range(num_tires):
		filename = tire_dirs[tire] + 'data_' + str(step) + '.dat'
		# print('filename: ' + str(filename))
		TireRender.Parse(tire,step,use_ancf,filename)