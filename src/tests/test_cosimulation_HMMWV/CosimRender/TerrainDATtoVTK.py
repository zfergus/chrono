import sys

# usage: python TerrainDATtoVTK.py <timestep> <list of files for this frame...>

# Expects the following file format for terrain:
# line 1: <timestep length>
# line 2: <num particles> <radius>
# lines >2: g x y z vx vy vz

def Parse(timestep,file_names,settling):
	timestep.zfill(4)

	rad = 0
	pos = []
	vel = []

	# For each input file
	for filename in file_names:
		file = open(filename, 'r')
		lines = file.readlines()

		# Skip empty data files
		if len(lines) <= 2:
			break

		rad = lines[1].split()[1]

		# For each line after the 2-line header
		j = 2
		while j < len(lines):
			line = lines[j]
			tokens = line.split()
			pos.append((tokens[1], tokens[2], tokens[3]))
			vel.append((tokens[4], tokens[5], tokens[6]))
			j = j + 1

		file.close()

	# Write VTK file with all positions and velocities
	if settling:
		prefix = 'settling_'
	else:
		prefix = 'terrain_'

	outfilename = prefix + timestep + '.vtk'
	outfile = open(outfilename, 'w')
	outfile.write('# vtk DataFile Version 1.0\n')
	outfile.write('Unstructured Grid Example\n')
	outfile.write('ASCII\n')
	outfile.write('\n\n')
	outfile.write('DATASET UNSTRUCTURED_GRID\n')
	outfile.write('POINTS ' + str(len(pos)) + ' float\n');
	for x in pos:
		outfile.write(x[0] + ' ' + x[1] + ' ' + x[2] + '\n')

	outfile.write('\n')
	outfile.write('POINT_DATA ' + str(len(pos)) + '\n')
	outfile.write('SCALARS Radii float \n')
	outfile.write('LOOKUP_TABLE default \n')
	for x in pos:
		outfile.write(rad + '\n')

	outfile.write('\n')
	outfile.write('SCALARS Height float \n')
	outfile.write('LOOKUP_TABLE default \n')
	for x in pos:
	    outfile.write(x[2] + '\n')

	outfile.write('\n')
	outfile.write('VECTORS Velocity float \n')
	for v in vel:
	    outfile.write(v[0] + ' ' + v[1] + ' ' + v[2] + '\n')

	outfile.close()