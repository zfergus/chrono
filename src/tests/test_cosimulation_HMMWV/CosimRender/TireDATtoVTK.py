import sys

# usage: python TerrainDATtoVTK.py <tire number> <timestep> <ancf_flag> <list of files for this frame...>

def Parse(tire_num,timestep,use_ancf,file_name):
	timestep.zfill(4)

	pos = []
	vel = []
	connectivity = []
	strain = []

	# For each input file
	# print('file: ' + file_name)
	file = open(file_name, 'r')
	lines = file.readlines()

	if use_ancf:
		num_lines_pos = int(lines[2].strip().split()[1])
	else:
		num_lines_pos = int(lines[2].strip().split()[0])

	i = 3 # Skip the header
	start = i

	# Read in positions
	while i < num_lines_pos + start:
		if use_ancf:
			pos.append(lines[i].strip())
			pos.append(lines[i+1].strip())
			pos.append(lines[i+2].strip())
			i = i + 6
		else:
			tokens = lines[i].strip().split()
			pos.append(tokens[0])
			pos.append(tokens[1])
			pos.append(tokens[2])
			i = i + 1

	start = i
	# Read in velocities
	while i < start + num_lines_pos:
		if use_ancf:
			vel.append(lines[i].strip())
			vel.append(lines[i+1].strip())
			vel.append(lines[i+2].strip())
			i = i + 6
		else:
			tokens = lines[i].strip().split()
			vel.append(tokens[0])
			vel.append(tokens[1])
			vel.append(tokens[2])
			i = i + 1

	# Skip blank lines
	while not lines[i].strip():
		i = i + 1

	tokens = lines[i].strip().split()
	if use_ancf:
		num_connectivity = int(tokens[1])
	else:
		# print(tokens)
		num_connectivity = int(tokens[1])

	i = i + 1

	start = i
	# Read in connectivity
	while i < start + num_connectivity:
		tokens = lines[i].strip().split()
		connectivity.append(tokens)
		i = i + 1

	if use_ancf:
		# Skip blank lines
		while not lines[i].strip():
			i = i + 1

		# Skip the strain header
		i = i + 1

		# Read in strains
		start = i
		while i < start + len(pos)/3:
			tokens = lines[i].strip().split()
			strain.append((tokens[0], tokens[1], tokens[2]))
			i = i + 1

	file.close()

	# Write VTK file with all positions and velocities
	outfilename = str(tire_num) + '_tire_' + timestep + '.vtk'
	outfile = open(outfilename, 'w')
	outfile.write('# vtk DataFile Version 1.0\n')
	outfile.write('Unstructured Grid Example\n')
	outfile.write('ASCII\n')
	outfile.write('\n\n')

	outfile.write('DATASET UNSTRUCTURED_GRID\n')
	if use_ancf:
		outfile.write('POINTS ' + str(int(num_lines_pos / 6)) + ' float\n')
	else:
		outfile.write('POINTS ' + str(int(num_lines_pos)) + 'float\n')

	i = 0
	while i < len(pos):
		outfile.write(pos[i] + ' ' + pos[i+1] + ' ' + pos[i+2] + '\n')
		i = i + 3

	outfile.write('\n')
	if use_ancf:
		outfile.write('CELLS ' + str(len(connectivity)) + ' ' + str(len(connectivity) * 5) + '\n')
	else:
		outfile.write('CELLS ' + str(len(connectivity)) + ' ' + str(len(connectivity) * 4) + '\n')

	for c in connectivity:
		if use_ancf:
			outfile.write('4 ' + c[0] + ' ' + c[1] + ' ' + c[2] + ' ' + c[3] + '\n')
		else:
			outfile.write('3 ' + c[0] + ' ' + c[1] + ' ' + c[2] + '\n')

	outfile.write('\n')
	outfile.write('CELL_TYPES ' + str(len(connectivity)) + '\n')
	for c in connectivity:
		outfile.write('9\n')

	outfile.write('\n')
	outfile.write('CELL_DATA ' + str(len(connectivity)) + '\n')
	outfile.write('POINT_DATA ' + str(int(len(pos)/3)) + '\n')

	outfile.write('\n')
	outfile.write('VECTORS Velocity float \n')
	i = 0
	while i < len(vel):
		outfile.write(vel[i] + ' ' + vel[i+1] + ' ' + vel[i+2] + '\n')
		i = i + 3

	outfile.write('\n')
	if use_ancf:
		outfile.write('VECTORS Strain float\n')
		for s in strain:
			outfile.write(s[0] + ' ' + s[1] + ' ' + s[2] + '\n')

	outfile.close()
