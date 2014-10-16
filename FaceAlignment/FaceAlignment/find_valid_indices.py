import subprocess
import os.path

basename = 'image_'
img_extension = '.png'
pts_extension = '.pts'
digits = 4
#path = '/Users/phg/Data/lfpw/testset/'
path = '/Users/phg/Data/lfpw/trainset/'
imgcount = 872 

def padWith(s, c, L):
	while len(s) < L:
		s = c + s
	return s

valid_indices = []

for i in range(imgcount):
	idx_str = padWith(str(i+1), '0', digits)
	imgfile = path + basename + idx_str + img_extension
	ptsfile = path + basename + idx_str + pts_extension
	#print imgfile, ptsfile

	if os.path.isfile(imgfile):
		valid_indices.append(i+1)

print 'valid files:', len(valid_indices)

with open('valid_indices.txt', 'w') as f:
	for idx in valid_indices:
		f.write(str(idx) + ' ')
	f.write('\n')
