import numpy as np
import cv2 as cv
import re
import sys
from struct import unpack

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts):
    verts = verts.reshape(-1, 3)
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f')
        
def read_pfm(file):
        # Adopted from https://stackoverflow.com/questions/48809433/read-pfm-format-in-python
        with open(file, "rb") as f:
            # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
            type = f.readline().decode('latin-1')
            if "PF" in type:
                channels = 3
            elif "Pf" in type:
                channels = 1
            else:
                sys.exit(1)
            # Line 2: width height
            line = f.readline().decode('latin-1')
            width, height = re.findall('\d+', line)
            width = int(width)
            height = int(height)

            # Line 3: +ve number means big endian, negative means little endian
            line = f.readline().decode('latin-1')
            BigEndian = True
            if "-" in line:
                BigEndian = False
            # Slurp all binary data
            samples = width * height * channels;
            buffer = f.read(samples * 4)
            # Unpack floats with appropriate endianness
            if BigEndian:
                fmt = ">"
            else:
                fmt = "<"
            fmt = fmt + str(samples) + "f"
            img = unpack(fmt, buffer)
        return img, height, width



def main():
    print('loading images...')
    img_gt = read_pfm('../data/Motorcycle-imperfect/disp0.pfm')
    
    img_gt = np.array(img_gt)
    depth_map = 193.001 * 3997.684 / (img_gt + 131.111)
    
    out_points = depth_map
    out_fn = 'out.ply'
    write_ply(out_fn, out_points)
    print('%s saved' % out_fn)

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()