"""This code allows the user to click to click on a mesh and extract coordinates
for landmarking of a statistical shape model.

Click anywhere on the mesh and press c.
"""
import argparse

import numpy as np
import vedo as v 
from vedo import printc

def get_inputs():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('-i','--inp', 
                      default=False, 
                      type=str, 
                      help='input segmentation directory'
                      )
  parser.add_argument('-o','--out', 
                      default="landmarks.txt", 
                      type=str, 
                      help='out landmark file'
                      )
  
  return parser.parse_args()
  
def click_landmarks_on_mesh(key):
  """taken from vedo/examples/basic/keypress.py
  """
  global landmarks
  mesh = vp.clickedActor
  if not mesh or key != "c":
      printc("click a mesh and press c.", c="r")
      printc("press 'esc' to exit.", c="r")
      return
  printc("clicked mesh    :", mesh.filename[-40:], c=4)
  printc("clicked 3D point:", mesh.picked3d, c=4)
  printc("clicked renderer:", [vp.renderer], c=2)
  landmarks.append(mesh.picked3d)

  # may need to vary 'r' based on mesh size
  vp.add(v.Sphere(pos=mesh.picked3d, r=0.1, c="v")) 

args = get_inputs()
landmarks = [] #initialise landmark list

# set up vedo plotter object to render mesh
vp = v.Plotter()
# enable interaction with user and plotter
vp.keyPressFunction = click_landmarks_on_mesh  

# check if stl file provided. Otherwise, use a generic shape as mesh to landmark
if args.inp:
  vp.load(args.inp)
else:
  vp += v.Star() 

vp += __doc__ # show docstring in plotter
vp.show()

landmarks = np.vstack(landmarks)
# show landmarks to user in terminal and write file
print(landmarks)
np.savetxt(args.out, landmarks, header="x y z")
