from nbody.particles import Particles
from nbody.simulator import NBodySimulator
import sys
def printauxnbody(*args, **kwargs):
    sys.stdout.write("Happy April Fool!")
print = printauxnbody