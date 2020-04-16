import numpy as np

class SimpleAvoidedCrossing:
	## Two state model
	def __init__(self, a = 0.01, b = 1.6, c = 0.005, d = 1.0, mass = 2000.0):
		self.A = a
		self.B = b
		self.C = c
		self.D = d
		self.mass = mass
		self.name = "simple"

	## V[x]
	def V(self, x):
		V11  = np.sign(x) * self.A * (1 - np.exp(-self.B*np.abs(x)))
		V22  = -V11
		V12  = self.C * np.exp( -self.D*x*x )
		return np.array([ [V11,V12],[V12,V22] ]) 

	## d/dx V[x]
	def dV(self,x):
		dV11 = self.A*self.B* np.exp(-self.B*np.abs(x))
		dV22 = -dV11
		dV12 = -2.0*self.C*self.D*x*np.exp(-self.D*x*x)
		return np.array( [ [dV11,dV12],[dV12,dV22] ] )
