import numpy as np

class ExtendedCouplingWithReflection:
	## Two state model
	def __init__(self, a = 6.0e-4, b = 0.10, c = 0.90, mass = 2000.0):
		self.A = a
		self.B = b
		self.C = c
		self.mass = mass
		self.name = 'extended'

	## V[x]
	def V(self, x):
		# V11 = n.psign(x) * self.a * (1.0-np.exp(-self.b*self.abs(x))
		V11  = self.A * np.ones_like(x)
		V22  = -self.A * np.ones_like(x)

		V12  = np.piecewise( x, [x<0,x>=0], 
			[ lambda x: self.B*np.exp(self.C*x),
			  lambda x: self.B*(2-np.exp(-self.C*x))
			])

		return np.asarray([ [V11,V12],[V12,V22] ]) 

	## d/dx V[x]
	def dV(self,x):
		dV11 = np.zeros_like(x)
		dV22 = np.zeros_like(x)
		dV12 = self.B*self.C*np.exp(-self.C*np.abs(x))
		return np.asarray( [ [dV11,dV12],[dV12,dV22] ] )
