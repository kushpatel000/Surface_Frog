import numpy as np

class DualAvoidedCrossing:
	## Two state model
	def __init__(self, a = 0.1, b = 0.28, c = 0.015, d = 0.06, e=0.05, mass = 2000.0):
		self.A = a
		self.B = b
		self.C = c
		self.D = d
		self.mass = mass
		self.E0 = e
		self.name = "dual"

	## V[x]
	def V(self, x):
		# V11 = n.psign(x) * self.a * (1.0-np.exp(-self.b*self.abs(x))
		V11  = np.zeros_like(x)
		V22  = -self.A*np.exp( -self.B*x*x ) + self.E0
		V12  = self.C*np.exp( -self.D*x*x )
		return np.asarray([ [V11,V12],[V12,V22] ]) 

	## d/dx V[x]
	def dV(self,x):
		dV11 = np.zeros_like(x)
		dV22 = 2.0*self.A*self.B*x*np.exp( -self.B*x*x )
		dV12 = -2.0*self.C*self.D*x*np.exp(-self.D*x*x)
		return np.asarray( [ [dV11,dV12],[dV12,dV22] ] )
