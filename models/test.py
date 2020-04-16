import numpy as np
from scipy.special import erf

class test:
	def __init__(self, a = 0.25, b = 0.5, mass = 2000.0):
		self.a = a
		self.b = b
		self.mass = mass
		self.name = 'test'

	def V(self, x):
		V11 = erf( self.a*x )
		V22 = -V11
		V12 = np.cosh(self.b*x) ** -2
		return np.array([ [V11,V12],[V12,V22] ])

	def dV(self,x):
		dV11 = 2*self.a / np.sqrt(np.pi) * np.exp( -(self.a*x)**2 )
		dV22 = -dV11
		dV12 = -2*self.b*np.tanh(self.b*x)/np.cosh(self.b*x)**2
		return np.array([ [dV11,dV12],[dV12,dV22] ])
