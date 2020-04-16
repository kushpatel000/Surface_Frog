import numpy as np

class StackedOscillators:
	def __init__(self, a = 0.0005, b = 0.0001, c = 2.5,
						d = 0.005, e = 5.0, deltaE = 0.0025, mass = 2000.0):
		self.a = a
		self.b = b
		self.c = c
		self.d = d
		self.e = e
		self.deltaE = deltaE
		self.mass = mass
		self.name = 'StackedOscillators'

	def V(self,x):
		V11 = self.a*x**2
		V22 = self.b*x**2 + self.deltaE
		V12 = self.d*np.exp(-self.e*(x-self.c)**2) + self.d*np.exp(-self.e*(x+self.c)**2)

		return np.array([[V11,V12],[V12,V22]])


	def dV(self,x):
		dV11 = 2*self.a*x
		dV22 = 2*self.b*x
		dV12 = -2*self.d*self.e*( (x-self.c)*np.exp(-self.e*(x-self.c)**2) + (x+self.c)*np.exp(-self.e*(x+self.c)**2) )

		return np.array([[dV11,dV12],[dV12,dV22]])


	def dAlt(self,x):
		h = 0.001
		Vm = self.V(x-0.5*h)
		Vp = self.V(x+0.5*h)

		dVout = (Vp-Vm)/h
		return dVout




if __name__ == '__main__':
	from matplotlib import pyplot as plt
	from SimpleAvoidedCrossing import SimpleAvoidedCrossing

	x = np.arange(-10,10,0.1)

	SAC = SimpleAvoidedCrossing()
	SAC_V = SAC.V(x)
	SO = StackedOscillators()
	SO_dV = SO.dV(x)
	SO_dV2 = SO.dAlt(x)
	
	plt.subplot(3,1,1)
	plt.plot(x,SO_dV[0,0],x,SO_dV2[0,0])
	plt.subplot(3,1,2)
	plt.plot(x,SO_dV[1,1],x,SO_dV2[1,1])
	plt.subplot(3,1,3)
	plt.plot(x,SO_dV[0,1],x,SO_dV2[0,1])
	plt.show()