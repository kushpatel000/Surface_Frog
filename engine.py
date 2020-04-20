from scipy.special import jn as besselj
import numpy as np
import json
import sys
import os
import importlib

global ii
ii = 1j

class engine:

	def __init__(self, jsonfile,pidx):
		self.results = False
		self.e_min = None
		self.e_max = None

		with open(jsonfile) as jf:
			data = json.load(jf)

			self.tag    = data['tag']	if 'tag'	in data else str(os.getppid())
			self.dt 	= data['dt']	if 'dt'		in data else 20.0
			self.Tmax	= data['Tmax']	if 'Tmax'	in data else 20000.0
			self.x0		= data['x0']	if 'x0'		in data else -5.0
			self.p0		= data['p0']	if 'p0'		in data else 5.0
			self.sg_x	= data['sg_x']	if 'sg_x'	in data else 0.0
			self.sg_p	= data['sg_p']	if 'sg_p'	in data else 0.0
			self.state0 = data['state0']if 'state0'	in data else 0
			self.logQ	= data['logQ']	if 'logQ'	in data else False
			self.mdl_nm	= data['model'] if 'model'	in data else 'SimpleAvoidedCrossing'
			
			self.collapseQ  = data['collapseQ']  if 'collapseQ'  in data else False
			self.hop_style  = data['hop_style']  if 'hop_style'  in data else "tully"
			self.propagator = data['propagator'] if 'propagator' in data else "standard"

			self.force_style = data['force_style'] if 'force_style' in data else "state"
			
			try:
				imp_mod    = importlib.import_module('models.'+self.mdl_nm)
				mdl_class  = getattr( imp_mod,self.mdl_nm )
				self.model = mdl_class()
			except:
				self.model = None
			if not self.model:
				print(str(pidx),': Model is not found. Exiting...')
				exit(0)

			self.nstates = self.model.V(0.0).shape[0]

		self.logfile = "LogFiles/{0}.{1}.{2:04d}".format(
				self.mdl_nm,self.tag,pidx )

	def _lowercaseParameters(self):
		self.hop_style   = self.hop_style.lower()
		self.propagator  = self.propagator.lower()
		self.force_style = self.force_style.lower()

	def _prep_initial_conditions(self):
		rs = np.random.RandomState()
		self.position	= rs.normal(loc=self.x0,scale=np.abs(self.sg_x))
		self.momentum	= rs.normal(loc=self.p0,scale=np.abs(self.sg_p))
		self.rho = np.zeros( [self.nstates,self.nstates],dtype=np.complex128 )
		self.rho[self.state0,self.state0] = 1.0
		self.state		= self.state0
		self.time       = 0.0

		self.zeta = np.random.rand() if self.hop_style == "conditioned" else None


	def _compute_force(self):
		V  = self.model.V(self.position)
		dV = self.model.dV(self.position)
		S,c = np.linalg.eigh(V)
		f = -np.einsum('ji,jm,mi->i',c,dV,c)
		
		if self.force_style == 'meanfield':
			# Weighted sum from population distribution

			# Tr[ rho.(-ct.dV.c) ]
			# f = -np.real( np.einsum('ij,jk,kl,li->',rho,c.T,dV,c) ) 
			# for some reason, this gives large (and only +ve) force vectors
			
			# Diag(rho) [dot] Diabatic Force Vector
			f = -np.real( np.einsum('ii,i->',self.rho,f ) )
			return f
		else:
			return f[self.state]

	def _compute_S_and_DCV(self):
		V  = self.model.V(self.position)
		dV = self.model.dV(self.position)
		S,c = np.linalg.eigh(V)
		dc = np.einsum("ij,jk,lk->il",c,dV,c)

		for j in range(self.nstates):
			for i in range(j):
				dE = S[j] - S[i]
				if abs(dE) < 1.0e-14 : dE = np.sign(dE)*1.0e-14
				dc[i,j] /= dE
				dc[j,i] /= -dE

		return S,dc

	def _propagate_rho(self,S,NAC,dt):
		W = S - ii*NAC
		dg,c = np.linalg.eigh(W)

		U   = np.einsum("ij,jk,lk->il",c,np.diag(np.exp(-ii*dg*dt)),c.conj() )
		out = np.einsum("ij,jk,lk->il",U,self.rho,U.conj() )
		
		self.rho = out

	def _LvN_evals(self,H):
		if int(self.Tmax/self.dt)%100 == 0 or not self.e_min or not self.e_max:
			S,c = np.linalg.eigh(H)
			self.e_max = 1.1*( np.max(S) - np.min(S) )
			self.e_min = -self.e_max
		return self.e_max,self.e_min

	def _LvN_propagate(self,S,NAC,dt):
		nexp = 7

		Heff = S - ii*NAC
		self._LvN_evals(Heff)

		R = dt*(self.e_max-self.e_min)/2
		G = dt*self.e_min
		X = -ii*self.dt*Heff/R

		c = np.full((nexp,),2)
		c[0] = 1
		phi = [ self.rho, np.dot(X,self.rho)-np.dot(self.rho,X) ]
		for k in range(2,nexp):
			phi.append( 2*(np.dot(X,phi[k-1])-np.dot(phi[k-1],X))+phi[k-2] )
		
		a = [ np.exp(ii*(R+G))*c[k]*besselj(k,R) for k in range(nexp) ]
				
		rho_dt = np.zeros_like(self.rho)
		for k in range(nexp):
			rho_dt += a[k]*phi[k]

		self.rho = rho_dt

	def _Tully_hop(self,NAC):
		probs = 2.0*self.dt*np.real( self.rho[self.state,:] )*NAC[self.state,:]/np.real( self.rho[self.state,self.state] )
		probs[ self.state ] = 0.0
		probs = probs.clip(0.0,1.0)
		zeta   = np.random.rand()
		cProbs = np.cumsum(probs)
		
		new_state = -1
		for i in range(len(cProbs)):
			if zeta < cProbs[i]:
				new_state = i
				break
		return new_state

	def _conditioned_hop(self):
		new_state = -1

		# if np.real(np.trace(self.rho)) < self.zeta:
		# I think this is supposed to be Tr[rho], but for closed quantum systems
		# then trace is preserved under evolution. I'll keep this as it is now.
		
		if self.rho[self.state,self.state] < self.zeta: 
			new_state = 0
			weights = np.copy(np.real(np.diag(self.rho)))
			weights[self.state] = 0.0
			cProbs = np.cumsum(weights)
			cProbs /= cProbs[-1]
			rando = np.random.rand()
			for i in range(len(cProbs)):
				if rando < cProbs[i]:
					new_state = i
					break
			# hopped to a state, choose new zeta
			self.zeta = np.random.rand()
		
		return new_state


	def _collapse_system(self):
		self.rho = np.zeros( [self.nstates,self.nstates],dtype=np.complex128 )
		self.rho[self.state,self.state] = 1.0

	def _compute_kinetic_energy(self,dc):
		# One dimensional system:
		comps = self.momentum
		# for higher than one dimension
		# comps = np.dot(dc,self.momentum)/np.dot(dc,dc) * dc
		
		ke = 0.5*comps*comps/self.model.mass
		return ke

	def _adjust_momentum(self,dc,reduction):
		vec   = dc / np.sqrt( dc*dc )
		invM  = 1/self.model.mass
		dcM   = dc*self.model.mass
		a     = invM*vec*vec
		b     = 2.0*vec*self.momentum/self.model.mass
		c     = -2.0*reduction
		roots = np.roots([a,b,c])
		scale = min( roots,key=lambda x:abs(x) )

		self.momentum += scale*vec

	def _header_format(self,tag,value):
		build = None
		if isinstance( value, int ):
			build = "{:17s} : {:11d}\n".format(tag,value)
		elif isinstance( value, float ):
			build = "{:17s} : {:11.3f}\n".format(tag,value)
		else:
			build = "{:17s} : {:11s}\n".format(tag,str(value))

		if build: return build
		else: return "\n"

	def _log_state(self):
		with open(self.logfile,'a') as wrtr:
			wrtr.write("{:10.3e}  {:10.3e}  {:10.3e}  {:5d}"
				.format(self.time,self.position,self.momentum,self.state))
			for i in range(self.nstates):
				wrtr.write("  {:13.3e}".format( np.real(self.rho[i,i]) ))
				for j in range(i+1,self.nstates):
					wrtr.write("  {:13.3e}".format( np.real(self.rho[i,j]) ))
					wrtr.write("  {:13.3e}".format( np.imag(self.rho[i,j]) ))
			wrtr.write("\n")

	def _write_header(self):
		with open(self.logfile,'w+') as wrtr:
			wrtr.write( self._header_format("Model"				,self.mdl_nm) )
			wrtr.write( self._header_format("Time Step"			,self.dt) )
			wrtr.write( self._header_format("Max Time"			,self.Tmax))
			wrtr.write( self._header_format("Initial Position"	,self.position) )
			wrtr.write( self._header_format("Initial Momentum"	,self.momentum) )
			wrtr.write( self._header_format("Initial State"		,self.state0))
			wrtr.write( self._header_format("CollapseQ"			,self.collapseQ))
			wrtr.write( self._header_format("Hop Style"			,self.hop_style))
			wrtr.write( self._header_format("Propagator"		,self.propagator))
			wrtr.write("\n\n")

			wrtr.write("{:10s}  {:10s}  {:10s}  {:5s}"
				.format("Time","Position","Momentum","State") )
			for i in range(self.nstates):
				wrtr.write( "{:>15s}".format("rho({0:2d},{0:2d})".format(i)) )
				for j in range(i+1,self.nstates):
					wrtr.write( "{:>15s}".format("Re@rho({:2d},{:2d})".format(i,j)) )
					wrtr.write( "{:>15s}".format("Im@rho({:2d},{:2d})".format(i,j)) )
			wrtr.write('\n')
			# Header finished
		# Print t=0 conditions
		self._log_state()

	def simulate(self):
		self._lowercaseParameters()
		self._prep_initial_conditions()

		if self.logQ: self._write_header()
		
		# HALF STEP
		force = self._compute_force()
		dp    = 0.5*self.dt*force
		self.momentum += dp
		last_momentum  = self.p0 - dp
		S, dc = self._compute_S_and_DCV()
		NAC   = dc*0.5*(self.momentum+last_momentum)/self.model.mass

		if self.propagator == 'chebyshev':
			self._LvN_propagate( np.diag(S),NAC,0.5*self.dt )
		else:
			self._propagate_rho( np.diag(S),NAC,0.5*self.dt )	


		while self.time < self.Tmax:
		#	Step 2: evolve classica and quantum trajectories
			self.position += self.momentum*self.dt/self.model.mass
			last_S, last_dc, last_rho = S,dc,self.rho

			S, dc = self._compute_S_and_DCV()
			force = self._compute_force()
			last_momentum = self.momentum
			self.momentum += force*self.dt
			NAC   = dc*0.5*(self.momentum+last_momentum)/self.model.mass
			
			if self.propagator == 'chebyshev':
				self._LvN_propagate( np.diag(S),NAC,self.dt )
			else:
				self._propagate_rho( np.diag(S),NAC,self.dt )	
			
			self.rho = 0.5*( self.rho + self.rho.T.conj() )

		#	Step 3: Compute switching probabilities, determine switch
			hop = -1
			if self.hop_style == 'tully':
				hop = self._Tully_hop(NAC)
			elif self.hop_style == 'conditioned':
				hop = self._conditioned_hop()
		#	else: Ehrenfest Dynamics (no hops)
			
		#	Step 4: if switch, adjust atomic velocities or prevent switch
			if hop > -1:
				delV = S[hop] - S[self.state]
				KE   = self._compute_kinetic_energy( dc[hop,self.state] )
				if delV <= KE: # Sufficient KE to hop states
					self._adjust_momentum(dc[hop,self.state],-delV)
					self.state = hop
					if self.collapseQ: self._collapse_system()

		#	Advance Time
			self.time += self.dt
		#	Log stats, if requested
			if self.logQ: self._log_state()
		#	If far enough in any direction and corresponding momentum, exit
			if abs(self.position) > 15.0 and np.sign(self.position)==np.sign(self.momentum):
				self.time = self.Tmax + self.dt

	#	Finish simulations
		self.results = np.zeros( (self.nstates,2),dtype=np.int )
		if self.position > 0:
			self.results[ self.state,1 ] = 1
		else:
			self.results[ self.state,0 ] = 1

		return self.results
