from . import pyFWIX
import pyOperator as Op
import genericIO
import numpy as np
from pyVector import superVector
import SepVector
import numba.cuda as cu
from concurrent.futures import ThreadPoolExecutor
import Hypercube

class CudaOperator(Op.Operator):
	def __init__(self,model,data):
		self.setDomainRange(model,data)
		self.stream = cu.stream().handle.value

class cuFFT2d(CudaOperator):
	def __init__(self,model,data):
		super().__init__(model, data)
		self.cppMode = pyFWIX.cuFFT2d(model.getHyper().cppMode)

	def forward(self,add,model,data):
		self.cppMode.forward(add, model.cppMode, data.cppMode)

	def adjoint(self,add,model,data):
		self.cppMode.adjoint(add, model.cppMode, data.cppMode)

class Spline4D (Op.Operator):
	def __init__(self,model, data, type='CR-spline',taper=[0,0,0,0]):
		if type == 'CR-spline':
			self.a = 0.5
			self.b = 0
		if type == 'B-spline':
			self.a = 0
			self.b = 1
		if type == 'MN-spline':
			self.a = 1/3
			self.b = 1/3

		self.taper = taper
		self.cppMode = pyFWIX.Spline4D(model.cppMode, data.cppMode, self.a, self.b, taper)
		self.domain = model
		self.range = data

	def forward(self,add,model,data):
		self.cppMode.forward(add,model.cppMode,data.cppMode)

	def adjoint(self,add,model,data):
		self.cppMode.adjoint(add,model.cppMode,data.cppMode)


class SplitOperator(Op.Operator):
	"""
	Bridge Operator:
	- Domain: SuperVector([4D_Slowness, 4D_Density])  (Monolithic)
	- Range:  SuperVector([Band0, Band1, ...])        (Split by Freq)
	
	Handles arbitrary chunking of the frequency axis.
	"""
	def __init__(self, model4d, n_splits=None):
		
		# 1. Validation
		if len(model4d.vecs) != 2:
			raise ValueError("Domain must be SuperVector([Slow4D, Den4D])")
			
		if n_splits is None:
			n_splits = model4d.vecs[0].getHyper().getAxis(3).n 
		else:
			n_splits = n_splits
		model_bands = self._create_range_from_domain(model4d, n_splits)

		self.splits = [] # List of tuples (start_idx, end_idx)
		curr_idx = 0
		for band_vec in model_bands.vecs:
			nf_local = band_vec.vecs[0].getHyper().getAxis(3).n 
			self.splits.append( (curr_idx, curr_idx + nf_local) )
			curr_idx += nf_local
			
		# Verify total size matches Monolithic input
		total_nf_input = model4d.vecs[0].getHyper().getAxis(3).n
		if curr_idx != total_nf_input:
			raise ValueError(f"Total band frequencies ({curr_idx}) mismatch 4D input ({total_nf_input})")

		self.domain = model4d
		self.range = model_bands

	def _create_range_from_domain(self, model4d, n_splits):
		"""
		Factory: Creates the 'Physical Model' (Bands) from the 'Monolithic Model'.
		Replaces 'create_split_model'.
		"""
		# 1. Inspect Input
		slow_4d = model4d.vecs[0]
		hyp = slow_4d.getHyper()
		axes = hyp.axes # [X, Y, F, Z]
		
		nx, ny, nf, nz = axes[0].n, axes[1].n, axes[2].n, axes[3].n
		dx, dy, df, dz = axes[0].d, axes[1].d, axes[2].d, axes[3].d
		ox, oy, of, oz = axes[0].o, axes[1].o, axes[2].o, axes[3].o
		
		n_band_size = nf // n_splits
		band_supervectors = []

		# 2. Create Geometry for each Band
		for i in range(n_splits):
			idx_start = i * n_band_size
			if i == n_splits - 1:
				idx_end = nf
			else:
				idx_end = idx_start + n_band_size
			
			current_nf = idx_end - idx_start
			current_of = of + (idx_start * df)
			
			# Create Axes for this band
			# Note: We keep the 4D structure even for bands
			band_axes = [
				axes[0], # X
				axes[1], # Y
				Hypercube.axis(n=current_nf, o=current_of, d=df), # F (Sliced)
				axes[3]  # Z
			]
			band_hyper = Hypercube.hypercube(axes=band_axes)
			
			# Create Components
			s_vec = SepVector.getSepVector(band_hyper, storage='dataComplex')
			d_vec = SepVector.getSepVector(band_hyper, storage='dataComplex')
			
			band_supervectors.append(superVector(s_vec, d_vec))
			
		# 3. Return Master SuperVector
		return superVector(*band_supervectors)

	def forward(self, add, model, data):
		"""
		Scatter: [Slow4D, Den4D] -> [Band0, Band1...]
		"""
		self.checkDomainRange(model, data)
		if not add: 
			data.zero()

		# Numpy View: [nz, nf, ny, nx] (Standard SepVector layout)
		# OR [n4, n3, n2, n1] -> [nz, nf, ny, nx]
		slow_4d = model.vecs[0].getNdArray()
		den_4d  = model.vecs[1].getNdArray()
		
		def _fwd_worker(i):
			# i is band index
			start, end = self.splits[i]
			# Map Slowness
			data.vecs[i].vecs[0].getNdArray()[:] += slow_4d[:, start:end, :, :]
			# Map Density
			data.vecs[i].vecs[1].getNdArray()[:] += den_4d[:, start:end, :, :]

		with ThreadPoolExecutor() as executor:
			list(executor.map(_fwd_worker, range(len(data.vecs))))

	def adjoint(self, add, model, data):
		"""
		Gather: [Band0, Band1...] -> [Slow4D, Den4D]
		"""
		self.checkDomainRange(model, data)
		if not add: 
			model.zero()

		slow_4d = model.vecs[0].getNdArray()
		den_4d  = model.vecs[1].getNdArray()

		def _adj_worker(i):
			start, end = self.splits[i]
			
			# Gather Slowness
			slow_4d[:, start:end, :, :] += data.vecs[i].vecs[0].getNdArray()[:]
			# Gather Density
			den_4d[:, start:end, :, :]  += data.vecs[i].vecs[1].getNdArray()[:]

		with ThreadPoolExecutor() as executor:
			list(executor.map(_adj_worker, range(len(data.vecs))))


class ScaleDepth(Op.Operator):
	"""
	Depth Scaling Operator: m_out = m_in * (z^power)
	
	Domain: SuperVector([Slow4D, Den4D])
	Range:  SuperVector([Slow4D, Den4D])
	"""
	def __init__(self, model, power=0.0):
		self.domain = model
		self.range = model
		self.power = power
		
		# 1. Inspect Geometry from Slowness Component
		slow_vec = model.vecs[0]
		hyp = slow_vec.getHyper()

		ax_z = hyp.axes[3] 
		nz, dz, oz = ax_z.n, ax_z.d, ax_z.o
		
		depths = oz + np.arange(nz) * dz
		w = np.power(depths, self.power)
		self.weights = w.reshape(-1, 1, 1, 1)

	def forward(self, add, model, data):
		self.checkDomainRange(model, data)
		if not add: 
			data.zero()
			
		# Apply to Slowness and Density
		for i in range(len(model.vecs)):
			m_arr = model.vecs[i].getNdArray()
			d_arr = data.vecs[i].getNdArray()
			d_arr[:] += m_arr * self.weights

	def adjoint(self, add, model, data):
		self.forward(add, data, model)


# This class now implements a smooth min(x, xmax)
class Softmax(Op.Operator):
	def __init__(self, model, data, xmax, tau=.001):
		self.domain = model
		self.range = data
		self.tau = tau
		self.tau2 = tau*tau
		self.xmax = xmax

	def forward(self, add, model, data):
		if not add: data.scale(0)
		
		# Unpack slowness and density components
		mod = model.getNdArray()
		dat = data.getNdArray()

		# Apply smooth minimum to the real part of the slowness
		x = mod.real
		dat.real[:] += 0.5 * (x + self.xmax - np.sqrt((x - self.xmax)**2 + self.tau2))
		# Pass the imaginary part of slowness through
		dat.imag[:] += mod.imag[:]

# This class now implements a smooth max(x, xmin)
class Softmin(Op.Operator):
	def __init__(self, model, data, xmin, tau=.001):
		self.domain = model
		self.range = data
		self.tau = tau
		self.tau2 = tau*tau
		self.xmin = xmin

	def forward(self, add, model, data):
		if not add: data.scale(0)
		
		# Unpack slowness and density components
		mod = model.getNdArray()
		dat = data.getNdArray()

		x = mod.real
		dat.real[:] += 0.5 * (x + self.xmin + np.sqrt((x - self.xmin)**2 + self.tau2))
		# Pass the imaginary part of slowness through
		dat.imag[:] += mod.imag[:]

# Derivative of Softmax
class dSoftmax(Op.Operator):
	def __init__(self, model, data, xmax, tau=.001):
		self.domain = model
		self.range = data
		self.tau = tau
		self.tau2 = tau*tau
		self.xmax = xmax
		self.bg = np.copy(model.getNdArray().real)

	def forward(self, add, model, data):
		if not add: data.scale(0)
		
		# Unpack slowness and density components
		mod = model.getNdArray()
		dat = data.getNdArray()

		# Apply the derivative (a scaling factor) to the real part of the slowness perturbation
		x_bg = self.bg
		scaling = 0.5 * (1.0 - (x_bg - self.xmax) / np.sqrt((x_bg - self.xmax)**2 + self.tau2))
		dat.real[:] += mod.real[:] * scaling
		dat.imag[:] += mod.imag[:]

	def adjoint(self, add, model, data):
		self.forward(add, data, model)

	def set_background(self, bg):
		self.bg[:] = bg.getNdArray().real

# Derivative of Softmin
class dSoftmin(Op.Operator):
	def __init__(self, model, data, xmin, tau=.001):
		self.domain = model
		self.range = data
		self.tau = tau
		self.tau2 = tau*tau
		self.xmin = xmin
		self.bg = np.copy(model.getNdArray().real)

	def forward(self, add, model, data):
		if not add: data.scale(0)
		
		# Unpack slowness and density components
		mod = model.getNdArray()
		dat = data.getNdArray()

		# Apply the derivative (a scaling factor) to the real part of the slowness perturbation
		x_bg = self.bg
		scaling = 0.5 * (1.0 + (x_bg - self.xmin) / np.sqrt((x_bg - self.xmin)**2 + self.tau2))
		dat.real[:] += mod.real[:] * scaling
		# Pass the imaginary part of slowness through
		dat.imag[:] += mod.imag[:]

	def adjoint(self, add, model, data):
		self.forward(add, data, model)

	def set_background(self, bg):
		self.bg[:] = bg.getNdArray().real

# Your factory function remains the same, but now uses the corrected classes.
# Note: I removed the 'l' parameter as it's not needed in this formulation.
def SoftMinMax(model, data, xmin, xmax, tau=[0.001, 0.001]):
	maxOp = Op.NonLinearOperator(Softmax(model, data, xmax, tau=tau[0]), dSoftmax(model, data, xmax, tau=tau[0]))
	minOp = Op.NonLinearOperator(Softmin(model, data, xmin, tau=tau[1]), dSoftmin(model, data, xmin, tau=tau[1]))
	return Op.CombNonlinearOp(maxOp, minOp)


def h(x, l, tau):
	return l*x - np.sqrt(l*l*x*x + tau**2)

def dh(x, l, tau):
	return l - l*l*x/np.sqrt(l*l*x*x + tau**2)

class HyperbolicPenalty(Op.Operator):
	"""docstring for HyperbolicPenalty."""

	def __init__(self, model, data, l=1, tau=.001):
		self.domain = model
		self.range = data
		self.l = l / 2
		self.tau = tau

	def forward(self, add, model, data):
		if not add: data.scale(0)
		# slownesss
		modNd = model[0].getNdArray()
		datNd = data[0].getNdArray()
		datNd.real[:] += modNd.real[:]
		datNd.imag[:] += h(modNd.imag, self.l, self.tau)
		# density 
		modNd = model[1].getNdArray()
		datNd = data[1].getNdArray()
		datNd[:] += modNd[:]

class Softclip(Op.Operator):
	"""docstring for Softclip."""

	def __init__(self, model, data, l=1, tau=.001):
		self.domain = model
		self.range = data
		self.l = l / 2
		self.tau = tau
		self.bg = np.copy(model[0].getNdArray().imag)

	def forward(self, add, model, data):
		if not add: data.scale(0)
		# slowness
		modNd = model[0].getNdArray()
		datNd = data[0].getNdArray()
		datNd.real[:] += modNd.real[:]
		sc = dh(self.bg, self.l, self.tau)
		datNd.imag[:] += modNd.imag[:]*sc
		# density 
		modNd = model[1].getNdArray()
		datNd = data[1].getNdArray()
		datNd[:] += modNd[:]
		

	def adjoint(self, add, model, data):
		self.forward(add,data,model)

	def set_background(self,bg):
		self.bg[:] = bg[0].getNdArray().imag


class Derivative(Op.Operator):
    def __init__(self, model, data, dw=1.0):
        """
        Regularization Operator along Frequency Axis with i*d/dw and Reflect Boundaries.
        Handles nested SuperVectors (Freq -> [Slow, Den]).
        """
        self.setDomainRange(model, data)
        self.inv_dw = 1.0 / dw

    def forward(self, add, model, data):
        if not add:
            data.zero()
        
        n_bands = len(model.vecs)
        scalar = 1j * self.inv_dw
        
        # 1. Standard Forward Difference (0 to N-2)
        for i in range(n_bands - 1):
            # Access the Frequency Bands (SuperVectors)
            band_curr = model.vecs[i]
            band_next = model.vecs[i+1]
            band_out  = data.vecs[i]

            # Iterate over components (Slowness, Density) inside the band
            # We zip them to handle them in pairs
            for vec_curr, vec_next, vec_out in zip(band_curr.vecs, band_next.vecs, band_out.vecs):
                arr_curr = vec_curr.getNdArray()
                arr_next = vec_next.getNdArray()
                arr_out  = vec_out.getNdArray()
                
                # Now we are subtracting numpy arrays, which works!
                arr_out[:] += scalar * (arr_next - arr_curr)

        # 2. Boundary Condition at N-1 (Reflect)
        if n_bands >= 2:
            band_prev = model.vecs[n_bands - 2]
            band_curr = model.vecs[n_bands - 1]
            band_out  = data.vecs[n_bands - 1]
            
            for vec_prev, vec_curr, vec_out in zip(band_prev.vecs, band_curr.vecs, band_out.vecs):
                arr_prev = vec_prev.getNdArray()
                arr_curr = vec_curr.getNdArray()
                arr_out  = vec_out.getNdArray()
                
                arr_out[:] += scalar * (arr_prev - arr_curr)

    def adjoint(self, add, model, data):
        if not add:
            model.zero()
            
        n_bands = len(data.vecs)
        scalar = -1j * self.inv_dw
        
        # 1. Adjoint of Standard Difference (0 to N-2)
        for i in range(n_bands - 1):
            band_res  = data.vecs[i]
            band_curr = model.vecs[i]
            band_next = model.vecs[i+1]
            
            for vec_res, vec_curr, vec_next in zip(band_res.vecs, band_curr.vecs, band_next.vecs):
                arr_res  = vec_res.getNdArray()
                arr_curr = vec_curr.getNdArray()
                arr_next = vec_next.getNdArray()
                
                term = scalar * arr_res
                
                # m[i+1] += term
                arr_next[:] += term
                # m[i]   -= term
                arr_curr[:] -= term

        # 2. Adjoint of Boundary Condition at N-1
        if n_bands >= 2:
            band_res  = data.vecs[n_bands - 1]
            band_prev = model.vecs[n_bands - 2]
            band_curr = model.vecs[n_bands - 1]
            
            for vec_res, vec_prev, vec_curr in zip(band_res.vecs, band_prev.vecs, band_curr.vecs):
                arr_res  = vec_res.getNdArray()
                arr_prev = vec_prev.getNdArray()
                arr_curr = vec_curr.getNdArray()
                
                term = scalar * arr_res
                
                # Add to m[N-2]
                arr_prev[:] += term
                # Subtract from m[N-1]
                arr_curr[:] -= term