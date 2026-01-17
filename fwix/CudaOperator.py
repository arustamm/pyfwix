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
		self.setDomainRange(model, data)

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

		super(SplitOperator, self).__init__(model4d, model_bands)

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


	

