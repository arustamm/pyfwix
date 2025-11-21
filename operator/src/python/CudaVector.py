import pyCudaOperator
import pyOperator as Op
import genericIO
import numpy as np
from pyVector import superVector
import numba.cuda as cu

class CudaOperator(Op.Operator):
	def __init__(self,model,data):
		self.setDomainRange(model,data)
		self.stream = cu.stream().handle.value

class cuFFT2d(CudaOperator):
	def __init__(self,model,data):
		super().__init__(model, data)
		self.cppMode = pyCudaOperator.cuFFT2d(model.getHyper().cppMode, self.stream)

	def forward(self,add,model,data):
		self.cppMode.forward(add, model.cppMode, data.cppMode)

	def adjoint(self,add,model,data):
		self.cppMode.adjoint(add, model.cppMode, data.cppMode)


	

