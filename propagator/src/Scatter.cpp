#include "Scatter.h"
#include <prop_kernels.cuh>
#include <cuda.h>

Scatter::Scatter(
  const std::shared_ptr<hypercube>& domain, 
  std::shared_ptr<complex4DReg> slow,
  std::shared_ptr<paramObj> par,
complex_vector* model, complex_vector* data, dim3 grid, dim3 block, cudaStream_t stream) 
: CudaOperator<complex4DReg, complex4DReg>(domain, domain, model, data, grid, block, stream), _slow_(slow) {
  
  _eps_ = par->getFloat("eps_scat",0.04);
  _dz_ = slow->getHyper()->getAxis(4).d;
  _ntaylor = par->getInt("taylor", 1);

  _grid_ = {32, 4, 4};
  _block_ = {16, 16, 4};

  // Initialize FFT for spatial dimensions (nx, ny)
  fft2d = std::make_unique<cuFFT2d>(domain, model_vec, data_vec, _grid_, _block_, _stream_);

  // Initialize kernel launchers
  launch_mult_kxky = Mult_kxky(&mult_kxky, _grid_, _block_, _stream_);
  launch_scale_by_slow = Slow_scale(&slow_scale_fwd, &slow_scale_adj, _grid_, _block_, _stream_);
  launch_scale_by_iw = Scale_by_iw(&scale_by_iw_fwd, &scale_by_iw_adj, _grid_, _block_, _stream_);
  launch_pad = Pad_launcher(&pad_fwd, &pad_adj, _grid_, _block_, _stream_);

  // Fill wavenumber and frequency arrays
  d_kx = fill_in_k(domain->getAxis(1)); // x spatial axis
  d_ky = fill_in_k(domain->getAxis(2)); // y spatial axis  
  d_w = fill_in_w(domain->getAxis(3)); // frequency axis

  auto pad_hyper = std::make_shared<hypercube>(
    std::vector<axis>{
      domain->getAxis(1), 
      domain->getAxis(2), 
      domain->getAxis(3)
    }
  );
  
   auto unpad_hyper = std::make_shared<hypercube>(
    std::vector<axis>{
      slow->getHyper()->getAxis(1), 
      slow->getHyper()->getAxis(2), 
      slow->getHyper()->getAxis(3)
    }
  );
  
  // Allocate device vectors for slowness
  _slow_slice = make_complex_vector(unpad_hyper, _grid_, _block_, _stream_);
  _padded_slow_slice = make_complex_vector(pad_hyper, _grid_, _block_, _stream_);

  // Allocate temporary wavefields
  _wfld_k = model_vec->cloneSpace();
  _wfld_scaled = model_vec->cloneSpace();
  _wfld_k->set_grid_block(_grid_, _block_);
  _wfld_scaled->set_grid_block(_grid_, _block_);
  
};

void Scatter::set_grid_block(dim3 grid, dim3 block) {
  launch_mult_kxky.set_grid_block(grid, block);
  launch_scale_by_slow.set_grid_block(grid, block);
  launch_scale_by_iw.set_grid_block(grid, block);
  launch_pad.set_grid_block(grid, block);
}

void Scatter::set_depth(int iz) {
  size_t offset = iz*_slow_slice->nelem;
  CHECK_CUDA_ERROR(cudaMemcpyAsync(_slow_slice->mat, _slow_->getVals() + offset, 
                                  _slow_slice->nelem * sizeof(cuFloatComplex), 
                                  cudaMemcpyHostToDevice, _stream_));
  launch_pad.run_fwd(_slow_slice, _padded_slow_slice);
}

void Scatter::cu_forward (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {
  if (!add) data->zero();
  
  fft2d->cu_forward(false, model, _wfld_k);
  
  for (int it=0; it<=_ntaylor; it++) {
		launch_mult_kxky.run_fwd(_wfld_k, _wfld_scaled, d_w, d_kx, d_ky, it); //scaling by k2/w2
    fft2d->cu_adjoint(_wfld_scaled);
    launch_scale_by_slow.run_fwd(_wfld_scaled, _wfld_scaled, _padded_slow_slice, coef[it], it, _eps_);
    // theres a sum inside this kernel
    launch_scale_by_iw.run_fwd(_wfld_scaled, data, d_w, _dz_);
	}
};


void Scatter::cu_adjoint (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {
  if (!add) model->zero();

  for (int it=0; it<=_ntaylor; it++) {
    _wfld_scaled->zero();
    launch_scale_by_iw.run_adj(_wfld_scaled, data, d_w, _dz_);
    launch_scale_by_slow.run_adj(_wfld_scaled, _wfld_scaled, _padded_slow_slice, coef[it], it, _eps_);
    fft2d->cu_forward(_wfld_scaled);
		launch_mult_kxky.run_fwd(_wfld_scaled, _wfld_k, d_w, d_kx, d_ky, it); //scaling by k2/w2    
    fft2d->cu_adjoint(true, model, _wfld_k);
	}
}

void Scatter::cu_forward (complex_vector* __restrict__ model) {
  
  fft2d->cu_forward(false, model, _wfld_k);
  model->zero(); // zero the model vector before accumulating results

  for (int it=0; it<=_ntaylor; it++) {
		launch_mult_kxky.run_fwd(_wfld_k, _wfld_scaled, d_w, d_kx, d_ky, it); //scaling by k2/w2
    fft2d->cu_adjoint(_wfld_scaled);
    launch_scale_by_slow.run_fwd(_wfld_scaled, _wfld_scaled, _padded_slow_slice, coef[it], it, _eps_);
    // theres a sum inside this kernel
    launch_scale_by_iw.run_fwd(_wfld_scaled, model, d_w, _dz_);
	}
};


void Scatter::cu_adjoint (complex_vector* __restrict__ data) {

  _wfld_scaled->zero();
  launch_scale_by_iw.run_adj(_wfld_scaled, data, d_w, _dz_);
  data->zero();

  for (int it=0; it<=_ntaylor; it++) {
    launch_scale_by_slow.run_adj(_wfld_k, _wfld_scaled, _padded_slow_slice, coef[it], it, _eps_);
    fft2d->cu_forward(_wfld_k);
		launch_mult_kxky.run_fwd(_wfld_k, _wfld_k, d_w, d_kx, d_ky, it); //scaling by k2/w2    
    fft2d->cu_adjoint(true, data, _wfld_k);
	}

  
}

