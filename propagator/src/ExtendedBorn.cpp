#include <ExtendedBorn.h>

ExtendedBorn::ExtendedBorn (
  const std::shared_ptr<hypercube>& domain, 
  const std::shared_ptr<hypercube>& range,
  const std::vector<std::shared_ptr<complex4DReg>>& slow_den, 
  std::shared_ptr<Propagator> propagator,
  dim3 grid, dim3 block, cudaStream_t stream) :
_propagator(propagator),
CudaOperator<complex4DReg, complex2DReg>(domain, range, grid, block, stream) {

  // Initialize the propagator
  bg_reflect = propagator->getReflect();
  bg_down = propagator->getDown();
  bg_up = propagator->getUp();
  inj_rec = propagator->getInjRec();
  wfld_slice_gpu = propagator->get_wfld_slice_buffer();

  auto wfld_hyper = propagator->getWfldSliceHyper();
  auto par = propagator->getDown()->getPar();

  _slow = slow_den[0];
  _den = slow_den[1];

  CHECK_CUDA_ERROR(cudaHostRegister(_slow->getVals(), _slow->getHyper()->getN123()*sizeof(std::complex<float>), cudaHostRegisterDefault));
  CHECK_CUDA_ERROR(cudaHostRegister(_den->getVals(), _den->getHyper()->getN123()*sizeof(std::complex<float>), cudaHostRegisterDefault));

  auto run_id = propagator->getRunId();
  run_id.insert(0, "extended_born_");
	auto wfld_pool = std::make_shared<WavefieldPool>(wfld_hyper, par, run_id);

  down = std::make_shared<Downward>(wfld_hyper, 
    _slow->getHyper(), par, 
    propagator->getRefSampler(), wfld_pool, "down",
    inj_rec->data_vec, inj_rec->data_vec, _grid_, _block_, _stream_);

	up = std::make_shared<Upward>(wfld_hyper, 
    _slow->getHyper(), par, 
    propagator->getRefSampler(), wfld_pool, "up",
    inj_rec->data_vec, inj_rec->data_vec, _grid_, _block_, _stream_);

  ax = domain->getAxes();

  // Allocatge the data vector
  // Here we use the objects from the propagator that contains background wavefields
  auto subhyper3d = std::make_shared<hypercube>(std::vector<axis>{ax[0], ax[1], ax[2]});
  hslow = std::make_shared<complex3DReg>(subhyper3d);
  dslow = make_complex_vector(subhyper3d, _grid_, _block_, _stream_);
  down_scattering = std::make_shared<DownScattering>(subhyper3d, wfld_hyper, _slow, bg_down, dslow, inj_rec->data_vec, grid, block, stream);

  // Re-use the data_vec from DownScattering
  up_scattering = std::make_shared<UpScattering>(subhyper3d, wfld_hyper, _slow, bg_up, dslow, inj_rec->data_vec, grid, block, stream);

  auto subhyper5d = std::make_shared<hypercube>(std::vector<axis>{ax[0], ax[1], ax[2], axis(2, 0, 1), axis(2, 0, 1)});
  hmodel = std::make_shared<complex5DReg>(subhyper5d);
  dmodel = make_complex_vector(subhyper5d, _grid_, _block_, _stream_);
  back_scattering = std::make_shared<BackScattering>(subhyper5d, wfld_hyper, slow_den, bg_down, dmodel, inj_rec->data_vec, grid, block, stream);

  local_data = std::make_shared<complex2DReg>(range);
  CHECK_CUDA_ERROR(cudaHostRegister(local_data->getVals(), getRangeSizeInBytes(), cudaHostRegisterDefault));
  CHECK_CUDA_ERROR(cudaHostRegister(hslow->getVals(), hslow->getHyper()->getN123()*sizeof(std::complex<float>), cudaHostRegisterDefault));
  CHECK_CUDA_ERROR(cudaHostRegister(hmodel->getVals(), hmodel->getHyper()->getN123()*sizeof(std::complex<float>), cudaHostRegisterDefault));
};

size_t ExtendedBorn::getSliceSize() const {
  // return the size of the slice
  return static_cast<size_t>(ax[0].n * ax[1].n * ax[2].n);
}

size_t ExtendedBorn::getSliceSizeInBytes() const {
  // return the size of the slice in bytes
  return getSliceSize() * sizeof(std::complex<float>);
}

void ExtendedBorn::set_background_model(std::vector<std::shared_ptr<complex4DReg>> model) {
	_propagator->set_background_model(model);
  // down_scattering->set_background_model(model[0]);
  // up_scattering->set_background_model(model[0]);
  // back_scattering->set_background_model(model);
}

void ExtendedBorn::forward(bool add, std::vector<std::shared_ptr<complex4DReg>> model, std::shared_ptr<complex2DReg> data) {

  CHECK_CUDA_ERROR(cudaHostRegister(model[0]->getVals(), model[0]->getHyper()->getN123()*sizeof(std::complex<float>), cudaHostRegisterDefault));
  CHECK_CUDA_ERROR(cudaHostRegister(model[1]->getVals(), model[1]->getHyper()->getN123()*sizeof(std::complex<float>), cudaHostRegisterDefault));

  // always zero out the internal data_vec that records the data
	_propagator->data_vec->zero();
  down_scattering->data_vec->zero();
  if(!add) data->zero();

  downward_scattering_fwd(model);
  downward_reflected_scattering_fwd(model);
  upward_scattering_fwd(model);
  backward_scattering_fwd(model);

  // Check if the decompression queue is empty
  if (bg_down->get_decomp_queue_size() > 0) 
    throw std::runtime_error("Decompression queue after background down not empty");
  if (bg_up->get_decomp_queue_size() > 0) 
    throw std::runtime_error("Decompression queue after background up not empty");
  if (down->get_decomp_queue_size() > 0) 
    throw std::runtime_error("Decompression queue after down-scattering not empty");
  if (up->get_decomp_queue_size() > 0) 
    throw std::runtime_error("Decompression queue after up-scattering not empty");
  

  CHECK_CUDA_ERROR(cudaMemcpyAsync(local_data->getVals(), _propagator->data_vec->mat, getRangeSizeInBytes(), cudaMemcpyDeviceToHost, _stream_));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(_stream_));
  std::transform(
    local_data->getVals(), local_data->getVals() + getRangeSize(), 
    data->getVals(), data->getVals(),
    std::plus<std::complex<float>>()
  );

    // unpin the memory
  CHECK_CUDA_ERROR(cudaHostUnregister(model[0]->getVals()));
  CHECK_CUDA_ERROR(cudaHostUnregister(model[1]->getVals()));

}



void ExtendedBorn::adjoint(bool add, std::vector<std::shared_ptr<complex4DReg>> model, std::shared_ptr<complex2DReg> data) {

  CHECK_CUDA_ERROR(cudaHostRegister(data->getVals(), getRangeSizeInBytes(), cudaHostRegisterDefault));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(_propagator->data_vec->mat, data->getVals(), getRangeSizeInBytes(), cudaMemcpyHostToDevice, _stream_));

  // zero out the wavefield
  down_scattering->data_vec->zero();
  if(!add) std::for_each(model.begin(), model.end(), [](std::shared_ptr<complex4DReg> m){m->zero();});

  upward_scattering_adj(model);
  downward_reflected_scattering_adj(model);
  downward_scattering_adj(model);
  backward_scattering_adj(model);

  // Check if the decompression queue is empty
  if (bg_down->get_decomp_queue_size() > 0) 
    throw std::runtime_error("Decompression queue after background down not empty");
  if (bg_up->get_decomp_queue_size() > 0) 
    throw std::runtime_error("Decompression queue after background up not empty");
  if (down->get_decomp_queue_size() > 0) 
    throw std::runtime_error("Decompression queue after down-scattering not empty");
  if (up->get_decomp_queue_size() > 0) 
    throw std::runtime_error("Decompression queue after up-scattering not empty");


    // unpin the memory
  CHECK_CUDA_ERROR(cudaHostUnregister(data->getVals()));

}

void ExtendedBorn::downward_scattering_fwd(const std::vector<std::shared_ptr<complex4DReg>>& model) {
  if (bg_down->get_decomp_queue_size() > 0) 
    throw std::runtime_error("Decompression queue before down-scattering not empty");
  bg_down->start_decompress_from_top();
  down_scattering->data_vec->zero();
  for (int iz = 0; iz < ax[3].n; iz++) {

    // Copy the current depth slice of the model (slowness)
    size_t offset = iz * this->getSliceSize();
    CHECK_CUDA_ERROR(cudaMemcpyAsync(dslow->mat, model[0]->getVals() + offset, this->getSliceSizeInBytes(), cudaMemcpyHostToDevice, _stream_));

    // Get the decompressed background wavefields and remove them from the queue
    down_scattering->set_depth(iz);
    // Schedule more decompression 
    bg_down->add_decompresss_from_top(iz);

    // Record the wavefield
    down->compress_slice(iz, down_scattering->data_vec);
    inj_rec->set_depth(iz);
    inj_rec->cu_adjoint(true, _propagator->data_vec, down_scattering->data_vec);

    // Propagate wavefield
    down->one_step_fwd(iz, down_scattering->data_vec);
    down_scattering->cu_forward(true, dslow, down_scattering->data_vec);
  }
  down->wait_to_finish();
};

void ExtendedBorn::downward_scattering_adj(std::vector<std::shared_ptr<complex4DReg>>& model) {
  if (bg_down->get_decomp_queue_size() > 0) 
    throw std::runtime_error("Decompression queue before down-scattering not empty");
  bg_down->start_decompress_from_bottom();
  down_scattering->data_vec->zero();
  for (int iz = ax[3].n-1; iz >= 0; iz--) {
    // Schedule more decompression (except for the last depth)
    down_scattering->set_depth(iz);
    bg_down->add_decompresss_from_bottom(iz);
    
    down_scattering->cu_adjoint(false, dslow, down_scattering->data_vec);
    down->one_step_adj(iz, down_scattering->data_vec);

    // for direct forward scattering
    inj_rec->set_depth(iz);
    inj_rec->cu_forward(true, _propagator->data_vec, down_scattering->data_vec);
  
    // Copy the current depth slice of the model (slowness) back to host
    size_t offset = iz * this->getSliceSize();
    CHECK_CUDA_ERROR(cudaMemcpyAsync(hslow->getVals(), dslow->mat, this->getSliceSizeInBytes(), cudaMemcpyDeviceToHost, _stream_));
    
    CHECK_CUDA_ERROR(cudaStreamSynchronize(_stream_)); 
    std::transform(
      hslow->getVals(), hslow->getVals() + getSliceSize(), 
      model[0]->getVals() + offset, model[0]->getVals() + offset,
      std::plus<std::complex<float>>());
    
  }
};
    
void ExtendedBorn::downward_reflected_scattering_fwd(const std::vector<std::shared_ptr<complex4DReg>>& model) {
  wfld_slice_gpu->zero();
  down_scattering->data_vec->zero();
  if (down->get_decomp_queue_size() > 0) 
    throw std::runtime_error("Decompression queue before reflected down-scattering not empty");
	down->start_decompress_from_bottom();
  for (int iz=ax[3].n-1; iz >= 0; --iz) {

    std::shared_ptr<complex4DReg> down_wfld_host = down->get_next_wfld_slice();

		// Schedule the next decompression task to maintain the look-ahead window.
		down->add_decompresss_from_bottom(iz);

		// Asynchronously copy the decompressed data from host to GPU.
		CHECK_CUDA_ERROR(cudaMemcpyAsync(wfld_slice_gpu->mat, down_wfld_host->getVals(), down->getDomainSizeInBytes(), cudaMemcpyHostToDevice, _stream_));

		// Enqueue GPU work for the current slice `iz`.
		up->one_step_fwd(iz, down_scattering->data_vec);

		bg_reflect->set_depth(iz);
		bg_reflect->cu_forward(true, wfld_slice_gpu, down_scattering->data_vec);

		inj_rec->set_depth(iz);
		inj_rec->cu_adjoint(true, _propagator->data_vec, down_scattering->data_vec);
    // Save the wavefield
		// up->compress_slice(iz, up->data_vec);
  }
  up->wait_to_finish();
};

void ExtendedBorn::downward_reflected_scattering_adj(std::vector<std::shared_ptr<complex4DReg>>& model) {
  if (bg_down->get_decomp_queue_size() > 0) 
    throw std::runtime_error("Decompression queue before down-scattering not empty");
  bg_down->start_decompress_from_bottom();
  // Be careful here as this will be overwritten by the up propagation
 wfld_slice_gpu->zero();
  down_scattering->data_vec->zero();
  if (up->get_decomp_queue_size() > 0) 
    throw std::runtime_error("Decompression queue before reflected up-scattering not empty");
	up->start_decompress_from_bottom();

  for (int iz = ax[3].n-1; iz >= 0; iz--) {
    // Get the decompressed background wavefields and remove them from the queue
    std::shared_ptr<complex4DReg> up_wfld_host = up->get_next_wfld_slice();
    // Asynchronously copy the decompressed data from host to GPU.
		CHECK_CUDA_ERROR(cudaMemcpyAsync(wfld_slice_gpu->mat, up_wfld_host->getVals(), up->getDomainSizeInBytes(), cudaMemcpyHostToDevice, _stream_));

    up->add_decompresss_from_bottom(iz);
    // Schedule more decompression (except for the last depth)
    down_scattering->set_depth(iz);
    bg_down->add_decompresss_from_bottom(iz);
    
    down_scattering->cu_adjoint(false, dslow, down_scattering->data_vec);
    down->one_step_adj(iz, down_scattering->data_vec);

    // inj_rec->set_depth(iz);
		// inj_rec->cu_forward(true, _propagator->data_vec, down_scattering->data_vec);

    bg_reflect->set_depth(iz);
		bg_reflect->cu_adjoint(true, down_scattering->data_vec,  wfld_slice_gpu);      
  
    // Copy the current depth slice of the model (slowness) back to host
    size_t offset = iz * this->getSliceSize();
    CHECK_CUDA_ERROR(cudaMemcpyAsync(hslow->getVals(), dslow->mat, this->getSliceSizeInBytes(), cudaMemcpyDeviceToHost, _stream_));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(_stream_)); 
    std::transform(
      hslow->getVals(), hslow->getVals() + getSliceSize(), 
      model[0]->getVals() + offset, model[0]->getVals() + offset,
      std::plus<std::complex<float>>());
  }
};

void ExtendedBorn::upward_scattering_fwd(const std::vector<std::shared_ptr<complex4DReg>>& model) {
  up_scattering->data_vec->zero();
  if (bg_up->get_decomp_queue_size() > 0) 
    throw std::runtime_error("Decompression queue before up-scattering not empty");
  bg_up->start_decompress_from_bottom();
  for (int iz=ax[3].n-1; iz >= 0; --iz) {

    // Copy the current depth slice of the model
    size_t offset = iz * this->getSliceSize();
    CHECK_CUDA_ERROR(cudaMemcpyAsync(dslow->mat, model[0]->getVals() + offset, this->getSliceSizeInBytes(), cudaMemcpyHostToDevice, _stream_));

    // Get the decompressed background wavefields and remove them from the queue
    up_scattering->set_depth(iz);
    // Schedule more decompression
    bg_up->add_decompresss_from_bottom(iz);

    // Record the wavefield
    inj_rec->set_depth(iz);
    inj_rec->cu_adjoint(true, _propagator->data_vec, up_scattering->data_vec);
    // Save the wavefield
		// up->compress_slice(iz, up_scattering->data_vec);

    // This is so that all the wavefields are removed from the decompression queue
    if (iz > 0) {
      up->one_step_fwd(iz-1, up_scattering->data_vec);
      // Up scattering gets recorded at the next depth
      up_scattering->cu_forward(true, dslow, up_scattering->data_vec);
    }
  }
};

void ExtendedBorn::upward_scattering_adj(std::vector<std::shared_ptr<complex4DReg>>& model) {
  up_scattering->data_vec->zero();
  dslow->zero();
  if (bg_up->get_decomp_queue_size() > 0) 
    throw std::runtime_error("Decompression queue before up-scattering not empty");
  bg_up->start_decompress_from_top();
  for (int iz=0; iz < ax[3].n; ++iz) {

    // Get the decompressed background wavefields and remove them from the queue
    up_scattering->set_depth(iz);
    // Schedule more decompression
    bg_up->add_decompresss_from_top(iz);

    // This is so that all the wavefields are removed from the decompression queue
    if (iz > 0) {
      // Up scattering gets recorded at the next depth
      up_scattering->cu_adjoint(false, dslow, up_scattering->data_vec);
      // Propagate wavefield
      up->one_step_adj(iz-1, up_scattering->data_vec);
    }

    // Inject into the wavefield
    inj_rec->set_depth(iz);
    inj_rec->cu_forward(true, _propagator->data_vec, up_scattering->data_vec);
    up->compress_slice(iz, up_scattering->data_vec);

    // Copy the current depth slice of the model
    size_t offset = iz * this->getSliceSize();
    CHECK_CUDA_ERROR(cudaMemcpyAsync(hslow->getVals(), dslow->mat, this->getSliceSizeInBytes(), cudaMemcpyDeviceToHost, _stream_));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(_stream_)); 
    std::transform(
      hslow->getVals(), hslow->getVals() + getSliceSize(), 
      model[0]->getVals() + offset, model[0]->getVals() + offset,
      std::plus<std::complex<float>>());
  }
  up->wait_to_finish();
};


void ExtendedBorn::backward_scattering_fwd(const std::vector<std::shared_ptr<complex4DReg>>& model) {
  back_scattering->data_vec->zero();
  if (bg_down->get_decomp_queue_size() > 0) 
    throw std::runtime_error("Decompression queue before back-scattering not empty");
	bg_down->start_decompress_from_bottom();
  for (int iz=ax[3].n-1; iz >= 0; --iz) {

    size_t offset = iz * this->getSliceSize();

    if (iz < ax[3].n-1) {
      // copy two slices of slowness
      CHECK_CUDA_ERROR(cudaMemcpyAsync(dmodel->mat, model[0]->getVals() + offset, 2*getSliceSizeInBytes(), cudaMemcpyHostToDevice, _stream_));
      // copy two slices of density
      CHECK_CUDA_ERROR(cudaMemcpyAsync(dmodel->mat + 2*getSliceSize(), model[1]->getVals() + offset, 2*getSliceSizeInBytes(), cudaMemcpyHostToDevice, _stream_));
    }
    else {
      // copy the same slice twice to handle the boundary condition
      CHECK_CUDA_ERROR(cudaMemcpyAsync(dmodel->mat, model[0]->getVals() + offset, getSliceSizeInBytes(), cudaMemcpyHostToDevice, _stream_));
      CHECK_CUDA_ERROR(cudaMemcpyAsync(dmodel->mat + getSliceSize(), model[0]->getVals() + offset, getSliceSizeInBytes(), cudaMemcpyHostToDevice, _stream_));
      // same for density
      CHECK_CUDA_ERROR(cudaMemcpyAsync(dmodel->mat + 2*getSliceSize(), model[1]->getVals() + offset, getSliceSizeInBytes(), cudaMemcpyHostToDevice, _stream_));
      CHECK_CUDA_ERROR(cudaMemcpyAsync(dmodel->mat + 3*getSliceSize(), model[1]->getVals() + offset, getSliceSizeInBytes(), cudaMemcpyHostToDevice, _stream_));
    }

    // Get the decompressed background wavefields and remove them from the queue
    back_scattering->set_depth(iz);
		// Schedule the next decompression task to maintain the look-ahead window.
		bg_down->add_decompresss_from_bottom(iz);

		// Enqueue GPU work for the current slice `iz`.
		up->one_step_fwd(iz, back_scattering->data_vec);

		back_scattering->cu_forward(true, dmodel, back_scattering->data_vec);

		inj_rec->set_depth(iz);
		inj_rec->cu_adjoint(true, _propagator->data_vec, back_scattering->data_vec);
  }
};

void ExtendedBorn::backward_scattering_adj(std::vector<std::shared_ptr<complex4DReg>>& model) {
  back_scattering->data_vec->zero();
  if (bg_down->get_decomp_queue_size() > 0) 
    throw std::runtime_error("Decompression queue before back-scattering not empty");
	bg_down->start_decompress_from_top();
  for (int iz=0; iz < ax[3].n; ++iz) {

    // Get the decompressed background wavefields and remove them from the queue
    back_scattering->set_depth(iz);
		// Schedule the next decompression task to maintain the look-ahead window.
		bg_down->add_decompresss_from_top(iz);

    inj_rec->set_depth(iz);
		inj_rec->cu_forward(true, _propagator->data_vec, back_scattering->data_vec);

    back_scattering->cu_adjoint(false, dmodel, back_scattering->data_vec);

    up->one_step_adj(iz, back_scattering->data_vec);

    size_t offset = iz * this->getSliceSize();

    // copy two slices of slowness
    CHECK_CUDA_ERROR(cudaMemcpyAsync(hmodel->getVals(), dmodel->mat, 2*getSliceSizeInBytes(), cudaMemcpyDeviceToHost, _stream_));
    // copy two slices of density
    CHECK_CUDA_ERROR(cudaMemcpyAsync(hmodel->getVals() + 2*getSliceSize(), dmodel->mat + 2*getSliceSize(), 2*getSliceSizeInBytes(), cudaMemcpyDeviceToHost, _stream_));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(_stream_)); 

    if (iz < ax[3].n-1) {
      // Accumulate slowness
      std::transform(
        hmodel->getVals(), hmodel->getVals() + 2*getSliceSize(), 
        model[0]->getVals() + offset, model[0]->getVals() + offset,
        std::plus<std::complex<float>>());
      // Accumulate density
      std::transform(
        hmodel->getVals() + 2*getSliceSize(), hmodel->getVals() + 4*getSliceSize(), 
        model[1]->getVals() + offset, model[1]->getVals() + offset,
        std::plus<std::complex<float>>());
    }
    else {
      // Accumulate slowness twice 
      std::transform(
        hmodel->getVals(), hmodel->getVals() + getSliceSize(), 
        model[0]->getVals() + offset, model[0]->getVals() + offset,
        std::plus<std::complex<float>>());
      // boundary condition
      std::transform(
        hmodel->getVals() + getSliceSize(), hmodel->getVals() + 2*getSliceSize(),
        model[0]->getVals() + offset, model[0]->getVals() + offset,
        std::plus<std::complex<float>>());
      // Accumulate density
      std::transform(
        hmodel->getVals() + 2*getSliceSize(), hmodel->getVals() + 3*getSliceSize(), 
        model[1]->getVals() + offset, model[1]->getVals() + offset,
        std::plus<std::complex<float>>());
        // BOUNDARY CONDITION?
        std::transform(
          hmodel->getVals() + 3*getSliceSize(), hmodel->getVals() + 4*getSliceSize(),
          model[1]->getVals() + offset, model[1]->getVals() + offset,
          std::plus<std::complex<float>>());
    }
  }
};
