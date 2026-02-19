#include <ExtendedMigration.h>

ExtendedMigration::ExtendedMigration (
  const std::shared_ptr<hypercube>& domain, 
  const std::shared_ptr<hypercube>& range,
  const std::vector<std::shared_ptr<complex4DReg>>& slow_den, 
  std::shared_ptr<Propagator> propagator,
  dim3 grid, dim3 block, cudaStream_t stream) :
_propagator(propagator),
CudaOperator<complex4DReg, complex2DReg>(domain, range, grid, block, stream) {

  // Initialize the propagator
  bg_down = propagator->getDown();
  inj_rec = propagator->getInjRec();

  auto wfld_hyper = propagator->getWfldSliceHyper();
  auto par = propagator->getDown()->getPar();

  _slow = slow_den[0];
  _den = slow_den[1];
  auto m_ax = _slow->getHyper()->getAxes();

  // CHECK_CUDA_ERROR(cudaHostRegister(_slow->getVals(), _slow->getHyper()->getN123()*sizeof(std::complex<float>), cudaHostRegisterDefault));
  // CHECK_CUDA_ERROR(cudaHostRegister(_den->getVals(), _den->getHyper()->getN123()*sizeof(std::complex<float>), cudaHostRegisterDefault));

  auto run_id = propagator->getRunId();
  run_id.insert(0, "extended_born_");
  auto _pool_up = std::make_shared<WavefieldPool>(wfld_hyper, par, run_id, m_ax[3].n);

	up = std::make_shared<Upward>(wfld_hyper, 
    _slow->getHyper(), par, 
    propagator->getRefSampler(), _pool_up, "up",
    inj_rec->data_vec, inj_rec->data_vec, _grid_, _block_, _stream_);

  ax = domain->getAxes();

  // Allocatge the data vector
  // Here we use the objects from the propagator that contains background wavefields
  auto subhyper3d = std::make_shared<hypercube>(std::vector<axis>{ax[0], ax[1], ax[2]});
  himage = std::make_shared<complex3DReg>(subhyper3d);
  dimage = make_complex_vector(subhyper3d, _grid_, _block_, _stream_);
  ic = std::make_shared<ImagingCondition>(subhyper3d, wfld_hyper, bg_down, dimage, inj_rec->data_vec, grid, block, stream);

  // CHECK_CUDA_ERROR(cudaHostRegister(himage->getVals(), himage->getHyper()->getN123()*sizeof(std::complex<float>), cudaHostRegisterDefault));
};

size_t ExtendedMigration::getSliceSize() const {
  // return the size of the slice
  return static_cast<size_t>(ax[0].n * ax[1].n * ax[2].n);
}

size_t ExtendedMigration::getSliceSizeInBytes() const {
  // return the size of the slice in bytes
  return getSliceSize() * sizeof(std::complex<float>);
}

void ExtendedMigration::migrate(bool add, std::shared_ptr<complex4DReg> image, std::shared_ptr<complex2DReg> data) {

  // CHECK_CUDA_ERROR(cudaHostRegister(data->getVals(), getRangeSizeInBytes(), cudaHostRegisterDefault));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(_propagator->data_vec->mat, data->getVals(), getRangeSizeInBytes(), cudaMemcpyHostToDevice, _stream_));

  // zero out the wavefield
  ic->data_vec->zero();
  if(!add) image->zero();

  for (int iz=0; iz < ax[3].n; ++iz) {

    ic->set_depth(iz);

    inj_rec->set_depth(iz);
    inj_rec->cu_forward(true, _propagator->data_vec, ic->data_vec);

    ic->cu_adjoint(false, dimage, ic->data_vec);

    up->one_step_adj(iz, ic->data_vec);

    size_t offset = iz * this->getSliceSize();

    // copy the image
    CHECK_CUDA_ERROR(cudaMemcpyAsync(himage->getVals(), dimage->mat, getSliceSizeInBytes(), cudaMemcpyDeviceToHost, _stream_));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(_stream_)); 

    // Accumulate slowness
    std::transform(
      himage->getVals(), himage->getVals() + getSliceSize(), 
      image->getVals() + offset, image->getVals() + offset,
      std::plus<std::complex<float>>());
    
  }

    // unpin the memory
  // CHECK_CUDA_ERROR(cudaHostUnregister(data->getVals()));

}

