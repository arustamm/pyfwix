
#include <OneWay.h>

using namespace SEP;

void OneWay::one_step_fwd(int iz, complex_vector* __restrict__ wfld) {

	prop->set_depth(iz);
	prop->cu_forward(wfld);

}

void OneWay::one_step_adj(int iz, complex_vector* __restrict__ wfld) {

	// propagate one step by changing the state of the wavefield
	prop->set_depth(iz);
	prop->cu_adjoint(wfld);
}

void Downward::cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

	if(!add) data->zero();

	for (int iz=0; iz < m_ax[3].n; ++iz) {
		this->compress_slice(iz, model);
		// Enqueue more work!
		this->one_step_fwd(iz, model);
	}

	_wfld_pool->wait_to_finish();

	data->add(model);

}

void Downward::cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

	if(!add) model->zero();

	for (int iz=m_ax[3].n; iz > 0; --iz) {
		this->compress_slice(iz, model);
		// propagate one step
		this->one_step_adj(iz-1, data);
	}

	model->add(data);


}

void Upward::cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

	if(!add) data->zero();

	// for (batches in z)

	for (int iz=m_ax[3].n; iz > 0; --iz) {
		// Enqueue more work!
		this->one_step_fwd(iz-1, model);
	}

  _wfld_pool->wait_to_finish();

	data->add(model);

}

void Upward::cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

	if(!add) model->zero();

	for (int iz=0; iz < m_ax[3].n; ++iz) {
		// propagate one step
		this->one_step_adj(iz, data);
	}

	model->add(data);

}
