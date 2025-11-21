
#include <OneStep.h>

using namespace SEP;

void NSPS::cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

	  if(!add) model->zero();

		// pad->forward(model,model_pad,0);
	  fft2d->cu_forward(0,data,model_k);

		for (int iref=0; iref < _nref_; ++iref) {

			ps->set_slow(_ref_->get_ref_slow(get_depth(),iref));
			ps->cu_adjoint(0, model_k, _wfld_ref);

			fft2d->cu_adjoint(_wfld_ref);
			// // taper->forward(_wfld_ref,_wfld_ref,1);
			
			select->set_value(iref);
			select->cu_forward(1, _wfld_ref,model);
		}

}

void NSPS::cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

		if(!add)  data->zero();
		model_k->zero();

		for (int iref=0; iref < _nref_; ++iref) {

			select->set_value(iref);
			select->cu_adjoint(0, _wfld_ref,model);

			fft2d->cu_forward(_wfld_ref);

			ps->set_slow(_ref_->get_ref_slow(get_depth(),iref));
			ps->cu_forward(1, _wfld_ref, model_k);
		}

		fft2d->cu_adjoint(1, data, model_k);

}
