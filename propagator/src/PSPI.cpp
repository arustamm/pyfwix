
#include <OneStep.h>

using namespace SEP;

void PSPI::cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

	if(!add) data->zero();

	taper->cu_forward(0,model,model_k);
	fft2d->cu_forward(model_k);

	for (int iref=0; iref < _nref_; ++iref) {

		ps->set_slow(_ref_->get_ref_slow(get_depth(),iref));
		ps->cu_forward(0, model_k, _wfld_ref);

		fft2d->cu_adjoint(_wfld_ref);
		
		select->set_value(iref);
		select->cu_forward(1, _wfld_ref,data);
	}

}

void PSPI::cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

		if(!add)  model->zero();
		model_k->zero();

		for (int iref=0; iref < _nref_; ++iref) {

			select->set_value(iref);
			select->cu_adjoint(0, _wfld_ref,data);

			fft2d->cu_forward(_wfld_ref);

			ps->set_slow(_ref_->get_ref_slow(get_depth(),iref));
			ps->cu_adjoint(1, model_k, _wfld_ref);
		}

		fft2d->cu_adjoint(1, model, model_k);

}

// void PSPI::cu_inverse(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

// 		if(!add)  model->zero();
// 		model_k->zero();

// 		for (int iref=0; iref < _nref_; ++iref) {

// 			select->set_value(iref);
// 			select->cu_adjoint(0, _wfld_ref,data);

// 			fft2d->cu_forward(_wfld_ref);

// 			ps->set_slow(_ref_->get_ref_slow(get_depth(),iref));
// 			ps->cu_inverse(1, model_k, _wfld_ref);
// 		}

// 		fft2d->cu_adjoint(1, model, model_k);

// }

void PSPI::cu_forward(complex_vector* __restrict__ model) {

		taper->cu_forward(0, model,model_k);
	  	fft2d->cu_forward(model_k);
		model->zero();

		for (int iref=0; iref < _nref_; ++iref) {

			ps->set_slow(_ref_->get_ref_slow(get_depth(),iref));
			ps->cu_forward(0, model_k, _wfld_ref);

			fft2d->cu_adjoint(_wfld_ref);
			
			select->set_value(iref);
			select->cu_forward(1, _wfld_ref, model);
		}

}

void PSPI::cu_adjoint(complex_vector* __restrict__ data) {

		model_k->zero();

		for (int iref=0; iref < _nref_; ++iref) {

			select->set_value(iref);
			select->cu_adjoint(0, _wfld_ref,data);

			fft2d->cu_forward(_wfld_ref);

			ps->set_slow(_ref_->get_ref_slow(get_depth(),iref));
			ps->cu_adjoint(1, model_k, _wfld_ref);
		}

		fft2d->cu_adjoint(model_k);
		taper->cu_adjoint(0, data, model_k);

}
