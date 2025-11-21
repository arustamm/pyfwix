#include <complex4DReg.h>
#include "PhaseShift.h"
#include <gtest/gtest.h>
#include <complex_vector.h>
#include <cuda_runtime.h>
#include  <Scatter.h>
#include <ImagingCondition.h>
#include <LinReflect.h>
#include <Scattering.h>
#include <Propagator.h>
#include <ExtendedBorn.h>

#include <jsonParamObj.h>
#include <random>

bool verbose = false;
double tolerance = 1e-5;

class Scatter_Test : public testing::Test {
 protected:
	void SetUp() override {
		auto ax = axis(100, 0, 0.01);
		auto ay = axis(100, 0, 0.01);
		auto az = axis(10, 0, 0.01);
		auto aw = axis(15, 1., 0.1);
		auto as = axis(5, 0, 1);;

		auto slow4d = std::make_shared<complex4DReg>(ax, ay, aw, az);
		slow4d->set(1.f);

		// create a vector of slowness values for each frequency
		auto domain = std::make_shared<hypercube>(ax, ay, aw, as);
		space4d = std::make_shared<complex4DReg>(domain);
		space4d->set(1.f);

		Json::Value root;
		auto par = std::make_shared<jsonParamObj>(root);
		dim3 grid = {32, 4, 4};
		dim3 block = {16, 16, 4};
		scatter = std::make_unique<Scatter>(domain, slow4d, par, nullptr, nullptr, grid, block);
		scatter->set_depth(5);
	}

	std::unique_ptr<Scatter> scatter;
	std::shared_ptr<complex4DReg> space4d;
};

TEST_F(Scatter_Test, fwd) {
	auto out = space4d->clone();
	for (int i=0; i < 3; ++i)
		ASSERT_NO_THROW(scatter->forward(false, space4d, out));
}

TEST_F(Scatter_Test, cu_fwd) {
	auto out = space4d->clone();
	ASSERT_NO_THROW(scatter->cu_forward(false, scatter->model_vec, scatter->data_vec));
	ASSERT_NO_THROW(scatter->cu_forward(scatter->data_vec));
}

TEST_F(Scatter_Test, dotTest) {
	auto err = scatter->dotTest(verbose);
	ASSERT_TRUE(err.first <= tolerance);
	ASSERT_TRUE(err.second <= tolerance);
}

class IC_Test : public testing::Test {
 protected:
	void SetUp() override {

		ax = {
			axis(100, 0, 0.01), // x-axis
			axis(100, 0, 0.01), // y-axis
			axis(15, 1., 0.1),  // w-axis
			axis(5, 0, 1),       // s-axis
			axis(10, 0, 0.01),  // z-axis
		};

		auto range = std::make_shared<hypercube>(ax[0], ax[1], ax[2], ax[3]);
		wfld1 = std::make_shared<complex4DReg>(range);
		wfld2 = std::make_shared<complex4DReg>(range);

		auto slow4d = std::make_shared<complex4DReg>(ax[0], ax[1], ax[2], ax[4]);
		slow4d->set(1.f);
		dslow = std::make_shared<complex3DReg>(ax[0], ax[1], ax[2]);

		Json::Value root;
		root["nref"] = 3;
		auto par = std::make_shared<jsonParamObj>(root);

		down = std::make_shared<Downward>(range, slow4d, par);
		// fill in the background wavefield
		wfld1->random();
		for (int i=0; i < ax[4].n; ++i) 
			down->forward(wfld1);

		ic = std::make_unique<ImagingCondition>(dslow->getHyper(), range, down);
	}

	std::unique_ptr<ImagingCondition> ic;
	std::shared_ptr<Downward> down;
	std::vector<axis> ax;
	std::shared_ptr<complex3DReg> dslow;
	std::shared_ptr<complex4DReg> wfld1, wfld2;
};

TEST_F(IC_Test, cu_fwd) {
	dslow->set(1.f);
	down->start_decompress_from_top();

	for (int i=0; i < ax[4].n; ++i) {
		ASSERT_NO_THROW(ic->set_depth(i));
		ASSERT_NO_THROW(ic->forward(false, dslow, wfld1));
		ASSERT_NO_THROW(down->add_decompresss_from_top(i));
		ASSERT_TRUE(std::real(wfld1->dot(wfld1)) > 0.0) << "Forward imaging condition failed at depth " << i;
	}
}

TEST_F(IC_Test, dotTest) {
	ASSERT_NO_THROW(down->add_decompresss_from_top(5));
	ic->set_depth(5);
	auto err = ic->dotTest(verbose);
	ASSERT_TRUE(err.first <= tolerance);
	ASSERT_TRUE(err.second <= tolerance);
}

class LinReflect_Test : public testing::Test {
  protected:
   void SetUp() override {
     nx = 100;
     auto ax1 = axis(nx, 0.f, 0.01f);
     ny = 100;
     auto ax2 = axis(ny, 0.f, 0.01f);
     nw = 10;
     auto ax3 = axis(nw, 1.f, 1.f);
     ns = 5;
     auto ax4 = axis(ns, 0.f, 1.f);
     nz = 10;
     auto ax5 = axis(nz, 0.f, 0.01f);
	
	 auto domain = std::make_shared<hypercube>(ax1, ax2, ax3, axis(2, 0, 1), axis(2, 0, 1));
     auto range = std::make_shared<hypercube>(ax1, ax2, ax3);
     model = std::make_shared<complex5DReg>(domain);
     data = std::make_shared<complex3DReg>(range);
 
     auto slow4d = std::make_shared<complex4DReg>(nx, ny, nw, nz);
     slow4d->random();
     auto den4d = std::make_shared<complex4DReg>(nx, ny, nw, nz);
     den4d->random();
     std::vector<std::shared_ptr<complex4DReg>> slow_den = {slow4d, den4d};

    lin_refl = std::make_unique<LinReflect>(domain, range, slow_den);
    lin_refl->set_grid({32, 4, 4});
    lin_refl->set_block({16, 16, 4});
    lin_refl->set_depth(5);
   }
 
   std::unique_ptr<LinReflect> lin_refl;
   int nx, ny, nz, nw, ns;
   std::shared_ptr<complex5DReg> model;
   std::shared_ptr<complex3DReg> data;
 };

 TEST_F(LinReflect_Test, set_depth) {
  for (int i=nz-1; i > 0; --i) 
  ASSERT_NO_THROW(lin_refl->set_depth(i));
}
 
 TEST_F(LinReflect_Test, fwd) { 
	model->random();
   for (int i=0; i < 3; ++i) {
     ASSERT_NO_THROW(lin_refl->forward(false, model, data));
	 std::cout << "Data norm after fwd: " << std::real(data->dot(data)) << "\n";
   }
 }

 TEST_F(LinReflect_Test, adj) { 
  for (int i=0; i < 3; ++i)
    ASSERT_NO_THROW(lin_refl->adjoint(false, model, data));
}

TEST_F(LinReflect_Test, dot) { 
  auto err = lin_refl->dotTest(verbose);
  ASSERT_TRUE(err.first <= tolerance);
  ASSERT_TRUE(err.second <= tolerance);
}

class ForwardScattering_Test : public testing::Test {
 protected:
	void SetUp() override {

		ax = {
			axis(100, 0, 0.01), // x-axis
			axis(100, 0, 0.01), // y-axis
			axis(15, 1., 0.1),  // w-axis
			axis(5, 0, 1),       // s-axis
			axis(10, 0, 0.01),  // z-axis
		};

		auto range = std::make_shared<hypercube>(ax[0], ax[1], ax[2], ax[3]);
		wfld1 = std::make_shared<complex4DReg>(range);
		wfld2 = std::make_shared<complex4DReg>(range);

		auto slow4d = std::make_shared<complex4DReg>(ax[0], ax[1], ax[2], ax[4]);
		slow4d->set(1.f);
		dslow = std::make_shared<complex3DReg>(ax[0], ax[1], ax[2]);

		Json::Value root;
		root["nref"] = 3;
		auto par = std::make_shared<jsonParamObj>(root);

		down = std::make_shared<Downward>(range, slow4d, par);
		// fill in the background wavefield
		wfld1->random();
		for (int i=0; i < ax[4].n; ++i) 
			down->forward(wfld1);

		fscat = std::make_unique<ForwardScattering>(dslow->getHyper(), range, slow4d, down);
	}

	std::unique_ptr<ForwardScattering> fscat;
	std::shared_ptr<Downward> down;
	std::vector<axis> ax;
	std::shared_ptr<complex3DReg> dslow;
	std::shared_ptr<complex4DReg> wfld1, wfld2;
};

TEST_F(ForwardScattering_Test, cu_fwd) {
	dslow->set(1.f);
	down->start_decompress_from_top();
	for (int i=0; i < ax[4].n; ++i) {
		ASSERT_NO_THROW(fscat->set_depth(i));
		ASSERT_NO_THROW(fscat->forward(false, dslow, wfld1));
		ASSERT_NO_THROW(down->add_decompresss_from_top(i));
		ASSERT_TRUE(std::real(wfld1->dot(wfld1)) > 0.0) << "Forward scattering failed at depth " << i;
	}
}

TEST_F(ForwardScattering_Test, dotTest) {
	ASSERT_NO_THROW(down->add_decompresss_from_top(5));
	fscat->set_depth(5);
	auto err = fscat->dotTest(verbose);
	ASSERT_TRUE(err.first <= tolerance);
	ASSERT_TRUE(err.second <= tolerance);
}

class BackScattering_Test : public testing::Test {
 protected:
	void SetUp() override {

		ax = {
			axis(100, 0, 0.01), // x-axis
			axis(100, 0, 0.01), // y-axis
			axis(15, 1., 0.1),  // w-axis
			axis(5, 0, 1),       // s-axis
			axis(10, 0, 0.01),  // z-axis
		};

		auto domain = std::make_shared<hypercube>(ax[0], ax[1], ax[2], axis(2, 0, 1), axis(2, 0, 1));
		auto range = std::make_shared<hypercube>(ax[0], ax[1], ax[2], ax[3]);
		wfld1 = std::make_shared<complex4DReg>(range);
		wfld2 = std::make_shared<complex4DReg>(range);

		auto slow4d = std::make_shared<complex4DReg>(ax[0], ax[1], ax[2], ax[4]);
		slow4d->set(1.f);
		auto den4d = slow4d->clone();
		auto model = std::vector<std::shared_ptr<complex4DReg>>{slow4d, den4d};
		
		dmodel = std::make_shared<complex5DReg>(domain);
		
		Json::Value root;
		root["nref"] = 3;
		auto par = std::make_shared<jsonParamObj>(root);

		down = std::make_shared<Downward>(range, slow4d, par);
		// fill in the background wavefield
		wfld1->random();
		for (int i=0; i < ax[4].n; ++i) 
			down->forward(wfld1);

		bscat = std::make_unique<BackScattering>(domain, range, model, down);
	}

	std::unique_ptr<BackScattering> bscat;
	std::shared_ptr<Downward> down;
	std::vector<axis> ax;
	std::shared_ptr<complex5DReg> dmodel;
	std::shared_ptr<complex4DReg> wfld1, wfld2;
};

TEST_F(BackScattering_Test, cu_fwd) {
	dmodel->random();
	down->start_decompress_from_top();
	for (int i=0; i < ax[4].n; ++i) {
		ASSERT_NO_THROW(bscat->set_depth(i));
		ASSERT_NO_THROW(bscat->forward(false, dmodel, wfld1));
		ASSERT_NO_THROW(down->add_decompresss_from_top(i));
		ASSERT_TRUE(std::real(wfld1->dot(wfld1)) > 0.0) << "Forward scattering failed at depth " << i;
	}
}

TEST_F(BackScattering_Test, dotTest) {
	ASSERT_NO_THROW(down->add_decompresss_from_top(5));
	bscat->set_depth(5);
	auto err = bscat->dotTest(verbose);
	ASSERT_TRUE(err.first <= tolerance);
	ASSERT_TRUE(err.second <= tolerance);
}

class ExtendedBorn_Test : public testing::Test {
protected:
	void SetUp() override {

		// Create fixed geometry
		int nsrc = 5;  // Number of source traces
		int nrec = 20; // Number of receivers (10 top + 10 bottom)
		int ntrace = nrec;
		
		ax = {
			axis(100, 0, 0.01), // x-axis
			axis(100, 0, 0.01), // y-axis
			axis(15, 1., 0.1),  // w-axis
			axis(nsrc, 0, 1),       // s-axis
			axis(10, 0, 0.01),  // z-axis
		};

		traces = std::make_shared<complex2DReg>(ax[2].n, ntrace);
		auto range = traces->getHyper();

		auto sources = std::make_shared<complex2DReg>(ax[2].n, nsrc);
		sources->set(1.f);

		auto slow4d = std::make_shared<complex4DReg>(ax[0], ax[1], ax[2], ax[4]);
		slow4d->set(1.f);
		// add a reflector at the bottom
		for (int iz = ax[4].n-3; iz < ax[4].n; ++iz)
			for (int iw = 0; iw < ax[2].n; ++iw)
				for (int iy = 0; iy < ax[1].n; ++iy)
					for (int ix = 0; ix < ax[0].n; ++ix)
						(*slow4d->_mat)[iz][iw][iy][ix] = {2.f, 0.f};

		auto den4d = std::make_shared<complex4DReg>(ax[0], ax[1], ax[2], ax[4]);
		den4d->set(1.f);
		slow_den = {slow4d, den4d};
		auto domain = slow4d->getHyper();

		// Calculate model bounds
		float x_min = ax[0].o + ax[0].d;                    // 0.01
		float x_max = ax[0].o + (ax[0].n - 2) * ax[0].d;         // 0.98
		float y_min = ax[1].o + ax[1].d;                    // 0.01
		float y_max = ax[1].o + (ax[1].n - 2) * ax[1].d;         // 0.98
		float z_min = ax[4].o + ax[4].d;                          // 0.01
		float z_max = ax[4].o + (ax[4].n - 2) * ax[4].d;               // 0.08

		// Source coordinates - evenly spaced at the top of the model
		std::vector<float> src_x(nsrc);
		std::vector<float> src_y(nsrc);
		std::vector<float> src_z(nsrc);
		std::vector<int> src_ids(nsrc);

		for (int i = 0; i < nsrc; ++i) {
			// Evenly distribute sources between x_min and x_max
			src_x[i] = x_min + i * (x_max - x_min) / (nsrc - 1);
			src_y[i] = (y_min + y_max) / 2.0f;          // Center in y
			src_z[i] = z_min + 0.02;                           // Top of model
			src_ids[i] = i % nsrc;                        // Cycle through source IDs
		}

		// Receiver coordinates - half at top, half at bottom
		std::vector<float> rec_x(nrec);
		std::vector<float> rec_y(nrec);
		std::vector<float> rec_z(nrec);
		std::vector<int> rec_ids(nrec);

		for (int i = 0; i < nrec; ++i) {
			// Evenly distribute receivers between x_min and x_max
			rec_x[i] = x_min + i * (x_max - x_min) / (nrec - 1);
			rec_y[i] = (y_min + y_max) / 2.0f;          // Center in y
			rec_ids[i] = i % nsrc;                        // Cycle through receiver IDs
			
			// Half receivers at top, half at bottom
			// if (i < nrec / 2) {
				rec_z[i] = z_min;                       // Top of model
			// } else {
			// 	rec_z[i] = z_max;                       // Bottom of model
			// }
		}

		Json::Value root;
		root["nref"] = 11;
		root["padx"] = ax[0].n;
		root["pady"] = ax[1].n;
		root["taperx"] = 0;
		root["tapery"] = 0;
		root["compress_error"] = 0;	// lossless compression
		auto par = std::make_shared<jsonParamObj>(root);

		auto prop = std::make_shared<Propagator>(
			sources->getHyper(), traces->getHyper(),
			slow4d->getHyper(), 
			sources, src_x, src_y, src_z, src_ids,
			rec_x, rec_y, rec_z, rec_ids,
			par
		);

		prop->forward(false, slow_den, traces);

		born = std::make_unique<ExtendedBorn>(domain, range, slow_den, prop);
		dmodel = {slow4d->clone(), den4d->clone()};
		dmodel[0]->zero();
		dmodel[1]->zero();
		(*dmodel[0]->_mat)[5][0][50][50] = {0.1f, 0.f};
}

	std::unique_ptr<ExtendedBorn> born;
	std::vector<axis> ax;
	std::shared_ptr<complex2DReg> traces;
	std::vector<std::shared_ptr<complex4DReg>> slow_den;
	std::vector<std::shared_ptr<complex4DReg>> dmodel;
};

TEST_F(ExtendedBorn_Test, fwd) {
	ASSERT_TRUE(std::real(traces->dot(traces)) > 0.0) << "The data after prop is zero";
	ASSERT_NO_THROW(born->forward(false, dmodel, traces));
	ASSERT_TRUE(std::real(traces->dot(traces)) > 0.0) << "The data after born is zero";
}

TEST_F(ExtendedBorn_Test, adj) {
	ASSERT_NO_THROW(born->adjoint(false, dmodel, traces));
	ASSERT_TRUE(std::real(dmodel[0]->dot(dmodel[0])) > 0.0) << "The model after born adjoint is zero";
}

TEST_F(ExtendedBorn_Test, dotTest) {
	auto err = born->dotTest(verbose);
	ASSERT_TRUE(err.first <= tolerance);
	ASSERT_TRUE(err.second <= tolerance);
}


int main(int argc, char **argv) {
	// Parse command-line arguments
	for (int i = 1; i < argc; ++i) {
		if (std::string(argv[i]) == "--verbose") {
			verbose = true;
		}
		else if (std::string(argv[i]) == "--tolerance" && i + 1 < argc) {
			tolerance = std::stod(argv[i + 1]);
		}
	}
		testing::InitGoogleTest(&argc, argv);
		return RUN_ALL_TESTS();
}