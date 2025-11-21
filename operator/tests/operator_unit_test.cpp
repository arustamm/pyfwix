#include <complex4DReg.h>
#include "FFT.h"
#include <gtest/gtest.h>
#include <complex_vector.h>
#include <cuda_runtime.h>
#include "StreamingOperator.h"

bool verbose = false;
double tolerance = 1e-5;

class FFTTest : public testing::Test {
 protected:
  void SetUp() override {
    n1 = 100;
    n2 = 100;
    n3 = 20;
    n4 = 10;
    auto hyper = std::make_shared<hypercube>(n1, n2, n3, n4);
    space4d = std::make_shared<complex4DReg>(hyper);
    space4d->set(1.f);
    cuFFT = std::make_unique<cuFFT2d>(hyper);
  }

  std::unique_ptr<cuFFT2d> cuFFT;
  std::shared_ptr<complex4DReg> space4d;
  int n1, n2, n3, n4;
};

TEST_F(FFTTest, no_memory) {
  auto input = space4d->clone();
  auto output = space4d->clone();
  auto fft2 = cuFFT2d(space4d->getHyper(), cuFFT->model_vec, cuFFT->data_vec);
  ASSERT_NO_THROW(fft2.forward(false, input, output));
}

TEST_F(FFTTest, forward_inverse) {
  auto input = space4d->clone();
  auto output = space4d->clone();
  auto inv = space4d->clone();
  input->random();
  input->scale(2.f);
  cuFFT->forward(false, input, output);
  cuFFT->adjoint(false, inv, output);
  for (int i = 0; i < space4d->getHyper()->getN123(); ++i) {
    EXPECT_NEAR(input->getVals()[i].real(), inv->getVals()[i].real(), 1e-6);
    EXPECT_NEAR(input->getVals()[i].imag(), inv->getVals()[i].imag(), 1e-6);
  }
}

TEST_F(FFTTest, constant_signal) {
  auto input = space4d->clone();
  auto output = space4d->clone();
  cuFFT->forward(false, input, output);
  for (int i4 = 0; i4 < n4; ++i4) {
    for (int i3 = 0; i3 < n3; ++i3) {
      for (int i2 = 0; i2 < n2; ++i2) {
        for (int i1 = 0; i1 < n1; ++i1) {
          // check DC component
          if (i1 == 0 && i2 == 0) EXPECT_NEAR((*output->_mat)[i4][i3][i2][i1].real(), n1*n2/sqrtf(n1*n2), 1e-6);
          EXPECT_NEAR((*output->_mat)[i4][i3][i2][i1].imag(), 0.0f, 1e-6);
        }
      }
    }
  }
}

TEST_F(FFTTest, mono_plane_wave) {
    int k1 = 10;
    int k2 = 20; 
    auto input = space4d->clone();
    auto output = space4d->clone();
    for (int i4 = 0; i4 < n4; ++i4) {
      for (int i3 = 0; i3 < n3; ++i3) {
        for (int i2 = 0; i2 < n2; ++i2) {
          for (int i1 = 0; i1 < n1; ++i1) {
            float phase = 2 * M_PI * (float(k1*i1)/n1 + float(k2*i2)/n2);
            (*input->_mat)[i4][i3][i2][i1] = std::exp(std::complex<float>(0, phase));
          }
        }
      }
    }
    cuFFT->forward(false, input, output);

    // Check frequency components
    for (int i4 = 0; i4 < n4; ++i4) {
      for (int i3 = 0; i3 < n3; ++i3) {
        for (int i2 = 0; i2 < n2; ++i2) {
          for (int i1 = 0; i1 < n1; ++i1) {
            auto val = (*output->_mat)[i4][i3][i2][i1];
            // expect only a spike at k1, k2
            if (i1 == k1 && i2 == k2) {
              auto conj_val = (*output->_mat)[i4][i3][n2-i2][n1-i1];
              EXPECT_NEAR(val.real(), n1*n2/sqrtf(n1*n2), 1e-6);
            }
          }
        }
      }
    }
}

TEST_F(FFTTest, dotTest) { 
  auto err = cuFFT->dotTest(verbose);
  ASSERT_TRUE(err.first <= tolerance);
  ASSERT_TRUE(err.second <= tolerance);
}

class StreamingTest : public testing::Test {
 protected:
  void SetUp() override {
    n1 = 100;
    n2 = 100;
    n3 = 20;
    n4 = 10;
    int nstreams = 4;
    dim3 grid = {32, 4, 4};
    dim3 block = {16, 16, 4};
    
    auto hyper = std::make_shared<hypercube>(n1, n2, n3, n4);
    space4d = std::make_shared<complex4DReg>(hyper);
    space4d->set(1.f);
    streamingFFT = std::make_unique<StreamingOperator<cuFFT2d>>(hyper,grid,block,nstreams);
  }

  std::unique_ptr<StreamingOperator<cuFFT2d>> streamingFFT;
  std::shared_ptr<complex4DReg> space4d;
  int n1, n2, n3, n4;
};

TEST_F(StreamingTest, forward_inverse) {
  auto input = space4d->clone();
  auto output = space4d->clone();
  auto inv = space4d->clone();
  input->random();
  input->scale(2.f);
  streamingFFT->forward(false, input, output);
  streamingFFT->adjoint(false, inv, output);
  for (int i = 0; i < space4d->getHyper()->getN123(); ++i) {
    EXPECT_NEAR(input->getVals()[i].real(), inv->getVals()[i].real(), 1e-6);
    EXPECT_NEAR(input->getVals()[i].imag(), inv->getVals()[i].imag(), 1e-6);
  }
}

TEST_F(StreamingTest, constant_signal) {
  auto input = space4d->clone();
  auto output = space4d->clone();
  streamingFFT->forward(false, input, output);
  for (int i4 = 0; i4 < n4; ++i4) {
    for (int i3 = 0; i3 < n3; ++i3) {
      for (int i2 = 0; i2 < n2; ++i2) {
        for (int i1 = 0; i1 < n1; ++i1) {
          // check DC component
          if (i1 == 0 && i2 == 0) EXPECT_NEAR((*output->_mat)[i4][i3][i2][i1].real(), n1*n2/sqrtf(n1*n2), 1e-6);
          EXPECT_NEAR((*output->_mat)[i4][i3][i2][i1].imag(), 0.0f, 1e-6);
        }
      }
    }
  }
}

TEST_F(StreamingTest, dotTest) { 
  auto err = streamingFFT->dotTest(verbose);
  ASSERT_TRUE(err.first <= tolerance);
  ASSERT_TRUE(err.second <= tolerance);
}


int main(int argc, char **argv) {
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