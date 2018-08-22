/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Henrik Forsten

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "config.h"

#ifdef USE_CUDNN
#include <cassert>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <iterator>
#include <limits>
#include <stdexcept>

#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "CuDNN.h"
#include "Network.h"
#include "GTP.h"
#include "Utils.h"

using namespace Utils;

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

void int8_to_float(int8_t *input, float *output, size_t N);
void scale_float_tensor(float *input, float scale, size_t N);

template <typename net_t>
void CuDNN<net_t>::initialize(const int channels, const int gpu,
                       bool silent) {

	/* For compatibility with OpenCL implementation */
	(void)channels;

    auto best_bandwidth = 0.0f;
    auto found_device = false;
    auto nDevices = 0;
    auto best_device_id = 0;
    cudaDeviceProp best_device;

    cudaGetDeviceCount(&nDevices);

    if (!silent) {
        myprintf("Detected %d CUDA devices.\n", nDevices);
    }

    auto id = 0;

    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        auto bandwidth = 2.0f*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6;
        if (!silent) {
            myprintf("Device Number: %d\n", i);
            myprintf("  Device name: %s\n", prop.name);
            myprintf("  Compute capability: %d.%d\n", prop.major, prop.minor);
            myprintf("  Peak memory bandwidth (GB/s): %.1f\n\n",
                   bandwidth);
        }

        bool preferred = (gpu == id);

        if ( (bandwidth > best_bandwidth) || preferred) {
            best_bandwidth = bandwidth;
            best_device = prop;
            best_device_id = i;
            if (preferred) {
                best_bandwidth = std::numeric_limits<decltype(best_bandwidth)>::max();
            } else {
                best_bandwidth = bandwidth;
            }
            found_device = true;
        }
        id++;
    }

    if (!found_device) {
        throw std::runtime_error("No suitable CUDA device found.");
    }

    myprintf("Selected device: %s\n", best_device.name);
    myprintf("with compute capability %d.%d.\n", best_device.major, best_device.minor);

    cudaSetDevice(best_device_id);

    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    m_handle = cudnn;
    m_init_ok = true;
}

template <typename net_t>
void CuDNN<net_t>::convolve(void *bufferIn,
					 void *bufferOut,
					 void *weights,
					 void *workspace,
					 size_t workspace_bytes,
					 const conv_descriptor& conv_desc,
                     float alpha) {

    const float beta = 0.0f;
	checkCUDNN(cudnnConvolutionForward(m_handle,
                                   &alpha,
                                   conv_desc.input_descriptor,
                                   bufferIn,
                                   conv_desc.kernel_descriptor,
                                   weights,
                                   conv_desc.convolution_descriptor,
                                   conv_desc.convolution_algorithm,
                                   workspace,
                                   workspace_bytes,
                                   &beta,
                                   conv_desc.output_descriptor,
                                   bufferOut));
}

template <typename net_t>
void CuDNN<net_t>::convolveActivation(void *bufferIn,
					 void *bufferOut,
					 void *weights,
					 void *residualBuffer,
					 void *biases,
					 void *workspace,
					 size_t workspace_bytes,
					 const conv_descriptor& conv_desc,
                     float alpha1,
                     float alpha2) {

	void *residual = bufferOut;

	float _alpha2 = 0.0f;
	if (residualBuffer != nullptr) {
		_alpha2 = alpha2;
		residual = residualBuffer;
	}

    /* y = act ( alpha1 * conv(x) + alpha2 * z + bias ) */

	checkCUDNN(cudnnConvolutionBiasActivationForward(
			/* handle */m_handle,
			/* alpha1 */&alpha1,
			/* xDesc */conv_desc.input_descriptor,
		    /* x */bufferIn,
			/* wDesc */conv_desc.kernel_descriptor,
			/* w */weights,
			/* convDesc */conv_desc.convolution_descriptor,
            /* algo */conv_desc.convolution_algorithm,
            /* workSpace */workspace,
            /* workSapceSize */workspace_bytes,
			/* alpha2 */&_alpha2,
			/* zDesc */conv_desc.output_descriptor,
			/* z */residual,
			/* biasDesc */conv_desc.bias_descriptor,
			/* bias */biases,
			/* activationDesc */conv_desc.activation_descriptor,
			/* yDesc */conv_desc.output_descriptor,
		    /* y */bufferOut));
}

template <typename net_t>
size_t CuDNN<net_t>::convolve_init(int channels, int outputs, int kernel_size,
		conv_descriptor& conv_desc) {

    cudnnDataType_t data_type;
    cudnnDataType_t bias_type;
    cudnnDataType_t compute_type;
    cudnnTensorFormat_t tensor_format;

    if (typeid(net_t) == typeid(float) ||
        (typeid(net_t) == typeid(int8_t) && kernel_size == 1) ) {
            /* Convolve1 layers are calculated in single precision when using int8. */
            data_type = CUDNN_DATA_FLOAT;
            bias_type = CUDNN_DATA_FLOAT;
            compute_type = CUDNN_DATA_FLOAT;
            tensor_format = CUDNN_TENSOR_NCHW;
            if (typeid(net_t) == typeid(int8_t)) {
                tensor_format = CUDNN_TENSOR_NHWC;
            }
    } else if (typeid(net_t) == typeid(half_float::half)) {
        data_type = CUDNN_DATA_HALF;
        bias_type = CUDNN_DATA_HALF;
        /* TODO: Use half computation if supported */
        compute_type = CUDNN_DATA_FLOAT;
        tensor_format = CUDNN_TENSOR_NCHW;
    } else {
        data_type = CUDNN_DATA_INT8;
        bias_type = CUDNN_DATA_FLOAT;
        compute_type = CUDNN_DATA_INT32;
        tensor_format = CUDNN_TENSOR_NHWC;
        if (channels % 4 != 0 || outputs % 4 != 0) {
            throw std::runtime_error(
                "Channels and outputs must be divisible by 4 for int8 precision");
        }
    }

	checkCUDNN(cudnnCreateTensorDescriptor(&conv_desc.input_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(conv_desc.input_descriptor,
                                      /*format=*/tensor_format,
                                      /*dataType=*/data_type,
                                      /*batch_size=*/1,
                                      /*channels=*/channels,
                                      /*image_height=*/BOARD_SIZE,
                                      /*image_width=*/BOARD_SIZE));

	checkCUDNN(cudnnCreateTensorDescriptor(&conv_desc.output_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(conv_desc.output_descriptor,
                                      /*format=*/tensor_format,
                                      /*dataType=*/data_type,
                                      /*batch_size=*/1,
                                      /*channels=*/outputs,
                                      /*image_height=*/BOARD_SIZE,
                                      /*image_width=*/BOARD_SIZE));

	checkCUDNN(cudnnCreateTensorDescriptor(&conv_desc.bias_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(conv_desc.bias_descriptor,
                                      /*format=*/tensor_format,
                                      /*dataType=*/bias_type,
                                      /*batch_size=*/1,
                                      /*channels=*/outputs,
                                      /*image_height=*/1,
                                      /*image_width=*/1));

	checkCUDNN(cudnnCreateFilterDescriptor(&conv_desc.kernel_descriptor));
	checkCUDNN(cudnnSetFilter4dDescriptor(conv_desc.kernel_descriptor,
									  /*dataType=*/data_type,
									  /*format=*/tensor_format,
									  /*out_channels=*/outputs,
									  /*in_channels=*/channels,
									  /*kernel_height=*/kernel_size,
									  /*kernel_width=*/kernel_size));

	checkCUDNN(cudnnCreateActivationDescriptor(&conv_desc.activation_descriptor));
	checkCUDNN(cudnnSetActivationDescriptor(conv_desc.activation_descriptor,
									  /*mode=*/CUDNN_ACTIVATION_RELU,
									  /*NanPropagation=*/CUDNN_NOT_PROPAGATE_NAN,
									  /*coef=*/0));

	auto pad_size = 0;

	if (kernel_size == 1) {
		pad_size = 0;
	} else if (kernel_size == 3) {
		pad_size = 1;
	}

	checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc.convolution_descriptor));
	checkCUDNN(cudnnSetConvolution2dDescriptor(conv_desc.convolution_descriptor,
											   /*pad_height=*/pad_size,
											   /*pad_width=*/pad_size,
											   /*vertical_stride=*/1,
											   /*horizontal_stride=*/1,
											   /*dilation_height=*/1,
											   /*dilation_width=*/1,
											   /*mode=*/CUDNN_CROSS_CORRELATION,
											   /*computeType=*/compute_type));

    if (typeid(net_t) == typeid(int8_t)) {
        conv_desc.convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    } else {
        int returnedAlgoCount;

        cudnnConvolutionFwdAlgoPerf_t perf[7];

        checkCUDNN(
            cudnnFindConvolutionForwardAlgorithm(m_handle,
                                                conv_desc.input_descriptor,
                                                conv_desc.kernel_descriptor,
                                                conv_desc.convolution_descriptor,
                                                conv_desc.output_descriptor,
                                                7,
                                                &returnedAlgoCount,
                                                &perf[0]));

        conv_desc.convolution_algorithm = perf[0].algo;

        //myprintf("Layer %d %d %d\n", channels, outputs, kernel_size);
        //for (auto i = 0; i < returnedAlgoCount; i++) {
        //	myprintf("Algo %d\n", perf[i].algo);
        //	myprintf("Time %f\n", perf[i].time);
        //	myprintf("Memory %zu\n", perf[i].memory);
        //	myprintf("\n");
        //}
    }

	size_t workspace_bytes = 0;
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(m_handle,
													   conv_desc.input_descriptor,
													   conv_desc.kernel_descriptor,
													   conv_desc.convolution_descriptor,
													   conv_desc.output_descriptor,
													   conv_desc.convolution_algorithm,
													   &workspace_bytes));
	return workspace_bytes;
}

template <typename net_t>
void CuDNN_Network<net_t>::add_weights_float(size_t layer,
                                 size_t size,
                                 const float * weights) {
    if (layer >= m_layers.size()) {
        m_layers.push_back(CuDNN_Layer());
    }

    auto weightSize = size * sizeof(float);

	void *device_mem;
	auto err = cudaMalloc((void**)&device_mem, weightSize);
	if (err != cudaSuccess) {
		throw std::runtime_error("cudaMalloc failed.");
	}
	err = cudaMemcpy(device_mem, (net_t*)&weights[0], weightSize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		throw std::runtime_error("cudaMemcpy failed.");
	}
    m_layers.back().weights.emplace_back(device_mem);
}

template <typename net_t>
void CuDNN_Network<net_t>::add_weights(size_t layer,
                                 size_t size,
                                 const float * weights) {
    if (layer >= m_layers.size()) {
        m_layers.push_back(CuDNN_Layer());
    }

    auto converted_weights = std::vector<net_t>();
    for(auto i = size_t{0}; i < size; i++) {
        converted_weights.emplace_back(weights[i]);
    }

    auto weightSize = size * sizeof(typename decltype(converted_weights)::value_type);

	void *device_mem;
	auto err = cudaMalloc((void**)&device_mem, weightSize);
	if (err != cudaSuccess) {
		throw std::runtime_error("cudaMalloc failed.");
	}
	err = cudaMemcpy(device_mem, (net_t*)&converted_weights[0], weightSize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		throw std::runtime_error("cudaMemcpy failed.");
	}
    m_layers.back().weights.emplace_back(device_mem);
}

static std::vector<float> zeropad_weights_KRSC(const std::vector<float> &w,
                                   unsigned int K,
                                   unsigned int R,
                                   unsigned int S,
                                   unsigned int C,
                                   unsigned int padding) {
    /* Filter layout KRSC: output, rows, columns, inputs */
    if (C % padding == 0 && K % padding == 0) {
        return w;
    }

    auto C_pad = ceilMultiple(C, padding);
    auto K_pad = ceilMultiple(K, padding);

    std::vector<float> w_pad(K_pad * R * S * C_pad);

    for (auto k = size_t{0}; k < K_pad; k++) {
        for (auto c = size_t{0}; c < C_pad; c++) {
            for (auto r = size_t{0}; r < R; r++) {
                for (auto s = size_t{0}; s < S; s++) {
                    float x = 0.0f;
                    if (k < K && c < C) {
                        x = w[k*R*S*C + r*S*C + s*C + c];
                    }
                    w_pad[k*R*S*C_pad + r*S*C_pad + s*C_pad + c] = x;
                }
            }
        }
    }
    return w_pad;
}

template <typename net_t>
void CuDNN_Network<net_t>::push_input_convolution(unsigned int filter_size,
                   unsigned int channels,
                   unsigned int outputs,
                   const std::vector<float>& weights,
                   const std::vector<float>& biases,
                   float scale) {

    std::vector<float> weights_pad(weights);

    if (typeid(net_t) == typeid(int8_t)) {
        if (outputs % 4 != 0) {
            throw std::runtime_error("Number of filters must be divisible by 4 for int8");
        }
        /* Pad input channels to multiple of 4 */
        weights_pad = zeropad_weights_KRSC(weights, outputs, filter_size,
                                           filter_size, channels, 4);
        channels = ceilMultiple(channels, 4);
    }

    /* int8 has bias as float */
    bool convert_bias = typeid(net_t) != typeid(int8_t);

    size_t layer = get_layer_count();
    push_weights(layer, weights_pad);
    push_weights(layer, biases, convert_bias);
    m_layers[layer].is_input_convolution = true;
    m_layers[layer].outputs = outputs;
    m_layers[layer].filter_size = filter_size;
    m_layers[layer].channels = channels;
    m_layers[layer].scale_1 = 1.0f/scale;
    m_layers[layer].scale_2 = 1.0f/scale;
    m_layers[layer].scale_3 = 1.0f;

    conv_descriptor conv_desc;
    auto wsize = m_cudnn.convolve_init(channels, outputs, filter_size, conv_desc);
    m_layers[layer].conv_desc = conv_desc;
    m_layers[layer].workspace_size = wsize;
}

template <typename net_t>
void CuDNN_Network<net_t>::push_residual(unsigned int filter_size,
                   unsigned int channels,
                   unsigned int outputs,
                   const std::vector<float>& weights_1,
                   const std::vector<float>& biases_1,
                   const std::vector<float>& weights_2,
                   const std::vector<float>& biases_2,
                   float scale_1,
                   float scale_2,
                   float scale_3) {

    /* int8 has bias as float */
    bool convert_bias = typeid(net_t) != typeid(int8_t);

    size_t layer = get_layer_count();
    push_weights(layer, weights_1);
    push_weights(layer, biases_1, convert_bias);
    push_weights(layer, weights_2);
    push_weights(layer, biases_2, convert_bias);
    m_layers[layer].is_residual_block = true;
    m_layers[layer].outputs = outputs;
    m_layers[layer].filter_size = filter_size;
    m_layers[layer].channels = channels;
    m_layers[layer].scale_1 = 1.0f/scale_1;
    m_layers[layer].scale_2 = 1.0f/scale_2;
    m_layers[layer].scale_3 = 1.0f/scale_3;

    conv_descriptor conv_desc;
    auto wsize = m_cudnn.convolve_init(channels, outputs, filter_size, conv_desc);
    m_layers[layer].conv_desc = conv_desc;
    m_layers[layer].workspace_size = wsize;
}

template <typename net_t>
void CuDNN_Network<net_t>::push_convolve(unsigned int filter_size,
                   unsigned int channels,
                   unsigned int outputs,
                   const std::vector<float>& weights) {

    std::vector<float> weights_pad(weights);

    size_t layer = get_layer_count();
    /* Convolve1 is calculated in single precision when using int8 */
    bool convert_weights = typeid(net_t) != typeid(int8_t);

    push_weights(layer, weights_pad, convert_weights);
    m_layers[layer].is_convolve1 = true;
    m_layers[layer].outputs = outputs;
    m_layers[layer].channels = channels;
    m_layers[layer].filter_size = filter_size;

    conv_descriptor conv_desc;
    auto wsize = m_cudnn.convolve_init(channels, outputs, filter_size, conv_desc);
    m_layers[layer].conv_desc = conv_desc;
    m_layers[layer].workspace_size = wsize;
}

template <typename T>
static std::vector<T> NHWC_to_NCHW(const std::vector<T> &x,
                                   unsigned int N,
                                   unsigned int H,
                                   unsigned int W,
                                   unsigned int C) {

    std::vector<T> x_out(N * H * W * C);

    for (auto n = size_t{0}; n < N; n++) {
        for (auto h = size_t{0}; h < H; h++) {
            for (auto w = size_t{0}; w < W; w++) {
                for (auto c = size_t{0}; c < C; c++) {
                    x_out[n*H*W*C + c*H*W + h*W + w] =
                        x[n*H*W*C + h*W*C + w*C + c];
                }
            }
        }
    }
    return x_out;
}

template <typename T>
static std::vector<T> NCHW_to_NHWC(const std::vector<T> &x,
                                   unsigned int N,
                                   unsigned int H,
                                   unsigned int W,
                                   unsigned int C) {

    std::vector<T> x_out(N * H * W * C);

    for (auto n = size_t{0}; n < N; n++) {
        for (auto h = size_t{0}; h < H; h++) {
            for (auto w = size_t{0}; w < W; w++) {
                for (auto c = size_t{0}; c < C; c++) {
                    x_out[n*H*W*C + h*W*C + w*C + c] =
                        x[n*H*W*C + c*H*W + h*W + w];
                }
            }
        }
    }
    return x_out;
}

template <typename net_t>
void CuDNN_Network<net_t>::activation_statistics(const void *InBuffer, const void *ResidualBuffer, Activations<float> &activations, size_t N) {
    std::vector<net_t> output(N);
    std::vector<net_t> residual(N);

    std::vector<float> activations_out;

    cudaMemcpy(&output[0], InBuffer, N * sizeof(net_t), cudaMemcpyDeviceToHost);

    if (ResidualBuffer) {
        cudaMemcpy(&residual[0], ResidualBuffer, N * sizeof(net_t), cudaMemcpyDeviceToHost);
    }

    float dev = 0.0f;
    float max = 0.0f;
    for (auto i = size_t{0}; i < N; i++) {
        auto res = net_t(0.0f);
        if (ResidualBuffer) {
            res = residual[i];
        }
        auto out = float(std::abs(net_t(output[i] - res)));
        max = std::max(out, max);
        dev += out * out;
        activations_out.emplace_back(out);
    }
    dev = std::sqrt(dev / (N - 1));
    //myprintf("max %.2f, dev %.2f\n", float(max), float(dev));
    activations.emplace_back(activations_out);
}

template <typename net_t>
void CuDNN_Network<net_t>::set_scales(const std::vector<float> activations, const int activation_scale) {
    /* Keeps track of how much we scaled the activations so that we can undo
     * the scaling at the end */
    float total_gain = 1.0f;

    int i = 0;
	for (auto iter = std::begin(m_layers); iter != std::end(m_layers); iter++) {
        auto& layer = *iter;

        if (layer.is_input_convolution) {
            auto conv_biases = begin(layer.weights) + 1;

            float g = activation_scale/activations[i];
            total_gain *= g;
            layer.scale_1 *= g;
            scale_float_tensor((float *)conv_biases[0], total_gain, layer.outputs);

            myprintf("Input: g %.2f\n", total_gain);
        } else if (layer.is_residual_block) {
            auto conv1_biases = begin(layer.weights) + 1;
            auto conv2_biases = begin(layer.weights) + 3;

            float g1 = activation_scale/(total_gain * activations[i]);
            total_gain *= g1;
            scale_float_tensor((float *)conv1_biases[0], total_gain, layer.outputs);
            myprintf("Res1: g %.2f\n", total_gain);
            i++;
            layer.scale_1 *= g1;

            float g2 = activation_scale/(total_gain * (activations[i]));
            total_gain *= g2;
            scale_float_tensor((float *)conv2_biases[0], total_gain, layer.outputs);
            myprintf("Res2: g %.2f\n", total_gain);
            layer.scale_2 *= g2;
            i++;

            /* Residual add scaling = residual block scaling */
            /* y = g3 * x + g1 * g2 * x */
            layer.scale_3 *= g1 * g2;

            float g3 = activation_scale/(total_gain * (activations[i]));
            total_gain *= g3;

            myprintf("Res2 add: g %.2f\n", total_gain);
            myprintf("gs %.2f %.2f %.2f\n", g1, g2, g3);
            scale_float_tensor((float *)conv2_biases[0], g3, layer.outputs);
            layer.scale_2 *= g3;
            layer.scale_3 *= g3;

        } else {
            /* Undo scaling at the end */
            myprintf("End: g %.2f\n", total_gain);
            layer.scale_1 = 1.0f/total_gain;
        }
        i++;
    }
}

template <typename net_t>
void CuDNN_Network<net_t>::forward_activations(const std::vector<float>& input,
                                            std::vector<float>& output_pol,
                                            std::vector<float>& output_val,
                                            CuDNNContext & cudnn_context,
                                            Activations<float> *activations,
                                            const int batch_size) {

    /* Always allocates enough space for floats */
    constexpr auto one_plane = BOARD_SQUARES * sizeof(float);
    const auto pol_elements = m_layers[m_layers.size()-2].outputs * BOARD_SQUARES;
    const auto val_elements = m_layers.back().outputs * BOARD_SQUARES;

    auto pol_net_t = std::vector<float>(pol_elements);
    auto val_net_t = std::vector<float>(val_elements);

    if (!cudnn_context.m_buffers_allocated) {
        auto max_wsize = size_t{0};
        auto max_channels = unsigned{0};
        for (const auto& layer : m_layers) {
            max_wsize = std::max(max_wsize, layer.workspace_size);
			max_channels = std::max(max_channels,
								std::max(layer.channels, layer.outputs));
        }
		auto alloc_insize = max_channels * one_plane;

		void *d_workspace;
		auto err = cudaMalloc((void**)&d_workspace, max_wsize);
		if (err != cudaSuccess) {
			throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
		}

		void *d_InBuffer;
		err = cudaMalloc((void**)&d_InBuffer, alloc_insize);
		if (err != cudaSuccess) {
			throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
		}

		void *d_OutBuffer;
		err = cudaMalloc((void**)&d_OutBuffer, alloc_insize);
		if (err != cudaSuccess) {
			throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
		}

		void *d_ResidualBuffer;
		err = cudaMalloc((void**)&d_ResidualBuffer, alloc_insize);
		if (err != cudaSuccess) {
			throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
		}
		cudnn_context.m_workspace = d_workspace;
		cudnn_context.m_InBuffer = d_InBuffer;
		cudnn_context.m_OutBuffer = d_OutBuffer;
		cudnn_context.m_ResidualBuffer = d_ResidualBuffer;
        cudnn_context.m_buffers_allocated = true;

    }

	auto workspace = cudnn_context.m_workspace;
	auto InBuffer = cudnn_context.m_InBuffer;
	auto OutBuffer = cudnn_context.m_OutBuffer;
	auto ResidualBuffer = cudnn_context.m_ResidualBuffer;

    /* Input must be padded with zeros when using int8 since the channels
     * are padded to 20 to be multiple of 4 */
	const auto inSize = sizeof(net_t) * m_layers[0].channels * BOARD_SQUARES;
    auto input_net_t = std::vector<net_t>(m_layers[0].channels * BOARD_SQUARES);

    auto output_t_size = sizeof(net_t);
    if (typeid(net_t) == typeid(int8_t)) {
        output_t_size = sizeof(float);
    }

    float input_scale = 1.0f;
    if (typeid(net_t) == typeid(int8_t)) {
        input_scale = input_scale_int8;
    }

    for (auto i = size_t{0}; i < input.size(); i++) {
        input_net_t[i] = input[i];
    }

    if (typeid(net_t) == typeid(int8_t)) {
        input_net_t = NCHW_to_NHWC<net_t>(input_net_t, batch_size, BOARD_SIZE, BOARD_SIZE, m_layers[0].channels);
    }

	auto err = cudaMemcpy(InBuffer, (net_t*)&input_net_t[0], inSize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
	}

	for (auto iter = std::begin(m_layers); iter != std::end(m_layers); iter++) {
        const auto& layer = *iter;
        const auto niter = std::next(iter);

        if (layer.is_input_convolution) {
            assert(niter != std::end(m_layers));
            auto conv_weights = begin(layer.weights);
            auto conv_biases = begin(layer.weights) + 1;
            m_cudnn.convolveActivation(InBuffer,
					 OutBuffer,
                     conv_weights[0],
					 nullptr,
                     conv_biases[0],
					 workspace,
					 layer.workspace_size,
					 layer.conv_desc,
                     layer.scale_1);
            if (activations != nullptr) {
                activation_statistics(OutBuffer, nullptr, activations[0], layer.outputs * BOARD_SQUARES);
            }

            //std::vector<net_t> output(layer.outputs * BOARD_SQUARES);
            //cudaMemcpy(&output[0], OutBuffer, layer.outputs * BOARD_SQUARES * sizeof(net_t), cudaMemcpyDeviceToHost);
            //if (typeid(net_t) == typeid(int8_t)) {
            //    output = NHWC_to_NCHW<net_t>(output, batch_size, BOARD_SIZE, BOARD_SIZE, layer.outputs);
            //}

            //myprintf("Input:\n");
            //for (auto i = 0; i < BOARD_SIZE; i++) {
            //    for (auto j = 0; j < BOARD_SIZE; j++) {
            //        myprintf("%4d ", int(output[i*BOARD_SIZE + j]));
            //    }
            //    myprintf("\n");
            //}
            //myprintf("\n");
            //if (err != cudaSuccess) {
            //    throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
            //}

			std::swap(InBuffer, OutBuffer);
        } else if (layer.is_residual_block) {
            assert(layer.channels == layer.outputs);
            assert(niter != std::end(m_layers));
            auto conv1_weights = begin(layer.weights);
            auto conv1_biases = begin(layer.weights) + 1;
            auto conv2_weights = begin(layer.weights) + 2;
            auto conv2_biases = begin(layer.weights) + 3;

            m_cudnn.convolveActivation(InBuffer,
					 OutBuffer,
                     conv1_weights[0],
					 nullptr,
                     conv1_biases[0],
					 workspace,
					 layer.workspace_size,
					 layer.conv_desc,
                     layer.scale_1);

            if (activations != nullptr) {
                activation_statistics(OutBuffer, nullptr, activations[0], layer.outputs * BOARD_SQUARES);
            }
            //if (typeid(net_t) == typeid(int8_t)) {
            //    Activations<float> act;
            //    activation_statistics(OutBuffer, nullptr, act, layer.outputs * BOARD_SQUARES);
            //}

            m_cudnn.convolveActivation(OutBuffer,
                         ResidualBuffer,
                         conv2_weights[0],
                         InBuffer,
                         conv2_biases[0],
                         workspace,
                         layer.workspace_size,
                         layer.conv_desc,
                         layer.scale_2,
                         layer.scale_3);

            if (activations != nullptr) {
                /* Convolve activation */
                activation_statistics(ResidualBuffer, InBuffer, activations[0], layer.outputs * BOARD_SQUARES);
                /* Trunk activation */
                activation_statistics(ResidualBuffer, nullptr, activations[0], layer.outputs * BOARD_SQUARES);
            }

            //if (typeid(net_t) == typeid(int8_t)) {
            //    Activations<float> act;
            //    activation_statistics(ResidualBuffer, InBuffer, act, layer.outputs * BOARD_SQUARES);
            //    activation_statistics(ResidualBuffer, nullptr, act, layer.outputs * BOARD_SQUARES);
            //}

            std::swap(InBuffer, ResidualBuffer);
            if (typeid(net_t) == typeid(int8_t) && niter->is_convolve1) {
                /* Convert to float for convolve1 */
                int8_to_float((int8_t*)InBuffer, (float*)OutBuffer, layer.outputs * BOARD_SQUARES);
                std::swap(InBuffer, OutBuffer);
            }
        } else {
            assert(layer.is_convolve1);
            //if (niter == std::end(m_layers)) {
            //    std::vector<float> output(layer.channels * BOARD_SQUARES);
            //    cudaMemcpy(&output[0], InBuffer, layer.channels * BOARD_SQUARES * sizeof(float), cudaMemcpyDeviceToHost);
            //    if (typeid(net_t) == typeid(int8_t)) {
            //        output = NHWC_to_NCHW<float>(output, batch_size, BOARD_SIZE, BOARD_SIZE, layer.channels);
            //    }
            //    myprintf("Output:\n");
            //    for (auto i = 0; i < BOARD_SIZE; i++) {
            //        for (auto j = 0; j < BOARD_SIZE; j++) {
            //            myprintf("%4d ", int(10.0f * output[i*BOARD_SIZE + j] / input_scale));
            //        }
            //        myprintf("\n");
            //    }
            //    myprintf("\n");
            //}

			m_cudnn.convolve(InBuffer,
				 OutBuffer,
				 layer.weights[0],
				 workspace,
				 layer.workspace_size,
				 layer.conv_desc,
                 layer.scale_1);

            if (niter == std::end(m_layers)) {
				/* Value */
				auto err = cudaMemcpy(&val_net_t[0], OutBuffer, val_elements * output_t_size, cudaMemcpyDeviceToHost);
				if (err != cudaSuccess) {
					throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
				}
            } else {
				/* Policy */
				auto err = cudaMemcpy(&pol_net_t[0], OutBuffer, pol_elements * output_t_size, cudaMemcpyDeviceToHost);
				if (err != cudaSuccess) {
					throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
				}
            }
        }
    }

    if (typeid(net_t) == typeid(int8_t)) {
        val_net_t = NHWC_to_NCHW<float>(val_net_t, batch_size, BOARD_SIZE, BOARD_SIZE, 1);
        pol_net_t = NHWC_to_NCHW<float>(pol_net_t, batch_size, BOARD_SIZE, BOARD_SIZE, 2);
    }

    for (auto i = size_t{0}; i < val_elements; i++) {
        output_val[i] = float(val_net_t[i]);
    }

    for (auto i = size_t{0}; i < pol_elements; i++) {
        output_pol[i] = float(pol_net_t[i]);
    }
}

template <typename net_t>
void CuDNN_Network<net_t>::forward(const std::vector<float>& input,
                                            std::vector<float>& output_pol,
                                            std::vector<float>& output_val,
                                            CuDNNContext & cudnn_context,
                                            const int batch_size) {
    forward_activations(input, output_pol, output_val, cudnn_context, nullptr, batch_size);
}

template class CuDNN<float>;
template class CuDNN_Network<float>;
#ifdef USE_HALF
template class CuDNN<half_float::half>;
template class CuDNN_Network<half_float::half>;
#endif
template class CuDNN<int8_t>;
template class CuDNN_Network<int8_t>;

#endif
