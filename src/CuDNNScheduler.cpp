/*
    This file is part of Leela Zero.
    Copyright (C) 2018 Junhee Yoo and contributors

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
#include "GTP.h"
#include "Random.h"
#include "Network.h"
#include "CuDNNScheduler.h"
#include "Utils.h"

static void bn_stddivs_to_conv(std::vector<float>& w,
                               const std::vector<float>& bn_stddivs,
                               std::vector<float>& bn_means,
                               const int outputs, const int channels) {

    for(auto o = 0; o < outputs; o++) {
        for(auto c = 0; c < channels; c++) {
            for(auto i = 0; i < 9; i++) {
                w[o*channels*9 + c*9 + i] *= bn_stddivs[o];
            }
        }
        // Multiply by -1 to convert to bias
        bn_means[o] *= -bn_stddivs[o];
    }
}

template <typename T>
static std::vector<T> KCRS_to_KRSC(const std::vector<T> &x,
                                   unsigned int K,
                                   unsigned int C,
                                   unsigned int R,
                                   unsigned int S) {
    /* K = outputs
     * C = inputs
     * R = rows
     * S = columns */

    std::vector<T> x_out(K * C * R * S);

    for (auto k = size_t{0}; k < K; k++) {
        for (auto c = size_t{0}; c < C; c++) {
            for (auto r = size_t{0}; r < R; r++) {
                for (auto s = size_t{0}; s < S; s++) {
                    x_out[k*R*S*C + r*S*C + s*C + c] =
                        x[k*C*R*S + c*R*S + r*S + s];
                }
            }
        }
    }
    return x_out;
}

static std::vector<float> scale_weights(const std::vector<float> &w,
                                        float scale, bool quantize) {

    std::vector<float> w_out(w.size());

    for (auto i = size_t{0}; i < w.size(); i++) {
        float x = scale * w[i];
        if (quantize) {
            if (x > 127) {
                x = 127;
            } else if (x < -127) {
                x = -127;
            }
            x = std::round(x);
        }
        w_out[i] = x;
    }
    return w_out;
}

static float abs_max(const std::vector<float> &w) {
    auto max = 0.0f;
    for (auto i = size_t{0}; i < w.size(); i++) {
        max = std::max(std::abs(w[i]), max);
    }
    return max;
}

static float mean(const std::vector<float> &w) {
    auto mean = 0.0f;
    for (auto i = size_t{0}; i < w.size(); i++) {
        mean += w[i];
    }
    return mean / w.size();
}

static float std_dev(const std::vector<float> &w) {
    auto dev = 0.0f;
    auto mu = mean(w);

    for (auto i = size_t{0}; i < w.size(); i++) {
        dev += (w[i] - mu) * (w[i] - mu);
    }
    return std::sqrt(dev / (w.size() - 1));
}

template <typename net_t>
void CuDNNScheduler<net_t>::initialize(const int channels) {
    // multi-gpu?
    auto gpus = cfg_gpus;

    // an empty GPU list from the command line represents autodetect.
    // put a minus one GPU index here.
    if (gpus.empty()) {
        gpus = {-1};
    }

    auto silent{false};
    auto gnum = size_t{0};

    // launch the worker thread.  round_up(cfg_num_threads / gpus.size()) threads
    // so that we only have enough contexts to achieve full parallelism.
    const auto num_threads = (cfg_num_threads + gpus.size() - 1) / gpus.size();
    m_context_pool.resize(num_threads);

    for (auto gpu : gpus) {
        auto cudnn = std::make_unique<CuDNN<net_t>>();
        auto net = std::make_unique<CuDNN_Network<net_t>>(*cudnn);
        cudnn->initialize(channels, gpu, silent);
        m_cudnn.push_back(std::move(cudnn));
        m_networks.push_back(std::move(net));

        // starting next GPU, let's not dump full list of GPUs
        silent = true;

        for (auto i = size_t{0}; i < num_threads; i++) {
            m_context_pool[i].emplace_back(std::make_shared<ContextPoolEntry>(gnum));
        }
        gnum++;
    }
}

template <typename net_t>
void CuDNNScheduler<net_t>::push_input_convolution(unsigned int filter_size,
                                                    unsigned int channels,
                                                    unsigned int outputs,
                                                    const std::vector<float>& weights,
                                                    const std::vector<float>& means,
                                                    const std::vector<float>& variances) {
    for (const auto& cudnn_net : m_networks) {
        std::vector<float> weights_conv = std::vector<float>(weights);
        std::vector<float> means_conv = std::vector<float>(means);

        bn_stddivs_to_conv(weights_conv,
                           variances,
                           means_conv,
                           outputs, channels);

        float scale = 1.0f;
        if (typeid(net_t) == typeid(int8_t)) {
            scale = 127.0f/abs_max(weights_conv);
            weights_conv = scale_weights(weights_conv, scale, true);
            weights_conv = KCRS_to_KRSC<float>(weights_conv,
                                               outputs,
                                               channels,
                                               filter_size,
                                               filter_size);

            means_conv = scale_weights(means_conv, input_scale_int8, false);
        }

        cudnn_net->push_input_convolution(
            filter_size, channels, outputs,
            weights_conv, means_conv, scale
        );
    }
}

template <typename net_t>
void CuDNNScheduler<net_t>::push_residual(unsigned int filter_size,
                                           unsigned int channels,
                                           unsigned int outputs,
                                           const std::vector<float>& weights_1,
                                           const std::vector<float>& means_1,
                                           const std::vector<float>& variances_1,
                                           const std::vector<float>& weights_2,
                                           const std::vector<float>& means_2,
                                           const std::vector<float>& variances_2) {
    for (const auto& cudnn_net : m_networks) {
        std::vector<float> weights_1_conv = std::vector<float>(weights_1);
        std::vector<float> means_1_conv = std::vector<float>(means_1);
        std::vector<float> weights_2_conv = std::vector<float>(weights_2);
        std::vector<float> means_2_conv = std::vector<float>(means_2);

        bn_stddivs_to_conv(weights_1_conv,
                           variances_1,
                           means_1_conv,
                           outputs, channels);

        bn_stddivs_to_conv(weights_2_conv,
                           variances_2,
                           means_2_conv,
                           outputs, channels);

        /* Convolution alpha */
        float scale_1 = 1.0f;
        float scale_2 = 1.0f;
        /* Residual add alpha */
        float scale_3 = 1.0f;

        if (typeid(net_t) == typeid(int8_t)) {
            scale_1 = 127.0f/abs_max(weights_1_conv);
            scale_2 = 127.0f/abs_max(weights_2_conv);
            weights_1_conv = scale_weights(weights_1_conv, scale_1, true);
            weights_2_conv = scale_weights(weights_2_conv, scale_2, true);

            weights_1_conv = KCRS_to_KRSC<float>(weights_1_conv,
                                                 outputs,
                                                 channels,
                                                 filter_size,
                                                 filter_size);

            weights_2_conv = KCRS_to_KRSC<float>(weights_2_conv,
                                                 outputs,
                                                 channels,
                                                 filter_size,
                                                 filter_size);

            means_1_conv = scale_weights(means_1_conv, input_scale_int8, false);
            means_2_conv = scale_weights(means_2_conv, input_scale_int8, false);
        }

        cudnn_net->push_residual(filter_size, channels, outputs,
                                  weights_1_conv,
                                  means_1_conv,
                                  weights_2_conv,
                                  means_2_conv,
                                  scale_1,
                                  scale_2,
                                  scale_3);
    }
}

template <typename net_t>
void CuDNNScheduler<net_t>::push_convolve(unsigned int filter_size,
                                           unsigned int channels,
                                           unsigned int outputs,
                                           const std::vector<float>& weights) {
    for (const auto & cudnn_net : m_networks) {
        cudnn_net->push_convolve(filter_size, channels, outputs, weights);
    }
}

template <typename net_t>
void CuDNNScheduler<net_t>::set_scales(const std::vector<float>& activations, const float activation_scale) {
    for (const auto & cudnn_net : m_networks) {
        cudnn_net->set_scales(activations, activation_scale);
    }
}

template <typename net_t>
void CuDNNScheduler<net_t>::activations(const std::vector<float>& input,
                                        Activations<float>& activations,
                                        std::vector<float>& output_pol,
                                        std::vector<float>& output_val) {
    std::shared_ptr<ContextPoolEntry> ctx;
    auto queue_num = size_t{0};
    {
        LOCK(m_context_pool_mutex, lock);
        while (queue_num < m_context_pool.size()) {
            if (!m_context_pool[queue_num].empty()) {
                ctx = std::move(m_context_pool[queue_num].front());
                m_context_pool[queue_num].pop_front();
                break;
            }
            queue_num++;
        }
        // if this failed, it means we ran out of contexts
        // which should be more than or equal to the number of threads
        assert(ctx != nullptr);
    }

    m_networks[ctx->net_index]->forward_activations(input, output_pol, output_val, ctx->context, &activations, 1);

    {
        LOCK(m_context_pool_mutex, lock);
        m_context_pool[queue_num].push_back(std::move(ctx));
    }
}

template <typename net_t>
void CuDNNScheduler<net_t>::forward(const std::vector<float>& input,
                                     std::vector<float>& output_pol,
                                     std::vector<float>& output_val) {
    std::shared_ptr<ContextPoolEntry> ctx;
    auto queue_num = size_t{0};
    {
        LOCK(m_context_pool_mutex, lock);
        while (queue_num < m_context_pool.size()) {
            if (!m_context_pool[queue_num].empty()) {
                ctx = std::move(m_context_pool[queue_num].front());
                m_context_pool[queue_num].pop_front();
                break;
            }
            queue_num++;
        }
        // if this failed, it means we ran out of contexts
        // which should be more than or equal to the number of threads
        assert(ctx != nullptr);
    }

    m_networks[ctx->net_index]->forward(input, output_pol, output_val, ctx->context);

    {
        LOCK(m_context_pool_mutex, lock);
        m_context_pool[queue_num].push_back(std::move(ctx));
    }
}

template class CuDNNScheduler<float>;
#ifdef USE_HALF
template class CuDNNScheduler<half_float::half>;
#endif
template class CuDNNScheduler<int8_t>;

#endif
