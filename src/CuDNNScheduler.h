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

#ifndef CUDNNSCHEDULER_H_INCLUDED
#define CUDNNSCHEDULER_H_INCLUDED
#include "config.h"

#include <list>
#include <vector>

#include "SMP.h"
#include "ForwardPipe.h"
#include "CuDNN.h"
#include "ThreadPool.h"


template <typename net_t>
class CuDNNScheduler : public ForwardPipe {
    class ContextPoolEntry {
    public:
        size_t net_index;
        CuDNNContext context;
        ContextPoolEntry(size_t index) : net_index(index) {}
    };
    class ForwardQueueEntry {
    public:
      std::mutex mutex;
      std::condition_variable cv;
      const std::vector<float>& in;
      std::vector<float>& out_p;
      std::vector<float>& out_v;
      ForwardQueueEntry(const std::vector<float>& input,
                        std::vector<float>& output_pol,
                        std::vector<float>& output_val) : in(input), out_p(output_pol), out_v(output_val)
        {}
    };
public:
    virtual void initialize(const int channels);
    virtual void forward(const std::vector<float>& input,
                         std::vector<float>& output_pol,
                         std::vector<float>& output_val);

    void activations(const std::vector<float>& input,
                     Activations<float>& activations,
                     std::vector<float>& output_pol,
                     std::vector<float>& output_val);

    virtual void push_input_convolution(unsigned int filter_size,
                                        unsigned int channels,
                                        unsigned int outputs,
                                        const std::vector<float>& weights,
                                        const std::vector<float>& means,
                                        const std::vector<float>& variances);

    virtual void push_residual(unsigned int filter_size,
                               unsigned int channels,
                               unsigned int outputs,
                               const std::vector<float>& weights_1,
                               const std::vector<float>& means_1,
                               const std::vector<float>& variances_1,
                               const std::vector<float>& weights_2,
                               const std::vector<float>& means_2,
                               const std::vector<float>& variances_2);

    virtual void push_convolve(unsigned int filter_size,
                               unsigned int channels,
                               unsigned int outputs,
                               const std::vector<float>& weights);

    void set_scales(const std::vector<float>& activations, const float activation_scale);

private:
    bool m_running = true;
    std::vector<std::vector<std::unique_ptr<CuDNN_Network<net_t>>>> m_networks;
    std::vector<std::vector<std::unique_ptr<CuDNN<net_t>>>> m_cudnn;

    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::list<std::shared_ptr<ForwardQueueEntry>> m_forward_queue;
    std::list<std::thread> m_worker_threads;

    void batch_worker(const size_t gnum);
};

#endif
