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
#include <iterator>
#include <limits>
#include <stdexcept>

#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <zlib.h>

#include "CuDNN.h"
#include "Network.h"
#include "GTP.h"
#include "Utils.h"

using namespace Utils;

static int gcd(int a, int b) {
    int t;
    while ( b != 0) {
        t = b;
        b = a % b;
        a = t;
    }
    return a;
}

static void add_to_hist(std::vector<int> &hist, std::vector<float> &data, int hist_max) {
    const auto bins = hist.size();

    for (auto i = size_t{0}; i < data.size(); i++) {
        auto x = data[i];
        if (x >= hist_max) {
            hist[bins - 1]++;
            continue;
        }
        if (x < 0.0f) {
            throw std::runtime_error("Got negative activation in calibration\n");
        }
        size_t bin = std::round(bins * x / hist_max);
        hist[bin]++;
    }
}

static std::vector<float> quantize_distribution(std::vector<float> hist, size_t q_size) {
    auto hist_size = hist.size();
    if (q_size == hist_size) {
        return hist;
    }

    std::vector<float> q(hist_size);

    /* Handle fractional bins by first upsampling by repeating values such that
     * we can divide it equally */
    auto g = gcd(hist_size, q_size);
    auto upsample = size_t(q_size / g);
    auto downsample = size_t((upsample * hist_size) / q_size);

    std::vector<float> up(upsample * hist_size);
    auto idx = size_t{0};
    for (auto i = size_t{0}; i < hist_size; i++) {
        for (auto j = size_t{0}; j < upsample; j++) {
            up[idx++] = hist[i];
        }
    }

    auto acc = 0.0f;

    idx = 0;

    std::vector<float> down(q_size);
    for (auto i = size_t{0}; i < upsample * hist_size; i++) {
        acc += up[i];
        if (i % downsample == downsample - 1){
            down[idx++] = acc / downsample;
            acc = 0.0f;
        }
    }

    /* Now we have downsampled to q_size array, expand back to original size */
    std::swap(upsample, downsample);

    std::vector<float> up2(upsample * q_size);
    idx = 0;
    for (auto i = size_t{0}; i < q_size; i++) {
        for (auto j = size_t{0}; j < upsample; j++) {
            up2[idx++] = down[i];
        }
    }

    acc = 0.0f;
    idx = 0;
    for (auto i = size_t{0}; i < upsample * q_size; i++) {
        acc += up2[i];
        if (i % downsample == downsample - 1){
            q[idx++] = acc / downsample;
            acc = 0.0f;
        }
    }

    /* TODO: Better hole filling */
    for (auto i = size_t{0}; i < hist_size; i++) {
        if (hist[i] == 0.0f) {
            q[i] = 0.0f;
        }
    }

    return q;
}

static float kl_divergence(std::vector<float> p, std::vector<float> q) {
    auto kl = 0.0f;
    for (auto i = size_t{0}; i < p.size(); i++) {
        if (p[i] > 1e-10) {
            kl += -p[i]*std::log(q[i]/p[i]);
        }
    }
    return kl;
}

static float entropy_calibration(std::vector<int> &hist, int hist_max, float outlier_weight) {
    auto constexpr ref_size = size_t{128};

    if (hist.size() < ref_size) {
        throw std::runtime_error("Histogram size is too small.");
    }
    auto min_i = -1;
    auto min_kl = 1e9;

    /* Zero bin messes everything up so get rid of it. */
    hist[0] = hist[1];

    for (auto i = ref_size; i < hist.size(); i+=5) {
        std::vector<float> reference(i);
        std::vector<float> candidate(i);

        auto sum = 0.0f;
        for (auto j = size_t{0}; j < i; j++) {
            reference[j] = hist[j] + 0.5f;
            candidate[j] = hist[j] + 0.5f;
            sum += hist[j];
        }
        auto outliers = 0;
        for (auto j = size_t{i}; j < hist.size(); j++) {
            outliers += hist[j];
        }
        outliers *= outlier_weight;

        reference[i - 1] += outliers;
        sum += outliers;
        for (auto j = size_t{0}; j < i; j++) {
            reference[j] /= sum;
            candidate[j] /= (sum - outliers);
        }
        auto q_candidate = quantize_distribution(candidate, ref_size);

        sum = 0.0f;
        for (auto j = size_t{0}; j < i; j++) {
            sum += q_candidate[j];
        }
        for (auto j = size_t{0}; j < i; j++) {
            q_candidate[j] /= sum;
        }
        auto kl = kl_divergence(reference, q_candidate);
        if (min_i == -1 || kl < min_kl) {
            min_i = i;
            min_kl = kl;
        }
    }
    return (min_i + 0.5f) * float(hist_max) / float(hist.size());
}

static int hex_to_int(const unsigned char c) {
    switch (c) {
        case '0': return 0;
        case '1': return 1;
        case '2': return 2;
        case '3': return 3;
        case '4': return 4;
        case '5': return 5;
        case '6': return 6;
        case '7': return 7;
        case '8': return 8;
        case '9': return 9;
        case 'a': return 10;
        case 'b': return 11;
        case 'c': return 12;
        case 'd': return 13;
        case 'e': return 14;
        case 'f': return 15;
        default: throw std::runtime_error("Invalid hex digit");
    }
}

std::vector<float> get_activations(CuDNNScheduler<float> &scheduler) {
    GameState state;
    state.init_game(BOARD_SIZE, 7.5);

    std::vector<float> policy_data(Network::OUTPUTS_POLICY * BOARD_SQUARES);
    std::vector<float> value_data(Network::OUTPUTS_VALUE * BOARD_SQUARES);

    std::vector<float> activations_acc;

    auto constexpr bins = 2048;

    std::vector<std::vector<int>> histogram;
    std::vector<float> hist_max;

    std::string cal_filename = "int8_cal.gz";
    auto gzhandle = gzopen(cal_filename.c_str(), "rb");
    if (gzhandle == nullptr) {
        myprintf("Could not open calibration file: %s\n", cal_filename.c_str());
        return {0, 0};
    }

    // Stream the gz file in to a memory buffer stream.
    auto buffer = std::stringstream{};
    constexpr auto chunkBufferSize = 64 * 1024;
    std::vector<char> chunkBuffer(chunkBufferSize);
    while (true) {
        auto bytesRead = gzread(gzhandle, chunkBuffer.data(), chunkBufferSize);
        if (bytesRead == 0) break;
        if (bytesRead < 0) {
            myprintf("Failed to decompress or read: %s\n", cal_filename.c_str());
            gzclose(gzhandle);
            return {0, 0};
        }
        assert(bytesRead <= chunkBufferSize);
        buffer.write(chunkBuffer.data(), bytesRead);
    }
    gzclose(gzhandle);

    auto line = std::string{};

    std::vector<std::vector<float>> planes;

    auto n = 0;
    std::vector<float> plane(BOARD_SQUARES * 18);

    auto p = 0;

    /* Examples to skip */
    auto skip = 11;

    /* Skip counter */
    auto skip_n = 0;

    while (std::getline(buffer, line)) {
        if ( n % 19 > 16) {
            /* Skip policy vector and winner */
            n++;
            p = 0;
            continue;
        }
        for (auto i = size_t{0}; i < line.size(); i++) {
            auto h = hex_to_int(line[i]);
            /* History planes */
            if ( n % 19 < 16 ) {
                if (i == BOARD_SQUARES/4 - 1) {
                    plane[p++] = bool(h & (1 << 0));
                } else {
                    plane[p++] = bool(h & (1 << 3));
                    plane[p++] = bool(h & (1 << 2));
                    plane[p++] = bool(h & (1 << 1));
                    plane[p++] = bool(h & (1 << 0));
                }
                /* Last digit has only one bit */
            } else if ( n % 19 == 16 ) {
                /* To move planes */
                for (auto j = 0; j < BOARD_SQUARES; j++) {
                    plane[p++] = !bool(h);
                }
                for (auto j = 0; j < BOARD_SQUARES; j++) {
                    plane[p++] = bool(h);
                }
                if (skip_n == skip) {
                    planes.emplace_back(plane);
                    skip_n = 0;
                } else {
                    skip_n++;
                }
                break;
            }
        }

        n++;
    }

    myprintf("%d calibration examples\n", planes.size());

    for (auto && input_data: planes) {

        Activations<float> activations;
        scheduler.activations(input_data, activations, policy_data, value_data);

        if (activations_acc.size() != activations.size()) {
            activations_acc.resize(activations.size());
            histogram.resize(activations.size());
            hist_max.resize(activations.size());
            for (auto i = size_t{0}; i < histogram.size(); i++) {
                histogram[i].resize(bins);
            }
        }

        float max = 0.0f;
        for (auto i = size_t{0}; i < activations.size(); i++) {
            for (auto j = size_t{0}; j < activations[i].size(); j++) {
                float a = activations[i][j];
                max = std::max(max, a);
            }
            if (hist_max[i] == 0.0f) {
                hist_max[i] = 2.0f * max;
            }
            max = 0.0f;
            add_to_hist(histogram[i], activations[i], hist_max[i]);
        }
    }

    myprintf("max %f\n", hist_max[0]);
    for (auto i = 0; i < bins; i++) {
        myprintf("%d ", histogram[0][i]);
    }
    myprintf("\n");

    myprintf("max %f\n", hist_max[hist_max.size()-2]);
    for (auto i = 0; i < bins; i++) {
        myprintf("%d ", histogram[hist_max.size()-2][i]);
    }
    myprintf("\n");

    myprintf("max %f\n", hist_max[hist_max.size()-1]);
    for (auto i = 0; i < bins; i++) {
        myprintf("%d ", histogram[hist_max.size()-1][i]);
    }
    myprintf("\n");

    myprintf("Ths: \n");

    // Extra importance to assign for saturating activations
    auto outlier_weight = 3.0f;
    for (auto i = size_t{0}; i < histogram.size(); i++) {
        auto th = entropy_calibration(histogram[i], hist_max[i], outlier_weight);
        activations_acc[i] = th;
        //activations_acc[i] = hist_max[i];
        myprintf("%.2f, ", th);
    }
    myprintf("\n");

    return activations_acc;
}

#endif
