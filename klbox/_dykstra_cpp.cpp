#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double kTinyPos = std::numeric_limits<double>::min();

struct DykstraResult {
    std::vector<double> p_star;
    int cycles;
    double final_V;
    double elapsed_s;
};

std::vector<double> to_double_vector(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& array
) {
    auto buf = array.request();
    if (buf.ndim != 1) {
        throw std::invalid_argument("expected a 1D float64 array");
    }
    const auto* ptr = static_cast<const double*>(buf.ptr);
    return std::vector<double>(ptr, ptr + buf.shape[0]);
}

std::vector<int> to_int_vector(
    const py::array_t<int, py::array::c_style | py::array::forcecast>& array
) {
    auto buf = array.request();
    if (buf.ndim != 1) {
        throw std::invalid_argument("expected a 1D int array");
    }
    const auto* ptr = static_cast<const int*>(buf.ptr);
    return std::vector<int>(ptr, ptr + buf.shape[0]);
}

double l1_norm(const std::vector<double>& x) {
    double max_value = -std::numeric_limits<double>::infinity();
    for (double value : x) {
        max_value = std::max(max_value, value);
    }

    if (!std::isfinite(max_value) || max_value <= 0.0) {
        double sum = 0.0;
        for (double value : x) {
            sum += value;
        }
        return sum;
    }

    double scaled_sum = 0.0;
    for (double value : x) {
        scaled_sum += value / max_value;
    }
    return max_value * scaled_sum;
}

void normalize_to_simplex_inplace(std::vector<double>& x) {
    const double s = l1_norm(x);
    if (!std::isfinite(s) || s <= 0.0) {
        throw std::runtime_error("normalize_to_simplex: sum must be positive and finite");
    }
    for (double& value : x) {
        value /= s;
    }
}

std::vector<double> reorder_by_sigma(const std::vector<double>& x, const std::vector<int>& sigma) {
    if (x.size() != sigma.size()) {
        throw std::invalid_argument("reorder_by_sigma: x and sigma must have the same length");
    }

    std::vector<double> out(x.size());
    std::vector<bool> seen(x.size(), false);
    for (std::size_t r = 0; r < sigma.size(); ++r) {
        const int idx = sigma[r];
        if (idx < 0 || static_cast<std::size_t>(idx) >= x.size()) {
            throw std::invalid_argument("sigma contains an out-of-range index");
        }
        if (seen[static_cast<std::size_t>(idx)]) {
            throw std::invalid_argument("sigma contains a duplicate index");
        }
        seen[static_cast<std::size_t>(idx)] = true;
        out[r] = x[static_cast<std::size_t>(idx)];
    }
    return out;
}

std::vector<double> undo_sigma_order(const std::vector<double>& ordered, const std::vector<int>& sigma) {
    if (ordered.size() != sigma.size()) {
        throw std::invalid_argument("undo_sigma_order: ordered and sigma must have the same length");
    }
    std::vector<double> out(ordered.size());
    for (std::size_t r = 0; r < sigma.size(); ++r) {
        out[static_cast<std::size_t>(sigma[r])] = ordered[r];
    }
    return out;
}

void normalize_clipped_copy(const std::vector<double>& z, std::vector<double>& out, double clip_eps) {
    out.resize(z.size());
    for (std::size_t i = 0; i < z.size(); ++i) {
        out[i] = std::max(z[i], clip_eps);
    }
    normalize_to_simplex_inplace(out);
}

void project_prefix_constraint(
    const std::vector<double>& z,
    std::size_t prefix_len,
    double b,
    double clip_eps,
    std::vector<double>& out
) {
    normalize_clipped_copy(z, out, clip_eps);

    double rho = 0.0;
    for (std::size_t i = 0; i < prefix_len; ++i) {
        rho += out[i];
    }
    if (rho >= b) {
        return;
    }

    const double denom_in = std::max(rho, kTinyPos);
    const double denom_out = std::max(1.0 - rho, kTinyPos);
    const double scale_in = b / denom_in;
    const double scale_out = (1.0 - b) / denom_out;

    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] *= (i < prefix_len) ? scale_in : scale_out;
    }
}

void project_gap_constraint(
    const std::vector<double>& z,
    int i,
    int j,
    double delta,
    double clip_eps,
    std::vector<double>& out,
    double feasibility_tol = 0.0
) {
    normalize_clipped_copy(z, out, clip_eps);

    const double s = out[static_cast<std::size_t>(i)] - out[static_cast<std::size_t>(j)];
    if (s >= delta - feasibility_tol) {
        return;
    }

    const double zi = out[static_cast<std::size_t>(i)];
    const double zj = out[static_cast<std::size_t>(j)];
    const double u = 1.0 - zi - zj;

    const double A = zi * (1.0 - delta);
    const double B = -delta * u;
    const double C = -zj * (1.0 + delta);
    if (A <= 0.0) {
        return;
    }

    double disc = B * B - 4.0 * A * C;
    disc = std::max(disc, 0.0);

    const double E = (-B + std::sqrt(disc)) / (2.0 * A);
    if (!std::isfinite(E) || E <= 0.0) {
        return;
    }

    const double D = zi * E + zj / E + u;
    if (!std::isfinite(D) || D <= 0.0) {
        return;
    }

    for (double& value : out) {
        value /= D;
    }
    out[static_cast<std::size_t>(i)] = (E / D) * zi;
    out[static_cast<std::size_t>(j)] = (1.0 / (E * D)) * zj;
}

double violation_from_sigma_order(
    const std::vector<double>& p_ordered,
    const std::vector<double>& tilde_pi,
    const std::vector<double>& underline,
    const std::vector<double>& overline,
    bool include_prefix_constraints,
    bool include_lower_constraints,
    bool include_upper_constraints
) {
    const std::size_t n = p_ordered.size();
    double violation = 0.0;

    if (include_prefix_constraints) {
        double prefix_sum = 0.0;
        for (std::size_t s = 1; s < n; ++s) {
            prefix_sum += p_ordered[s - 1];
            const double b_pref = 1.0 - tilde_pi[s];
            violation = std::max(violation, std::max(b_pref - prefix_sum, 0.0));
        }
    }

    if (include_lower_constraints || include_upper_constraints) {
        for (std::size_t r = 0; r + 1 < n; ++r) {
            const double diff = p_ordered[r] - p_ordered[r + 1];
            if (include_lower_constraints) {
                violation = std::max(violation, std::max(underline[r] - diff, 0.0));
            }
            if (include_upper_constraints) {
                violation = std::max(violation, std::max(diff - overline[r], 0.0));
            }
        }
    }

    return violation;
}

DykstraResult dykstra_kl_project_cpp_impl(
    const std::vector<double>& q,
    const std::vector<int>& sigma,
    const std::vector<double>& tilde_pi,
    const std::vector<double>& underline,
    const std::vector<double>& overline,
    double tau,
    int K_max,
    double log_clip_eps,
    bool include_prefix_constraints,
    bool include_lower_constraints,
    bool include_upper_constraints
) {
    const std::size_t n = q.size();
    if (sigma.size() != n || tilde_pi.size() != n) {
        throw std::invalid_argument("q, sigma, and tilde_pi must have the same length");
    }
    if (underline.size() + 1 != n || overline.size() + 1 != n) {
        throw std::invalid_argument("underline and overline must have length n - 1");
    }
    if (K_max < 0) {
        throw std::invalid_argument("K_max must be non-negative");
    }

    std::vector<double> q_ordered = reorder_by_sigma(q, sigma);
    for (double& value : q_ordered) {
        value = std::max(value, log_clip_eps);
    }
    normalize_to_simplex_inplace(q_ordered);

    const std::size_t n_minus_1 = (n > 0 ? (n - 1) : 0);
    const std::size_t m =
        (include_prefix_constraints ? n_minus_1 : 0) +
        (include_lower_constraints ? n_minus_1 : 0) +
        (include_upper_constraints ? n_minus_1 : 0);

    const std::size_t prefix_end = include_prefix_constraints ? n_minus_1 : 0;
    const std::size_t lower_begin = prefix_end;
    const std::size_t lower_end = lower_begin + (include_lower_constraints ? n_minus_1 : 0);
    const std::size_t upper_begin = lower_end;
    const std::size_t upper_end = upper_begin + (include_upper_constraints ? n_minus_1 : 0);

    std::vector<double> z_prev = q_ordered;
    std::vector<double> z_new(n, 0.0);
    std::vector<double> u(n, 0.0);
    std::vector<double> d_ring(m * n, 0.0);

    const auto t0 = std::chrono::steady_clock::now();
    if (m == 0) {
        const auto t1 = std::chrono::steady_clock::now();
        return DykstraResult{
            undo_sigma_order(z_prev, sigma),
            0,
            0.0,
            std::chrono::duration<double>(t1 - t0).count(),
        };
    }

    for (int cycle = 1; cycle <= K_max; ++cycle) {
        for (std::size_t h = 0; h < m; ++h) {
            const double* d_row = d_ring.data() + h * n;

            double max_log_u = -std::numeric_limits<double>::infinity();
            for (std::size_t k = 0; k < n; ++k) {
                const double z_clip = std::max(z_prev[k], log_clip_eps);
                const double value = std::log(z_clip) + d_row[k];
                u[k] = value;
                max_log_u = std::max(max_log_u, value);
            }
            for (double& value : u) {
                value = std::exp(value - max_log_u);
            }

            if (h < prefix_end) {
                const std::size_t prefix_len = h + 1;
                const double b_pref = 1.0 - tilde_pi[prefix_len];
                project_prefix_constraint(u, prefix_len, b_pref, kTinyPos, z_new);
            } else if (h < lower_end) {
                const std::size_t r = h - lower_begin;
                project_gap_constraint(u, static_cast<int>(r), static_cast<int>(r + 1), underline[r], kTinyPos, z_new);
            } else if (h < upper_end) {
                const std::size_t r = h - upper_begin;
                project_gap_constraint(u, static_cast<int>(r + 1), static_cast<int>(r), -overline[r], kTinyPos, z_new);
            } else {
                throw std::logic_error("invalid constraint index in C++ Dykstra loop");
            }

            double* d_row_mut = d_ring.data() + h * n;
            for (std::size_t k = 0; k < n; ++k) {
                const double z_prev_clip = std::max(z_prev[k], log_clip_eps);
                const double z_new_clip = std::max(z_new[k], log_clip_eps);
                d_row_mut[k] += std::log(z_prev_clip / z_new_clip);
            }

            for (double value : z_new) {
                if (!std::isfinite(value)) {
                    throw std::runtime_error("non-finite z_new encountered in C++ Dykstra loop");
                }
            }
            for (std::size_t k = 0; k < n; ++k) {
                if (!std::isfinite(d_row_mut[k])) {
                    throw std::runtime_error("non-finite dual state encountered in C++ Dykstra loop");
                }
            }

            z_prev.swap(z_new);
        }

        const double V = violation_from_sigma_order(
            z_prev,
            tilde_pi,
            underline,
            overline,
            include_prefix_constraints,
            include_lower_constraints,
            include_upper_constraints
        );
        if (V <= tau) {
            const auto t1 = std::chrono::steady_clock::now();
            return DykstraResult{
                undo_sigma_order(z_prev, sigma),
                cycle,
                V,
                std::chrono::duration<double>(t1 - t0).count(),
            };
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double V = violation_from_sigma_order(
        z_prev,
        tilde_pi,
        underline,
        overline,
        include_prefix_constraints,
        include_lower_constraints,
        include_upper_constraints
    );
    return DykstraResult{
        undo_sigma_order(z_prev, sigma),
        K_max,
        V,
        std::chrono::duration<double>(t1 - t0).count(),
    };
}

py::dict result_to_dict(const DykstraResult& result) {
    py::array_t<double> p_star(result.p_star.size());
    auto view = p_star.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(result.p_star.size()); ++i) {
        view(i) = result.p_star[static_cast<std::size_t>(i)];
    }

    py::dict out;
    out["p_star"] = std::move(p_star);
    out["cycles"] = result.cycles;
    out["final_V"] = result.final_V;
    out["elapsed_s"] = result.elapsed_s;
    return out;
}

int resolve_n_threads(int n_threads, std::size_t batch_size) {
    if (batch_size == 0) {
        return 1;
    }
    if (n_threads <= 0) {
        const unsigned hw = std::thread::hardware_concurrency();
        n_threads = static_cast<int>(hw == 0 ? 1 : hw);
    }
    n_threads = std::max(n_threads, 1);
    n_threads = std::min<int>(n_threads, static_cast<int>(batch_size));
    return n_threads;
}

py::dict dykstra_kl_project_cpp_batch_raw(
    py::array_t<double, py::array::c_style | py::array::forcecast> q_batch,
    py::array_t<int, py::array::c_style | py::array::forcecast> sigma_batch,
    py::array_t<double, py::array::c_style | py::array::forcecast> tilde_pi_batch,
    py::array_t<double, py::array::c_style | py::array::forcecast> underline_batch,
    py::array_t<double, py::array::c_style | py::array::forcecast> overline_batch,
    double tau,
    int K_max,
    double log_clip_eps,
    int n_threads,
    bool include_prefix_constraints,
    bool include_lower_constraints,
    bool include_upper_constraints
) {
    auto q_buf = q_batch.request();
    auto sigma_buf = sigma_batch.request();
    auto tilde_buf = tilde_pi_batch.request();
    auto under_buf = underline_batch.request();
    auto over_buf = overline_batch.request();

    if (q_buf.ndim != 2 || sigma_buf.ndim != 2 || tilde_buf.ndim != 2 ||
        under_buf.ndim != 2 || over_buf.ndim != 2) {
        throw std::invalid_argument(
            "batch projection expects 2D arrays for q, sigma, tilde_pi, underline, and overline"
        );
    }

    const std::size_t batch = static_cast<std::size_t>(q_buf.shape[0]);
    const std::size_t n = static_cast<std::size_t>(q_buf.shape[1]);
    const std::size_t n_minus_1 = n > 0 ? n - 1 : 0;

    if (static_cast<std::size_t>(sigma_buf.shape[0]) != batch ||
        static_cast<std::size_t>(tilde_buf.shape[0]) != batch ||
        static_cast<std::size_t>(under_buf.shape[0]) != batch ||
        static_cast<std::size_t>(over_buf.shape[0]) != batch) {
        throw std::invalid_argument("batch projection arrays must share the same batch dimension");
    }
    if (static_cast<std::size_t>(sigma_buf.shape[1]) != n ||
        static_cast<std::size_t>(tilde_buf.shape[1]) != n) {
        throw std::invalid_argument("sigma and tilde_pi batch arrays must have shape [batch, n]");
    }
    if (static_cast<std::size_t>(under_buf.shape[1]) != n_minus_1 ||
        static_cast<std::size_t>(over_buf.shape[1]) != n_minus_1) {
        throw std::invalid_argument("underline and overline batch arrays must have shape [batch, n-1]");
    }

    const auto* q_ptr = static_cast<const double*>(q_buf.ptr);
    const auto* sigma_ptr = static_cast<const int*>(sigma_buf.ptr);
    const auto* tilde_ptr = static_cast<const double*>(tilde_buf.ptr);
    const auto* under_ptr = static_cast<const double*>(under_buf.ptr);
    const auto* over_ptr = static_cast<const double*>(over_buf.ptr);

    std::vector<DykstraResult> results(batch);
    std::exception_ptr worker_error = nullptr;
    std::mutex worker_error_mutex;

    {
        py::gil_scoped_release release;

        const int threads_to_use = resolve_n_threads(n_threads, batch);
        std::vector<std::thread> workers;
        workers.reserve(static_cast<std::size_t>(threads_to_use));

        auto worker = [&](int thread_id) {
            try {
                for (std::size_t b = static_cast<std::size_t>(thread_id);
                     b < batch;
                     b += static_cast<std::size_t>(threads_to_use)) {
                    std::vector<double> q_row(q_ptr + b * n, q_ptr + (b + 1) * n);
                    std::vector<int> sigma_row(sigma_ptr + b * n, sigma_ptr + (b + 1) * n);
                    std::vector<double> tilde_row(tilde_ptr + b * n, tilde_ptr + (b + 1) * n);
                    std::vector<double> under_row(under_ptr + b * n_minus_1, under_ptr + (b + 1) * n_minus_1);
                    std::vector<double> over_row(over_ptr + b * n_minus_1, over_ptr + (b + 1) * n_minus_1);

                    results[b] = dykstra_kl_project_cpp_impl(
                        q_row,
                        sigma_row,
                        tilde_row,
                        under_row,
                        over_row,
                        tau,
                        K_max,
                        log_clip_eps,
                        include_prefix_constraints,
                        include_lower_constraints,
                        include_upper_constraints
                    );
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(worker_error_mutex);
                if (!worker_error) {
                    worker_error = std::current_exception();
                }
            }
        };

        for (int t = 0; t < threads_to_use; ++t) {
            workers.emplace_back(worker, t);
        }
        for (auto& worker_thread : workers) {
            worker_thread.join();
        }
    }

    if (worker_error) {
        std::rethrow_exception(worker_error);
    }

    py::array_t<double> p_star_batch({static_cast<py::ssize_t>(batch), static_cast<py::ssize_t>(n)});
    py::array_t<int> cycles_batch(static_cast<py::ssize_t>(batch));
    py::array_t<double> final_v_batch(static_cast<py::ssize_t>(batch));
    py::array_t<double> elapsed_batch(static_cast<py::ssize_t>(batch));

    auto p_star_view = p_star_batch.mutable_unchecked<2>();
    auto cycles_view = cycles_batch.mutable_unchecked<1>();
    auto final_v_view = final_v_batch.mutable_unchecked<1>();
    auto elapsed_view = elapsed_batch.mutable_unchecked<1>();

    for (py::ssize_t b = 0; b < static_cast<py::ssize_t>(batch); ++b) {
        const auto& result = results[static_cast<std::size_t>(b)];
        for (py::ssize_t j = 0; j < static_cast<py::ssize_t>(n); ++j) {
            p_star_view(b, j) = result.p_star[static_cast<std::size_t>(j)];
        }
        cycles_view(b) = result.cycles;
        final_v_view(b) = result.final_V;
        elapsed_view(b) = result.elapsed_s;
    }

    py::dict out;
    out["p_star"] = std::move(p_star_batch);
    out["cycles"] = std::move(cycles_batch);
    out["final_V"] = std::move(final_v_batch);
    out["elapsed_s"] = std::move(elapsed_batch);
    out["n_threads"] = resolve_n_threads(n_threads, batch);
    return out;
}



py::dict dykstra_kl_project_cpp_raw(
    py::array_t<double, py::array::c_style | py::array::forcecast> q,
    py::array_t<int, py::array::c_style | py::array::forcecast> sigma,
    py::array_t<double, py::array::c_style | py::array::forcecast> tilde_pi,
    py::array_t<double, py::array::c_style | py::array::forcecast> underline,
    py::array_t<double, py::array::c_style | py::array::forcecast> overline,
    double tau,
    int K_max,
    double log_clip_eps,
    bool include_prefix_constraints,
    bool include_lower_constraints,
    bool include_upper_constraints
) {
    const std::vector<double> q_vec = to_double_vector(q);
    const std::vector<int> sigma_vec = to_int_vector(sigma);
    const std::vector<double> tilde_vec = to_double_vector(tilde_pi);
    const std::vector<double> under_vec = to_double_vector(underline);
    const std::vector<double> over_vec = to_double_vector(overline);

    DykstraResult result;
    {
        py::gil_scoped_release release;
        result = dykstra_kl_project_cpp_impl(
            q_vec,
            sigma_vec,
            tilde_vec,
            under_vec,
            over_vec,
            tau,
            K_max,
            log_clip_eps,
            include_prefix_constraints,
            include_lower_constraints,
            include_upper_constraints
        );
    }

    return result_to_dict(result);
}


}  // namespace


PYBIND11_MODULE(_dykstra_cpp, m) {
    m.doc() = "C++ pybind11 implementation of the KL-Dykstra hot loop for klbox.";

    m.def(
        "dykstra_kl_project_cpp_raw",
        &dykstra_kl_project_cpp_raw,
        py::arg("q"),
        py::arg("sigma"),
        py::arg("tilde_pi"),
        py::arg("underline"),
        py::arg("overline"),
        py::arg("tau"),
        py::arg("K_max"),
        py::arg("log_clip_eps"),
        py::arg("include_prefix_constraints") = true,
        py::arg("include_lower_constraints") = true,
        py::arg("include_upper_constraints") = true,
        R"pbdoc(
Compute one KL-Dykstra projection in C++.
)pbdoc"
    );

    m.def(
        "dykstra_kl_project_cpp_batch_raw",
        &dykstra_kl_project_cpp_batch_raw,
        py::arg("q_batch"),
        py::arg("sigma_batch"),
        py::arg("tilde_pi_batch"),
        py::arg("underline_batch"),
        py::arg("overline_batch"),
        py::arg("tau"),
        py::arg("K_max"),
        py::arg("log_clip_eps"),
        py::arg("n_threads") = 0,
        py::arg("include_prefix_constraints") = true,
        py::arg("include_lower_constraints") = true,
        py::arg("include_upper_constraints") = true,
        R"pbdoc(
Compute a batch of independent KL-Dykstra projections in C++.
Each row uses its own sigma-ordered KL-box description.
)pbdoc"
    );
}

